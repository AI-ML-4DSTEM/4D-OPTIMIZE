#!/usr/bin/env python
# coding: utf-8
# %%

# Python 2 compat?
#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

# standard libraries
import argparse
import logging
import os
import time

from typing import Dict

# 3rd party
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

try:
    import GPUtil
except ImportError:
    GPUtil = None

print('Loading ray Tune ... \n')
import ray
from ray import tune
from ray.tune import CLIReporter, Callback
from ray.tune.integration.keras import TuneReportCheckpointCallback
from ray.tune.integration.tensorflow import (DistributedTrainableCreator, get_num_workers)
from ray.tune.logger import LoggerCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.hyperopt import HyperOptSearch

print('Loading complex convolution, pooling libraries ... \n')
#Custom libraries - User defined
from conv_utils import parseDataset
from conv_utils import convSpec2d_block, conv2d_block
from conv_utils import cross_correlate_iff, cross_correlate_ff

logger = logging.getLogger(__name__)


class LRScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, alpha=10, warmup_steps=50):
        super(LRScheduler, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.alpha = alpha
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.alpha * self.d_model) * tf.math.minimum(arg1, arg2)


class TuneLoggerCallback(LoggerCallback):
    def __init__(self, filename: str = "trial_log.txt"):
        logger.info("Init LoggerCallback")

    def log_trial_start(self, trial: "Trial"):
        logger.info("Trial started: {}".format(trial))

    def log_trial_result(self, iiteration: int, trial: "Trial", result: Dict):
        logger.info("Trial Result for trial: {}, iteration: {} -- result: {}".format(trial, iteration, result))

    def log_trial_end(self, trial: "Trial", failed: bool = False):
        if failed:
            logger.warn("Trial fail - trial: {}".format(trial))
        else:
            logger.info("Trial success - trial: {}".format(trial))

# %%
#Build siamese type template matching model for Bragg peak detection
# Using same weight shareing U-net network since version 5 - three different types are - without cross-corr; cross correlation (ver 6) 
# as the input to the u-net and third (ver 7) is cross-correlate the output of u-net network

def build_generator(input_channel, filter_size, n_depth, dp_rate = 0.1, act = 'relu'):
    
    input_size = (256, 256, input_channel // 2)
    inputsA = tf.keras.Input(shape=input_size)   # input for CBED
    inputsB = tf.keras.Input(shape=input_size)   # input for probe
    skips = []

    print("Cross correlation layer ... \n")
    cc1 = tf.keras.layers.Lambda(cross_correlate_ff)([inputsA, inputsB])
    #cc1 = tf.keras.layers.BatchNormalization()(cc1)
    
    print("\n")
    print("Building the comlpex U-net ... \n")
    
    pool = convSpec2d_block(cc1, filter_size, n_depth=n_depth, dp_rate=dp_rate, activation=act) 
    
    # conv u-net for 4DSTEM cbed pattern
    for encode in range(int(np.log2(256))-1):
        conv = convSpec2d_block(pool, np.minimum(filter_size * (2**encode), 256), n_depth=n_depth, dp_rate=dp_rate, activation=act)
        pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv)
        skips.append(conv)
    
    for decode in reversed(range(int(np.log2(256))-1)):
        conv = convSpec2d_block(pool, np.minimum(filter_size * (2**(decode+1)), 256), n_depth=n_depth, dp_rate=dp_rate, activation=act)
        pool = tf.keras.layers.UpSampling2D(size=(2, 2))(conv)
        pool = tf.keras.layers.Concatenate(axis=-1)([skips[decode], pool])

    unet_final = convSpec2d_block(pool, filter_size, n_depth=n_depth, dp_rate=dp_rate, activation=act) 

    print("\n")
    print("Building the inverse fft layer ... \n")
    unet_out = tf.keras.layers.Lambda(cross_correlate_iff, dtype='float32')(unet_final)
    
    print("\n")
    print("Building the one more spatial convolution layers ... \n")
    conv_real = conv2d_block(unet_out, filter_size, n_depth=n_depth, dp_rate=dp_rate, activation=act)

    print("\n")
    print("Building the final layer ... \n")
    pred_out = tf.keras.layers.Conv2D(filters = 1, kernel_size = (3, 3),padding='same', kernel_initializer = 'he_uniform')(conv_real)
    pred_out = tf.keras.layers.Activation('linear', dtype='float32')(pred_out)
    
    model = tf.keras.Model(inputs=[inputsA, inputsB], outputs=pred_out)

    return model

# may not be needed now
def wait_for_gpu(gpu_id=None, target_util=0.001, retry = 10, delay_s=10):
    if GPUtil is None:
        raise RuntimeError(
            "GPUtil must be installed if calling `wait_for_gpu`.")
        
    if gpu_id is None:
        gpu_id_list = ray.get_gpu_ids()
        print('here are all the GPU IDs available to Ray: \n')
        print(gpu_id_list)
        print('\n')
        if not gpu_id_list:
            raise RuntimeError("No GPU ids found from `ray.get_gpu_ids()`. "
                               "Did you set Tune resources correctly?")
        gpu_id = gpu_id_list[0]

    gpu_attr = "id"
    if isinstance(gpu_id, str):
        if gpu_id.isdigit():
            # GPU ID returned from `ray.get_gpu_ids()` is a str representation
            # of the int GPU ID
            print("GPU ID {} is found \n".format(gpu_id))
            gpu_id = int(gpu_id)
        else:
            # Could not coerce gpu_id to int, so assume UUID
            # and compare against `uuid` attribute e.g.,
            # 'GPU-04546190-b68d-65ac-101b-035f8faed77d'
            print("Replacing gpu attribute to uuid \n")
            gpu_attr = "uuid"
    elif not isinstance(gpu_id, int):
        raise ValueError(f"gpu_id ({type(gpu_id)}) must be type str/int.")

    def gpu_id_fn(g):
        # Returns either `g.id` or `g.uuid` depending on
        # the format of the input `gpu_id`
        return getattr(g, gpu_attr)
    
    gpu_ids = {gpu_id_fn(g) for g in GPUtil.getGPUs()}
    if gpu_id not in gpu_ids:
        raise ValueError(
            f"{gpu_id} not found in set of available GPUs: {gpu_ids}. "
            "`wait_for_gpu` takes either GPU ordinal ID (e.g., '0') or "
            "UUID (e.g., 'GPU-04546190-b68d-65ac-101b-035f8faed77d').")

    for i in range(int(retry)):
        gpu_object = next(
            g for g in GPUtil.getGPUs() if gpu_id_fn(g) == gpu_id)
        logger.info(f"GPU MEMORY Util: {gpu_object.memoryUtil:0.3f}. ")
        if gpu_object.memoryUtil > target_util:
            print('Waiting for GPU as the memory utilization is {} \n'.format(gpu_object.memoryUtil))
            logger.info(f"Waiting for GPU util to reach {target_util}. "
                        f"Util: {gpu_object.memoryUtil:0.3f}")
            time.sleep(delay_s)
        else:
            return True
    raise RuntimeError("GPU memory was not freed.")


# %%
#template matching using Siamese identical u-net
def train_model(config, checkpoint_dir=None):
    '''
    train loop
    config: filter_size, depth, dropout, activation, alpha, batch_size, warmup, optimizer,
    '''

    print('Loading Tensorflow GPU: \n')
    # issue with ray actors with connecting to the autograph
    import tensorflow as tf

    print("train_model")

    #print('Checking if GPU is available before trial starts... \n')
    #TODO: Build a new wait for gpu
    #wait_for_gpu()

    # necessary to prevent workers from exhausting threads on the host
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    #Enable Automatic Mixed Precision
    #if args.enable_amp:
    #    print("Enabling mixed precision... \n")
    #    from tensorflow.keras import mixed_precision
    #    policy = mixed_precision.Policy('mixed_float16')
    #    mixed_precision.set_global_policy(policy)

    #Load Tensorflow XLA
    #if args.enable_xla:
    #    print("Enabling Tensorflow XLA... \n")
    #    tf.keras.backend.clear_session()
    #    tf.config.optimizer.set_jit(False) # Enable XLA.
    
    #Load TFRecords to TF batch dataset
    #TODO: Use TF Generators
    batch_size = 4
    prepTrainDataset = parseDataset(filepath = config["train_dataset"])
    train_dataset = prepTrainDataset.read(batch_size = batch_size, augment = False, mode = config["data_mode"])
    prepTestDataset = parseDataset(filepath = config["test_dataset"])
    val_dataset = prepTestDataset.read(batch_size = batch_size, augment = False, mode = config["data_mode"])

    # Disable AutoShard.
    if not args.enable_shard:
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        train_dataset = train_dataset.with_options(options)
        val_dataset = val_dataset.with_options(options)
    
    learning_rate = LRScheduler( batch_size, alpha =  config["alpha"], warmup_steps = config["warmup"])
    
    if config["optimizer"] == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=  learning_rate)
    elif config["optimizer"] == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate= learning_rate, momentum= config["momentum"])
    elif config["optimizer"] == 'adadelta':
        optimizer = tf.keras.optimizers.Adadelta(learning_rate= learning_rate)
    elif config["optimizer"] == 'lamb':
        optimizer = tfa.optimizers.LAMB(learning_rate= learning_rate)
    else:
        # TODO - set default here or before the if clause
        logger.warning("optimizer not set!")
        logger.info("Setting optimizer to default of 'adam'")
        optimizer = tf.keras.optimizers.Adam(learning_rate=  learning_rate)

    with strategy.scope():
        input_channel = 2
        model = build_generator(input_channel, config["filter_size"], config["depth"], dp_rate = config["dropout"], act = config["activation"])
        model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])

    #tboard_callback = tf.keras.callbacks.TensorBoard(log_dir='/global/cscratch1/sd/mhenders/ncem_ml_runs/', histogram_freq=1,profile_batch = '1,1000')

    print("Calling model.fit")
    EPOCHS = 10
    model.fit(
        train_dataset,
        callbacks=[TuneReportCheckpointCallback({"mean_mae": "mae"})],
        epochs=EPOCHS,
        validation_data = val_dataset)

def tune_model(args):
    print("tune model")
    print(args)

    scratch_dir = os.path.abspath(os.path.expandvars("$SCRATCH"))
    run_dir = os.path.join(scratch_dir, "ncem_ml_runs")

    num_workers = args.num_ray_hosts * args.num_ray_workers_per_host
    sched = ASHAScheduler(time_attr="training_iteration", max_t=400, grace_period=20)
    hyperopt_search = HyperOptSearch(metric="mean_mae", mode="min")
    hyperopt_search = ConcurrencyLimiter(hyperopt_search, max_concurrent=num_workers)
    # sched = ASHAScheduler(metric="mean_accuracy", mode="max")

    tf.keras.backend.clear_session()
    tf.config.run_functions_eagerly(False)
    tf.config.optimizer.set_jit(False)

    # with tf.profiler.experimental.Trace(
    #    ','.join(node_ips), '/global/cscratch1/sd/mhenders/ncem_ml_runs/', 60):
    tf_trainable = DistributedTrainableCreator(
        train_model,
        num_workers=num_workers,
        num_workers_per_host=args.num_ray_workers_per_host,
        num_gpus_per_worker=args.num_ray_gpus_per_worker,
        num_cpus_per_worker=args.num_ray_cpus_per_worker)
    analysis = tune.run(
        tf_trainable,
        verbose=2,
        name="raytune_experiment",
        local_dir=run_dir,
        max_failures=10,
        queue_trials=False,
        log_to_file=True,
        progress_reporter=CLIReporter(),
        scheduler=sched,
        search_alg=hyperopt_search,
        metric="mae",
        mode="min",
        stop={
            "mean_mae": 0.1,
            "training_iteration": args.num_training_iterations
            },
        num_samples=-1,
        # checkpoint_at_end=True,
        sync_config=tune.SyncConfig(sync_to_driver=False),
        config={
            "train_dataset": args.train_dataset,
            "test_dataset": args.test_dataset,
            "data_mode": args.data_mode,
            "filter_size": tune.randint(4, 64),
            "depth": tune.randint(2, 8),
            "dropout": tune.uniform(0.1, 0.5),
            "momentum": tune.uniform(0.1, 0.9),
            "alpha": tune.randint(1, 10000),
            "warmup": tune.randint(1, 50),
            "activation": tune.choice(["relu", "tanh"]),
            "optimizer": tune.choice(["adam", "sgd", "adadelta", "lamb"])
            },
        # resources_per_trial=tune.PlacementGroupFactory([{"CPU": args.num_ray_cpus_per_worker, "GPU": args.num_ray_gpus_per_worker}]),
        callbacks=[TuneLoggerCallback()])
    return analysis


if __name__ == "__main__":
    logging.basicConfig(level = logging.DEBUG)

    # Parse arguments usinf argparse
    parser = argparse.ArgumentParser(description='Disk detection complex CNN network',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train-dataset', type=str,
                        help='provide the data file path for trianing datasets')
    parser.add_argument('--test-dataset', type=str,
                        help='provide the data file path for validation datasets')

    parser.add_argument('--enable-shard', action='store_true', default=False,
                        help='enable sharding during training')
    parser.add_argument('--enable-amp', action='store_true', default=True,
                        help='enable automated mixed precision during training')
    parser.add_argument('--enable-xla', action='store_true', default=True,
                        help='enable tensorflow XLA during training')

    parser.add_argument('--use-checkpoint', action='store_true', default=True,
                        help='whether to use checkpoint tp load previously trained models ')

    parser.add_argument('--data-mode', type=str, default='default',
                        help='set data mode for the training to default, norm or multitask mode')

    parser.add_argument('--num-training-iterations', type=int, default=1,
                    help='Number of training iterations')

    parser.add_argument('--num-ray-hosts', type=int, default=8,
                    help='Number of workers')

    parser.add_argument('--num-ray-workers-per-host', type=int, default=8,
                    help='Number of workers per host')

    parser.add_argument('--num-ray-cpus-per-worker', type=int, default=9,
                    help='Number of cpus per worker')

    parser.add_argument('--num-ray-gpus-per-worker', type=int, default=1,
                    help='Number of gpus per worker')

    args = parser.parse_args()

    scratch_dir = os.path.abspath(os.path.expandvars("$SCRATCH"))

    # ip_head and redis_passwords are set by ray cluster shell scripts
    print(os.environ["ip_head"], os.environ["redis_password"])
    ray.init(
        address='auto',
        log_to_driver=True,
        _redis_max_memory=100*10**9,
        #_node_ip_address=os.environ["ip_head"].split(":")[0],
        _redis_password=os.environ["redis_password"],
        _temp_dir=os.path.join(scratch_dir, 'raytmp')
    )
    #import pathlib
    #print(pathlib.Path(__file__).parent.absolute())
    #print(pathlib.Path().absolute())
    analysis = tune_model(args)
    print("Best hyperparameters found were: ", analysis.best_config)
