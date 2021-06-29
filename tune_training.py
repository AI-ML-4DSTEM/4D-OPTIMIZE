#!/usr/bin/env python
# coding: utf-8
# %%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import uuid
import logging
import h5py
import argparse
import numpy as np
from tqdm import tqdm

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
import tensorflow as tf

import tensorflow_addons as tfa

print('Loading complex convolution, pooling libraries ... \n')

#Custom libraries - User defined
import convSpectral
import poolSpectral
import augmentation
from conv_utils import parseDataset
from conv_utils import convSpec2d_block, conv2d_block
from conv_utils import cross_correlate_iff, cross_correlate_ff
from conv_utils import CustomSSIML1ELoss

print('Loading ray Tune ... \n')
from filelock import FileLock

import ray
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.keras import TuneReportCallback

logger = logging.getLogger(__name__)

def _import_gputil():
    try:
        import GPUtil
    except ImportError:
        GPUtil = None
    return GPUtil

#Parse arguments usinf argparse
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

parser.add_argument('--num-training-iterations', type=int, default=10,
                    help='set data mode for the training to default, norm or multitask mode')

args = parser.parse_args()

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

def wait_for_gpu(gpu_id=None, target_util=0.001, retry = 10, delay_s=10):
    GPUtil = _import_gputil()
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
def train_model(config):
    '''
    train loop
    config: filter_size, depth, dropout, activation, alpha, batch_size, warmup, optimizer,
    '''
    
    print('Checking if GPU is available before trial starts... \n')    
    #TODO: Build a new wait for gpu
    wait_for_gpu()
    
    #Load Tensorflow on GPUs
    print('Loading Tensorflow GPU: \n')
    import tensorflow as tf
    import tensorflow_addons as tfa

    print('Loading complex convolution, pooling libraries ... \n')

    #Custom libraries - User defined
    import convSpectral
    import poolSpectral
    import augmentation
    from conv_utils import parseDataset
    from conv_utils import convSpec2d_block, conv2d_block
    from conv_utils import cross_correlate_iff, cross_correlate_ff
    from conv_utils import CustomSSIML1ELoss
    
    #Enable Automatic Mixed Precision
    if args.enable_amp:
        print("Enabling mixed precision... \n")
        from tensorflow.keras import mixed_precision
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)

    #Load Tensorflow XLA
    if args.enable_xla:
        print("Enabling Tensorflow XLA... \n")
        tf.keras.backend.clear_session()
        tf.config.optimizer.set_jit(True) # Enable XLA.
    
    #Load TFRecords to TF batch dataset
    #TODO: Use TF Generators
    batch_size = 8
    prepTrainDataset = parseDataset(filepath = args.train_dataset)
    train_dataset = prepTrainDataset.read(batch_size = batch_size, augment = False, mode = args.data_mode)
    prepTestDataset = parseDataset(filepath = args.test_dataset)
    val_dataset = prepTestDataset.read(batch_size = batch_size, augment = False, mode = args.data_mode)

    # Disable AutoShard.
    if not args.enable_shard:
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        train_dataset = train_dataset.with_options(options)
        val_dataset = val_dataset.with_options(options)
    
    input_channel = 2
    EPOCHS =10
    model = build_generator(input_channel, config["filter_size"], config["depth"], dp_rate = config["dropout"], act = config["activation"])
    
    class LRScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, d_model, alpha =10, warmup_steps=50):
            super(LRScheduler, self).__init__()

            self.d_model = d_model
            self.d_model = tf.cast(self.d_model, tf.float32)
            self.alpha = alpha
            self.warmup_steps = warmup_steps

        def __call__(self, step):
            arg1 = tf.math.rsqrt(step)
            arg2 = step * (self.warmup_steps ** -1.5)

            return tf.math.rsqrt(self.alpha*self.d_model) * tf.math.minimum(arg1, arg2)
    
    learning_rate = LRScheduler( batch_size, alpha =  config["alpha"], warmup_steps = config["warmup"])
    
    if config["optimizer"] == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=  learning_rate)
    elif config["optimizer"] == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate= learning_rate, momentum= config["momentum"])
    elif config["optimizer"] == 'adadelta':
        optimizer = tf.keras.optimizers.Adadelta(learning_rate= learning_rate)
    elif config["optimizer"] == 'lamb':
        optimizer = tfa.optimizers.LAMB(learning_rate= learning_rate)
     
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    model.fit(train_dataset, callbacks=[TuneReportCallback({
            "mean_mae": "mae"})], epochs=EPOCHS, validation_data = val_dataset)
    
def tune_model(num_training_iterations):
    sched = ASHAScheduler(time_attr="training_iteration", max_t=400, grace_period=20)
    hyperopt_search = HyperOptSearch(metric="mean_mae", mode="min")
   # sched = ASHAScheduler(metric="mean_accuracy", mode="max")

    analysis = tune.run(
        train_model,
        name="exp",
        scheduler=sched,
        search_alg=hyperopt_search,
        metric="mae",
        mode="min",
        stop={
            "mean_mae": 0.1,
            "training_iteration": num_training_iterations
        },
        num_samples=-1,
        checkpoint_at_end=True,
        resources_per_trial={
            "cpu": 10,
            "gpu": 1
        },
        config={
            "filter_size": tune.randint(4, 64),
            "depth": tune.randint(2, 8),
            "dropout": tune.uniform(0.1, 0.5),
            "momentum": tune.uniform(0.1, 0.9),
            "alpha": tune.randint(1,10000),
            "warmup": tune.randint(1,50),
            "activation": tune.choice(["relu", "tanh"]),
            "optimizer": tune.choice(["adam", "sgd", "adadelta", "lamb"])
        })
    
    print("Best hyperparameters found were: ", analysis.best_config)

if __name__ == "__main__":
    # ip_head and redis_passwords are set by ray cluster shell scripts
    print(os.environ["ip_head"], os.environ["redis_password"])
    ray.init(address='auto', _node_ip_address=os.environ["ip_head"].split(":")[0], _redis_password=os.environ["redis_password"])
    import pathlib
    print(pathlib.Path(__file__).parent.absolute())
    print(pathlib.Path().absolute())
    tune_model(args.num_training_iterations)
