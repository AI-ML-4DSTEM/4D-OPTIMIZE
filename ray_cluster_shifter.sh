#!/bin/bash
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --gpus-per-task=8
#SBATCH --image=nersc/tensorflow:ngc-21.05-tf2-v0
#SBATCH --exclusive
#SBATCH -t 02:00:00
#SBTACH -J disk_det
#SBATCH -o ray_test-%j.out
#SBATCH -e ray_error-%j.err
#SBATCH -A m2571
#SBATCH -L SCRATCH

#set up modules
module load cgpu
module load cuda/shifter

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

export train_dataset="unrot_augmented_train.tfrecords"
export test_dataset="unrot_augmented_test.tfrecords"

#set up environment
export PATH=/opt/shifter/bin:${PATH}
export LD_LIBRARY_PATH=/opt/shifter/lib:${LD_LIBRARY_PATH}
export PYTHONUSERBASE=/global/cfs/projectdirs/m3795/4dstem/matt/

#set up environment

################# DON NOT CHANGE THINGS HERE UNLESS YOU KNOW WHAT YOU ARE DOING ###############
# This script is a modification to the implementation suggest by gregSchwartz18 here:
# https://github.com/ray-project/ray/issues/826#issuecomment-522116599
redis_password=$(uuidgen)
export redis_password

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )

head_num_cpus=$(($SLURM_CPUS_PER_TASK - 4))

node_1=${nodes_array[0]}
ip=$(hostname -i | cut -d' ' -f1) # making redis-address
port=6379
ip_head=$ip:$port
export ip_head ip_port
echo "IP Head: $ip_head"

# Optional step to monitor the memory and activity of the head node while running
#nvidia-smi --query-gpu=index,timestamp,name,pci.bus_id,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 5 -f "smi_profile-${SLURM_JOB_ID}.csv" &

#export RAY_BACKEND_LOG_LEVEL=debug
#export RAY_PROFILING=1
# this enables the flag for GPU memory to be allocated as needed
export TF_FORCE_GPU_ALLOW_GROWTH=true
# this limits the number of tune results that will be queued to limit memory usage
export TUNE_RESULT_BUFFER_LENGTH=10

# This sets up the tensorflow config for the head node
TF_NODES=$(jq -ncR '[inputs]' <<< "${nodes}")
export TF_CONFIG="{\"cluster\": {\"worker\": $TF_NODES, \"task\": {\"type\": \"worker\", \"index\": 0}}"

# Notes:
# the ulimit is needed to avoid file limits from all of the task workers that can be created
# num_cpus and num_gpus should be explicitly set to avoid overallocation problems
echo "STARTING HEAD at $node_1 $ip:$port with cpus: ${head_num_cpus}, gpus: ${SLURM_GPUS_PER_TASK}"
shifter --env PYTHONUSERBASE=${PYTHONUSERBASE} \
     bash -c 'ulimit -n 65536; ray start -v --head --node-ip-address "$ip" --dashboard-host "$ip" --port "$port" \
     --redis-password "$redis_password" --temp-dir "$SCRATCH/raytmp" \
     --num-cpus "$head_num_cpus" --num-gpus "$head_num_gpus" --block' &
echo "Head started, waiting before starting $worker_num workers..."
sleep 30

worker_num=$(($SLURM_JOB_NUM_NODES - 1)) #number of nodes other than the head node
echo "NODE OTHER THAN HEAD NODE $worker_num"
for ((  i=1; i<=$worker_num; i++ ))
do
  node_i=${nodes_array[$i]}
  # set the tensorflow config for this worker, it will be passed in to shifter and then reset
  export TF_CONFIG="{\"cluster\": {\"worker\": $TF_NODES,\"task\": {\"type\": \"worker\", \"index\": $i}}"

  echo "STARTING WORKER $i at $node_i with $node_i_gpus gpus"
  srun -C gpu --nodes=1 --ntasks=1 -c ${SLURM_CPUS_PER_TASK} --cpu-bind=cores --gpus-per-task=${SLURM_GPUS_PER_TASK} -w "$node_i" \
       shifter --env PYTHONUSERBASE=${PYTHONUSERBASE} \
       bash -c 'ulimit -n 65536; ray start -v --address "$ip_head" --redis-password "$redis_password" --temp-dir "$SCRATCH/raytmp" \
       --num-cpus ${SLURM_CPUS_PER_TASK} --num-gpus "$node_i_gpus" --block' &  sleep 5
  echo "WORKER $i STARTED"
  # pause to give the workers time to sync with the head node
  sleep 30
done
##############################################################################################

#### call your code below
# reset the head node tensorflow config again for the client code
export TF_CONFIG="{\"cluster\": {\"worker\": $TF_NODES, \"task\": {\"type\": \"worker\", \"index\": 0}}"

start_time="$(date -u +%s)"
echo "STARTING TASKS..."
export RAY_ADDRESS=$ip_head
RAY_NUM_CPUS_PER_WORKER=9
RAY_NUM_GPUS_PER_WORKER=1
export ray_num_workers=$((${SLURM_JOB_NUM_NODES} * ${SLURM_GPUS_PER_TASK}))
# set the tensorflow concurrency limit for trials to the total number of workers
export TUNE_MAX_PENDING_TRIALS_PG=${ray_num_workers}
shifter --env PYTHONUSERBASE=${PYTHONUSERBASE} \
        --env redis_password="$redis_password" \
        bash -c 'python -u tune_training.py --train-dataset "$train_dataset" --test-dataset "$test_dataset" --num-ray-hosts ${#nodes[@]} --num-ray-workers-per-host ${SLURM_GPUS_PER_TASK} --num-ray-cpus-per-worker "$RAY_NUM_CPUS_PER_WORKER" --num-ray-gpus-per-worker "$RAY_NUM_GPUS_PER_WORKER"'
end_time="$(date -u +%s)"
elapsed="$(($end_time-$start_time))"
echo "TASKS COMPLETED after $elapsed seconds"

# clean up the head process
shifter --env PYTHONUSERBASE=${PYTHONUSERBASE} \
     bash -c 'ray stop'

#pkill nvidia-smi

exit
