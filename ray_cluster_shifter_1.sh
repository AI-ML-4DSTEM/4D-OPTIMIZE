#!/bin/bash
#SBATCH -C gpu
#SBATCH -c 80
#SBATCH -G 8
#SBATCH --nodes 2
#SBATCH --ntasks-per-node=1
#SBATCH --image=nersc/tensorflow:ngc-21.03-tf2-v0
#SBATCH --exclusive
#SBATCH -t 02:00:00
#SBTACH -J disk_det
#SBATCH -o ray_test.out
#SBATCH -e ray_error.err
#SBATCH -A m2571

#set up modules
module load cgpu
module load cuda/shifter

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

train=$(readlink -f unrot_augmented_train.tfrecords)
test=$(readlink -f unrot_augmented_test.tfrecords)

#set up environment
export PATH=/opt/shifter/bin:${PATH}
export LD_LIBRARY_PATH=/opt/shifter/lib:${LD_LIBRARY_PATH}

#set up environment

################# DON NOT CHANGE THINGS HERE UNLESS YOU KNOW WHAT YOU ARE DOING ###############
# This script is a modification to the implementation suggest by gregSchwartz18 here:
# https://github.com/ray-project/ray/issues/826#issuecomment-522116599
redis_password=$(uuidgen)
export redis_password

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )

node_1=${nodes_array[0]} 
ip=$(srun --nodes=1 --ntasks=1 -w $node_1 hostname --ip-address) # making redis-address
port=6379
ip_head=$ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "STARTING HEAD at $node_1"
srun --nodes=1 --ntasks=1 -w $node_1 shifter ray start --block --head --node-ip-address=$ip --port=6379 --redis-password=$redis_password &
sleep 30

worker_num=$(($SLURM_JOB_NUM_NODES - 1)) #number of nodes other than the head node
echo "NODE OTHER THAN HEAD NODE $worker_num"
for ((  i=1; i<=$worker_num; i++ ))
do
  node_i=${nodes_array[$i]}
  echo "STARTING WORKER $i at $node_i"
  srun --nodes=1 --ntasks=1 -w $node_i shifter ray start --block --address=$ip_head --redis-password=$redis_password &
  sleep 5
done
##############################################################################################

#### call your code below

shifter python -u tune_training.py --train-dataset=$train --test-dataset=$test
exit
