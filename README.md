# CNN on CIFAR-10 with Horovod

## How to install Horovod on Plafrim

- Install all frameworks needed : pip install torch==1.7 tensorflow==2.3.0 matplotlib==3.3.3 Keras==2.4.3 mxnet numpy==1.18.5

**Be careful of the version of frameworks you are installing. It might make an error while installing Horovod**

- Load all modules needed : module load compiler/cuda/10.1 dnn/cudnn/10.0-v7.6.4.38 mpi/openmpi/3.1.4-all build/cmake/3.18.4 compiler/gcc/8.3.0 language/python/3.8.0

Then you can :

- Install horovod : HOROVOD_WITH_TENSORFLOW=1 pip install --no-cache-dir horovod

## How to execute

### CNN TensorFlow-GPU :

  - python3 CNN.py
  
### CNN with Horovod GPU:
  
   - horovodrun -np 2 python3 CNN-Horovod.py (Replace 2 by the number of GPU's you want to use)
   
### CNN with Horovod CPU (Really bad performance):
   
   - CUDA_VISIBLE_DEVICES=-1 horovodrun -np 4 python3 CNN-Horovod.py (Replace 2 by the number of CPU's you want to use)
   
### CNN Basic:

  You can run the basic version of CNN by unloading cuda and dnn :
      - module unload dnn/cudnn/10.0-v7.6.4.38 compiler/cuda/10.1
  And then run:
      - python3 CNN.py
