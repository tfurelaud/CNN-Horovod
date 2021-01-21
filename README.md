# CNN on CIFAR-10 with Horovod

## How to compute on Plafrim
<em>important</em>
- Install all frameworks needed : pip install torch==1.7 tensorflow==2.3.0 matplotlib==3.3.3 Keras==2.4.3 mxnet pickle numpy 

- Load all modules needed : module load compiler/cuda/10.1 dnn/cudnn/10.0-v7.6.4.38 mpi/openmpi/3.1.4-all mpi/openmpi/3.1.4-all build/cmake/3.18.4 compiler/gcc/8.3.0 language/python/3.8.0


- Install horovod : HOROVOD_WITH_TENSORFLOW=1 pip install --no-cache-dir horovod

- Then you can execute with : horovodrun -np 2 python3 CNN-Horovod.py
