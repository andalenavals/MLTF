import tensorflow as tf
from datetime import datetime
from timeit import default_timer as timer


if float(tf.__version__[:3]) < 2.0:
    print("Using eager execution")
    #tf.compat.v1.enable_eager_execution()
    tf.enable_eager_execution()
# find if GPU or CPU is used
#tf.debugging.set_log_device_placement(True)

'''
def get_tensors():
    N=1000
    a2 = tf.ones((N,N))
    b2 = tf.ones((N,N))
    return a2,b2
def complex_operations(a,b):
    c = tf.matmul(a, b)
'''


def get_tensors():
    N=10000
    a2 = tf.ones((N,N))
    b2 = tf.ones((N,N))
    return a2,b2
def complex_operations(a,b):
    with tf.device('/gpu:1'):
        for i in range(100):
            c = tf.matmul(a, b)
    



#memory intense process in cpu, calculation in gpu
def mipcpu_operationgpu():
    start = timer()
    with tf.device('/CPU:0'):
        a2,b2=get_tensors()
    mid= timer()
    complex_operations(a2,b2)
    end = timer()
    duration=end-start
    opet=end-mid
    mipt=mid-start
    print("mipcpu_operationgpu took %.3f ms, mip %.3f ms and operations %.3f ms " % (1e3*duration,1e3*mipt, 1e3*opet ))

#memory intense process in gpu, calculation in gpu
def allgpu():
    start = timer()
    a2,b2=get_tensors()
    mid= timer()
    complex_operations(a2,b2)
    end = timer()
    duration=end-start
    opet=end-mid
    mipt=mid-start
    print("all gpu took %.3f ms, mip %.3f ms and operations %.3f ms " % (1e3*duration,1e3*mipt, 1e3*opet ))

def main():
    ## DO NOT RUN BOTH TEST SIMULTANEOUSLY
    
    allgpu()
    #mipcpu_operationgpu()
    #mipcpu_operationgpu()
    #mipcpu_operationgpu()

    #allgpu()
    #allgpu()
    #allgpu()
    
    '''
    #gpus = tf.config.list_physical_devices('GPU')
    gpus=["GPU:0", "GPU:1"]
    strategy = tf.distribute.MirroredStrategy(gpus)
    start = timer()
    with strategy.scope():
        a = tf.ones((N,N))
        b = tf.ones((N,N))
        c = tf.matmul(a, b)
    duration = timer() -start
    print("This operation 4 took %.3f ms" % (1e3*duration))
    '''


if __name__ == "__main__":
    main()
