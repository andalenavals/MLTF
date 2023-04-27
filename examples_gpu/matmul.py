import tensorflow as tf
from datetime import datetime
from timeit import default_timer as timer
#print("This operation took %.3f ms" % (1e3*(endtime - starttime).total_seconds())); datetime.now()


if float(tf.__version__[:3]) < 2.0:
    print("Using eager execution")
    #tf.compat.v1.enable_eager_execution()
    tf.enable_eager_execution()
# find if GPU or CPU is used
tf.debugging.set_log_device_placement(True)

N=10000

# dummy ##comment each block and execute at the time
start = timer()
with tf.device('/CPU:0'):
    a2 = tf.ones((N,N))
    b2 = tf.ones((N,N))
c = tf.matmul(a2, b2)
duration = timer() -start
print("This operation 1 took %.3f ms" % (1e3*duration))


# From here the real test begin
start = timer()
with tf.device('/CPU:0'):
    a2 = tf.ones((N,N))
    b2 = tf.ones((N,N))
c = tf.matmul(a2, b2)
duration = timer() -start
print("This operation 2 took %.3f ms" % (1e3*duration))

start= timer()
a3 = tf.ones((N,N))
b3 = tf.ones((N,N))
c3 = tf.matmul(a3, b3)
duration = timer() -start
print("This operation 3 took %.3f ms" % (1e3*duration))


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


