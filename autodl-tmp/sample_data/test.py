# test whether GPU can be used
import tensorflow as tf

# 查看版本号
tf.__version__
# 查看gpu能否使用
tf.test.is_gpu_available()
# tf.config.list_physical_devices('GPU')
