'''
ResNet-like CNN models.

@author Emma Wang
'''

import tensorflow as tf
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer
import numpy as np
import time

import resnet_model

tf.flags.DEFINE_bool("use_tpu", True, "Use TPUs or not.")
# TPU only
tf.flags.DEFINE_string(
    "gcp_project", default="",
    help="Project name for the Cloud TPU project. If not specified, "
          "the GCE project metadata will be used.")
tf.flags.DEFINE_string(
    "tpu_zone", default="",
    help="GCE zone where the Cloud TPU is located in.")
tf.flags.DEFINE_string(
    "tpu_name", default=None,
    help="Name of the Cloud TPU for Cluster Resolvers.")

tf.flags.DEFINE_string("model_dir", None, "Estimator model_dir")
tf.flags.DEFINE_integer("iterations", 100,
                        "Number of iterations per TPU training loop.")
tf.flags.DEFINE_integer("num_shards", 8, "Number of shards (TPU chips).")
tf.flags.DEFINE_string("machine", 'tpu/2x2', "machine")

tf.flags.DEFINE_integer("train_steps", 100, "Total number of steps.")
tf.flags.DEFINE_integer("batch_size", 8,
                        "Mini-batch size for the training.")
tf.flags.DEFINE_string("mode", "train", "train of infer.")
tf.flags.DEFINE_string("optimizer", "rms", "Choose among rms, sgd, and momentum.")
tf.flags.DEFINE_string("data_type", "float32", "bfloat16 is only for TPU, and float16 only for GPU with Tensor Cores.")

tf.flags.DEFINE_integer("input_size", 224, "")
tf.flags.DEFINE_integer("output_size", 1000, "")
tf.flags.DEFINE_integer("filters", 16, "")
tf.flags.DEFINE_string("resnet_layers", "1,1,1,1", "residual blocks in each group")
tf.flags.DEFINE_string("block_fn", "residual", "choose from residual and bottleneck")

FLAGS = tf.flags.FLAGS
input_size = [FLAGS.input_size, FLAGS.input_size, 3]
output_size = FLAGS.output_size
batch_size = FLAGS.batch_size
resnet_layers = [int(i) for i in FLAGS.resnet_layers.split(',')]
block_fn = FLAGS.block_fn
filters = FLAGS.filters

opt = FLAGS.optimizer
RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.

def get_input_fn(input_size, output_size):

  def input_fn(params):
    batch_size = params['batch_size']
    if FLAGS.data_type == 'float32':
      dataset = tf.data.Dataset.range(1).repeat().map(
        lambda x: (tf.cast(tf.constant(np.random.random_sample(input_size).astype(np.float32), tf.float32), tf.float32),
                 tf.constant(np.random.randint(output_size, size=(1,))[0], tf.int32)))
    elif FLAGS.data_type == 'float16':
      dataset = tf.data.Dataset.range(1).repeat().map(
        lambda x: (tf.cast(tf.constant(np.random.random_sample(input_size).astype(np.float32), tf.float16), tf.float16),
                 tf.constant(np.random.randint(output_size, size=(1,))[0], tf.int32)))
    elif FLAGS.data_type == 'bfloat16':
      dataset = tf.data.Dataset.range(1).repeat().map(
        lambda x: (tf.cast(tf.constant(np.random.random_sample(input_size).astype(np.float32), tf.bfloat16), tf.bfloat16),
                 tf.constant(np.random.randint(output_size, size=(1,))[0], tf.int32)))
    
    dataset = dataset.prefetch(batch_size)

    dataset = dataset.apply(
        tf.contrib.data.batch_and_drop_remainder(batch_size))

    dataset = dataset.prefetch(4)     # Prefetch overlaps in-feed with training
    images, labels = dataset.make_one_shot_iterator().get_next()
    return images, labels
  return input_fn


def get_custom_getter():
  def inner_custom_getter_float16(getter, *args, **kwargs):
    cast_to_16 = False
    requested_dtype = kwargs['dtype']
    if requested_dtype == tf.float16:
      kwargs['dtype'] = tf.float32
      cast_to_16 = True
    var = getter(*args, **kwargs)
    if cast_to_16:
      var = tf.cast(var, tf.float16)
      print('tf.float16', args, kwargs, var)
    return var

  def inner_custom_getter_bfloat16(getter, *args, **kwargs):
    cast_to_16 = False
    requested_dtype = kwargs['dtype']
    if requested_dtype == tf.bfloat16:
      kwargs['dtype'] = tf.float32
      cast_to_16 = True
    var = getter(*args, **kwargs)
    if cast_to_16:
      var = tf.cast(var, tf.bfloat16)
      print('tf.bfloat16', args, kwargs, var)
    return var

  if FLAGS.data_type == 'float16':
    return inner_custom_getter_float16
  elif FLAGS.data_type == 'bfloat16':
    return inner_custom_getter_bfloat16
  

def model_fn(features, labels, mode, params):
  output_size = params['output_size']
  net = features

  if FLAGS.data_type == 'float32':
    network = resnet_model.resnet_v1(
      resnet_layers,
      block_fn,
      num_classes=output_size,
      data_format='channels_last',
      filters=filters)

    net = network(
      inputs=features, is_training=True)
  else:
    with tf.variable_scope('cg', custom_getter=get_custom_getter()):
      network = resnet_model.resnet_v1(
        resnet_layers,
        block_fn,
        num_classes=output_size,
        data_format='channels_last',
        filters=filters)
  
      net = network(
        inputs=features, is_training=True)
      net = tf.cast(net, tf.float32)
    
  onehot_labels=tf.one_hot(labels, output_size)
  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=net)

  learning_rate = tf.train.exponential_decay(
      0.1, tf.train.get_global_step(), 25000, 0.97)
  if opt == 'sgd':
      tf.logging.info('Using SGD optimizer')
      optimizer = tf.train.GradientDescentOptimizer(
          learning_rate=learning_rate)
  elif opt == 'momentum':
      tf.logging.info('Using Momentum optimizer')
      optimizer = tf.train.MomentumOptimizer(
          learning_rate=learning_rate, momentum=0.9)
  elif opt == 'rms':
      tf.logging.info('Using RMS optimizer')
      optimizer = tf.train.RMSPropOptimizer(
          learning_rate,
          RMSPROP_DECAY,
          momentum=RMSPROP_MOMENTUM,
          epsilon=RMSPROP_EPSILON)
  if FLAGS.use_tpu:
    optimizer = tpu_optimizer.CrossShardOptimizer(optimizer)

  train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

  param_stats = tf.profiler.profile(
    tf.get_default_graph(),
    options=ProfileOptionBuilder.trainable_variables_parameter())
  fl_stats = tf.profiler.profile(
    tf.get_default_graph(),
    options=tf.profiler.ProfileOptionBuilder.float_operation())

  return tpu_estimator.TPUEstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op)

ProfileOptionBuilder = tf.profiler.ProfileOptionBuilder
def main(unused_argv):
  del unused_argv

  start = time.time()
  tf.logging.set_verbosity(tf.logging.INFO)
  print('Tensorflow version: ' + str(tf.__version__))
  for k,v in iter(tf.app.flags.FLAGS.flag_values_dict().items()):
    print("***%s: %s" % (k, v))

  if FLAGS.use_tpu == True:
    if FLAGS.tpu_name is None:
      raise RuntimeError("You must specify --tpu_name.")
  
    else:
      if '1.6.0' in tf.__version__:
        tpu_cluster_resolver = (
          tf.contrib.cluster_resolver.TPUClusterResolver(
            tpu_names=[FLAGS.tpu_name],
            zone=FLAGS.tpu_zone,
            project=FLAGS.gcp_project))
      else:
        tpu_cluster_resolver = (
          tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name,
            zone=FLAGS.tpu_zone,
            project=FLAGS.gcp_project))
      tpu_grpc_url = tpu_cluster_resolver.get_master()
  else:
    tpu_grpc_url = ''

  run_config = tpu_config.RunConfig(
      master=tpu_grpc_url,
      evaluation_master=tpu_grpc_url,
      model_dir=FLAGS.model_dir,
      save_checkpoints_secs=None,
      session_config=tf.ConfigProto(
          allow_soft_placement=True, log_device_placement=False, gpu_options=tf.GPUOptions(allow_growth=True)),
      tpu_config=tpu_config.TPUConfig(iterations_per_loop=FLAGS.iterations, num_shards=FLAGS.num_shards),
  )
  estimator = tpu_estimator.TPUEstimator(
      model_fn=model_fn,
      params={"output_size": output_size, "input_size": input_size},
      use_tpu=FLAGS.use_tpu,
      train_batch_size=batch_size,
      config=run_config)
  estimator.train(input_fn=get_input_fn(input_size, output_size), max_steps=FLAGS.train_steps)

  total_time = time.time() - start
  example_per_sec = batch_size * FLAGS.train_steps / total_time
  global_step_per_sec = FLAGS.train_steps / total_time
  print("Total time: " + str(total_time))
  #tf.logging.info("global_step/sec: %s" % global_step_per_sec)
  #tf.logging.info("examples/sec: %s" % example_per_sec)


if __name__ == "__main__":
  tf.app.run()
