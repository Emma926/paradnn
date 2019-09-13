'''
RNN models.

@author Emma Wang
'''

import tensorflow as tf
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer
import time

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
tf.flags.DEFINE_integer("batch_size", 128,
                        "Mini-batch size for the training.")
tf.flags.DEFINE_string("mode", "train", "train of infer.")
tf.flags.DEFINE_integer("warmup_steps", 300, "warmup steps")
tf.flags.DEFINE_string("optimizer", "rms", "Choose among rms, sgd, and momentum.")

tf.flags.DEFINE_string("input_size", "50,1024,100", "max_length, vocabulary size, embedding size")
tf.flags.DEFINE_integer("layer", 5, "number of hidden layers")
FLAGS = tf.flags.FLAGS

opt = FLAGS.optimizer
RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.
batch_size = FLAGS.batch_size
train_steps = FLAGS.train_steps

rnncell = 'rnn'
num_layers = FLAGS.layer
# [max_length, vocabulary size, embedding size]
input_size = [int(i) for i in FLAGS.input_size.split(',')]
output_size = input_size[0]


def get_input_fn(input_size, output_size):

  def input_fn(params):
    batch_size = params['batch_size']
    word_ids = tf.random_uniform(
        [batch_size, input_size[0] + 1], maxval=input_size[1], dtype=tf.int32)

    features = {}
    features['inputs'] = tf.slice(
        word_ids, [0, 0], [batch_size, input_size[0]])
    features['lengths'] = tf.random_uniform([batch_size], maxval=input_size[0], dtype=tf.int32)

    labels = tf.slice(
        word_ids, [0, 1], [batch_size, output_size])

    return (features, labels)
  return input_fn
 
def model_fn(features, labels, mode, params):

  output_size = params['output_size']
  input_size = params['input_size']
  batch_size = params["batch_size"]
  embedding_size = input_size[2]
  vocab_size = input_size[1]
  max_length = input_size[0]
  inputs = features['inputs']
  lengths = features['lengths']

  embeddings = tf.get_variable(
      "embeddings", [vocab_size, embedding_size], dtype=tf.float32)

  input_embeddings = tf.nn.embedding_lookup(embeddings, inputs)

  def rnn_with_dropout_cell():
    if rnncell == 'lstm':
      cell = tf.contrib.rnn.BasicLSTMCell(
        embedding_size,
        forget_bias=0.0,
        state_is_tuple=True)
    elif rnncell == 'rnn':
      cell = tf.contrib.rnn.BasicRNNCell(embedding_size)
    elif rnncell == 'gru':
      cell = tf.contrib.rnn.GRUCell(embedding_size)
    return cell

  cell_network = tf.contrib.rnn.MultiRNNCell(
      [rnn_with_dropout_cell() for _ in range(num_layers)],
      state_is_tuple=True)
  network_zero_state = cell_network.zero_state(batch_size, dtype=tf.float32)

  outputs, _ = tf.nn.dynamic_rnn(
        cell_network, input_embeddings, initial_state=network_zero_state, swap_memory=True)


  outputs_flat = tf.reshape(outputs, [-1, embedding_size])
  logits_flat = tf.contrib.layers.linear(outputs_flat, vocab_size)
  labels_flat = tf.reshape(labels, [-1])
  mask = tf.sequence_mask(lengths,
                          maxlen=max_length)
  mask = tf.cast(mask, tf.float32)
  mask_flat = tf.reshape(mask, [-1])
  num_logits = tf.to_float(tf.reduce_sum(lengths))

  softmax_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels_flat, logits=logits_flat)
  loss = tf.reduce_sum(mask_flat * softmax_cross_entropy) / num_logits

  # Configuring the optimization step.
  learning_rate = tf.train.exponential_decay(
      0.1,
      tf.train.get_global_step(),
      10000,
      0.9)
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

  train_op = optimizer.minimize(
      loss,
      global_step=tf.train.get_global_step())

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
  estimator.train(input_fn=get_input_fn(input_size, output_size), max_steps=train_steps)

  total_time = time.time() - start
  example_per_sec = batch_size * train_steps / total_time
  global_step_per_sec = train_steps / total_time
  print("Total time: " + str(total_time))
  #tf.logging.info("global_step/sec: %s" % global_step_per_sec)
  #tf.logging.info("examples/sec: %s" % example_per_sec)


if __name__ == "__main__":
  tf.app.run()
