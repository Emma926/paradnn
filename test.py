''' A self-contained test file.

@author Emma Wang
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer
import time
import os

tf.flags.DEFINE_bool("use_tpu", True, "Use TPUs or not.")

# Cloud TPU Cluster Resolvers
tf.flags.DEFINE_string(
    "gcp_project", default="",
    help="Project name for the Cloud TPU project. If not specified, "
          "the GCE project metadata will be used.")
tf.flags.DEFINE_string(
    "tpu_zone", default="",
    help="GCE zone where the Cloud TPU is located in.")
tf.flags.DEFINE_string(
    "tpu_name", default="",
    help="Name of the Cloud TPU for Cluster Resolvers.")

tf.flags.DEFINE_integer("iterations", 10,
                        "Number of iterations per TPU training loop.")
tf.flags.DEFINE_integer("num_shards", 8, "Number of shards (TPU chips).")
tf.flags.DEFINE_string("model_dir", None, "Estimator model_dir")
tf.flags.DEFINE_integer("batch_size", 128,
                        "Global batch size for the training")
tf.flags.DEFINE_integer("train_steps", 100,
                        "Total number of steps.")

input_dim = [224, 224, 3]
output_dim = 1000

FLAGS = tf.flags.FLAGS
def get_input_fn(batch_size, input_dim, output_dim):

  def input_fn(params):
    size = [batch_size]
    for i in input_dim:
      size.append(i) 
    images = tf.random_uniform(
        size, minval=-0.5, maxval=0.5, dtype=tf.float32)
    labels = tf.random_uniform(
        [batch_size], maxval=output_dim, dtype=tf.int32) 
    labels = tf.one_hot(labels, output_dim)
    return images, labels
  return input_fn
  
def model_fn(features, labels, mode, params):
  output_dim = params['output_dim']
  net = features

  shp = net.get_shape().as_list()

  flattened_shape = shp[1] * shp[2] * shp[3]

  net = tf.reshape(net, [shp[0], flattened_shape])

  net = tf.layers.dense(
    inputs=net,
    units=4,
    activation=tf.nn.relu)

  net = tf.layers.dropout(
    inputs=net,
    rate=0.5)

  net = tf.layers.dense(
    inputs=net,
    units=output_dim,
    activation=None)


  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=labels, logits=net)

  learning_rate = tf.train.exponential_decay(
      0.01, tf.train.get_global_step(), 25000, 0.97)
  if FLAGS.use_tpu:
    optimizer = tpu_optimizer.CrossShardOptimizer(
        tf.train.GradientDescentOptimizer(learning_rate=learning_rate))
  else:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

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

  start = time.time()
  tf.logging.set_verbosity(tf.logging.INFO)
  
  if FLAGS.use_tpu:
      tf.logging.info("Using TPUs.")
  else:
      tf.logging.info("NOT using TPUs.")
      

  if FLAGS.use_tpu:
    tf.logging.info('tpu name:', FLAGS.tpu_name)
    if FLAGS.tpu_name is None:
      raise RuntimeError("You must specify --tpu_name.")
  
    else:
      if '1.6.0' in tf.__version__:
        tpu_cluster_resolver = (
          tf.contrib.cluster_resolver.TPUClusterResolver(
              tpu_names=[os.uname()[1]],
              zone=FLAGS.tpu_zone,
              project=FLAGS.gcp_project))
      else:
        tpu_cluster_resolver = (
          tf.contrib.cluster_resolver.TPUClusterResolver(
              os.uname()[1],
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
      tpu_config=tpu_config.TPUConfig(iterations_per_loop=FLAGS.iterations, num_shards=FLAGS.num_shards),
  )

  estimator = tpu_estimator.TPUEstimator(
      model_fn=model_fn,
      params={"bs": FLAGS.batch_size, "output_dim": output_dim, "input_dim": input_dim},
      use_tpu=FLAGS.use_tpu,
      train_batch_size=FLAGS.batch_size,
      config=run_config)
  estimator.train(input_fn=get_input_fn(FLAGS.batch_size, input_dim, output_dim), max_steps=FLAGS.train_steps)

  total = time.time() - start
  tf.logging.info("Total time: " + str(total))

if __name__ == "__main__":
  tf.app.run()
