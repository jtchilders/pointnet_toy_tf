import tensorflow as tf
import logging
import numpy as np
logger = logging.getLogger(__name__)

config = None


def get_datasets(config_file):
   global config
   config = config_file

   image_shape = config['data']['image_shape']
   channel_shape = config['data']['channels']
   batch_size = config['data']['batch_size']

   input_shape = image_shape

   output_shape = []
   output_shape.append(image_shape[0])

   train = random_generated_dataset(input_shape,output_shape,config)
   valid = random_generated_dataset(input_shape,output_shape,config)

   return train,valid


def random_gen():
    image_shape = config['data']['image_shape']
    input_shape = image_shape
    output_shape = image_shape[0]
    for i in range(config['data']['total_images']):
        yield np.ones(input_shape, np.float32),np.ones(output_shape, np.int32)


def random_generated_dataset(input_shape,output_shape,config):
    ds = tf.data.Dataset.from_generator(random_gen,(tf.float32,tf.int32))

    ds = ds.batch(config['data']['batch_size'], drop_remainder=True)
    # shard the data
    if config['hvd']:
        ds = ds.shard(config['hvd'].size(), config['hvd'].rank())
    # how many inputs to prefetch to improve pipeline performance
    ds = ds.prefetch(buffer_size=config['data']['prefectch_buffer_size'])

    return ds
