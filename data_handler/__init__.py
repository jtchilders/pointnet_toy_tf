import logging
from . import random_file_generator,h5_file_generator,csv_file_generator,random_input_generator
logger = logging.getLogger('data_handler')

__all__ = ['random_file_generator','h5_file_generator','csv_file_generator','random_input_generator']


def get_datasets(config):

   if config['data']['handler'] in globals():
      logger.info('using data handler %s',config['data']['handler'])
      handler = globals()[config['data']['handler']]
   else:
      raise Exception('failed to find data handler %s in globals %s' % (config['data']['handler'],globals()))

   return handler.get_datasets(config)
