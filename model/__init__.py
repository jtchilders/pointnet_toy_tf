import logging
logger = logging.getLogger('model')

__all__ = ['dummy', 'pointnet_semseg']
from . import dummy,pointnet_semseg


def get_model(input,is_training_pl,config):
   if config['model']['name'] in globals():
      logger.info('using model name %s',config['model']['name'])
      model = globals()[config['model']['name']]
   else:
      raise Exception('failed to find data handler %s in globals %s' % (config['model']['name'],globals()))

   return model.get_model(input,is_training_pl,config)

