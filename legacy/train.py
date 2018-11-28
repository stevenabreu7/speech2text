import yaml
import model

config_path = 'config/las_config.yaml'
conf = yaml.load(open(config_path, 'r'))

las = model.ListenAttendSpell(**conf['model_params'])
las.train(config_path)