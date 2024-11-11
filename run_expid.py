import os
import sys
import logging
import fuxictr_version
from fuxictr import datasets
from datetime import datetime
from fuxictr.utils import load_config, set_logger, print_to_json, print_to_list
from fuxictr.features import FeatureMap
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.pytorch.dataloaders import H5DataLoader
from fuxictr.preprocess import FeatureProcessor, build_dataset
import src as model_zoo
import gc
import argparse
from pathlib import Path

# Set the current working directory to the directory of this script
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Print the current working directory and its contents for debugging
current_directory = os.getcwd()
print("Current Working Directory:", current_directory)
print("Contents of Current Directory:", os.listdir(current_directory))

# Get the base directory where the script is located
base_dir = os.path.dirname(os.path.realpath(__file__))

# Define data paths based on base_dir
data_root = os.path.join(base_dir, 'data', 'final')

# print available data files
print("Available Data Files:", os.listdir(data_root))

train_data_path = os.path.join(data_root, 'train_data_final.csv')
valid_data_path = os.path.join(data_root, 'valid_data_final.csv')
test_data_path = os.path.join(data_root, 'test_data_final.csv')


# Check if data files exist before proceeding
if not all(os.path.exists(f) for f in [train_data_path, valid_data_path, test_data_path]):
    print("One or more data files do not exist.")
    print(f"Train Data Exists: {os.path.exists(train_data_path)}")
    print(f"Valid Data Exists: {os.path.exists(valid_data_path)}")
    print(f"Test Data Exists: {os.path.exists(test_data_path)}")
    sys.exit(1)  # Exit if files are missing

# Load configuration
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./config/', help='The config directory.')
parser.add_argument('--expid', type=str, default='DeepFM_test', help='The experiment id to run.')
parser.add_argument('--gpu', type=int, default=-1, help='The gpu index, -1 for cpu')
args = vars(parser.parse_args())

experiment_id = args['expid']
params = load_config(args['config'], experiment_id)
params['gpu'] = args['gpu']

# Update params with absolute paths
params['data_root'] = data_root
params['train_data'] = train_data_path
params['valid_data'] = valid_data_path
params['test_data'] = test_data_path

set_logger(params)
logging.info("Params: " + print_to_json(params))
seed_everything(seed=params['seed'])

data_dir = os.path.join(params['data_root'], params['dataset_id'])
feature_map_json = os.path.join(data_dir, "feature_map.json")

if params["data_format"] == "csv":
    # Build feature_map and transform h5 data
    feature_encoder = FeatureProcessor(**params)
    params["train_data"], params["valid_data"], params["test_data"] = \
        build_dataset(feature_encoder, **params)

feature_map = FeatureMap(params['dataset_id'], data_dir)
feature_map.load(feature_map_json, params)
logging.info("Feature specs: " + print_to_json(feature_map.features))

model_class = getattr(model_zoo, params['model'])
model = model_class(feature_map, **params)
model.count_parameters()  # Print number of parameters used in model

train_gen, valid_gen = H5DataLoader(feature_map, stage='train', **params).make_iterator()
model.fit(train_gen, validation_data=valid_gen, **params)

logging.info('****** Validation evaluation ******')
valid_result = model.evaluate(valid_gen)
del train_gen, valid_gen
gc.collect()

result_filename = Path(args['config']).name.replace(".yaml", "") + '.csv'
with open(result_filename, 'a+') as fw:
    fw.write(' {},[command] python {},[exp_id] {},[dataset_id] {},[train] {},[val] {} \n' \
        .format(datetime.now().strftime('%Y%m%d-%H%M%S'), 
                ' '.join(sys.argv), experiment_id, params['dataset_id'],
                "N.A.", print_to_list(valid_result)))