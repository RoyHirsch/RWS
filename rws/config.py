"""Config dict contains all the parametes for running an experiment."""

import os
import torch
import ml_collections
import yaml


def get_config(args):
    if args.config:
        config = load_yaml(args.config)
        
    else:
        config = get_default_config()
    config.config_file_path = args.config

    if args.data_path:
        config.data_path = args.data_path
    if args.exp_root:
        config.exp_root = args.exp_root
    if args.exp_name:
        config.exp_name = args.exp_name
    modify_config(config)
    return config


def get_default_config():
    """Get an experiment configuraion dict."""

    config = ml_collections.ConfigDict()

    # exp #
    config.exp_root = '/experiments/test'
    config.exp_name = 'split1'
    config.description = ''
    config.exp_dir = os.path.join(config.exp_root, config.exp_name)
    config.dump_log = True
    config.seed = 42
    config.gpu_num = 3
    config.checkpoint = None

    # data #    
    config.dataset = 'gtea'
    config.split = '4'
    config.data_root_dir = '/mnt/disks/tempseg/data/data'
    
    config.dataset_root = os.path.join(config.data_root_dir, config.dataset)
    config.train_videos_list = os.path.join(config.dataset_root, 'splits', 'train.split{}.bundle'.format(config.split))
    config.test_videos_list = os.path.join(config.dataset_root, 'splits', 'test.split{}.bundle'.format(config.split))
    config.features_path = os.path.join(config.dataset_root, 'features')
    config.gt_path = os.path.join(config.dataset_root, 'groundTruth')
    mapping_file = os.path.join(config.dataset_root, 'mapping.txt')

    config.actions_dict = get_action_mapping(mapping_file)
    config.num_classes = len(config.actions_dict)
    config.sample_rate = 2 if config.dataset == '50salads' else 1
    config.ignore_index = -100
    config.batch_size = 8
    config.num_workers = 4

    # temporal model #
    config.num_stages = 4
    config.num_layers = 10
    config.num_f_maps = 64
    config.features_dim = 2048

    # optim #
    config.lr = 5e-4
    config.weight_decay = 0.0
    config.num_epochs = 50
    config.start_epochs = 30 # todo
    config.device = torch.device('cuda:{}'.format(config.gpu_num) if torch.cuda.is_available() else 'cpu')
    config.eval_interval = 2

    # random walk #
    config.i3d_based_laplacian = False 
     # ['cosine', 'euclidean']
    config.similarity_method = 'euclidean'
     # ['power', 'exp', 'none']
    config.sharpening_method = 'exp'
    config.beta = 30
     # ['mean', 'min', 'max', 'none']
    config.averaging_method = 'min'
    config.num_neighbors = 30
    config.gamma = 0.0001
    config.smooth = 10
    
    # losses # 
     # ['random_walk', 'forward_backward']
    config.supervised_loss_name = 'forward_backward'
    # ['mse', 'laplace']
    config.smoothness_loss_name = 'mse'

    config.smoothness_weight = 0.15
    config.confidence_weight = 0.1

    return config


def dump_to_yaml(config, dir, file_name='config_test'):
    with open(os.path.join(dir, file_name + '.yaml'), 'w') as file:
        yaml.dump(config.to_dict(), file)


def load_yaml(filepath):
    with open(filepath, 'r') as f:
        d = yaml.load(f, Loader=yaml.UnsafeLoader)
    config = ml_collections.ConfigDict()
    for k, v in d.items():
        setattr(config, k, v)
    config.device = torch.device('cuda:{}'.format(config.gpu_num) if torch.cuda.is_available() else 'cpu')
    return config


def get_and_modify_config(modifications_dict):
    """Gets a dict with params to modify and modify the config"""

    config = get_default_config()
    for k, v in modifications_dict.items():
        if getattr(config, k, None) != None:
            setattr(config, k, v)

    modify_config(config)
    return config


def modify_config(config):
    config.dataset_root = os.path.join(config.data_root_dir, config.dataset)
    config.exp_dir = os.path.join(config.exp_root, config.exp_name)
    config.train_videos_list = os.path.join(config.dataset_root, 'splits', 'train.split{}.bundle'.format(config.split))
    config.test_videos_list = os.path.join(config.dataset_root, 'splits', 'test.split{}.bundle'.format(config.split))
    config.features_path = os.path.join(config.dataset_root, 'features')
    config.gt_path = os.path.join(config.dataset_root, 'groundTruth')
    mapping_file = os.path.join(config.dataset_root, 'mapping.txt')

    config.actions_dict = get_action_mapping(mapping_file)
    config.num_classes = len(config.actions_dict)
    config.sample_rate = 2 if config.dataset == '50salads' else 1
    config.device = torch.device('cuda:{}'.format(config.gpu_num) if torch.cuda.is_available() else 'cpu')


def get_action_mapping(mapping_file):
    file_ptr = open(mapping_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])
    return actions_dict
