import json
import inspect
import torch
import os
import sys
import yaml
from shutil import copy, copytree
from os.path import join, dirname, realpath, expanduser, isfile, isdir, basename


class Logger(object):

    def __getattr__(self, k):
        return print

log = Logger()

def training_config_from_cli_args():
    experiment_name = sys.argv[1]
    experiment_id = int(sys.argv[2])

    yaml_config = yaml.load(open(f'experiments/{experiment_name}'), Loader=yaml.SafeLoader)

    config = yaml_config['configuration']
    config = {**config, **yaml_config['individual_configurations'][experiment_id]}
    config = AttributeDict(config)
    return config


def score_config_from_cli_args():
    experiment_name = sys.argv[1]
    experiment_id = int(sys.argv[2])
    

    yaml_config = yaml.load(open(f'experiments/{experiment_name}'), Loader=yaml.SafeLoader)

    config = yaml_config['test_configuration_common']

    if type(yaml_config['test_configuration']) == list:
        test_id = int(sys.argv[3])
        config = {**config, **yaml_config['test_configuration'][test_id]}
    else:
        config = {**config, **yaml_config['test_configuration']}

    if 'test_configuration' in yaml_config['individual_configurations'][experiment_id]:
        config = {**config, **yaml_config['individual_configurations'][experiment_id]['test_configuration']}

    train_checkpoint_id = yaml_config['individual_configurations'][experiment_id]['name']

    config = AttributeDict(config)
    return config, train_checkpoint_id


def get_from_repository(local_name, repo_files, integrity_check=None, repo_dir='~/dataset_repository', 
                        local_dir='~/datasets'):
    """ copies files from repository to local folder.
    
    repo_files: list of filenames or list of tuples [filename, target path] 

    e.g. get_from_repository('MyDataset', [['data/dataset1.tar', 'other/path/ds03.tar'])
    will create a folder 'MyDataset' in local_dir, and extract the content of
    '<repo_dir>/data/dataset1.tar' to <local_dir>/MyDataset/other/path.
     """

    local_dir = realpath(join(expanduser(local_dir), local_name))

    dataset_exists = True

    # check if folder is available
    if not isdir(local_dir):
        dataset_exists = False

    if integrity_check is not None:
        try:
            integrity_ok = integrity_check(local_dir)
        except BaseException:
            integrity_ok = False

        if integrity_ok:
            log.hint('Passed custom integrity check')
        else:
            log.hint('Custom integrity check failed')

        dataset_exists = dataset_exists and integrity_ok

    if not dataset_exists:

        repo_dir = realpath(expanduser(repo_dir))

        for i, filename in enumerate(repo_files):

            if type(filename) == str:
                origin, target = filename, filename
                archive_target = join(local_dir, basename(origin))
                extract_target = join(local_dir)
            else:
                origin, target = filename
                archive_target = join(local_dir, dirname(target), basename(origin))
                extract_target = join(local_dir, dirname(target))
            
            archive_origin = join(repo_dir, origin)

            log.hint(f'copy: {archive_origin} to {archive_target}')

            # make sure the path exists
            os.makedirs(dirname(archive_target), exist_ok=True)

            if os.path.isfile(archive_target):
                # only copy if size differs
                if os.path.getsize(archive_target) != os.path.getsize(archive_origin):
                    log.hint(f'file exists but filesize differs: target {os.path.getsize(archive_target)} vs. origin {os.path.getsize(archive_origin)}')
                    copy(archive_origin, archive_target)
            else:
                copy(archive_origin, archive_target)

            extract_archive(archive_target, extract_target, noarchive_ok=True)

            # concurrent processes might have deleted the file
            if os.path.isfile(archive_target):
                os.remove(archive_target)


def extract_archive(filename, target_folder=None, noarchive_ok=False):
    from subprocess import run, PIPE

    if filename.endswith('.tgz') or filename.endswith('.tar'):
        command = f'tar -xf {filename}'
        command += f' -C {target_folder}' if target_folder is not None else ''
    elif filename.endswith('.tar.gz'):
        command = f'tar -xzf {filename}'
        command += f' -C {target_folder}' if target_folder is not None else ''
    elif filename.endswith('zip'):
        command = f'unzip {filename}'
        command += f' -d {target_folder}' if target_folder is not None else ''
    else:
        if noarchive_ok:
            return
        else:
            raise ValueError(f'unsuppored file ending of {filename}')

    log.hint(command)
    result = run(command.split(), stdout=PIPE, stderr=PIPE)
    if result.returncode != 0:
        print(result.stdout, result.stderr)


class AttributeDict(dict):
    """ 
    An extended dictionary that allows access to elements as atttributes and counts 
    these accesses. This way, we know if some attributes were never used. 
    """

    def __init__(self, *args, **kwargs):
        from collections import Counter
        super().__init__(*args, **kwargs)
        self.__dict__['counter'] = Counter()

    def __getitem__(self, k):
        self.__dict__['counter'][k] += 1
        return super().__getitem__(k)

    def __getattr__(self, k):
        self.__dict__['counter'][k] += 1
        return super().get(k)

    def __setattr__(self, k, v):
        return super().__setitem__(k, v)

    def __delattr__(self, k, v):
        return super().__delitem__(k, v)    

    def unused_keys(self, exceptions=()):
        return [k for k in super().keys() if self.__dict__['counter'][k] == 0 and k not in exceptions]

    def assume_no_unused_keys(self, exceptions=()):
        if len(self.unused_keys(exceptions=exceptions)) > 0:
            log.warning('Unused keys:', self.unused_keys(exceptions=exceptions))


def get_attribute(name):
    import importlib

    if name is None:
        raise ValueError('The provided attribute is None')
    
    name_split = name.split('.')
    mod = importlib.import_module('.'.join(name_split[:-1]))
    return getattr(mod, name_split[-1])



def filter_args(input_args, default_args):

    updated_args = {k: input_args[k] if k in input_args else v for k, v in default_args.items()}
    used_args = {k: v for k, v in input_args.items() if k in default_args}
    unused_args = {k: v for k, v in input_args.items() if k not in default_args}

    return AttributeDict(updated_args), AttributeDict(used_args), AttributeDict(unused_args)


def load_model(checkpoint_id, weights_file=None, strict=True, model_args='from_config', with_config=False):

    config = json.load(open(join('logs', checkpoint_id, 'config.json')))

    if model_args != 'from_config' and type(model_args) != dict:
        raise ValueError('model_args must either be "from_config" or a dictionary of values')

    model_cls = get_attribute(config['model'])

    # load model
    if model_args == 'from_config':
        _, model_args, _ = filter_args(config, inspect.signature(model_cls).parameters)

    model = model_cls(**model_args)

    if weights_file is None:
        weights_file = realpath(join('logs', checkpoint_id, 'weights.pth'))
    else:
        weights_file = realpath(join('logs', checkpoint_id, weights_file))

    if isfile(weights_file):
        weights = torch.load(weights_file)
        for _, w in weights.items():
            assert not torch.any(torch.isnan(w)), 'weights contain NaNs'
        model.load_state_dict(weights, strict=strict)
    else:
        raise FileNotFoundError(f'model checkpoint {weights_file} was not found')

    if with_config:
        return model, config
    
    return model


class TrainingLogger(object):

    def __init__(self, model, log_dir, config=None, *args):
        super().__init__()
        self.model = model
        self.base_path = join(f'logs/{log_dir}') if log_dir is not None else None

        os.makedirs('logs/', exist_ok=True)
        os.makedirs(self.base_path, exist_ok=True)

        if config is not None:
            json.dump(config, open(join(self.base_path, 'config.json'), 'w'))

    def iter(self, i, **kwargs):
        if i % 100 == 0 and 'loss' in kwargs:
            loss = kwargs['loss']
            print(f'iteration {i}: loss {loss:.4f}')

    def save_weights(self, only_trainable=False, weight_file='weights.pth'):
        if self.model is None:
            raise AttributeError('You need to provide a model reference when initializing TrainingTracker to save weights.')

        weights_path = join(self.base_path, weight_file)

        weight_dict = self.model.state_dict()

        if only_trainable:
            weight_dict = {n: weight_dict[n] for n, p in self.model.named_parameters() if p.requires_grad}
        
        torch.save(weight_dict, weights_path)
        log.info(f'Saved weights to {weights_path}')

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        """ automatically stop processes if used in a context manager """
        pass        