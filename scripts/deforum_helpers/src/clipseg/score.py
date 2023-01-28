from torch.functional import Tensor

import torch
import inspect
import json
import yaml
import time
import sys

from general_utils import log

import numpy as np
from os.path import expanduser, join, isfile, realpath

from torch.utils.data import DataLoader

from metrics import FixedIntervalMetrics

from general_utils import load_model, log, score_config_from_cli_args, AttributeDict, get_attribute, filter_args


DATASET_CACHE = dict()

def load_model(checkpoint_id, weights_file=None, strict=True, model_args='from_config', with_config=False, ignore_weights=False):

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

    if isfile(weights_file) and not ignore_weights:
        weights = torch.load(weights_file)
        for _, w in weights.items():
            assert not torch.any(torch.isnan(w)), 'weights contain NaNs'
        model.load_state_dict(weights, strict=strict)
    else:
        if not ignore_weights:
            raise FileNotFoundError(f'model checkpoint {weights_file} was not found')

    if with_config:
        return model, config
    
    return model


def compute_shift2(model, datasets, seed=123, repetitions=1):
    """ computes shift """
    
    model.eval()
    model.cuda()

    import random
    random.seed(seed)

    preds, gts = [], []
    for i_dataset, dataset in enumerate(datasets):

        loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False, drop_last=False)

        max_iterations = int(repetitions * len(dataset.dataset.data_list))
        
        with torch.no_grad():

            i, losses = 0, []
            for i_all, (data_x, data_y) in enumerate(loader):

                data_x = [v.cuda(non_blocking=True) if v is not None else v for v in data_x]
                data_y = [v.cuda(non_blocking=True) if v is not None else v for v in data_y]

                pred, = model(data_x[0], data_x[1], data_x[2])
                preds += [pred.detach()]
                gts += [data_y]

                i += 1
                if max_iterations and i >= max_iterations:
                    break
        
    from metrics import FixedIntervalMetrics
    n_values = 51
    thresholds = np.linspace(0, 1, n_values)[1:-1]
    metric = FixedIntervalMetrics(resize_pred=True, sigmoid=True, n_values=n_values)

    for p, y in zip(preds, gts):
        metric.add(p.unsqueeze(1), y)     
            
    best_idx = np.argmax(metric.value()['fgiou_scores'])
    best_thresh = thresholds[best_idx]

    return best_thresh


def get_cached_pascal_pfe(split, config):
    from datasets.pfe_dataset import PFEPascalWrapper
    try:
        dataset =  DATASET_CACHE[(split, config.image_size, config.label_support, config.mask)]
    except KeyError:
        dataset = PFEPascalWrapper(mode='val', split=split, mask=config.mask, image_size=config.image_size, label_support=config.label_support)
        DATASET_CACHE[(split, config.image_size, config.label_support, config.mask)] = dataset
    return dataset




def main():
    config, train_checkpoint_id = score_config_from_cli_args()

    metrics = score(config, train_checkpoint_id, None)

    for dataset in metrics.keys():
        for k in metrics[dataset]:
            if type(metrics[dataset][k]) in {float, int}:
                print(dataset, f'{k:<16} {metrics[dataset][k]:.3f}')


def score(config, train_checkpoint_id, train_config):

    config = AttributeDict(config)

    print(config)

    # use training dataset and loss
    train_config = AttributeDict(json.load(open(f'logs/{train_checkpoint_id}/config.json')))

    cp_str = f'_{config.iteration_cp}' if config.iteration_cp is not None else ''


    model_cls = get_attribute(train_config['model'])

    _, model_args, _ = filter_args(train_config, inspect.signature(model_cls).parameters)

    model_args = {**model_args, **{k: config[k] for k in ['process_cond', 'fix_shift'] if k in config}}

    strict_models = {'ConditionBase4', 'PFENetWrapper'}
    model = load_model(train_checkpoint_id, strict=model_cls.__name__ in strict_models, model_args=model_args, 
                        weights_file=f'weights{cp_str}.pth', )
                           

    model.eval()
    model.cuda()

    metric_args = dict()

    if 'threshold' in config:
        if config.metric.split('.')[-1] == 'SkLearnMetrics':
            metric_args['threshold'] = config.threshold

    if 'resize_to' in config:
        metric_args['resize_to'] = config.resize_to

    if 'sigmoid' in config:
        metric_args['sigmoid'] = config.sigmoid    

    if 'custom_threshold' in config:
        metric_args['custom_threshold'] = config.custom_threshold     

    if config.test_dataset == 'pascal':
        
        loss_fn = get_attribute(train_config.loss)
        # assume that if no split is specified in train_config, test on all splits, 
        
        if 'splits' in config:
            splits = config.splits 
        else:
            if 'split' in train_config and type(train_config.split) == int:
                # unless train_config has a split set, in that case assume train mode in training
                splits = [train_config.split]
                assert train_config.mode == 'train'
            else:
                splits = [0,1,2,3]
            
        log.info('Test on these splits', splits)

        scores = dict()
        for split in splits:

            shift = config.shift if 'shift' in config else 0

            # automatic shift
            if shift == 'auto':
                shift_compute_t = time.time()
                shift = compute_shift2(model, [get_cached_pascal_pfe(s, config) for s in range(4) if s != split], repetitions=config.compute_shift_fac)
                log.info(f'Best threshold is {shift}, computed on splits: {[s for s in range(4) if s != split]}, took {time.time() - shift_compute_t:.1f}s')

            dataset = get_cached_pascal_pfe(split, config)

            eval_start_t = time.time()

            loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False, drop_last=False)

            assert config.batch_size is None or config.batch_size == 1, 'When PFE Dataset is used, batch size must be 1'

            metric = FixedIntervalMetrics(resize_pred=True, sigmoid=True, custom_threshold=shift, **metric_args)

            with torch.no_grad():

                i, losses = 0, []
                for i_all, (data_x, data_y) in enumerate(loader):

                    data_x = [v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for v in data_x]
                    data_y = [v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for v in data_y]

                    if config.mask == 'separate':  # for old CondBase model
                        pred, = model(data_x[0], data_x[1], data_x[2])
                    else:
                        # assert config.mask in {'text', 'highlight'}
                        pred, _, _, _  = model(data_x[0], data_x[1], return_features=True)

                    # loss = loss_fn(pred, data_y[0])
                    metric.add(pred.unsqueeze(1) + shift, data_y)

                    # losses += [float(loss)]

                    i += 1
                    if config.max_iterations and i >= config.max_iterations:
                        break

            #scores[split] = {m: s for m, s in zip(metric.names(), metric.value())}

            log.info(f'Dataset length: {len(dataset)}, took {time.time() - eval_start_t:.1f}s to evaluate.')

            print(metric.value()['mean_iou_scores'])

            scores[split] = metric.scores()

            log.info(f'Completed split {split}')
        
        key_prefix = config['name'] if 'name' in config else 'pas'

        all_keys = set.intersection(*[set(v.keys()) for v in scores.values()])

        valid_keys = [k for k in all_keys if all(v[k] is not None and isinstance(v[k], (int, float, np.float)) for v in scores.values())]

        return {key_prefix: {k: np.mean([s[k] for s in scores.values()]) for k in valid_keys}}


    if config.test_dataset == 'coco':
        from datasets.coco_wrapper import COCOWrapper

        coco_dataset = COCOWrapper('test', fold=train_config.fold, image_size=train_config.image_size, mask=config.mask,
                                    with_class_label=True)

        log.info('Dataset length', len(coco_dataset))
        loader = DataLoader(coco_dataset, batch_size=config.batch_size, num_workers=2, shuffle=False, drop_last=False)
        
        metric = get_attribute(config.metric)(resize_pred=True, **metric_args)

        shift = config.shift if 'shift' in config else 0

        with torch.no_grad():

            i, losses = 0, []
            for i_all, (data_x, data_y) in enumerate(loader):
                data_x = [v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for v in data_x]
                data_y = [v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for v in data_y]

                if config.mask == 'separate':  # for old CondBase model
                    pred, = model(data_x[0], data_x[1], data_x[2])
                else:
                    # assert config.mask in {'text', 'highlight'}
                    pred, _, _, _  = model(data_x[0], data_x[1], return_features=True)

                metric.add([pred + shift], data_y)

                i += 1
                if config.max_iterations and i >= config.max_iterations:
                    break                

        key_prefix = config['name'] if 'name' in config else 'coco'      
        return {key_prefix: metric.scores()}
        #return {key_prefix: {k: v for k, v in zip(metric.names(), metric.value())}}


    if config.test_dataset == 'phrasecut':
        from datasets.phrasecut import PhraseCut

        only_visual = config.only_visual is not None and config.only_visual
        with_visual = config.with_visual is not None and config.with_visual

        dataset = PhraseCut('test', 
                            image_size=train_config.image_size,
                            mask=config.mask, 
                            with_visual=with_visual, only_visual=only_visual, aug_crop=False, 
                            aug_color=False)

        loader = DataLoader(dataset, batch_size=config.batch_size, num_workers=2, shuffle=False, drop_last=False)
        metric = get_attribute(config.metric)(resize_pred=True, **metric_args)

        shift = config.shift if 'shift' in config else 0


        with torch.no_grad():

            i, losses = 0, []
            for i_all, (data_x, data_y) in enumerate(loader):
                data_x = [v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for v in data_x]
                data_y = [v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for v in data_y]

                pred, _, _, _  = model(data_x[0], data_x[1], return_features=True)
                metric.add([pred + shift], data_y)

                i += 1
                if config.max_iterations and i >= config.max_iterations:
                    break                

        key_prefix = config['name'] if 'name' in config else 'phrasecut'      
        return {key_prefix: metric.scores()}
        #return {key_prefix: {k: v for k, v in zip(metric.names(), metric.value())}}

    if config.test_dataset == 'pascal_zs':
        from third_party.JoEm.model.metric import Evaluator
        from third_party.JoEm.data_loader import get_seen_idx, get_unseen_idx, VOC
        from datasets.pascal_zeroshot import PascalZeroShot, PASCAL_VOC_CLASSES_ZS

        from models.clipseg import CLIPSegMultiLabel

        n_unseen = train_config.remove_classes[1]

        pz = PascalZeroShot('val', n_unseen, image_size=352)
        m = CLIPSegMultiLabel(model=train_config.name).cuda()
        m.eval();

        print(len(pz), n_unseen)
        print('training removed', [c for class_set in PASCAL_VOC_CLASSES_ZS[:n_unseen // 2] for c in class_set])

        print('unseen', [VOC[i] for i in get_unseen_idx(n_unseen)])
        print('seen', [VOC[i] for i in get_seen_idx(n_unseen)])

        loader = DataLoader(pz, batch_size=8)
        evaluator = Evaluator(21, get_unseen_idx(n_unseen), get_seen_idx(n_unseen))

        for i, (data_x, data_y) in enumerate(loader):
            pred = m(data_x[0].cuda())
            evaluator.add_batch(data_y[0].numpy(), pred.argmax(1).cpu().detach().numpy())
            
            if config.max_iter is not None and i > config.max_iter: 
                break
                
        scores = evaluator.Mean_Intersection_over_Union()        
        key_prefix = config['name'] if 'name' in config else 'pas_zs'      

        return {key_prefix: {k: scores[k] for k in ['seen', 'unseen', 'harmonic', 'overall']}}

    elif config.test_dataset in {'same_as_training', 'affordance'}:
        loss_fn = get_attribute(train_config.loss)

        metric_cls = get_attribute(config.metric)
        metric = metric_cls(**metric_args)

        if config.test_dataset == 'same_as_training':
            dataset_cls = get_attribute(train_config.dataset)
        elif config.test_dataset == 'affordance':
            dataset_cls = get_attribute('datasets.lvis_oneshot3.LVIS_Affordance')
            dataset_name = 'aff'
        else:
            dataset_cls = get_attribute('datasets.lvis_oneshot3.LVIS_OneShot')
            dataset_name = 'lvis'

        _, dataset_args, _ = filter_args(config, inspect.signature(dataset_cls).parameters)

        dataset_args['image_size'] = train_config.image_size  # explicitly use training image size for evaluation

        if model.__class__.__name__ == 'PFENetWrapper':
            dataset_args['image_size'] = config.image_size

        log.info('init dataset', str(dataset_cls))
        dataset = dataset_cls(**dataset_args)

        log.info(f'Score on {model.__class__.__name__} on {dataset_cls.__name__}')

        data_loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=config.shuffle)

        # explicitly set prompts
        if config.prompt == 'plain':
            model.prompt_list = ['{}']
        elif config.prompt == 'fixed':
            model.prompt_list = ['a photo of a {}.']
        elif config.prompt == 'shuffle':
            model.prompt_list = ['a photo of a {}.', 'a photograph of a {}.', 'an image of a {}.', '{}.']
        elif config.prompt == 'shuffle_clip':
            from models.clip_prompts import imagenet_templates
            model.prompt_list = imagenet_templates

        config.assume_no_unused_keys(exceptions=['max_iterations'])

        t_start = time.time()

        with torch.no_grad():  # TODO: switch to inference_mode (torch 1.9)
            i, losses = 0, []
            for data_x, data_y in data_loader:

                data_x = [x.cuda() if isinstance(x, torch.Tensor) else x for x in data_x]
                data_y = [x.cuda() if isinstance(x, torch.Tensor) else x for x in data_y]

                if model.__class__.__name__ in {'ConditionBase4', 'PFENetWrapper'}:
                    pred, = model(data_x[0], data_x[1], data_x[2])
                    visual_q = None
                else:
                    pred, visual_q, _, _  = model(data_x[0], data_x[1], return_features=True)

                loss = loss_fn(pred, data_y[0])

                metric.add([pred], data_y)

                losses += [float(loss)]

                i += 1
                if config.max_iterations and i >= config.max_iterations:
                    break

        # scores = {m: s for m, s in zip(metric.names(), metric.value())}
        scores = metric.scores()

        keys = set(scores.keys())
        if dataset.negative_prob > 0 and 'mIoU' in keys:
            keys.remove('mIoU')

        name_mask = dataset.mask.replace('text_label', 'txt')[:3]
        name_neg = '' if dataset.negative_prob == 0 else '_' + str(dataset.negative_prob)
        
        score_name = config.name if 'name' in config else f'{dataset_name}_{name_mask}{name_neg}'

        scores = {score_name: {k: v for k,v in scores.items() if k in keys}}
        scores[score_name].update({'test_loss': np.mean(losses)})

        log.info(f'Evaluation took {time.time() - t_start:.1f}s')

        return scores
    else:
        raise ValueError('invalid test dataset')









if __name__ == '__main__':
    main()