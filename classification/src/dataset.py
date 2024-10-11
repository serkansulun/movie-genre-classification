import utils as u
import torch
import random
import numpy as np
from pathlib import Path

SINGULAR_FEATURES = ('asr_sentiment', 'ocr_sentiment')

DATA = None     # global, will be initialized later

def process_dataset(config):
    global DATA

    if config['debug'] or u.is_running_locally():
        data_path = Path(config['dataset_dir']) / 'features_debug.pkl'
    else:
        data_path = Path(config['dataset_dir']) / 'features.pkl'     # TODO

    print(f'Loading entire data from {data_path} ...', end=' ', flush=True)
    DATA = u.pickle_load(data_path)
    print('Loaded.', flush=True)

    stats = DATA['stats']
    device = 'cuda' if torch.cuda.is_available() and not config['no_cuda'] else 'cpu'

    for video_path in DATA['samples'].keys():
        # Add video path as another key because we will later convert this dict to a list.
        # So we save the keys in order not to lose them.
        DATA['samples'][video_path]['video_path'] = video_path

        sample_features = set(DATA['samples'][video_path]['features'].keys())

        # Remove unwanted features
        features_to_delete = list(sample_features - set(config['features']))
        # Also remove empty features
        for feature_name, tensor in DATA['samples'][video_path]['features'].items():
            if u.is_empty(tensor) and feature_name not in features_to_delete:
                features_to_delete.append(feature_name)
        for feature_name in features_to_delete:
            del DATA['samples'][video_path]['features'][feature_name]
        
        if config['normalize']:     # min-max normalization
            for feature_name in DATA['samples'][video_path]['features'].keys():
                DATA['samples'][video_path]['features'][feature_name] = u.normalize(
                    DATA['samples'][video_path]['features'][feature_name], 
                    stats[feature_name]['min'], 
                    stats[feature_name]['max']
                    )
        if config['standardize']:   # standardize to zero mean unit variance
            for feature_name in DATA['samples'][video_path]['features'].keys():
                DATA['samples'][video_path]['features'][feature_name] = u.standardize(
                    DATA['samples'][video_path]['features'][feature_name], 
                    stats[feature_name]['mean'], 
                    stats[feature_name]['std']
                    )
                
        # Add zeros if feature is missing
        features_to_add = set(config['features']) - set(DATA['samples'][video_path]['features'].keys())  
        for feature_name in features_to_add:
            DATA['samples'][video_path]['features'][feature_name] = torch.zeros((
                config['feature_lengths'][feature_name],
                config['feature_dims'][feature_name],
                ))
            
        # Pad with zeros if needed
        for feature_name in DATA['samples'][video_path]['features'].keys():

            if len(DATA['samples'][video_path]['features'][feature_name].shape) == 1:
                DATA['samples'][video_path]['features'][feature_name] = DATA['samples'][video_path]['features'][feature_name][np.newaxis, :]

            source_length = DATA['samples'][video_path]['features'][feature_name].shape[0]
            if feature_name in SINGULAR_FEATURES:
                target_length = 1
            else:
                target_length = config['feature_lengths'][feature_name]
            if target_length > source_length:
                pad_amount = target_length - source_length
                DATA['samples'][video_path]['features'][feature_name] = np.pad(DATA['samples'][video_path]['features'][feature_name], ((0, pad_amount), (0, 0)))
            
        # Convert all to Tensor, move to device
        for feature_name in DATA['samples'][video_path]['features'].keys():
            DATA['samples'][video_path]['features'][feature_name] = torch.Tensor(DATA['samples'][video_path]['features'][feature_name]).to(device)
        DATA['samples'][video_path]['label'] = torch.Tensor(DATA['samples'][video_path]['label']).to(device)

    # # SPLIT INTO TRAIN AND TEST

    # Convert to list
    DATA['samples'] = list(DATA['samples'].values())

    splits_fp = Path('preprocessing/data/splits.json')
    if splits_fp.exists():
        splits = u.json_load(splits_fp)
    else:
        splits = {}
        files = sorted([sample['video_path'] for sample in DATA['samples']])
        trn_split_index = int(round(len(DATA['samples']) * config['trn_val_tst_ratio'][0]))
        tst_split_index = -int(round(len(DATA['samples']) * config['trn_val_tst_ratio'][-1]))
        splits['trn'] = files[:trn_split_index]
        splits['val'] = files[trn_split_index:tst_split_index]
        splits['tst'] = files[tst_split_index:]
        u.json_save(splits, splits_fp)

    for split, splits in splits.items():
        DATA[split] = [sample for sample in DATA['samples'] if sample['video_path'] in splits]
    
    del DATA['samples']


def select_elements_with_distance(lst, n, random_frame=True):
    '''Selects random elements, from equal distanced segments.
    It first divides the list into segments, and then gets a random sample from each segment'''

    n_lst = len(lst)

    step = n_lst // n
    inds = list(range(0, n_lst - (n_lst % n), step))    # indices with equal distance

    if random_frame:
        # add the end, and pick randomly between indices
        inds.append(n_lst)
        selected_elements = [lst[random.randint(inds[i], inds[i+1] - 1)] for i in range(len(inds) - 1)]
    else:
        # just take with equal space
        selected_elements = [lst[ind] for ind in inds]     

    return selected_elements   


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, mode, feature_lengths, feature_dims):

        self.mode = mode
        self.feature_lengths = feature_lengths
        self.feature_dims = feature_dims
        
        assert self.mode in ('trn', 'val', 'tst'), 'Mode can only be trn, val, or tst.'
        self.data = DATA[self.mode]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]['features'].copy()
        video_path = self.data[idx]['video_path']
        label = self.data[idx]['label']
        for feature_name in sample.keys():
            target_length = self.feature_lengths[feature_name]
            source_length = sample[feature_name].shape[0]

            if source_length > target_length:
                if self.mode == 'train':    # Random frames
                    inds = sorted(random.sample(range(source_length), target_length))
                else:   # Equidistant frames
                    inds = u.equidistant_indices(source_length, target_length)
                sample[feature_name] = sample[feature_name][inds, :]

        return sample, label, video_path
    
    def get_label_counts(self):
        labels = [sample['label'] for sample in self.data]
        labels = torch.stack(labels)
        counts = torch.sum(labels, 0)
        return counts

