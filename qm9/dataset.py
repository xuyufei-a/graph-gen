from torch.utils.data import DataLoader
from qm9.data.args import init_argparse
from qm9.data.collate import PreprocessQM9
from qm9.data.utils import initialize_datasets
from mypy.dataset.dataset import ProcessedDataset
import os
import numpy as np
import torch


def retrieve_dataloaders(cfg):
    assert('qm9' in cfg.dataset), 'Only QM9 dataset is supported for now.'

    if 'qm9' in cfg.dataset:
        batch_size = cfg.batch_size
        num_workers = cfg.num_workers
        filter_n_atoms = cfg.filter_n_atoms
        # Initialize dataloader
        args = init_argparse('qm9')
        # data_dir = cfg.data_root_dir

        dataset_path = 'qm9/temp/qm9_srd'

        splits = ['train', 'test', 'valid']

        datasets = {}
        for split in splits:
            with np.load(os.path.join(dataset_path, split+'.npz')) as f:
                datasets[split] = {key: torch.from_numpy(val) for key, val in f.items()}
        datasets = {split: ProcessedDataset(data) for split, data in datasets.items()}

        # Construct PyTorch dataloaders from datasets
        preprocess = PreprocessQM9(load_charges=cfg.include_charges)
        dataloaders = {split: DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=args.shuffle if (split == 'train') else False,
                                         num_workers=num_workers,
                                         collate_fn=preprocess.collate_fn)
                             for split, dataset in datasets.items()}

    charge_scale = None
    return dataloaders, charge_scale


def filter_atoms(datasets, n_nodes):
    for key in datasets:
        dataset = datasets[key]
        idxs = dataset.data['num_atoms'] == n_nodes
        for key2 in dataset.data:
            dataset.data[key2] = dataset.data[key2][idxs]

        datasets[key].num_pts = dataset.data['one_hot'].size(0)
        datasets[key].perm = None
    return datasets