import os
import shutil
import torch
import random

import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from torchmimic.data import IHMDataset
from torchmimic.data import DecompensationDataset
from torchmimic.data import LOSDataset
from torchmimic.data import PhenotypingDataset

from libauc.sampler import DualSampler


def pad_colalte(batch):
    xx, yy, lens, mask, index = zip(*batch)
    x = pad_sequence(xx, batch_first=True)
    y = torch.FloatTensor(yy)
    # index = torch.Tensor(index)

    return x, y, lens, mask, index


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print(f"Experiment dir: {path}")

    if scripts_to_save is not None:
        if not os.path.exists(os.path.join(path, "scripts")):
            os.mkdir(os.path.join(path, "scripts"))
        for script in scripts_to_save:
            dst_file = os.path.join(path, "scripts", os.path.basename(script))
            shutil.copyfile(script, dst_file)


# Takes dict of random samples per task, returns a shuffled list of buffer_size length with an equal distribution of random samples from each task
def update_buffer(task_num, task_samples, buffer_size):
    buff = []

    # Calculate the number of samples to take from each list
    samples_per_list = buffer_size // (task_num + 1)

    for _, lst in enumerate(task_samples.values()):
        # Randomly sample values from the list
        sampled_values = random.sample(lst, samples_per_list)
        buff.extend(sampled_values)

    # Shuffle the list
    random.shuffle(buff)
    return buff


def get_samples(task_num, buffer_size, train_loader):
    # get specified number of random samples

    random_samples = []
    sample_idx = random.sample(range(len(train_loader)), buffer_size)
    for idx, (data, label, lens, mask, index) in enumerate(train_loader):
        if idx in sample_idx:
            random_samples.append((data, label, lens, mask, index, task_num))

    random.shuffle(random_samples)
    return random_samples


# returns task training loader
def get_train_loader(
    task_num,
    task_name,
    tasks,
    lf_map,
    train_batch_size,
    sample_size,
    workers,
    device,
    pAUC=False,
    region=0,
):
    ss = [1, 1, 1, 0.5, 0.25]
    if region == 0:
        clf = (
            (lf_map[task_num - 1] + "_train.csv")
            if (task_num > 0 and len(tasks) > 2)
            else "train_listfile.csv"
        )
    else:
        clf = f"{lf_map[region-1]}_train.csv" if task_num > 0 else "train_listfile.csv"
        ss = [1, ss[region]]

    if task_name == "ihm":
        train_dataset = IHMDataset(
            tasks[task_num],
            train=True,
            n_samples=sample_size,
            customListFile=clf,
        )
    elif task_name == "decomp":
        train_dataset = DecompensationDataset(
            tasks[task_num],
            train=True,
            n_samples=int((sample_size * ss[task_num]) * 0.7),
            customListFile=clf,
        )
    elif task_name == "los":
        train_dataset = LOSDataset(
            tasks[task_num],
            train=True,
            n_samples=int((sample_size * ss[task_num]) * 0.7),
            customListFile=clf,
        )
    elif task_name == "phen":
        train_dataset = PhenotypingDataset(
            tasks[task_num], train=True, n_samples=sample_size, customListFile=clf
        )
    kwargs = {"num_workers": workers, "pin_memory": True} if device else {}

    if pAUC:
        sampler = DualSampler(train_dataset, train_batch_size, sampling_rate=0.5)
        train_loader = DataLoader(
            train_dataset,
            sampler=sampler,
            batch_size=train_batch_size,
            # shuffle=True,
            collate_fn=pad_colalte,
            **kwargs,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            collate_fn=pad_colalte,
            **kwargs,
        )

    return train_loader


# Returns validation loaders for each task
def get_val_loaders(
    task_name, tasks, lf_map, val_batch_size, sample_size, workers, device, region
):
    val_loaders = []
    clf = None
    for task_num, task_data in enumerate(tasks):
        ss = [1, 1, 1, 0.5, 0.25]
        if region == 0:
            clf = (
                (lf_map[task_num - 1] + "_val.csv")
                if (task_num > 0 and len(tasks) > 2)
                else "val_listfile.csv"
            )
        else:
            clf = f"{lf_map[region-1]}_val.csv" if task_num > 0 else "val_listfile.csv"
            ss = [1, ss[region]]

        if task_name == "ihm":
            val_dataset = IHMDataset(
                task_data,
                train=False,
                n_samples=sample_size,
                customListFile=clf,
            )
        elif task_name == "decomp":
            val_dataset = DecompensationDataset(
                task_data,
                train=False,
                n_samples=int((sample_size * ss[task_num]) * 0.15),  # 100000
                customListFile=clf,
            )
        elif task_name == "los":
            val_dataset = LOSDataset(
                task_data,
                train=False,
                n_samples=int((sample_size * ss[task_num]) * 0.15),
                customListFile=clf,
            )
        elif task_name == "phen":
            val_dataset = PhenotypingDataset(
                task_data, train=False, n_samples=sample_size, customListFile=clf
            )

        kwargs = {"num_workers": workers, "pin_memory": True} if device else {}
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            drop_last=True,
            shuffle=False,
            collate_fn=pad_colalte,
            **kwargs,
        )

        val_loaders.append(val_loader)

    return val_loaders


def get_test_loaders(
    task_name, tasks, lf_map, test_batch_size, sample_size, workers, device, region
):
    test_loaders = []
    clf = None
    for task_num, task_data in enumerate(tasks):
        ss = [1, 1, 1, 0.5, 0.25]
        if region == 0:
            clf = (
                (lf_map[task_num - 1] + "_test.csv")
                if (task_num > 0 and len(tasks) > 2)
                else "test_listfile.csv"
            )
        else:
            clf = (
                f"{lf_map[region-1]}_test.csv" if task_num > 0 else "test_listfile.csv"
            )
            ss = [1, ss[region]]

        if task_name == "ihm":
            test_dataset = IHMDataset(
                task_data,
                train=False,
                n_samples=sample_size,
                customListFile=clf,
            )
        elif task_name == "decomp":
            test_dataset = DecompensationDataset(
                task_data,
                train=False,
                n_samples=int((sample_size * ss[task_num]) * 0.15),  # 100000
                customListFile=clf,
            )
        elif task_name == "los":
            test_dataset = LOSDataset(
                task_data,
                train=False,
                n_samples=int((sample_size * ss[task_num]) * 0.15),
                customListFile=clf,
            )
        elif task_name == "phen":
            test_dataset = PhenotypingDataset(
                task_data,
                train=False,
                n_samples=sample_size,
                customListFile=clf,
            )

        kwargs = {"num_workers": workers, "pin_memory": True} if device else {}
        test_loader = DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            drop_last=True,
            shuffle=False,
            collate_fn=pad_colalte,
            **kwargs,
        )

        test_loaders.append(test_loader)

    return test_loaders
