from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
from data_extractor import tasks
import pandas as pd
import numpy as np

def create_directories(eicu_path, task_path, task_list, regional=True, skip_labeling=False):

    #load predefined splits
    splits = {
        'val': set(pd.read_csv('resources/val_split.csv', dtype=str)['files']),
        'test': set(pd.read_csv('resources/test_split.csv', dtype=str)['files'])
    }

    patients = tasks.dataframe_from_csv(os.path.join(task_path, f"root/timeseries_info.csv"))

    mapper = tasks.dataframe_from_csv('resources/episode_mapper.csv')
    mapper = mapper.set_index('episode') 

    listfiles = {}

    if skip_labeling:
        for task in task_list:
            listfiles[task] = tasks.dataframe_from_csv(os.path.join(task_path, f"root/{task}_listfile.csv"))
    else:        
        listfiles = tasks.populate_root(eicu_path, task_list, splits, patients, mapper)
    

    fullname = {'pheno': 'phenotyping_split/',
                'ihm': 'in-hospital-mortality_split/',
                'decomp': 'decompensation_split/',
                'los': 'length-of-stay_split/',
            }

    for task in task_list:
        print("Creating", fullname[task])
        tasks.create_dir(task_path, task, listfiles[task], splits)

        if regional:
            print("Creating regional splits...")
            tasks.split(os.path.join(task_path, fullname[task]), eicu_path, write=True)


def main():
    parser = argparse.ArgumentParser(description="Create directories and labels for specified tasks")
    parser.add_argument('--eicu_dir', type=str, help="Path to eICU dataset", 
        default="/data/datasets/physionet.org/files/eicu-crd/2.0"
    )
    parser.add_argument('--task_dir', type=str, help="Directory where the task directories should be created",
        default="/data/datasets/eICU2MIMIC/new_split/"
    )
    
    parser.add_argument('--all', action='store_true', help="Flag to create all task directories and root/")
    
    parser.add_argument('--regional', action='store_true', help="Flag to perform regional splitting")

    parser.add_argument('--ihm', action='store_true', help="Flag to create the in-hospital-mortality_split/ directory")
    parser.add_argument('--pheno', action='store_true', help="Flag to create the phenotyping_split/ directory")
    parser.add_argument('--decomp', action='store_true', help="Flag to create the decompensation_split/ directory")
    parser.add_argument('--los', action='store_true', help="Flag to create the length-of-stay_split/ directory")
    
    parser.add_argument('--skip_labeling', action='store_true', help="Flag to skip to splitting. Should only be set after the root and all of its contents (like listfiles) for each task has been created")
    

    args, _ = parser.parse_known_args()

    task_list = ['ihm', 'pheno', 'decomp', 'los']

    if not (args.ihm or args.all):
        task_list.remove('ihm')
    if not (args.pheno or args.all):
        task_list.remove('pheno')
    if not (args.decomp or args.all):
        task_list.remove('decomp')
    if not (args.los or args.all):
        task_list.remove('los')

    create_directories(args.eicu_dir, args.task_dir, task_list, regional=args.regional, skip_labeling=args.skip_labeling)

if __name__ == '__main__':
    main()
