#! /usr/bin/env python3

from argparse import ArgumentParser
from collections import Iterable
import os

import numpy as np
import pandas as pd

KB_TO_MB = 1. / 1024

ACCURACY_DIR = 'data/accuracy'
THROUGHPUT_DIR = 'data/throughput'
MEMORY_DIR = 'data/memory'
OUTPUT_CSV = 'data/benchmarks-{}.csv'

RUNTIMES = {
    'nv_tx2': ('TF', 'Float32', 'Float16'),
    'qc_sd845': ('TF', 'CPU', 'GPU', 'G16', 'DSP'),
}
COLUMN_NAMES = ['net_name', 'runtime', 'throughput', 'peak_uss', 'peak_pss',
                'peak_rss', 'accuracy', 'precision', 'recall', 'platform']


def _calculate_performance_gain(df):
    gain = np.empty(len(df))                                                      
    grouped = df.groupby('net_name')                                               

    for key, indices in grouped.groups.items():                                    
        print(key)
        group = grouped.get_group(key)                                             
        tf_lat = group.loc[group['runtime'] == 'TF', 'throughput'].values[0]  
        lats = group['throughput'].values                                          
        for i, lat in zip(indices, lats):                                          
            gain[i] = lat / tf_lat                                                 
                                                                                   
    return gain


def build_dataframe(platform, mem_dir=MEMORY_DIR, tp_dir=THROUGHPUT_DIR, acc_dir=ACCURACY_DIR, columns=COLUMN_NAMES):
    records = []

    for runtime in RUNTIMES[platform]:
        mem_df = pd.read_csv(os.path.join(mem_dir, platform, runtime + '.csv'))
        tp_df = pd.read_csv(os.path.join(tp_dir, platform, runtime + '.csv'))
        acc_df = pd.read_csv(os.path.join(acc_dir, platform, runtime + '.csv'))

        assert set(mem_df.net_name.unique()) == set(tp_df.net_name.unique()) == set(acc_df.net_name.unique())
        for net_name in mem_df.net_name.unique():
            rec = {
                'net_name': net_name,
                'runtime': runtime,

                'throughput': tp_df.loc[tp_df['net_name'] == net_name, 'throughput'].values[0],

                'peak_uss': KB_TO_MB * mem_df.loc[mem_df['net_name'] == net_name, 'peak_uss'].values[0],
                'peak_pss': KB_TO_MB * mem_df.loc[mem_df['net_name'] == net_name, 'peak_pss'].values[0],
                'peak_rss': KB_TO_MB * mem_df.loc[mem_df['net_name'] == net_name, 'peak_rss'].values[0],

                'accuracy': acc_df.loc[acc_df['net_name'] == net_name, 'accuracy'].values[0],
                'precision': acc_df.loc[acc_df['net_name'] == net_name, 'precision'].values[0],
                'recall': acc_df.loc[acc_df['net_name'] == net_name, 'recall'].values[0],

                'platform': platform,
            }
            records.append(rec)
            
    df = pd.DataFrame(records, columns=columns)
    df['gain'] = _calculate_performance_gain(df)
    return df


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('platform', type=str, choices=['nv_tx2', 'qc_sd845'])
    args = parser.parse_args()

    output_csv = OUTPUT_CSV.format(args.platform)
    df = build_dataframe(args.platform)
    df.to_csv(output_csv, index=False)

    print(df)
