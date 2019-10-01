#! /usr/bin/env python3

from argparse import ArgumentParser
from collections import Iterable
import os

try:
    import matplotlib.pyplot as plt
except ImportError as ex:
    print("[-] matplotlib not found. Plotting functionality diasbled")
import numpy as np
import pandas as pd

KB_TO_MB = 1. / 1024

ACCURACY_DIR = 'data/accuracy'
THROUGHPUT_DIR = 'data/throughput'
MEMORY_DIR = 'data/memory'
OUTPUT_CSV = 'data/benchmarks.csv'

DATA_TYPES = ['tf', 'float', 'half']
COLUMN_NAMES = ['net_name', 'data_type', 'throughput',
                'peak_uss', 'peak_pss', 'peak_rss',
                'accuracy', 'precision', 'recall']


def build_dataframe(platform, mem_dir=MEMORY_DIR, tp_dir=THROUGHPUT_DIR, acc_dir=ACCURACY_DIR, columns=COLUMN_NAMES):
    records = []

    for data_type in DATA_TYPES:
        mem_df = pd.read_csv(os.path.join(mem_dir, platform, data_type + '.csv'))
        tp_df = pd.read_csv(os.path.join(tp_dir, platform, data_type + '.csv'))
        acc_df = pd.read_csv(os.path.join(acc_dir, platform, data_type + '.csv'))

        assert set(mem_df.net_name.unique()) == set(tp_df.net_name.unique()) == set(acc_df.net_name.unique())
        for net_name in mem_df.net_name.unique():
            rec = {
                'net_name': net_name,
                'data_type': data_type,

                'throughput': tp_df.loc[tp_df['net_name'] == net_name, 'throughput'].values[0],

                'peak_uss': int(KB_TO_MB * mem_df.loc[mem_df['net_name'] == net_name, 'peak_uss'].values[0]),
                'peak_pss': int(KB_TO_MB * mem_df.loc[mem_df['net_name'] == net_name, 'peak_pss'].values[0]),
                'peak_rss': int(KB_TO_MB * mem_df.loc[mem_df['net_name'] == net_name, 'peak_rss'].values[0]),

                'accuracy': acc_df.loc[acc_df['net_name'] == net_name, 'accuracy'].values[0],
                'precision': acc_df.loc[acc_df['net_name'] == net_name, 'precision'].values[0],
                'recall': acc_df.loc[acc_df['net_name'] == net_name, 'recall'].values[0],
            }
            records.append(rec)

    return pd.DataFrame(records, columns=columns)


def plot(df, column_name, config):
    fig, axs = plt.subplots(figsize=(24, 13.5), nrows=3, ncols=6)
    fig.tight_layout()
    fig.subplots_adjust(left=0.05, top=0.925, bottom=0.025, hspace=0.20, wspace=0.35)
    fig.suptitle(config['title'], fontsize=16)

    global_max = df[column_name].max() * 1.1
    if isinstance(global_max, Iterable):                                         
        global_max = max(global_max)                                              

    grouped = df.groupby('net_name')
    for key, ax in zip(sorted(grouped.groups.keys()), axs.flatten()):
        group = grouped.get_group(key)
        if 'xgroups' in config.keys():
            ind = np.arange(3)
            for i, (color, xgroup) in enumerate(zip(config['plot']['color'], config['xgroups'])):
                subgroup = group.loc[group['data_type'] == xgroup]
                ax.bar(ind+(i*0.25), subgroup[column_name].values[0],  width=0.25, color=color)
        else:
            group.plot(kind='bar', x='data_type', y=column_name, ax=ax, **config['plot'])
            ax.legend().remove()
        
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
        ax.set_title(key)
        ax.set_xlabel('')
        ax.set_ylabel(config['axis_name'])
        ax.set_yscale(**config['scale'])
        if config['ylims'] == 'global':
            ax.set_ylim(top=global_max)
        elif config['ylims'] == 'local_range':
            local_max = group[column_name].max() * 1.02
            local_min = group[column_name].min() * 0.98
            if isinstance(local_max, Iterable):                                         
                local_max = max(local_max)                                              
                local_min = min(local_min)                                              
            ax.set_ylim(top=local_max, bottom=local_min)

    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('platform', type=str, choices=['nv_tx2', 'qc_sd845'])
    args = parser.parse_args()

    if not os.path.isfile(OUTPUT_CSV):
        df = build_dataframe()
        df.to_csv(OUTPUT_CSV, index=False)
    else:
        df = pd.read_csv(OUTPUT_CSV)

    color = ('tab:red', 'tab:green', 'tab:blue')
    # Plot max memory usage
    metrics = ['peak_uss', 'peak_pss', 'peak_rss']
    plt_config = {
        'axis_name': 'Memory Footprint (MB)',
        'scale': {'value': 'linear'},
        'title': 'Peak Memory Usage',
        'xgroups': ('tf', 'trt-float', 'trt-half'),
        'plot': {
            'color': color,
        },
        'ylims': 'global',
    }
    plot(df, metrics, plt_config)

    # Plot throughput
    plt_config = {
        'axis_name': 'Throughput (infc/s)',
        'scale': {'value': 'linear'},
        'title': 'Throughput (Inferences per Second)',
        'ylims': 'default',
        'plot': {
            'color': color[0],
        }
    }
    plot(df, 'throughput', plt_config)

    # Plot accuracy metrics
    metrics = ['accuracy', 'precision', 'recall']
    plt_config = {
        'axis_name': 'Score',
        'scale': {'value': 'linear'},
        'title': 'Accuracy Metrics',
        'xgroups': ('tf', 'trt-float', 'trt-half'),
        'ylims': 'local_range',
        'plot': {
            'color': color,
            'width': 0.15
        },
    }
    plot(df, metrics, plt_config)
