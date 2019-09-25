#! /usr/bin/env python3

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
LATENCY_DIR = 'data/throughput'
MEMORY_DIR = 'data/memory'
OUTPUT_CSV = 'data/benchmarks.csv'

DATA_TYPES = ['tf', 'trt-float', 'trt-half']
COLUMN_NAMES = ['net_name', 'data_type', 'throughput',
                'peak_uss', 'peak_pss', 'peak_rss',
                'accuracy', 'precision', 'recall']


def build_dataframe(mem_dir=MEMORY_DIR, lat_dir=LATENCY_DIR, columns=COLUMN_NAMES, acc_dir=ACCURACY_DIR):
    records = []

    for data_type in DATA_TYPES:
        mem_df = pd.read_csv(os.path.join(mem_dir, data_type + '.csv'))
        lat_df = pd.read_csv(os.path.join(lat_dir, data_type + '.csv'))
        acc_df = pd.read_csv(os.path.join(acc_dir, data_type + '.csv'))

        assert set(mem_df.net_name.unique()) == set(lat_df.net_name.unique()) == set(acc_df.net_name.unique())
        for net_name in mem_df.net_name.unique():
            rec = {
                'net_name': net_name,
                'data_type': data_type,

                'throughput': lat_df.loc[lat_df['net_name'] == net_name, 'throughput'].values[0],

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
                ax.bar(ind+(i*0.25), subgroup[column_name].values[0], label=xgroup, width=0.25, color=color)
        else:
            group.plot(kind='bar', x='data_type', y=column_name, ax=ax, **config['plot'])
        
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

        if not config['plot'].get('label', ''):                                 
            ax.legend().remove()                                                

    plt.show()


if __name__ == '__main__':
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
        'title': 'Maximum Memory Footprint',
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
        'axis_name': 'Proportion correct',
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
