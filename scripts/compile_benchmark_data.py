#! /usr/bin/env python3

from collections import Iterable
import logging
import os
import sys

try:
    import matplotlib.pyplot as plt
except ImportError as ex:
    print("[-] matplotlib not found. Plotting functionality diasbled")
import numpy as np
import pandas as pd

from model_meta import NETS

KB_TO_MB = 1. / 1024

ACCURACY_DIR = 'data/accuracy'
LATENCY_DIR = 'data/throughput'
MEMORY_DIR = 'data/memory'
OUTPUT_CSV = 'data/benchmarks.csv'

DATA_TYPES = ['tf', 'trt-float', 'trt-half']
COLUMN_NAMES = ['net_name', 'data_type', 'throughput', 'peak_uss', 'peak_pss',
                'peak_rss', 'accuracy', 'precision', 'recall']


def _make_record_dict(net_name, data_type, df):
    return {
        'net_name': net_name,
        'data_type': data_type,
        'uss_avg': int(KB_TO_MB * df.uss.mean()),
        'uss_max': int(KB_TO_MB * df.uss.max()),
        'pss_avg': int(KB_TO_MB * df.pss.mean()),
        'pss_max': int(KB_TO_MB * df.pss.max()),
        'rss_avg': int(KB_TO_MB * df.rss.mean()),
        'rss_max': int(KB_TO_MB * df.rss.max()),
    }


def build_dataframe(mem_dir=MEMORY_DIR, lat_dir=LATENCY_DIR, columns=COLUMN_NAMES, acc_dir=ACCURACY_DIR):
    records = []

    for data_type in DATA_TYPES:
        mem_df = pd.read_csv(os.path.join(mem_dir, data_type + '.csv'))
        lat_df = pd.read_csv(os.path.join(lat_dir, data_type + '.csv'))
        acc_df = pd.read_csv(os.path.join(acc_dir, data_type + '.csv'))
        for net_name, net_meta in NETS.items():
            if 'exclude' in net_meta.keys() and net_meta['exclude'] is True:
                logging.info("Skipping {}".format(net_name))
                continue
    
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


def plot_common(df, column_name, config):
    fig, axs = plt.subplots(figsize=(24, 13.5), nrows=3, ncols=5)
    fig.tight_layout()
    fig.subplots_adjust(left=0.05, top=0.925, bottom=0.025, hspace=0.15, wspace=0.25)
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


def plot_grouped(df, column_name, config):
    fig, axs = plt.subplots(figsize=(24, 13.5), nrows=3, ncols=5)
    fig.tight_layout()
    fig.subplots_adjust(left=0.05, top=0.925, bottom=0.025, hspace=0.15, wspace=0.25)
    fig.suptitle(config['title'], fontsize=16)

    global_max = df[column_name].max() * 1.1
    if isinstance(global_max, Iterable):                                         
        global_max = max(global_max)                                              

    grouped = df.groupby('net_name')
    for key, ax in zip(sorted(grouped.groups.keys()), axs.flatten()):
        group = grouped.get_group(key)
        ind = np.arange(3)
        for i, (color, xgroup) in enumerate(zip(config['plot']['color'], config['xgroups'])):
            subgroup = group.loc[group['data_type'] == xgroup]
            ax.bar(ind+(i*0.25), subgroup[column_name].values[0], label=xgroup, width=0.25, color=color)
        
        ax.set_xticklabels(column_name)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
        ax.set_xticks(ind + 0.25 / 2)
        ax.set_title(key)
        ax.set_xlabel('')
        ax.set_ylabel(config['axis_name'])
        ax.set_yscale(**config['scale'])
        if config['ylims'] == 'global':
            limits = {'top': global_max}
        elif config['ylims'] == 'local_range':
            local_max = group[column_name].max() * 1.02
            local_min = group[column_name].min() * 0.98
            if isinstance(local_max, Iterable):                                         
                local_max = max(local_max)                                              
                local_min = min(local_min)                                              
            limits = {'top': local_max, 'bottom': local_min}
        else:
            limits = {}
        ax.set_ylim(**limits)                                          

        if not config['plot'].get('label', ''):                                 
            ax.legend().remove()                                                

    plt.show()
    

def plot(df, column_name, config):
    fig, axs = plt.subplots(figsize=(24, 13.5), nrows=3, ncols=5)               
    fig.tight_layout()                                                          
    fig.subplots_adjust(left=0.05, top=0.925, bottom=0.025, hspace=0.15, wspace=0.25)
    fig.suptitle(config['title'], fontsize=16)                                  
                                                                                
    global_max = df[column_name].max() * 1.1
    if isinstance(global_max, Iterable):                                         
        global_max = max(global_max)                                              
                                                                                
    grouped = df.groupby('net_name')                                            
    for key, ax in zip(sorted(grouped.groups.keys()), axs.flatten()):           
        group = grouped.get_group(key)                                          
        group.plot(kind='bar', x='data_type', y=column_name, ax=ax, **config['plot'])
                                                                                
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)                    
        ax.set_title(key)                                                       
        ax.set_xlabel('')                                                       
                                                                                
        ax.set_ylabel(config['axis_name'])                                      
        ax.set_yscale(**config['scale'])                                        
        if config['ylims'] == 'global':
            limits = {'top': global_max}
        elif config['ylims'] == 'local_range':
            local_max = group[column_name].max() * 1.02
            local_min = group[column_name].min() * 0.98
            if isinstance(local_max, Iterable):                                         
                local_max = max(local_max)                                              
                local_min = min(local_min)                                              
            limits = {'top': local_max, 'bottom': local_min}
        else:
            limits = {}
        ax.set_ylim(**limits)                                          
                                                                                
        if not config['plot'].get('label', ''):                                 
            ax.legend().remove()                                                
                                                                                
    plt.show()                                                                  


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if not os.path.isfile(OUTPUT_CSV):
        logger.info("Creating benchmark dataframe")
        df = build_dataframe()
        df.to_csv(OUTPUT_CSV)
    else:
        logger.info("Using existing benchmark dataframe")
        df = pd.read_csv(OUTPUT_CSV)

    # Plot max memory usage
    metrics = ['uss_max', 'pss_max', 'rss_max']
    plt_config = {
        'axis_name': 'Memory Footprint (MB)',
        'scale': {'value': 'linear'},
        'title': 'Maximum Memory Footprint',
        'xgroups': ('tf', 'float', 'half'),
        'plot': {
            'color': ('tab:blue', 'tab:red', 'tab:green'),
        },
        'ylims': 'global',
    }
    plot_grouped(df, metrics, plt_config)

    # Plot avg memory usage
    metrics = ['uss_avg', 'pss_avg', 'rss_avg']
    plt_config.update({
        'axis_name': 'Average Memory Footprint (MB)',
        'title': 'Average Memory Footprint',
    })
    plot_grouped(df, metrics, plt_config)

   # Plot throughput
    plt_config = {
        'axis_name': 'Throughput (infc/s)',
        'scale': {'value': 'linear'},
        'title': 'Throughput (Inferences per Second)',
        'ylims': 'default',
        'plot': {
            'color': 'tab:purple',
        }
    }
    plot(df, 'throughput', plt_config)

    # Plot performance multiplier
    plt_config = {
        'axis_name': 'Performance Gain',
        'scale': {'value': 'log', 'basey': 2},
        'title': 'Performance gain',
        'ylims': 'global',
        'plot': {
            'color': 'tab:orange',
        }
    }
    plot(df, 'gain', plt_config)
   
   # Plot accuracy metrics
    metrics = ['accuracy', 'precision', 'recall']
    plt_config = {
        'axis_name': 'Proportion correct',
        'scale': {'value': 'linear'},
        'title': 'Accuracy Metrics',
        'xgroups': ('tf', 'float', 'half'),
        'ylims': 'local_range',
        'plot': {
            'color': ('tab:brown', 'tab:gray', 'tab:olive'),
            'width': 0.15
        },
    }
    plot_grouped(df, metrics, plt_config)
