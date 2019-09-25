#! /usr/bin/env bash

export SNPE_TARGET_ARCH=aarch64-linux-gcc4.9
export PATH=$PATH:$HOME/snpe/snpeexample/$SNPE_TARGET_ARCH/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/snpe/snpeexample/$SNPE_TARGET_ARCH/lib
export ADSP_LIBRARY_PATH="${SNPE_ROOT}/lib/dsp;/usr/lib/rfsa/adsp;/dsp"

source venv/bin/activate
