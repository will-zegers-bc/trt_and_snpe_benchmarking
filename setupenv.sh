#! /usr/bin/env bash

ARCH=$(uname -m)
if [ ${ARCH} == "aarch64" ]; then
    export SNPE_TARGET_ARCH="aarch64-linux-gcc4.9"
elif [ ${ARCH} == "x86_64" ]; then
    export SNPE_TARGET_ARCH="x86_64-linux-clang"
else
    echo "Unsupported architecture"
    exit 1
fi

export PATH="${PATH}:${SNPE_ROOT}/bin/${SNPE_TARGET_ARCH}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${SNPE_ROOT}/lib/${SNPE_TARGET_ARCH}"
export ADSP_LIBRARY_PATH="${SNPE_ROOT}/lib/dsp;/usr/lib/rfsa/adsp;/dsp"
export PYTHONPATH="${PYTHONPATH}:${SNPE_ROOT}/lib/python"

source venv/bin/activate
