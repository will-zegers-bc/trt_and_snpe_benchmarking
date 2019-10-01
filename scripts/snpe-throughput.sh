#! /usr/bin/env bash

CONTAINER_DIR='data/dlc'

runtime=$1
duration=$2
profile=$3
case ${runtime} in
  'cpu')
    rt_flag='--use_cpu'
    suffix='.dlc'
    ;;
  'gpu')
    rt_flag='--use_gpu'
    suffix='.dlc'
    ;;
  'g16')
    rt_flag='--use_gpu_fp16'
    suffix='.dlc'
    ;;
  'dsp')
    rt_flag='--use_dsp'
    suffix='_quantized.dlc'
    ;;
esac
OUTPUT_FILE="data/throughput/"${runtime}".csv"

dlcs=$(ls --ignore '*_quantized.dlc' data/dlc)

echo "net_name,throughput" > ${OUTPUT_FILE}
for dlc in ${dlcs[@]}; do
  name=${dlc::-4} 
  container=${CONTAINER_DIR}/${name}${suffix}
  result=$(snpe-throughput-net-run --container ${container} --duration ${duration} --perf_profile ${profile} ${rt_flag})
  throughput=$(grep -oh "[0-9]*\.[0-9]*" <<< ${result} | head -n1)
  echo ${name},${throughput} | tee -a ${OUTPUT_FILE}
done
