#!/bin/bash
# perform parameter sweep for multinode scaling runs

# num_gpus=(1 2 4 8 16 32)
opt_level=(O0)
base_bs=14

mkdir -p .slurm_scripts
mkdir -p .slurm_logs

module load esslurm

# sweep
for ngpu in ${num_gpus[@]}; do
  for opt in ${opt_level[@]}; do
    echo "Processing: $ngpu gpus, $opt optim..."
    nnodes=$(python -c "from math import ceil; print(ceil($ngpu/8))")
    ngpu_per_node=$(python -c "print(min($ngpu, 8))")
    olev=$(echo $opt | cut -c2-3)
    bs=$((($olev + 1)*$base_bs))
    cp batchScript.sh submit_job_g${ngpu}_${opt}.sh
    sed -i "s/<ngpu>/${ngpu_per_node}/g" submit_job_g${ngpu}_${opt}.sh
    sed -i "s/<nnodes>/${nnodes}/g" submit_job_g${ngpu}_${opt}.sh
    sed -i "s/<output>/.slurm_logs\/ngpu_g${ngpu}_${opt}.out/g" submit_job_g${ngpu}_${opt}.sh
    sed -i "s/<opt>/${opt}/g" submit_job_g${ngpu}_${opt}.sh
    sed -i "s/<bs>/${bs}/g" submit_job_g${ngpu}_${opt}.sh
    sbatch submit_job_g${ngpu}_${opt}.sh
    mv submit_job_g${ngpu}_${opt}.sh .slurm_scripts
  done
done