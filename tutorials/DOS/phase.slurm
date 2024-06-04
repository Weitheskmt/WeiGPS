#!/bin/bash

#SBATCH --job-name=1_ph
#SBATCH -p amd_512
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -o array-job.%A.log # Output file

# 获取传递给脚本的参数
filename=$1

# 检查是否传递了参数
if [ -z "$filename" ]; then
    echo "Error: Missing filename parameter."
    exit 1
fi

python "$filename"
