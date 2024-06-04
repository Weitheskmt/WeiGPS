mkdir -p figs data

#!/bin/bash
OMEGA=0.1
DISORDER=2
SEED=1

for ((s=0; s<=32; s++)); do
    # 创建唯一的文件名
    filename="weigps_dynamics_$s.py"
    sed -e "s/VALUE_OMEGA/$OMEGA/g" -e "s/VALUE_DISORDER/$DISORDER/g" -e "s/VALUE_S/$SEED/g" -e "s/VALUE_UP/$s/g" 0_weigps_dynamics.py > "$filename"
    sbatch dynamics.slurm "$filename"
done
