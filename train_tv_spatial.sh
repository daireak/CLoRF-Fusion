#!/bin/bash
#SBATCH --job-name=INR_Train
#SBATCH --output=logs/inr_%j.log
#SBATCH --error=logs/inr_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00

# 激活您的虚拟环境
source /home/dengxiaogui/NTSR/venv/bin/activate

# 创建日志和模型权重保存文件夹 (train_tv_spatial 默认输出在 runs)
mkdir -p logs
mkdir -p runs

echo "=========================================================="
echo "Starting INR Spatial-Temporal Fusion Training Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "=========================================================="

# ================= 实验控制区 =================
# 想要跑哪个数据集，就把对应的注释(#)删掉，并给另外的加上注释即可。
# (注: Python启动命令已全部放在一行，方便整行注释/解注)

# 【1】PU 数据集
# 设置: PU数据, 缩放4倍, 秩(k)=48
python train_tv_spatial.py --data_path /home/dengxiaogui/Data/PU.mat --r_path /home/dengxiaogui/Data/R.mat --mat_key img --r_key R --scale 4 --inr siren --k 48 --lam 1.0 --alpha 0.01 --log_dir ./runs/PU_Experiment --num_iters 30000 --snr_hsi 30 --snr_msi 35

# 【2】WDC 数据集
# 设置: WDC数据, 缩放4倍, 秩(k)=48
#python train_tv_spatial.py --data_path /home/dengxiaogui/Data/WDC.mat --r_path /home/dengxiaogui/Data/R.mat --mat_key wdc --r_key R --scale 4 --inr siren --pos_enc --k 48 --lam 1.0 --alpha 0.01 --log_dir ./runs/WDC_Experiment --num_iters 30000

# 【3】Chikusei 数据集
# 设置: Chikusei数据, 缩放8倍, 秩(k)=90
#python train_tv_spatial.py --data_path /home/dengxiaogui/Data/Chikusei.mat --r_path /home/dengxiaogui/Data/R_chikusei.mat --mat_key chikusei --r_key R --scale 8 --inr relu --pos_enc --k 48 --lam 1.0 --alpha 0.01 --log_dir ./runs/Chikusei_Experiment --num_iters 10000

echo "Job finished."
