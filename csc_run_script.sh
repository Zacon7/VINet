#!/bin/bash
#SBATCH --job-name=hw_test
#SBATCH --account=project_2006308
#SBATCH --partition=gputest
#SBATCH --time=00:00:30
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-type=END
#SBATCH --output=output.txt

module load pytorch

pip3 install -r requirements.txt

srun python3 main.py
