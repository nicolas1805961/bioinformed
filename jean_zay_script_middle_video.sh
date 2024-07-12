#!/bin/bash
#SBATCH --array=1,   # Number of configuration files
#SBATCH --job-name=gpu_mono          # nom du job
#SBATCH -C v100-16g                  # reserver des GPU 16 Go seulement
##SBATCH --partition=gpu_p2          # de-commente pour la partition gpu_p2
#SBATCH --qos=qos_gpu-t3            # qos_gpu-t4 qos_gpu-dev qos_gpu-t3
#SBATCH --nodes=1                    # on demande un noeud
#SBATCH --ntasks-per-node=1          # avec une tache par noeud (= nombre de GPU ici)
#SBATCH --gres=gpu:1                 # nombre de GPU (1/4 des GPU)
#SBATCH --cpus-per-task=10           # nombre de coeurs CPU par tache (1/4 du noeud 4-GPU)
##SBATCH --cpus-per-task=3           # nombre de coeurs CPU par tache (pour gpu_p2 : 1/8 du noeud 8-GPU)
# /!\ Attention, "multithread" fait reference Ã  l'hyperthreading dans la terminologie Slurm
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=20:00:00          # 48:00:00 temps maximum d'execution demande (HH:MM:SS) 00:05:00 20:00:00  
#SBATCH --output=gpu_mono%j.out      # nom du fichier de sortie
#SBATCH --error=gpu_mono%j.out       # nom du fichier d'erreur (ici commun avec la sortie)

# nettoyage des modules charges en interactif et herites par defaut
module purge

# chargement des modules
module load pytorch-gpu/py3/1.11.0

# echo des commandes lancees
set -x

#python .\register_2_45.py --test_or_val test --dataset Lib --dirpath models\2024-02-13_14H02 --model models\2024-02-13_14H02\model_Lib_bmreg_seg_nup_0.4_bs_8_epoch_100_lr_0.0001_lmbd_0.05_gamma_0.01.pth -g 0
python train_2.py --losstype bmreg_seg --config config${SLURM_ARRAY_TASK_ID}.yaml