#!/bin/bash

#SBATCH --job-name=chemlactica-test
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=10:00
#SBATCH --output=train_job_%j.log
#SBATCH --nice=1


export TOKENIZERS_PARALLELISM=true
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
python3 -m accelerate.commands.launch --config_file chemlactica/config/config.yaml chemlactica/train.py --train_type sft --dir_data_type computed --from_pretrained /nfs/dgx/raid/chem/checkpoints/facebook/galactica-125m/26d322857a184fcbafda5d4a/checkpoint-118784 --model_config 125m --training_data_dir /auto/home/menuab/code/sft_data/ADME_HLM/train/ --valid_data_dir /auto/home/menuab/code/sft_data/ADME_HLM/valid/ --train_batch_size 16 --eval_steps 30 --save_steps 300 --max_steps 330 --dataloader_num_workers 16 --experiment_name gal125m_test --checkpoints_root_dir /nfs/dgx/raid/chem/checkpoints/ --no_track --flash_attn
