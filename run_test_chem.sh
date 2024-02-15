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
python3 -m accelerate.commands.launch --config_file src/config/accelerate_config.yaml src/train.py --train_type pretrain --dir_data_type computed --from_pretrained facebook/galactica-125m --model_config 125m --training_data_dir /raid/chem/Test_Jsons_16_shuf_v0_43/ --valid_data_dir /home/ysu/chem/data/valid --train_batch_size 2 --eval_steps 5 --save_steps 5 --max_steps 5 --dataloader_num_workers 16 --experiment_name gal125m_test --checkpoints_root_dir /raid/chem/checkpoints/ --no_track --flash_attn
