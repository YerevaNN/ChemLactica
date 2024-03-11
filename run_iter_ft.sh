#!/bin/bash

#SBATCH --job-name=ChemLactica_rej_sampling_ft  # Job name
#SBATCH --gres=gpu:1                # Number of GPUs
#SBATCH --cpus-per-task=4          # Number of CPU cores
#SBATCH --mem=50G                   # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --time=60:00:00             # Time limit hrs:min:sec
#SBATCH --output=../job_%j.log   # Standard output and error log (%j expands to jobId)
#SBATCH --nice=0                 # Nice value (the higher the value, the lower the priority)

python3 src/rejection_sampling_ft.py --from_pretrained /nfs/dgx/raid/chem/checkpoints/facebook/galactica-125m/26d322857a184fcbafda5d4a/checkpoint-118784 --model_config 125m --valid_data_dir /nfs/dgx/raid/chem/testing_data/comp_valid --train_batch_size 64 --valid_batch_size 16 --rounds 20 --steps_per_round 800 --eval_steps 800 --save_steps 800 --dataloader_num_workers 4 --gradient_accumulation_steps 1 --experiment_name gal125m_rejection_sampling_single_mol_databank --checkpoints_root_dir /nfs/dgx/raid/chem/checkpoints/ --track --track_dir /nfs/dgx/raid/chem/aim/ --flash_attn --device cuda:0 --max_learning_rate 1e-6
