xvfb-run -s "-screen 0 1400x900x24" python train_dbc.py \
    --domain_name cheetah \
    --agent deepmdp \
    --task_name run \
    --img_source video \
    --decoder_type reconstruction \
    --resource_files ~/Sandbox-SRL/environments/video/crowd-1.mp4 \
    --encoder_type pixel \
    --action_repeat 4 \
    --save_video \
    --save_tb \
    --save_wandb \
    --work_dir log/deepMDP \
    --seed 1 \
    --num_train_steps 100000 \
    
