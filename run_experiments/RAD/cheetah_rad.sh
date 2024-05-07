xvfb-run -s "-screen 0 1400x900x24" python train_rad.py \
    --domain_name cheetah \
    --task_name run \
    --img_source video \
    --resource_files ~/Sandbox-SRL/environments/video/crowd-1.mp4 \
    --encoder_type pixel \
    --work_dir log/RAD \
    --action_repeat 4 \
    --num_eval_episodes 10 \
    --agent rad_sac \
    --frame_stack 3 \
    --data_augs crop \
    --seed 1 \
    --num_train_steps 100000 \

# Initially tested on 8 action repeat