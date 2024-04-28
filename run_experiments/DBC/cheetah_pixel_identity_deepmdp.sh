xvfb-run -s "-screen 0 1400x900x24" python trains/train_dbc.py \
    --domain_name cheetah \
    --task_name run \
    --agent deepmdp \
    --img_source "video" \
    --resource_files "~/Sandbox-SRL/environments/videos/crowd-1.mp4" \
    --encoder_type pixel \
    --decoder_type reward \
    --action_repeat 4 \
    --save_video \
    --save_tb \
    --work_dir log \
    --seed 1 \
    --save_tb \
    --save_model \
    --save_buffer \
    --save_video  \
    --render
    