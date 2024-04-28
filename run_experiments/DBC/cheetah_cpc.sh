xvfb-run -s "-screen 0 1400x900x24" python train_dbc.py \
    --domain_name cheetah \
    --agent baseline \
    --task_name run \
    --img_source "video" \
    --resource_files "~/Sandbox-SRL/environments/videos/crowd-1.mp4" \
    --encoder_type pixel \
    --decoder_type contrastive \
    --action_repeat 4 \
    --save_video \
    --save_tb \
    --work_dir log \
    --seed 1