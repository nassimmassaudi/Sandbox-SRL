# xvfb-run -s "-screen 0 1400x900x24" python

xvfb-run -s "-screen 0 1400x900x24" python train_curl.py \
    --domain_name cheetah \
    --task_name run \
    --img_source "video" \
    --resource_files "~/Sandbox-SRL/environments/videos/crowd-1.mp4" \
    --encoder_type pixel \
    --action_repeat 8 \
    --save_tb --pre_transform_image_size 100 --image_size 84 \
    --work_dir log \
    --agent curl_sac --frame_stack 3 \
    --seed -1 --critic_lr 1e-3 --actor_lr 1e-3 --eval_freq 10000 --batch_size 128 --num_train_steps 1000000 