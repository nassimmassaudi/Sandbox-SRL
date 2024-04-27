xvfb-run -s "-screen 0 1400x900x24" python train_rad.py \
    --domain_name cheetah \
    --task_name run \
    --encoder_type pixel --work_dir log \
    --action_repeat 8 --num_eval_episodes 10 \
    --pre_transform_image_size 100 --image_size 84 \
    --agent rad_sac --frame_stack 3 --data_augs flip  \
    --seed 23 --critic_lr 1e-3 --actor_lr 1e-3 --eval_freq 10000 --batch_size 128 --num_train_steps 200000 &
 
absl-py==2.1.0
aiohttp==3.9.4
aiosignal==1.3.1
ale-py==0.8.1
annotated-types==0.6.0
antlr4-python3-runtime==4.9.3
anyio==4.3.0
appdirs==1.4.4
argon2-cffi==23.1.0
argon2-cffi-bindings==21.2.0
arrow==1.3.0
asttokens==2.4.1
async-lru==2.0.4
async-timeout==4.0.3
atari-py==0.2.9
attrs==23.2.0
awscrt==0.19.19
Babel==2.14.0
backoff==2.2.1
beautifulsoup4==4.12.3
bleach==6.1.0
boto3==1.34.84
botocore==1.34.84
cachetools==5.3.3
certifi==2024.2.2
cffi==1.16.0
charset-normalizer==3.3.2
click==8.1.7
cloudpickle==3.0.0
comm==0.2.2
contourpy==1.2.1
cycler==0.12.1
debugpy==1.8.1
decorator==5.1.1
defusedxml==0.7.1
dm-env==1.6
dm-tree==0.1.8
dm_control @ git+https://github.com/deepmind/dm_control.git@7a6a5309e3ef79a720081d6d90958a2fb78fd3fe
dmc2gym @ git+https://github.com/1nadequacy/dmc2gym.git@06f7e335d988b17145947be9f6a76f557d0efe81
docker-pycreds==0.4.0
docstring_parser==0.16
dotmap==1.3.30
# etils==1.7.0
exceptiongroup==1.2.0
executing==2.0.1
Farama-Notifications==0.0.4
fastapi==0.110.1
fastjsonschema==2.19.1
filelock==3.13.4
fire==0.6.0
fonttools==4.51.0
fqdn==1.5.1
frozenlist==1.4.1
fsspec==2024.3.1
gitdb==4.0.11
GitPython==3.1.43
glfw==2.7.0
google-auth==2.29.0
google-auth-oauthlib==1.2.0
grpcio==1.62.1
gym==0.26.2
gym-notices==0.0.8
gymnasium==0.29.1
h11==0.14.0
hydra-core==1.3.2
idna==3.7
imageio==2.34.0
imageio-ffmpeg==0.4.9
importlib_metadata==7.1.0
importlib_resources==6.4.0
ipdb==0.13.13
ipykernel==6.26.0
ipython==8.17.2
ipywidgets==8.1.1
isoduration==20.11.0
jedi==0.19.1
Jinja2==3.1.3
jmespath==1.0.1
joblib==1.4.0
json5==0.9.25
jsonpointer==2.4
jsonschema==4.21.1
jsonschema-specifications==2023.12.1
jupyter-events==0.10.0
jupyter-lsp==2.2.5
jupyter_client==8.6.1
jupyter_core==5.7.2
jupyter_server==2.14.0
jupyter_server_terminals==0.5.3
jupyterlab==4.0.6
jupyterlab_pygments==0.3.0
jupyterlab_server==2.26.0
jupyterlab_widgets==3.0.10
kiwisolver==1.4.5
kornia==0.7.2
kornia_rs==0.1.3
labmaze==1.0.6
lazy_loader==0.4
lightning==2.2.2
lightning-cloud==0.5.64
lightning-utilities==0.10.1
lightning_sdk==0.1.4
litdata==0.2.2
lxml==5.2.1
Markdown==3.6
markdown-it-py==3.0.0
MarkupSafe==2.1.5
matplotlib==3.8.2
matplotlib-inline==0.1.6
mdurl==0.1.2
mistune==3.0.2
mpmath==1.3.0
mujoco==3.1.4
multidict==6.0.5
nbclient==0.10.0
nbconvert==7.16.3
nbformat==5.10.4
nest-asyncio==1.6.0
# networkx==3.3
notebook_shim==0.2.4
numpy==1.26.2
nvidia-cublas-cu12==12.1.3.1
nvidia-cuda-cupti-cu12==12.1.105
nvidia-cuda-nvrtc-cu12==12.1.105
nvidia-cuda-runtime-cu12==12.1.105
nvidia-cudnn-cu12==8.9.2.26
nvidia-cufft-cu12==11.0.2.54
nvidia-curand-cu12==10.3.2.106
nvidia-cusolver-cu12==11.4.5.107
nvidia-cusparse-cu12==12.1.0.106
nvidia-ml-py==12.535.133
nvidia-nccl-cu12==2.19.3
nvidia-nvjitlink-cu12==12.4.127
nvitop==1.3.2
oauthlib==3.2.2
objprint==0.2.3
omegaconf==2.3.0
opencv-python==4.9.0.80
overrides==7.7.0
packaging==24.0
pandas==2.1.4
pandocfilters==1.5.1
parso==0.8.4
pexpect==4.9.0
pillow==10.3.0
platformdirs==4.2.0
prometheus_client==0.20.0
prompt-toolkit==3.0.43
protobuf==4.23.4
psutil==5.9.8
ptyprocess==0.7.0
pure-eval==0.2.2
pyarrow==15.0.2
pyasn1==0.6.0
pyasn1_modules==0.4.0
pycparser==2.22
pydantic==2.7.0
pydantic_core==2.18.1
pygame==2.5.2
Pygments==2.17.2
PyJWT==2.8.0
PyOpenGL==3.1.7
pyparsing==3.1.2
PyPrind==2.11.3
python-dateutil==2.9.0.post0
python-json-logger==2.0.7
python-multipart==0.0.9
pytorch-lightning==2.2.2
pytz==2024.1
PyYAML==6.0.1
pyzmq==25.1.2
recordclass==0.21.1
referencing==0.34.0
requests==2.31.0
requests-oauthlib==2.0.0
rfc3339-validator==0.1.4
rfc3986-validator==0.1.1
rich==13.7.1
-e git+https://github.com/astooke/rlpyt.git@b32d589d12d31ba3c8a9cfb7a3c85c6e350b2904#egg=rlpyt
rpds-py==0.18.0
rsa==4.9
s3transfer==0.10.1
# scikit-image==0.23.1
scikit-learn==1.3.2
# scikit-video==1.1.11
scipy==1.11.4
Send2Trash==1.8.3
sentry-sdk==1.45.0
setproctitle==1.3.3
shtab==1.7.1
simple-term-menu==1.6.4
six==1.16.0
smmap==5.0.1
sniffio==1.3.1
soupsieve==2.5
stable_baselines3==2.3.1
stack-data==0.6.3
starlette==0.37.2
sympy==1.12
tb-nightly==2.17.0a20240418
tensorboard==2.15.1
tensorboard-data-server==0.7.2
termcolor==2.4.0
terminado==0.18.1
threadpoolctl==3.4.0
tifffile==2024.4.18
tinycss2==1.2.1
tomli==2.0.1
torch
# torchmetrics==1.3.1
# torchvision==0.17.1+cu121
tornado==6.4
tqdm==4.66.2
# traitlets==5.14.2
# triton==2.2.0
# types-python-dateutil==2.9.0.20240316
# typing_extensions==4.11.0
tyro==0.8.3
# tzdata==2024.1
# uri-template==1.3.0
# urllib3==2.2.1
uvicorn==0.29.0
# viztracer==0.16.2
wandb==0.16.6
# wcwidth==0.2.13
# webcolors==1.13
# webencodings==0.5.1
# websocket-client==1.7.0
# Werkzeug==3.0.2
# widgetsnbextension==4.0.10
# yapf==0.40.2
# yarl==1.9.4
# zipp==3.18.1
sk-video
skimage