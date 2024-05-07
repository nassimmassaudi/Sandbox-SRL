# from hydra._internal.utils import _locate

# import sys
# print("PYTHONPATH:", sys.path)  # Output current Python path

# # Try locating the class to see if it raises an error
# try:
#   cls = _locate("agents.drq_agent.DRQAgent")
#   print("Class found:", cls)
# except Exception as e:
#   print("Error locating class:", e)
# import os 
# import glob
# import sys

# resource_files = "~/Sandbox-SRL/environments/video/crowd-1.mp4"

# files = glob.glob(os.path.expanduser(resource_files))

# print(os.path.expanduser(resource_files))
# print(files)

import os 
import utils.utils_dbc as utils


work_dir = "log/CPC/baseline-contrastive-cheetah-run-05-07-May-16-im84-b512-s1-pixel"

os.makedirs(work_dir, exist_ok=True)

video_dir = utils.make_dir(os.path.join(work_dir, 'video'))
model_dir = utils.make_dir(os.path.join(work_dir, 'model'))
buffer_dir = utils.make_dir(os.path.join(work_dir, 'buffer'))

import wandb

wandb.login(key="f302eb74e388e8173b2c053ef81436e9f6d87606")