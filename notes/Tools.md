# Technical Configuration and TODOs

Config Files : Tyro (Advantage of Python like yaml file and intuitive CLI) 

Tracking Experiences : W&B (Most popular and well documented with Open Source and Cloud storage for experiments)

Bar progress : rich progress

Agents : JAX and FLAX

Some utils (replay buffer, wrappers) : Stable Baseline3

Agents list that will be implemented :

- SAC 
- C51 (rainbow)
- DQN
- TD3

*An exhaustive Guide for [Markdown Notation](https://github.com/adam-p/markdown-here/wiki)*


# TODO List
```
[x] DQN Vanilla 
[x] Atari and Classic Control
[]  MuJoCo v3
[] MuJoCo v4 (pip install mujoco not working probably requires building from the git directly)
[] Cleaning the global pip packages (some unnecessary package have been installed by mistake) https://mujoco.readthedocs.io/en/stable/python.html
[] Minigrid and MiniWorld 
[x] W&B Experiment Tracking
[] Clear yml file for the environment
[] Writting 
[]
[] No Environment aggregation implemented at the moment
```




----

```
Currently Loaded Modules:
  1) CCconfig                 4) StdEnv/2020    (S)   7) ucx/1.8.0             10) cudacore/.11.7.0 (H,t)  13) python/3.10.2       (t)
  2) gentoo/2020     (S)      5) gcccore/.9.3.0 (H)   8) libfabric/1.10.1      11) cuda/11.7        (H,t)  14) ipykernel/2023b
  3) imkl/2020.1.217 (math)   6) gcc/9.3.0      (t)   9) openmpi/4.0.3    (m)  12) libffi/3.3              15) ipython-kernel/3.10

  Where:
   S:     Module is Sticky, requires --force to unload or purge
   m:     MPI implementations / Implémentations MPI
   math:  Mathematical libraries / Bibliothèques mathématiques
   t:     Tools for development / Outils de développement
   H:                Hidden Module
```

Currently used :
```
Currently Loaded Modules:
  1) CCconfig                 4) StdEnv/2020    (S)   7) ucx/1.8.0             10) cudacore/.11.7.0 (H,t)  13) python/3.10.2       (t)
  2) gentoo/2020     (S)      5) gcccore/.9.3.0 (H)   8) libfabric/1.10.1      11) cuda/11.7        (H,t)  14) ipykernel/2023b
  3) imkl/2020.1.217 (math)   6) gcc/9.3.0      (t)   9) openmpi/4.0.3    (m)  12) libffi/3.3              15) ipython-kernel/3.10

```

Updated to :
```
Currently Loaded Modules:
  1) CCconfig                 4) StdEnv/2020   (S)   7) ipykernel/2023b      10) gcccore/.11.3.0  (H)  13) cudacore/.11.8.0 (H,t)  16) cudnn/8.6.0.163 (math)
  2) gentoo/2020     (S)      5) libffi/3.3          8) ipython-kernel/3.10  11) gcc/11.3.0       (t)  14) cuda/11.8.0      (H,t)
  3) imkl/2020.1.217 (math)   6) python/3.10.2 (t)   9) gdrcopy/2.3          12) libfabric/1.10.1      15) ucx/1.8.0

```

Currently Loaded Modules:
  1) CCconfig                 4) StdEnv/2020    (S)   7) ucx/1.8.0             10) cudacore/.11.7.0 (H,t)  13) python/3.10.2       (t)
  2) gentoo/2020     (S)      5) gcccore/.9.3.0 (H)   8) libfabric/1.10.1      11) cuda/11.7        (H,t)  14) ipykernel/2023b
  3) imkl/2020.1.217 (math)   6) gcc/9.3.0      (t)   9) openmpi/4.0.3    (m)  12) libffi/3.3              15) ipython-kernel/3.10



Currently Loaded Modules:
  1) gdrcopy/2.3.1            5) StdEnv/2020     (S)   9) ipython-kernel/3.10      13) ucx/1.12.1        17) openmpi/4.1.4    (m)
  2) CCconfig                 6) libffi/3.3           10) gcccore/.11.3.0     (H)  14) libfabric/1.15.1  18) cudacore/.11.8.0 (H,t)
  3) gentoo/2020     (S)      7) python/3.10.2   (t)  11) gcc/11.3.0          (t)  15) pmix/4.1.2        19) cuda/11.8.0      (H,t)
  4) imkl/2020.1.217 (math)   8) ipykernel/2023b      12) hwloc/2.7.1              16) ucc/1.0.0         20) cudnn/8.6.0.163  (math)