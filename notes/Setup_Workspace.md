# Instructions for setting up the code

Setting up MUJOCO_PATH and MUJOCO_PLUGIN_PATH

pip install --upgrade gymnasium[atari] fixed

```bash
virtualenv srl
pip install "gymnasium[atari, accept-rom-license]"
```

```bash
Xvfb :1 -screen 0 1024x768x16 &
export DISPLAY=:1.0
```
