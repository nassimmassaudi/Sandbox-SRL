# State Representation Learning Framework

The objective of this project is to setup an concise, effective and modular framework with a premise focus on State Representation Learning (SRL). The project will consist of the implementation of a set of agents with a Vanilla baseline and some modular part (such as the loss, the embeddings, the number of Q Networks and so on ...). The choice of the different packages and tools used for this project have been listed in  [Tools](notes/Tools.md).



## Table of Contents
- [List of Implemented Environments](notes/Environments.md) 
- [Instructions for the Workspace's Setup](notes/Setup_Workspace.md)
- [List of Papers](notes/Papers.md)[^1]
- [Selected Tools](notes/Tools.md)
- [Useful Cheatsheet for Markdown](notes/Markdown_Cheatsheet.md)[^1]

[^1]: This will be removed/replaced for the release.

## Structure of the Framework

```
.
├── agents
│   ├── ddpg
│   ├── dqn
│   ├── rainbow
│   ├── sac
│   └── td3
├── environments
│   ├── Atari
│   └── MuJoCo
├── experiments
├── notes
├── runs
├── tests
├── utils
│   └── evals
├── videos
└── wandb
```




