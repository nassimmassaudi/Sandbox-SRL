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

```bash
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

## References

1. **Learning Invariant Representations for Reinforcement Learning without Reconstruction**  
   Amy Zhang, Rowan McAllister, Roberto Calandra, Yarin Gal, Sergey Levine, 2021  
   [arXiv:2006.10742](https://arxiv.org/abs/2006.10742)  
   [GitHub Repository](https://github.com/facebookresearch/deep_bisim4control)

2. **Data-Efficient Reinforcement Learning with Self-Predictive Representations**  
   Max Schwarzer, Ankesh Anand, Rishab Goel, R Devon Hjelm, Aaron Courville, Philip Bachman, 2021  
   Presented at the International Conference on Learning Representations  
   [OpenReview](https://openreview.net/forum?id=uCQfPZwRaUu)  
   [GitHub Repository](https://github.com/facebookresearch/deep_bisim4control)

3. **Reinforcement Learning with Augmented Data**  
   Michael Laskin, Kimin Lee, Adam Stooke, Lerrel Pinto, Pieter Abbeel, Aravind Srinivas, 2020  
   [arXiv:2004.14990](https://arxiv.org/abs/2004.14990)  
   [GitHub Repository](https://github.com/MishaLaskin/rad)

4. **CURL: Contrastive Unsupervised Representations for Reinforcement Learning**  
   Aravind Srinivas, Michael Laskin, Pieter Abbeel, 2020  
   [arXiv:2004.04136](https://arxiv.org/abs/2004.04136)  
   [GitHub Repository](https://github.com/MishaLaskin/curl)
   
6. **DeepMind Lab**  
   Charles Beattie, Joel Z. Leibo, Denis Teplyashin, Tom Ward, Marcus Wainwright, Heinrich Küttler, Andrew Lefrancq, Simon Green, Víctor Valdés, Amir Sadik, Julian Schrittwieser, Keith Anderson, Sarah York, Max Cant, Adam Cain, Adrian Bolton, Stephen Gaffney, Helen King, Demis Hassabis, Shane Legg, Stig Petersen, 2016  
   [arXiv:1612.03801](https://arxiv.org/abs/1612.03801)
   [GitHub Repository](https://github.com/google-deepmind/lab)
 
