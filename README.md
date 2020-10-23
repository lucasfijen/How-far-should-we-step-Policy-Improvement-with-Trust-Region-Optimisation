# How far should we step? Policy Improvement with Trust Region Optimisation
Students: Lucas Fijen, Pieter de Marez Oyens, Jonathan Mitnik, Guido Visser.

---------

## Experiment

To read more about our experiment in comparing TRPO with NPG, read our blog entry [here](https://rl-policy-trpo.netlify.app/).

## Setup

This codebase requires Python 3.6 (or higher). We recommend using Anaconda or Miniconda for setting up the virtual environment. Here's a walk through for the installation and project setup.

```
git clone https://github.com/lucasfijen/ReinforcementLearning
cd srcs
conda create -n rl_final_report python=3.6
conda activate rl_final_report
pip install -r requirements.txt
```

Assuming you are in `src/`, you can then run the environment using `python agent.py`. For information regarding the various arguments, see `arguments.py`, and also
the code repository [Deep Bayesian Quadrature Policy Optimization](https://github.com/Akella17/Deep-Bayesian-Quadrature-Policy-Optimization) for additional information.

## Credits
Our codebase is heavily based on Akella17's implementation of TRPO and NPG, from their code repository [Deep Bayesian Quadrature Policy Optimization](https://github.com/Akella17/Deep-Bayesian-Quadrature-Policy-Optimization),
