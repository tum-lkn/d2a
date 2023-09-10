# Source code for D2A: Operating a Service Function Chain Platform with Data-Driven Scheduling Policies

**Disclaimer**. The source code in this repository has been made publicly
available for transparency and as contribution for the scientific community. The
source code reflects in most parts the state in which the results for the referenced
publications were obtained. The source code has mostly been left as is.

This repository contains the source code that was used to learn assignments for
Containerized Network Functions (CNFs) to CPU cores and accompanies the publication
> P. Krämer, P. Diederich, C. Krämer, R. Pries, W. Kellerer, and A. Blenk, “D2A: Operating a Service Function Chain Platform with Data-Driven Scheduling Policies,” IEEE TNSM, pp. 1–15, 2022, doi: 10.1109/TNSM.2022.3177694.

This repository further depends on [OpenNetVM](https://github.com/sdnfv/openNetVM)
which was used to execute and run test-bed measurements, and [MoonGen](https://github.com/emmericp/MoonGen)
to generate traffic and record measurements.

The repository is organized as follows:

`ccode`. This folder contains code to evaluate the behavior of the Linux scheduler
using varying settings.

`cpu_assignment`. This Python package contains the Reinforcement Learning (RL) setup.
Specifically, this module contains the RL environment, reward functions, training
facilities, and evaluation scripts for trained models, among others.

`dataprep`. This package contains utility code for the analysis of data. For example,
this module contains code to load CNF statistics from result files, transform
problem configurations into graphs.

`environment`. This package contains code related to experiments on OpenNetVM.
Specifically, this package contains code to generate experiments, combine experiments
with the results of assignment algorithms, etc. Further, the module `tp_sim` includes
a simulation of the system.

`evaluation`. This package contains utility functions to make evaluations for
experiments and trained RL algorithms.

`layers`. This package contains Neural Network (NN) components. The modules implement
various attention layers. Specifically, the `attn.py` module implements a custom
version for Graph attention, operating on edge lists and allowing the batching
of arbitrary sized graphs.

`notebooks`. Contains a notebook exploring and visualizing results.

`scripts`. This folder contains scripts automating repititive tasks and evaluations.