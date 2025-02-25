# SIR on Networks

> Data assimilation of the SIR model based on social networks

Given the structure of a social network and the observed number of infected nodes, the parameters of the SIR model, such as \( \beta \) and/or \( \gamma \), can be estimated using the Ensemble Kalman Filter (EnKF) algorithm.


## Installation Guide for Python 3
To run the code, you need to install the ndlib and filterpy packages. These are essential for network diffusion simulations and filtering algorithms, respectively. Additional packages can be installed as needed.

## Code Description
- `codes/scen1`:the codes in Scenario 1, the core code is **sir_enkf.py**
- `codes/scen2`: the codes in Scenario 2, the core code is **hda_main.py**
- `codes/scen3`: the codes in Scenario 2,  the core code is **enkf_diffnet.py**
- `codes/scen*/EnKF_delta_*.py`: the EnKF algorithm, the primary difference lies in the distinct methods used to correct the state under various scenarios.
- `codes/scen1/origsir.py`: the original SIR model
- `codes/scen1/particle_filter.py`: particle filter

## Datasets Description
- `datasets/graph/`: gnutella p2p network
- `datasets/scene1_data/`: the results in Scenario 1
- `datasets/scene2_data/`: the results in Scenario 2
- `datasets/scene3_data/`: the results in Scenario 3

  
