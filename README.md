# Code for Learning Decision Variables in Many-Objective Optimization Problems


![Graphical abstract](/ProjectImages/graphical_abstract.png)

Resources and extra documentation for the manuscript "Learning Decision Variables in Many-Objective Optimization Problems" published in IEEE Latin America Transactions. The code is organized by functionalities. The scripts and folders description is as follows

1. **main.py**: Example of script for running a small example of both regression and optimization experiments.          
2. **regression_experiment.py**: Code for execution of the regression experiment.
3. **optimization_experiment.py**: Code for execution of the optimization experiment.
4. **dvl.py**: Source code of the DVL algorithm.
5. **dvl_utils.py**: Utility functions used by the DVL algorithm.
6. **models_utils.py**: Utility functions for creating machine learning models.
7. **problems_utils.py**: Utility functions used by optimization problems.
8. **\ProjectImages**. Some manuscript images and figures for the `README.m` file.

## Requirements
- Python 3.10 or later

## Screenshots

<div id="header" align="center">
  <img src="ProjectImages\regression_experiment_2.png" width="400"/>
</div>
<div id="header" align="center">
  <img src="ProjectImages\DTLZ1_10_rfroo.png" width="200"/>
  <img src="ProjectImages\DTLZ2_3_svr.png" width="200"/>
  <img src="ProjectImages\DTLZ5_3_svr.png" width="200"/>
  <img src="ProjectImages\DTLZ7_3_linear.png" width="200"/>
</div>

## Instructions for running 
1. Run `main.py` script to execute a demonstration of the experiments.
2. For the regression experiment a folder named Fronts need to be downloaded from [Fronts](https://1drv.ms/f/s!AoVbNqANaM4XgtZCgCuzyvLx7h0pDA?e=jA6Jzp) and added to the application's root directory.

