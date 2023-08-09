import warnings
from sklearn.exceptions import ConvergenceWarning
from regression_experiment import generateSpreadSheets
from optimization_experiment import generateDVLSpreadSheets

warnings.simplefilter("ignore", category=ConvergenceWarning)

num_objectives = 3
problem_labels = ['DTLZ1', 'DTLZ2']

#Simplier Configuration for running the Regression Experiment on problems DTLZ1 and DTLZ2 with 3 objectives.
scales = [250, 500, 1000, 1500, 10000]
generateSpreadSheets(problem_labels, scales, num_objectives=num_objectives, times=1)


#Simplier Configuration for running the Optimization Experiment with DVL on problems DTLZ1 and DTLZ2 with 3 objectives.
evaluations = [250, 500, 1000, 1500, 10000]
generateDVLSpreadSheets(['DTLZ1'], ['SVR'], evaluations, [50, 112, 132, 200, 300], [1, 2, 4, 6, 44], num_objectives, 1)
generateDVLSpreadSheets(['DTLZ2'], ['MLPSS'], evaluations, [50, 112, 132, 200, 7388], [1, 2, 4, 6, 12], num_objectives, 1)