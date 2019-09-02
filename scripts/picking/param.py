from enum import Enum


Bin = Enum('Bin', ['Left', 'Right', 'File'])

Mode = Enum('Mode', ['Measure', 'Evaluate', 'Perform'])

SelectionMethod = Enum('SelectionMethod', [
    'Max',
    'Min',
    'Top5',
    'Top5LowerBound',
    'Prob',
    'Random',
    'RandomInference',
    'Uncertain',  # Near 0.5
    'NotZero',
    'Bottom5',
    'Bayes',  # What was that?
    'BayesTop',
    'BayesProb',
])
