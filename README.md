# time_series_python
Some common functions in R to manipulate or simulate time series with AR, MA, ARMA, ARIMA, SARIMA translated into python

## Files
The functions are divided in different files depending on their goal.

### Visualisations
This file contains a set of function to visualise the data to get an idea of the best model to be fitted and functions to visualise the model and residues after fitting.

### Simulators
This file contains functions that can generate samples of AR, MA, ARMA, ARIMA and SARIMA processes.

### Tests
Test for the residues and fitte models to check that the model did fit properly to the data.

### Fitting functions
This file presents a function that fits multiple models to the data and selects the best one according to a selected metric, as well as, trend fitting functions.