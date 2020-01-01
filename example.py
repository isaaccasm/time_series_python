import numpy as np
import pandas as pd
from fitting_functions import fit_model, fit_trend_moving_average, fit_trend_splines
from simulators import generate_ARIMA
from visualisation import plot_analysis_ARIMA_results, plot_ARIMA_corr_coeff

def example_visualisation_plot_ARIMA_corr_coeff():
    data = generate_ARIMA(arparams=[0.7], maparams=[0.2], d=1, show=False)

    df = pd.DataFrame({'time':list(range(len(data))), 'value':data})
    plot_ARIMA_corr_coeff(df, s=1, log=False, der=2)

def example_visualisation_plot_ARIMA_corr_coeff():
    data = generate_ARIMA(arparams=[0.7, 0.1], maparams=[0.2], d=1, show=False)
    orders = [(1,0,1), (1,1,1), (2,0,1), (1,1,2), (3,0,1), (2,1,1)]
    model = fit_model(data, orders, 'aic', show=1)

    plot_analysis_ARIMA_results(model)

def example_remove_trend_spline():
    data = generate_ARIMA(arparams=[0.7, 0.1], maparams=[0.2], d=1, show=False)
    #Add trend
    trend = 0.05 * np.arange(len(data))
    trend[:len(trend)//2] = trend[:len(trend)//2] * 0.2
    data += trend

    #detect trend
    val = fit_trend_splines(data, 500)

    orders = [(1, 0, 1), (1, 1, 1), (2, 0, 1), (1, 1, 2), (3, 0, 1), (2, 1, 1)]
    model = fit_model(data, orders, 'aic', show=1)

    plot_analysis_ARIMA_results(model)

example_remove_trend_spline()