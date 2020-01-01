import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fitting_functions import fit_model, fit_trend_moving_average, fit_trend_splines
from simulators import generate_ARIMA
from test_results import test_forecast_SARIMA
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
    data = generate_ARIMA(arparams=[0.7, -0.1], maparams=[-0.8], d=1, show=False)
    #Add trend
    trend = 0.05 * np.arange(len(data))
    trend[:len(trend)//2] = trend[:len(trend)//2] * 0.2
    data += trend

    #detect trend
    estimated_trend = fit_trend_splines(data, 500)

    plt.plot(estimated_trend, 'b')
    plt.plot(trend, 'r')
    plt.legend(['estimated_trend', 'real trend'])
    plt.show()

    data -= estimated_trend

    orders = [(1, 0, 1), (1, 1, 1), (2, 0, 1), (1, 1, 2), (2, 1, 1)]
    model = fit_model(data, orders, 'aic', show=1, method='css', start_params=[[0.2, -0.1, 0.1], [0.2, -0.1, 0.1], [0.2, 0.1, -0.1, 0.1], [0.2, 0.3, -0.1, 0.1], [0.2, 0.1, -0.1, 0.1]])

    plot_analysis_ARIMA_results(model)

def example_remove_trend_moving_average():
    data = generate_ARIMA(arparams=[0.7, -0.1], maparams=[-0.8], d=1, show=False, n=2000)
    #Add trend
    trend = 0.05 * np.arange(len(data))
    trend[:len(trend)//2] = trend[:len(trend)//2] * 0.2
    data += trend

    #detect trend
    estimated_trend = fit_trend_moving_average(data, 100)

    plt.plot(estimated_trend, 'b')
    plt.plot(trend, 'r')
    plt.legend(['estimated_trend', 'real trend'])
    plt.show()

    data -= estimated_trend

    orders = [(1, 0, 1), (1, 1, 1), (2, 0, 1), (1, 1, 2), (2, 1, 1)]
    model = fit_model(data, orders, 'aic', show=1, method='css', start_params=[[0.2, -0.1, 0.1], [0.2, -0.1, 0.1], [0.2, 0.1, -0.1, 0.1], [0.2, 0.3, -0.1, 0.1], [0.2, 0.1, -0.1, 0.1]])

    plot_analysis_ARIMA_results(model)

def example_test_test_forecast_SARIMA():
    data = generate_ARIMA(arparams=[0.7, 0.1], maparams=[0.2], d=1, show=False, n=110)
    df = pd.DataFrame({'time': list(range(len(data))), 'value': data})

    orders = [(1,0,1), (1,1,1), (2,0,1), (1,1,2), (3,0,1), (2,1,1)]
    res_model, model = fit_model(data[:100], orders, 'aic', show=1, return_untrained_model=True)

    test_forecast_SARIMA(model, df)

example_test_test_forecast_SARIMA()