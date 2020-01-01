import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import LSQUnivariateSpline

import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

def fit_model(data, orders, metric, s=None, min_p=0.01, return_untrained_model=False, show=0, **kwargs):
    """
    Fit a set of models and return the model that minimises a given metric that must be provided by the class ARMAResults
    :param data: The time series to analyse as a 1D numpy array
    :param orders: A list of tuples, each with 3 or 6 values (p,d,q) or (p,d,q,sp,sd,sq). These are the number of parameters of the ARIMA to fit.
                    These values should be chosen by using the parc and acf of the data and the diff of the data. Notice
                    that the log of the data may be also applied in some cases.
    :param metric: The type of error to use for returning the model that minimises this metric. Use 'aic', 'bic', 'ssd'
    :param s: seasonality. If None, then ARIMA is used instead of SARIMA model.
    :param min_p: The minimum p-value to accept a model. Notice that the null hypothesis is that all the acf are 0. Its rejection means that
                at least one of the ACF cannot be 0. So, the p value is the probability of the acf to have the value they have
                and still being actually 0. So, we want a high p-value, since we want the null hypothesis to be True.
    :param return_untrained_model: Not only returned the trained model, but also the model before the fitting it.
    :param show: int -> 0 do not show anything, 1 show one line with a summary, 2-> Show all the information except exceptions.
                        3 shows all the information.
    :param kwargs: Parameters for the fitting function. Check:
                    https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_model.ARIMA.fit.html#statsmodels.tsa.arima_model.ARIMA.fit
    :return: The trained ARMAResults model that minimises the given metric.
    """
    start_params = []
    if 'disp' not in kwargs:
        kwargs['disp'] = -1
    if 'start_params' in kwargs:
        if hasattr(kwargs['start_params'], '__len__'):
            if not hasattr(kwargs['start_params'], '__len__'):
                start_params = kwargs['start_params']

    values = []
    for i, order in enumerate(orders):
        if s is None:
            model = ARIMA(data, order=order)
        else:
            model = SARIMAX(data, order=order[:3], seasonal_order=order[3:]+[s])

        try:
            if start_params:
                kwargs['start_params'] = start_params[i]
            results_AR = model.fit(**kwargs)

            residues = results_AR.resid
            results_AR.sse = np.sum(residues**2)

            v = acorr_ljungbox(residues, int(np.log(len(data))), boxpierce=True)
            if show == 2:
                print('Chi2 value: {}, p-value: {} Box Chi2 value: {} and Box piercing p-value: {}'.format(*[vi[-1] for vi in v]))
            p_value = np.min(v[3])

            #The part int(p_value < min_p)*1e6 adds a huge penalty in the case the p_value is so small that the residues cannot
            #be considered as stationary. In the case that all of the models have the same problem, the returned model will still be
            #the one that minimises the metric given by the user.
            values.append(getattr(results_AR, metric.lower()) + int(p_value < min_p)*1e6)

            if show > 0:
                print('order:{} - AIC: {:.2f} - BIC: {:.2f} - SSE: {:.2f} - p-value {:.4f}'.format(order, results_AR.aic, results_AR.bic, results_AR.sse, p_value))

        except Exception as err:# ValueError:
            #results_AR = model.fit(start_params = [0., 0, 0, 1], disp=-1)
            if show > 0:
                print('order:{} -- Failed'.format(order))
            if show > 2:
                print(repr(err))
            values.append(1e20)

    if show > 0:
        print('Selected order: {}'.format(orders[np.argmin(values)]))

    if s is None:
        if not return_untrained_model:
            return ARIMA(data, order=orders[np.argmin(values)]).fit(disp=-1)
        else:
            return ARIMA(data, order=orders[np.argmin(values)]).fit(disp=-1), lambda data: ARIMA(data, order=orders[np.argmin(values)])
    else:
        order = orders[np.argmin(values)]
        if not return_untrained_model:
            return SARIMAX(data, order=order[:3], seasonal_order=order[3:] + [s]).fit(disp=-1)
        else:
            return SARIMAX(data, order=order[:3], seasonal_order=order[3:] + [s]).fit(disp=-1), lambda data: SARIMAX(data, order=order[:3], seasonal_order=order[3:] + [s])


def fit_trend_splines(data, nsegs=2):
    """
    Fit the trend with splines.
    :return:
    """

    index = np.arange(len(data))
    nknots = max(2, nsegs + 1)
    knots = np.linspace(index[0], index[-1], nknots + 2)[1:-2]
    fun = LSQUnivariateSpline(index, data, knots)
    return fun(index)

def _extrapolate_signals(y1, y2, num_samples=1):
    """
    Extrapolate numsamples using two samples that are separated by 1 unit.
    :param y1: float -> Value of the first point (x=0)
    :param y2: float -> Value of the second point (x=1)
    :param num_samples: int ->  Numbers of samples to extrapolate (x=2, 3 ...)
    :return: The values of the extrapolated points
    """
    m = y2 - y1
    n = y1
    return m * np.arange(2, num_samples+2) + n


def fit_trend_moving_average(data, length=5):
    """
    Fit the data using the moving average. It uses linear regression for the borders
    :param data: numpy array -> The temporal data
    :param length: int -> The length of the moving average
    :return: A numpy array with the same length as the data and the values of the trend.
    """
    n = length // 2
    average = np.convolve(data, np.ones((length,)) / length, mode='valid')
    init = _extrapolate_signals(average[1], average[0], num_samples=n)
    end =  _extrapolate_signals(average[-2], average[-1], num_samples=n)

    return np.array(init.tolist() + average.tolist() + end.tolist())


def decompose_trend_seasonal_moving_average(data, show=False):
    """

    :param data: Pandas series or dataframe. The indices must be the time and the values the values of the serie
    :param show: True when the decomposition must be plotted.
    :return: The residues, seasonal component and trend respectively
    """
    # deal with missing values. see issue

    res = sm.tsa.seasonal_decompose(data)
    if show:
        res.plot()
        plt.show()

    return [res.resid, res.seasonal, res.trend]
