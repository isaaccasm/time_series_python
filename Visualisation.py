import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import learning_curve
from statsmodels.tsa.stattools import acf, pacf

def plot_ARIMA_corr_coeff(data, s=1, log=False, der=2):
    """
    Plot the original data, the first and second derivative and for these three
    curves, plot the autocorrelation coefficients and partial acf.
    :param data: pandas dataframe. The original data
    :param s: The seasonality in case the differencing must be done in the seasonality. By default s=1.
    :param log: When true, apply the log operation in the data before computing all the values.
    :param der: The number of derivatives that are going to be performed
    :return:
    """
    if log:
        data['value'] = np.log(data['value'])

    axss = []
    for i in range(der+1):
        if i == 0:
            name = 'Original data'
        else:
            name = 'Derivative {}'.format(i)
        r = acf(data['value'])
        pr = pacf(data['value'])

        fig, axs = plt.subplots(3, 1)
        axss.append(axs)
        axss[i][0].plot(data['time'], data['value'])
        axss[i][0].set_title(name)
        axss[i][1].stem(r)
        axss[i][1].plot([0,len(r)],[0.2,0.2],'--k')
        axss[i][1].plot([0, len(r)], [-0.2, -0.2], '--k')
        axss[i][1].set_title('Auto correlation coefficients')
        axss[i][2].stem(pr)
        axss[i][2].plot([0, len(r)], [0.2, 0.2], '--k')
        axss[i][2].plot([0, len(r)], [-0.2, -0.2], '--k')
        axss[i][2].set_title('Partial Auto correlation coefficients')

        data = pd.DataFrame({'time':data.loc[:, 'time'].values[s:],
            'value':data.loc[:, 'value'].values[s:] - data.loc[:,'value'].values[:-s]} )
        #data.loc[s:, 'value'] = data.loc[:, 'value'].values[s:] - data.loc[:,'value'].values[:-s] #np.diff(data['value'])
        #data.loc[:s, 'value'] = 0
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.4)

    plt.show()

def plot_QQ(model, fit=False, *args, **kwargs):
    """
    Plot the QQ plot.
    :param model: The statmodel model
    :param fit: When True the line that is shown as True line is the fitting line. This is useful sometimes since a straight
                line different to y=x means that the distribution is probably the same but the parameters are the same
                For instance, a Gaussian with different mean or sigma.
    :param args: Parameters for the qqplot method from statmodels. The most important one is the first parametrs
                which represents a model different to Gaussian (use: scipy.stats.t for t distribution and so on).
                Check: http://www.statsmodels.org/dev/generated/statsmodels.graphics.gofplots.qqplot.html
    :param kwargs: Other parameters for qqplot
    :return: None
    """
    res = model.resid  # residuals
    xmin = np.min(res)
    xmax = np.max(res)

    if 'fit' not in kwargs:
        kwargs['fit'] = fit

    #reg = LinearRegression().fit(np.arange(len(res)).reshape(-1,1), res)
    #print('Fitting line coefficients: {} and intercepts'.format(reg.coef_, reg.intercept_))


    fig = sm.qqplot(res, line='r', *args, **kwargs)
    plt.plot([xmin,xmax],[xmin,xmax], 'r')
    plt.show()

def plot_analysis_ARIMA_results(model, *args, **kwargs):
    """
    Plot the residues, acf, pacf and QQ plot of the residues
    :param model: The statmodel model
    :return:
    """
    res = model.resid#.iloc[1:]  # residuals
    pr = pacf(res)
    r = acf(res)

    fig, axs = plt.subplots(2, 2)
    axs[0][0].plot(res)
    axs[0][0].set_title('Residues between the data and those fitted by ARIMA')
    axs[0][1].stem(r)
    axs[0][1].set_title('Auto correlation coefficients')
    axs[1][0].stem(pr)
    axs[1][0].set_title('Partial Auto correlation coefficients')

    kwargs['ax'] = axs[1][1]
    kwargs['fit'] = True
    sm.graphics.qqplot(res, line='r', *args, **kwargs)
    axs[1][1].set_title('Q-Q plot')

    plt.show()