import matplotlib.pyplot as plt
import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox

def compute_statistics(data):
    """

    :param data: numpy array -> The residues of the temporal data
    :return:
    """
    v = acorr_ljungbox(data, int(np.log(len(data))), boxpierce=True)
    print('Chi2 value: {}, p-value: {} Box Chi2 value: {} and Box piercing p-value: {}'.format(*[vi[-1] for vi in v]))
    return v

def test_forecast_SARIMA(model, data, column='value', init=10):
    """
    Forecast the values of the model
    :param model: The SARIMA model already defined but not necessarily fit. The idea is to return a model from statsmodel
                    as lambda data: ARMIMA(data, order=order), so that the model is defined but not fitted.
    :param data: The data to train the model. Notice that the estimations will be used to forecast new values. It must be
                a pandas series with indexes as time.
    :param init: Number of samples to test the model
    :return: A pandas series with the new values
    """
    #if isinstance(column, (int, float)):
    #    column = data.columns[int(column)]

    data = data.astype(float)

    # history = y.values[:-init].tolist()
    test = data.iloc[-init:]
    data['predictions'] = float('nan')
    data['lowerBound'] = float('nan')
    data['upperBound'] = float('nan')
    for t in range(len(test)):
        history = data.loc[:data.index[-init + t], column]
        new_model = model(history)
        model_fit = new_model.fit(disp=0)
        output = model_fit.forecast()  # Only next samples. model.predict(fittedValues, start=length-init,end=length-1)
        yhat = output[0]
        bound = 1.96 * np.std(model_fit.res)
        data['predictions'].iloc[-init + t] = yhat
        data['lowerBound'].iloc[-init + t] = yhat - bound
        data['upperBound'].iloc[-init + t] = yhat + bound
        obs = test.iloc[t]
        # history.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs))

    data[[column, 'predictions']].plot()
    plt.plot(data['upperBound'], 'k')
    plt.plot(data['lowerBound'], 'k')
    plt.show()

    return data


def R_squared(model, Xt, yt, prob=False):
    """
    Compute the R squared coefficient for a linear model
    :param model: The scikit learn model
    :param Xt: A numpy array with the features. Notice that if one feature is used, Xt.reshape(-1,1) must be used
    :param yt: The labels
    :param prob: Boolean. When True the estimated probability from the model (predict_proba) is used instead.
    :return: The R-squared coefficient
    """
    if prob:
        yt2 = model.predict_proba(Xt)[:,1]
    else:
        yt2 = model.predict(Xt)

    SSE_error = np.sum((yt - yt2)**2)
    SSE_reg = np.sum((yt - np.mean(yt))**2)

    return 1 - SSE_error/SSE_reg

def adjusted_R_squared(model, Xt, yt, prob=False):
    """
    Compute the R squared coefficient for a linear model
    :param model: The scikit learn model
    :param Xt: A numpy array with the features. Notice that if one feature is used, Xt.reshape(-1,1) must be used
    :param yt: The labels
    :param prob: Boolean. When True the estimated probability from the model (predict_proba) is used instead.
    :return: The R-squared coefficient
    """

    shape_coeff = model.coef_.shape
    if len(shape_coeff) > 1:
        num_coeff = shape_coeff[1]
    else:
        num_coeff = shape_coeff[0]

    num_samples = len(yt)
    R2 = R_squared(model, Xt, yt)

    return 1 - (1 - R2)*(num_samples - 1)/(num_samples- num_coeff - 1)