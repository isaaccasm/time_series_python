import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import acovf, ccovf, acf, pacf
from statsmodels.tsa.arima_process import arma_generate_sample


def generate_MA(weights=(0.7,0.2), number_samples=int(1e4), Zt=None, show=False):
    """
    Generate a Moving average process (MA) with weights weights
    :param weights: The weights of the MA process, they generally are < 1. No restriction for the number
    :param number_samples: The number os samples
    :param show: Plot the figure and the auto correlation of the data.
    :return: The MA process
    """
    # Generate noise
    if Zt is None:
        Zt = np.random.randn(number_samples)

    #X0 = Z0
    #X1 = Z1 + weight[0]*Z0
    #X2 = Z2 + weight[0]*Z1 + weight[1]*Z0
    #.
    #.
    #.
    #Xt = Zt + sum_w=0...m weight[w]*Zt-w-1
    # Introduce a variable
    Xt = np.copy(Zt)
    for i,weight in enumerate(weights):
        Xt[i+1:] += weight * Zt[:-i-1]

    moving_average_process = Xt

    # plot the process and plot its ACF
    if show:
        fig, axs = plt.subplots(2,1)
        axs[0].plot(moving_average_process)
        axs[0].set_title('A moving average process of order {} with weights {}'.format(len(weights), weights))

        acf_MA = acf(moving_average_process)
        axs[1].stem(acf_MA)
        axs[1].set_title('Correlogram of a moving average process of order {}'.format(len(weights)))
        plt.show()

    return moving_average_process


def generate_AR(weights=(0.7,0.2), number_samples=int(1e4), noise=None, show=False, mean=0, sigma=1):
    """
    Generate a Autocorrelation process (AR) with weights weights
    :param weights: The weights of the AR process, they generally are < 1. No restriction for the number
    :param number_samples: The number os samples
    :param show: Plot the figure and the auto correlation of the data.
    :return: The MA process
    """
    # Generate noise
    if noise is None:
        noise = sigma*np.random.randn(number_samples) + mean
    else:
        number_samples = len(noise)

    data = []
    for i in range(number_samples):
        value = noise[i]
        for ii, weight in enumerate(weights):
            if len(data) > ii:
                value += weight*data[-ii-1]
        data.append(value)

    # Introduce a variable
    AR_p = np.array(data)

    # plot the process and plot its ACF
    if show:
        fig, axs = plt.subplots(2,1)
        axs[0].plot(AR_p)
        axs[0].set_title('An auto-regressive process of order {} with weights {}'.format(len(weights), weights))

        acf_MA = pacf(AR_p)
        axs[1].stem(acf_MA)
        axs[1].set_title('Partial Correlogram of an auto-regressive process of order {}'.format(len(weights)))
        plt.show()

    return AR_p


def _generate_ARMA_plot(arparams=[1.0, -0.7], maparams=[1.0, 0.2], n=int(1e4), show=False):
    """
    Generate ARMA samples. This function uses the standard from statsmodels. DO NOT USE, use the
    generate_ARMA instead. This function was created as a helper for plotting the samples.
    :param arparams: list -> The ar parameters
    :param maparams: list -> The ma parameters
    :param n:
    :param show:
    :return:
    """
    y = arma_generate_sample(np.array(arparams), np.array(maparams), n)

    if show:
        if not np.any(np.isnan(y)):
            r = acf(y)
            pr = pacf(y)

            fig, ax = plt.subplots(3, 1)
            ax[0].plot(y, label="Data")
            ax[1].stem(r, label="ACF")
            ax[2].stem(pr, label="PACF")
            plt.show()

    return y


def generate_ARMA(arparams=[0.7], maparams=[0.2], n=int(1e4), show=False):
    """
    Generate samples from a given ARMA model.
    :param arparams: list -> The ar parameters
    :param maparams: list -> The ma parameters
    :param n: int -> Number of samples
    :param show: boolean -> To show the samples (together with autocorrelation coefficients (ACF) and partial ACF (PACF)
    :return:
    """
    return _generate_ARMA_plot(arparams=[1] + [-a for a in arparams], maparams=[1] + maparams, n=n, show=show)


def _extend_parameters(params, extension, s=1):
    """
    Auxiliar function to extend the parameters of MA or AR model. It generally use to add the seasonal component to
    a MA or AR model or to add the difference component to a AR model (passing from ARMA to ARIMA or SARMA to SARIMA)
    :param params: list -> The parameters to extend
    :param extension: list or integer -> Mix the parameters params with these new parameters or add the
    :return:
    """
    if isinstance(extension, (int, float)):
        num_extensions = extension
        extension = []
        if num_extensions > 0:
            aux_ext = [1] + [0 for _ in range(s - 1)] + [-1]
            for i in range(num_extensions - 1):
                extension = [aux_ext + [0]]
                extension.append([0] + [-a for a in aux_ext])
                aux_ext = np.sum(extension, axis=0).tolist()

    if len(params) == 0:
        return extension
    elif len(extension) == 0:
        return params
    else:
        all_params = np.array(params) * np.array(extension).reshape(-1,1)
        d = len(extension)
        extended_params = []
        for i,time_params in enumerate(all_params):
            extended_params.append([0 for _ in range(i)] + time_params.tolist() + [0 for _ in range(d-i-1)])

        params = np.sum(extended_params, axis=0)

    return params


def generate_ARIMA(arparams=[], maparams=[], d=0, n=int(1e4), show=False):
    """
    Generate samples of an ARIMA model.
    :param arparams: list -> The ar parameters
    :param maparams: list -> The ma parameters
    :param d: int -> The difference component
    :param n: int -> The number of samples to generate
    :param show: bool -> Whether to plot the curve and the autocorrelation coefficients and partial ones.
    :return: The samples.
    """
    arparams = [1] + [-a for a in arparams]
    arparams = _extend_parameters(arparams, d)  # From AR to ARI (ARMA to ARIMA)
    return _generate_ARMA_plot(arparams=arparams, maparams=[1] + maparams, n=n, show=show)


def generate_SARIMA(arparams=[], maparams=[], sarparams=[], smaparams=[], d=0, sd=0, s=1, n=int(1e4), show=False):
    """
    Generate samples of a sarima model.
    :param arparams: list -> The ar parameters
    :param maparams: list -> The ma parameters
    :param sarparams: list -> The ar parameters of the seasonal component
    :param smaparams: list -> The ma parameters of the seasonal component
    :param d: int -> The difference component
    :param sd:  int -> The difference component of the seasonal part
    :param s: int -> The seasonal component
    :param n: int -> The number of samples to generate
    :param show: bool -> Whether to plot the curve and the autocorrelation coefficients and partial ones.
    :return: The samples.
    """
    arparams = [1] + [-a for a in arparams]
    sarparams_aux = list(sarparams)
    sarparams = [1] + [0 for _ in range(0, len(sarparams) * s)] #The -1 is because we need to add the 1 at the beginning
    for i in range(len(sarparams_aux)):
        sarparams[(i+1)*s] = -sarparams_aux[i]

    smaparams_aux = list(smaparams)
    smaparams = [1] + [0 for _ in range(0, len(smaparams) * s)]  # The -1 is because we need to add the 1 at the beginning
    for i in range(len(smaparams_aux)):
        smaparams[(i + 1) * s] = smaparams_aux[i]

    arparams = _extend_parameters(arparams, d) #From AR to ARI (ARMA to ARIMA)
    sarparams = _extend_parameters(sarparams, sd, s) #FROM SAR to SARI

    arparams = _extend_parameters(arparams, sarparams) #Mix AR and SAR
    #arparams = [-a for a in arparams[1:]]

    maparams = _extend_parameters([1]+maparams, smaparams) #MIX MA and SMA
    #maparams = [a for a in maparams[1:]]

    return _generate_ARMA_plot(arparams=arparams, maparams=maparams, n=n, show=show)

if __name__ == '__main__':
    generate_ARIMA(arparams=[0.7], maparams=[0.2], d=1, show=True)