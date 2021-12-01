import numpy as np
import scipy.stats as sts


def Mean(x):
    return np.sum(x) / len(x)


def Median(x):
    N = len(x)
    sorted = np.sort(x)
    if N % 2 == 0:
        return (sorted[int(N/2) - 1] + sorted[int(N/2)]) / 2.0
    else:
        return sorted[int((N-1)/2)]


def Std(x):
    mean = Mean(x)
    variance = np.sum((x - mean)**2) / (len(x) - 1)
    return np.sqrt(variance)


def Skewness(x):
    N = len(x)
    mean = Mean(x)
    std = Std(x)
    sum = np.sum((x - mean) ** 3)
    return sum / (N*(std**3))


def Kurtosis(x):
    N = len(x)
    mean = Mean(x)
    std = Std(x)
    sum = np.sum((x - mean) ** 4)
    return sum / (N*(std**4)) - 3.0


def antiKurtosis(x):
    k = Kurtosis(x)
    return 1.0 / np.sqrt(k + 3)


def mean_confidence_interval(x, error):
    t = sts.t.ppf(1.0 - error/2.0, len(x) - 1)
    mean = Mean(x)
    mean_std = Std(x) / np.sqrt(len(x))
    meanl = mean - t * mean_std
    meanh = mean + t * mean_std
    return (meanl, meanh, mean_std)


def median_confidence_interval(x, error):
    N = len(x)
    u = sts.norm.ppf(1.0-error/2.0)
    j = int(N/2.0 - u * np.sqrt(N)/2.0)
    k = int(N/2.0 + u * np.sqrt(N)/2.0)
    t = np.sort(x)
    return (t[j], t[k])


class KDE:
    def __init__(self, x, bandwidth):
        self.xvalues = x
        self.bw = bandwidth
        pass

    def value(self, x):
        # func = [gaussian_kernel((x - xi) / self.bw) for xi in self.xvalues]
        # sum = np.sum(func)
        sum = 0.0
        for xi in self.xvalues:
            sum += gaussian_kernel((x - xi) / self.bw)
        return sum / (len(self.xvalues) * self.bw)


def gaussian_kernel(u: float):
    return np.exp(-(u**2) / 2.0) / np.sqrt(2.0 * np.pi)


def skewnessStd(n: int) -> float:
    return np.sqrt((6*n*(n-1))/((n-2)*(n+1)*(n+3)))


def kurtosisStd(n: int) -> float:
    return np.sqrt((24*n*((n-1)**2))/((n-3)*(n-2)*(n+3)*(n+5)))


def skewnessTest(x, error=0.05) -> bool:
    sk = Skewness(x)
    std = skewnessStd(len(x))
    ua = sk / std
    u = sts.norm.ppf(1.0 - error/2.0)
    return (np.abs(ua) <= u)


def kurtosisTest(x, error=0.05) -> bool:
    kt = Kurtosis(x)
    std = kurtosisStd(len(x))
    ue = (kt/std)
    u = sts.norm.ppf(1.0 - error/2.0)
    return (np.abs(ue) <= u)


def myNormalTest(x, error=0.05):
    return (skewnessTest(x, error) and kurtosisTest(x, error))


def median_conf_interval(x, error=0.05):
    N = len(x)
    u = sts.norm.ppf(1.0-error/2.0)
    j = int(N/2.0 - u * np.sqrt(N)/2.0)
    k = int(N/2.0 + u * np.sqrt(N)/2.0)
    t = np.sort(x)
    return (t[j], t[k])


def norm_std_confidence_interval(x, error=0.05):
    N = len(x)
    std = np.std(x, ddof=1)
    t = sts.t.ppf(1.0-error/2.0, df=N-1)
    std_std = std/np.sqrt(2.0*N)
    return (std - t * std_std, std + t * std_std)


def norm_mean_confidence_interval(x, error=0.05):
    N = len(x)
    mean = np.mean(x)
    t = sts.t.ppf(1.0-error/2.0, df=N-1)
    std = np.std(x, ddof=1)
    mean_std = std / np.sqrt(N)
    return (mean - t * mean_std, mean + t * mean_std)
