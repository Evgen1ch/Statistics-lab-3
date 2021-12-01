import pandas as pd
import numpy as np
import scipy.stats as stats


def mean_equality_dep(x, y, alpha=0.05):
    if len(x) != len(y):
        raise ValueError("x and y must have same length")
    if len(x) == 0:
        raise ValueError("x and y must have non-zero length")

    x = np.array(x)
    y = np.array(y)
    n = len(x)
    z = x - y
    z_mean = np.mean(z)
    z_std = np.std(z, ddof=1)

    if z_std == 0:
        t = 0.0
    else:
        t = (z_mean * np.sqrt(n)) / z_std

    crit_t = stats.t.ppf(1.0-alpha/2.0, df=n-1)
    return (abs(t) <= crit_t, t, crit_t)


def mean_equality_indep(x, y, alpha=0.05, welch=False):
    # todo z division
    x = np.array(x)
    y = np.array(y)
    Nx = len(x)
    Ny = len(y)

    zMean = np.mean(x) - np.mean(y)
    VarX = np.var(x, ddof=1)
    VarY = np.var(y, ddof=1)
    
    if welch:
        VarXm = VarX/Nx
        VarYm = VarY/Ny
        VarZm = VarXm + VarYm
        StdZm = np.sqrt(VarZm)
        if StdZm == 0.0:
            t = 0.0
        else:
            t = zMean / StdZm
            df = (VarX / Nx + VarY / Ny)**2 * ((1.0/(Nx - 1)) *
                                            (VarX/Nx)**2 + (1.0/(Ny - 1))*(VarY/Ny)**2) ** (-1)
    else:
        S2 = ((Nx - 1) * VarX + (Ny - 1) * VarY) / (Nx + Ny - 2)
        denom = np.sqrt((S2 / Nx) + (S2 / Ny))
        if denom == 0.0:
            t = 0.0
        else:
            t = (zMean) / denom
        df = Nx + Ny - 2

    crit_t = stats.t.ppf(1.0-alpha/2.0, df=df)
    return (np.abs(t) <= crit_t, t, crit_t)


def variance_equality(x, y, alpha=0.05):
    x = np.array(x)
    y = np.array(y)
    n1 = len(x)
    n2 = len(y)
    VarX = np.var(x, ddof=1)
    VarY = np.var(y, ddof=1)

    f = VarX/VarY if VarX >= VarY else VarY/VarX

    v1 = n1 - 1 if VarX >= VarY else n2 - 1
    v2 = n2 - 1 if VarX >= VarY else n1 - 1

    crit_f = stats.f.ppf(1.0-alpha, v1, v2)

    return (f <= crit_f, f, crit_f)


def WilcoxonSignedRanksTest(x, y, alpha=0.05):
    if len(x) != len(y):
        raise ValueError("x and y must have same length")
    z = np.array(x - y)

    z = z[z != 0]

    if len(z) == 0:
        u = 0.0
    else:
        df = pd.DataFrame()
        df['z'] = z

        N = len(z)
        s = []
        for zi in z:
            s.append(int(zi > 0))
        df['s'] = s
        df['z_abs'] = np.abs(df['z'])
        df = df.sort_values(by='z_abs', axis=0)

        z = df['z_abs'].values
        r = []
        prev = z[0]
        c = 1
        cr = 1
        for zi in z[1:]:
            if zi == prev:
                c += 1
            if zi != prev:
                r = r + [np.mean(np.arange(cr - c + 1, cr + 1, 1))
                        for i in range(c)]
                c = 1
            cr += 1
        r = r + [np.mean(np.arange(cr - c + 1, cr + 1, 1))
                for i in range(c)]
        r = np.array(r)
        df['r'] = r

        T = np.sum(df['r'].values * df['s'].values)
        Et = 0.25 * N * (N+1)
        Dt = 1.0 / 24.0 * N * (N+1) * (2.0*N + 1)
        u = (T - Et) / np.sqrt(Dt)

    crit_u = stats.norm.ppf(1.0 - alpha / 2.0)
    return (np.abs(u) <= crit_u, u, crit_u)


def MannWhitneyTest(x, y, alpha=0.05):
    x = np.array(x)
    y = np.array(y)
    n1 = len(x)
    n2 = len(y)
    ranks = []
    for xi in x:
        cur = 0.0
        for yi in y:
            if xi > yi:
                cur += 1.0
            elif xi == yi:
                cur += 0.5
        ranks.append(cur)

    U = np.sum(ranks)
    Eu = 0.5 * n1 * n2
    Du = 1.0 / 12.0 * n1 * n2 * (n1 + n2 + 1)
    u = (U - Eu) / np.sqrt(Du)
    crit_u = stats.norm.ppf(1.0 - alpha / 2.0)
    return (np.abs(u) <= crit_u, u, crit_u)
