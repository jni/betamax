import sys, os
from numpy import *
import numpy as np
from scipy.stats.distributions import norm, expon
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d

def array_or_bust(v):
    """Ensure a value ends up in an array, be it an array, list, or scalar."""
    try:
        n = len(v)
    except TypeError:
        v = [v]
    return array(v)

def p0(ms, xbars):
    """Estimate the probability that each mu=0."""
    px0 = exp(-(ms * xbars**2)/2)
    return px0 / (1 + px0)

def betanorm(ms, xbars, alpha=0.05):
    """Return the power of a Z-test for given means and sample sizes."""
    za = norm.ppf(1 - alpha/2)
    sms = sqrt(ms)
    bn = norm.cdf(-za+sms*xbars) + norm.cdf(-za-sms*xbars)
    return bn

def betamax(ms, xbars, alpha=0.05):
    ebeta = betanorm(ms, xbars, alpha)
    p = p0(ms, xbars)
    b = alpha*p**2 + (1-p**2)*ebeta
    return b

def p_values(ms, xbars):
    """Return the Z-test p-values for given means and sample sizes."""
    return 2*norm.cdf(-sqrt(ms)*abs(xbars))

def reject(ms, xbars, alpha=0.05):
    """Return rejected tests given means, sample sizes and alpha level."""
    return p_values(ms, xbars) < alpha

def delta_betamax_delta_m(ms, xbars, alpha=0.05, past_samples=None):
    # ensure we have arrays
    if type(ms) != ndarray:
        if type(ms) != list:
            ms = [ms]
            xbars = [xbars]
        ms = array(ms)
        xbars = array(xbars)
    delta = zeros(len(ms))
    bm0 = betamax(ms, xbars, alpha)
    for i in range(len(ms)):
        ms[i] += 1
        if past_samples is None:
            delta[i] = betamax(ms[i], xbars[i], alpha) - bm0[i]
        else:
            deltas = []
            for s in past_samples[i]:
                xbar = mean(past_samples[i]+[s])
                deltas.append(betamax(ms[i], xbar, alpha) - bm0[i])
            delta[i] = median(deltas)
        ms[i] -= 1
    return delta

def plot_idealized_betamax_horizon(mu1, mu2, M=40, alpha=0.05):
    mus = array([mu1, mu2])
    return plt.plot(array(
        [betamax(array([i, M-i]), mus, alpha=alpha).sum() 
        for i in range(M+1)]), 'r-')

def plot_empirical_betamax_horizon(mu1, mu2, M=40, nreps=10, alpha=0.05):
    mus = array([mu1, mu2])
    for i in range(nreps):
        _, surf = power_surface(mus, M, alpha)
        _ = plt.plot(array([surf[i, M-i-1] for i in range(M)]), 'b--')

def cumulative_mean(a):
    b = zeros_like(a)
    for i in range(len(a)):
        b[i] = a[:i+1].mean()
    return b

def cumulative_median(a):
    b = zeros_like(a)
    for i in range(len(a)):
        b[i] = median(a[:i+1])
    return b

def idealized_power_surface(mu=1.0, mmax=100, alpha=0.05):
    mu = array_or_bust(mu)
    n = len(mu)
    surface = zeros((mmax+1)*ones(n))
    for i in range(surface.size):
        ms = array(unravel_index(i, surface.shape))
        surface.ravel()[i] = betamax(ms, mu, alpha).sum()
    return surface

def power_surface(mu=1.0, mmax=100, alpha=0.05, sorted_sample=False):
    mu = array_or_bust(mu)
    n = len(mu)
    surface = zeros((mmax+1)*ones(n))
    xs = random.multivariate_normal(mu, eye(n), mmax).T
    if sorted_sample:
        for i in range(len(xs)):
            xs[i] = sorted(xs[i], key=lambda x: abs(x-mu[i]))
    for i in range(surface.size):
        ms = array(unravel_index(i, surface.shape)) + ones(n)
        means = array([xs[j][0:ms[j]].mean() for j in range(n)])
        surface.ravel()[i] = betamax(ms, means, alpha).sum()
    return xs, surface

def plot_surface(s, truncate=True, color=False):
    fig = plt.figure()
    ax = Axes3D(fig)
    s = s.copy()
    for i in range(s.shape[0]):
        for j in range(s.shape[0]-i, s.shape[1]):
            if truncate:
                s[i,j] = nan
    horizon = []
    for i in range(s.shape[0]):
        j = s.shape[0]-i-1
        horizon.append((i, j, s[j,i]))
    x, y = meshgrid(arange(s.shape[0]), arange(s.shape[1]))
    if color:
        ax.plot_surface(x, y, s, rstride=1, cstride=1, cmap=plt.cm.jet)
    else:
        plot_wireframe(ax, x, y, s, rstride=1, cstride=1)
    ax.plot(*zip(*horizon), c='b', linestyle='-', linewidth=2)
    plt.show()
    return ax

def add_arrival_histogram(ax3d, s, line3ds, maxh=1.0):
    hist_dict = {}
    for l in line3ds:
        try:
            hist_dict[tuple(l[:,-1])] += 1
        except KeyError:
            hist_dict[tuple(l[:,-1])] = 1.0
    m = maxh/max(hist_dict.values())
    for k, v in hist_dict.items():
        v *= m
        x, y, z = k
        ax3d.bar3d(x, y, z, 0.5, 0.5, v, color='r')



def line3d_from_history(h, s):
    x, y, z = [2], [2], [s[2,2]]
    for i in h:
        if i == 0:
            x.append(x[-1])
            y.append(y[-1]+1)
        else:
            x.append(x[-1]+1)
            y.append(y[-1])
        z.append(s[y[-1],x[-1]])
    return array([x,y,z])

def plot_wireframe(ax, X, Y, Z, *args, **kwargs):
    '''
    Plot a 3D wireframe.

    ==========  ================================================
    Argument    Description
    ==========  ================================================
    *X*, *Y*,   Data values as numpy.arrays
    *Z*
    *rstride*   Array row stride (step size)
    *cstride*   Array column stride (step size)
    ==========  ================================================

    Keyword arguments are passed on to
    :func:`matplotlib.collections.LineCollection.__init__`.

    Returns a :class:`~mpl_toolkits.mplot3d.art3d.Line3DCollection`
    '''

    rstride = kwargs.pop("rstride", 1)
    cstride = kwargs.pop("cstride", 1)

    had_data = ax.has_data()
    rows, cols = Z.shape

    tX, tY, tZ = np.transpose(X), np.transpose(Y), np.transpose(Z)

    rii = [i for i in range(0, rows, rstride)]+[rows-1]
    cii = [i for i in range(0, cols, cstride)]+[cols-1]
    xlines = [X[i] for i in rii]
    ylines = [Y[i] for i in rii]
    zlines = [Z[i] for i in rii]

    txlines = [tX[i] for i in cii]
    tylines = [tY[i] for i in cii]
    tzlines = [tZ[i] for i in cii]

    lines = [zip(xl, yl, zl) for xl, yl, zl in \
            zip(xlines, ylines, zlines)]
    lines += [zip(xl, yl, zl) for xl, yl, zl in \
            zip(txlines, tylines, tzlines)]

    for line in lines:
        remove = []
        for i, point in enumerate(line):
            if isnan(point[-1]):
                remove.append(i)
        remove.reverse()
        for i in remove:
            _ = line.pop(i)

    linec = art3d.Line3DCollection(lines, *args, **kwargs)
    ax.add_collection(linec)
    ax.auto_scale_xyz(X, Y, Z, had_data)

    return linec

def real_power(ms, mus, alpha=0.05):
    za = norm.ppf(1-alpha/2)
    sms = sqrt(ms)
    true_hits = (mus != 0).astype(double)
    return true_hits.dot(norm.cdf(-za+sms*mus) + norm.cdf(-za-sms*mus))

def confusion_matrix(called_hits, actual_hits):
    tp = (called_hits*actual_hits).sum()
    tn = ((1-called_hits)*(1-actual_hits)).sum()
    fp = (called_hits*(1-actual_hits)).sum()
    fn = ((1-called_hits)*actual_hits).sum()
    conf = array([[tn, fp], [fn, tp]])
    return conf

def tpr(conf):
    tpr = float(conf[1,1])/(conf[1,1]+conf[1,0])
    return tpr

def fpr(conf):
    fpr = float(conf[0,1])/(conf[0,0]+conf[0,1])
    return fpr

def simulate(mbar, mus, ms=None, alpha=0.05):
    mus = array_or_bust(mus)
    if ms is None:
        ms = 2*ones(len(mus))
    n = len(ms)
    ms = ms.copy()
    samples = []
    for i in range(n):
        samples.append([])
        for j in range(int(ms[i])):
            samples[i].append(random.normal(loc=mus[i]))
    x_bars = array(map(mean, samples))
    bm = betamax(ms, x_bars, alpha)
    db = delta_betamax_delta_m(ms, x_bars, alpha, samples)
    history = []
    for curr_M in range(int(sum(ms)), int(mbar*n)):
        i = db.argmax()
        history.append(i)
        samples[i].append(random.normal(loc=mus[i]))
        ms[i] += 1
        x_bars[i] = mean(samples[i])
        bm[i] = betamax(ms[i], x_bars[i], alpha)
        db[i] = delta_betamax_delta_m(ms[i:i+1], x_bars[i:i+1], alpha, None)
                                                       # [samples[i]])
    power = real_power(ms, mus, alpha)
    naive_power = real_power(mbar*ones((n,)), mus, alpha)
    pvals = p_values(ms, x_bars)
    called_hits = reject(ms, x_bars, alpha)
    actual_hits = mus > 0
    conf = confusion_matrix(called_hits, actual_hits)
    naive_xbars = random.multivariate_normal(mus, eye(len(mus)), mbar).mean(axis=0)
    naive_pvals = p_values(mbar*ones(len(mus)), naive_xbars)
    naive_called_hits = reject(mbar*ones(len(mus)), naive_xbars, alpha)
    naive_conf = confusion_matrix(naive_called_hits, actual_hits)
    return history, ms, bm, power, naive_power, pvals, naive_pvals, conf, naive_conf

def roc(mbars, mus, ms, alphas, reps=5):
    import progressbar as pb
    res = zeros((2,2,len(alphas)))
    progress = pb.ProgressBar(maxval=len(alphas))
    for i,alpha in progress(enumerate(alphas)):
        for j in range(reps):
            conf, naive_conf = simulate(mbars, mus, ms, alpha)[-2:]
            res[:,0,i] += tpr(conf), fpr(conf)
            res[:,1,i] += tpr(naive_conf), fpr(naive_conf)
    res /= reps
    return res
    

def make_fig1(res, mus):
    plot = plt.plot

    res_mean = concatenate([r[1][newaxis,:] for r in res],0).mean(axis=0)
    for i in range(len(res)):
        l0 = plot(res[i][1], color='0.8', linestyle='-')
    
    l1 = plot(res_mean, 'r-')
    l2 = plot(6*ones(200), 'b:')
    plt.ylim(0,50)
    plt.xlabel('Line Index')
    plt.ylabel('Number of Samples')
    
    ax2 = plt.twinx()
    l3 = plot(mus, 'k--')
    plt.ylim(0,mus.max()*2)
    plt.ylabel('True Line Mean')
    
    plt.legend([l0, l1, l2, l3], ['Individual Trials', 'Average Trial', 'Naive Sampling', 'Line Means'], 'upper left')

def compare_ms_for_different_alphas(ress, mus, alphas):
    plot = plt.plot

    res_means = []
    for res in ress:
        res_means.append(
            concatenate([r[1][newaxis,:] for r in res],0).mean(axis=0))
    ls = [plot(res_mean) for res_mean in res_means]
    ln = plot(6*ones(len(mus)), 'b:')
    plt.ylim(0, max([res_mean.max() for res_mean in res_means])+5)
    plt.xlabel('Line Index')
    plt.ylabel('Number of Samples')

    ax2 = plt.twinx()
    l3 = plot(mus, 'k--')
    plt.ylim(0,mus.max()*2)
    plt.ylabel('True Line Mean')

    plt.legend(ls+ln, ['alpha %f'%a for a in alphas] + ['Naive Sampling'], 'upper left')
