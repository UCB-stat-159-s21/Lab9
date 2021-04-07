import os
import sys
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def sim_gauss_data(loc=0, scale=1, size=250):
    '''
    Simulate draws from a Normal distribution.
    '''
    a = ss.norm.rvs(loc=loc, scale=scale, size=size)
    return a

def gaussian(vals, loc=0, scale=1):
    '''
    Wrapper for Normal pdf used for curve fitting.
    '''
    return ss.norm.pdf(vals, loc=loc, scale=scale)

def fit_gaussian_from_hist(bins, heights):
    '''
    Fit a Normal curve to a given histogram. 
    
    Arguments:
        bins    : np.array. 
                  bins defines the endpoints of the histogram bins
        heights : np.array
                  heights is teh height of the histogram (assumed to be in terms of density not raw counts)
    Returns:
        
    '''
    bin_mids = np.array([(bins[i] + bins[i+1]) / 2 for i in range(bins.shape[0] - 1)])
    popt, pcov = curve_fit(gaussian, bin_mids, heights)
    est_loc, est_scale = popt
    return bin_mids, est_loc, est_scale

def plot_data_and_generating_distrib(data, loc=0, scale=1, size=250, bins=20, fit=False, filepath=None):
    '''
    Plot data along with generating distribution.
    '''
    hist, bins, _ = plt.hist(data, bins=bins, density=True)
    start = ss.norm.ppf(0.001, loc=loc, scale=scale)
    end = ss.norm.ppf(0.999, loc=loc, scale=scale)
    x = np.linspace(start, end, size*2)
    plt.plot(x, ss.norm.pdf(x, loc=loc, scale=scale), 'k-', lw=2, label='generating dist')
    if fit:
        bin_mids, est_loc, est_scale = fit_gaussian_from_hist(bins, hist)
        plt.plot(bin_mids, gaussian(bin_mids, est_loc, est_scale), 'm-', lw=2, label='fit')
    plt.legend()
    if filepath:
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.exists(os.path.dirname(filepath)))
        plt.savefig(filepath)
        
        
if __name__ == '__main__':
    data = sim_gauss_data()
    plot_data_and_generating_distrib(data, fit=True, filepath=sys.argv[1])
