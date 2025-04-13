import numpy as np
from celerite import terms
import celerite
from scipy.optimize import minimize
import autograd

"""
This project includes code from the repository:
https://github.com/burke86/taufit, authored by Colin J. Burke.
"""

def hampel_filter(x, y, window_size, n_sigmas=3):
    """
    Perform outlier rejection using a Hampel filter
    
    x: time (list or np array)
    y: value (list or np array)
    window_size: window size to use for Hampel filter
    n_sigmas: number of sigmas to reject outliers past
    
    returns: x, y, mask [lists of cleaned data and outlier mask]
        
    Adapted from Eryk Lewinson
    https://towardsdatascience.com/outlier-detection-with-hampel-filter-85ddf523c73d
    """
    
    # Ensure data are sorted
    if np.all(np.diff(x) > 0):
        ValueError('Data are not sorted!')
        
    x0 = x[0]
    
    n = len(x)
    outlier_mask = np.zeros(n)
    k = 1.4826 # MAD scale factor for Gaussian distribution

    # Loop over data points
    for i in range(n):
        # Window mask
        mask = (x > x[i] - window_size) & (x < x[i] + window_size)
        if len(mask) == 0:
            idx.append(i)
            continue
        # Compute median and MAD in window
        y0 = np.median(y[mask])
        S0 = k*np.median(np.abs(y[mask] - y0))
        # MAD rejection
        if (np.abs(y[i] - y0) > n_sigmas*S0):
            outlier_mask[i] = 1

    outlier_mask = outlier_mask.astype(bool)
    
    return np.array(x)[~outlier_mask], np.array(y)[~outlier_mask], outlier_mask

def drw_mle(x, y, yerr, bounds, supress_warn=False, verbose=True):
    """
    Perform Maximum Likelihood Estimation (MLE) for a damped random walk (DRW) model.

    Parameters:
    x (np.ndarray): Time data
    y (np.ndarray): Observed data values
    yerr (np.ndarray): Errors associated with the observed data
    bounds (str or list): If 'default', the function will set bounds automatically. 
                          If a list, the bounds should specify the log-scale bounds for 
                          amplitude, cadence, and variance parameters.
    supress_warn (bool, optional): Whether to suppress warnings during computation. Default is False.
    verbose (bool, optional): Whether to print detailed output, including log-likelihood values. Default is True.

    Returns:
    tuple: 
        - gp (celerite.GP): The fitted Gaussian process model
        - initial (np.ndarray): The optimal parameters found using MLE
        - None: Placeholder for future return value
    
    The function fits a DRW model using the `celerite` package to the provided data 
    and returns the maximum likelihood parameters. The model includes a real term 
    for the DRW process and a jitter term for white noise.
    """

    # Sort data
    ind = np.argsort(x)
    x = x[ind]; y = y[ind]; yerr = yerr[ind]
    baseline = x[-1]-x[0]

    # Set bounds if 'default' is provided
    if bounds == 'defalt':
        min_precision = np.min(yerr)
        amplitude = np.max(y+yerr)-np.min(y-yerr)
        amin = np.log(0.001*min_precision)
        amax = np.log(1000*amplitude)
        min_cadence = np.clip(np.min(np.diff(x)), 1e-8, None)
        cmin = np.log(1/(1e9))
        cmax = np.log(1/(1e-3))
        smin = -10
        smax = np.log(amplitude)
        bounds = [amin, amax, cmin, cmax, smin, smax]
    amin = bounds[0]
    amax = bounds[1]
    cmin = bounds[2]
    cmax = bounds[3]
    log_a = np.mean([amin,amax])
    log_c = np.mean([cmin,cmax])
    smin = bounds[4]
    smax = bounds[5]
    log_s = np.mean([smin,smax])

    # Define the kernel for the Gaussian process
    kernel = terms.RealTerm(log_a=log_a, log_c=log_c,
                            bounds=dict(log_a=(amin, amax), log_c=(cmin, cmax)))
    # Add jitter term
    kernel += terms.JitterTerm(log_sigma=log_s, bounds=dict(log_sigma=(smin, smax)))

    # Suppress warnings if needed    
    if supress_warn:
        warnings.filterwarnings("ignore")
        
    # Create the Gaussian process with the defined kernel and data
    gp = celerite.GP(kernel, mean=np.mean(y), fit_mean=False)
    gp.compute(x, yerr)
    
    if verbose:
        print("Initial log-likelihood: {0}".format(gp.log_likelihood(y)))

    # Define the negative log-likelihood function and its gradient
    def neg_log_like(params, y, gp):
        gp.set_parameter_vector(params)
        return -gp.log_likelihood(y)

    def grad_neg_log_like(params, y, gp):
        gp.set_parameter_vector(params)
        return -gp.grad_log_likelihood(y)[1]

    # Fit for the maximum likelihood parameters
    initial_params = gp.get_parameter_vector()
    bounds = gp.get_parameter_bounds()
    
    def _mle(y, gp, initial_params, bounds):
        # Perform MLE optimization
        soln = minimize(neg_log_like, initial_params, jac=grad_neg_log_like,
                        method="L-BFGS-B", bounds=bounds, args=(y, gp))
        initial = np.array(soln.x)
        if verbose:
            print("Final log-likelihood: {0}".format(-soln.fun))
        return initial
        
    initial = _mle(y, gp, initial_params, bounds)   
    gp.set_parameter_vector(initial)

    return gp, initial, None

def simulate_drw(x, tau=50, sigma=0.2, ymean=0, size=1, seed=None):
    """
    Simulate a damped random walk (DRW) process using a Gaussian process (GP) model.

    Parameters:
    x (np.ndarray): Time data (1D array)
    tau (float, optional): Timescale of the DRW process. Default is 50.
    sigma (float, optional): Amplitude of the DRW process. Default is 0.2.
    ymean (float, optional): Mean value of the simulated data. Default is 0.
    size (int, optional): Number of realizations to simulate. Default is 1.
    seed (int, optional): Random seed for reproducibility. Default is None.

    Returns:
    np.ndarray: Simulated data array of shape (size, len(x))

    The function generates a realization of a damped random walk process based on
    the provided timescale (tau), amplitude (sigma), and other optional parameters
    using a Gaussian process model from the `celerite` library.
    """
    
    np.random.seed(seed)
    log_a = np.log(2*sigma**2)
    log_c = np.log(1/tau)
    kernel = terms.RealTerm(log_a=log_a, log_c=log_c)

    # Simulate
    gp = celerite.GP(kernel, mean=ymean)
    gp.compute(x)
    y = gp.sample(size=size)
    return y

def drw_fit_mle(x, y, yerr):
    """
    Simulates a Damped Random Walk (DRW) light curve and fits it using a maximum likelihood estimator (MLE).

    Parameters:
    -----------
    x : array-like
        Time values of a light curve.
    y : array-like
        Data values of a light curve.
    yerr : float or array-like, optional (default=0.2)
        Measurement uncertainties (errors). If a single float is provided, it is applied to all points.

    Returns:
    --------
    log_tau_out : float
        Estimated log timescale from the DRW MLE fit.
    Delta_AIC_low : float
        Change in AIC when forcing a short timescale fit.
    Delta_AIC_hi : float
        Change in AIC when forcing a long timescale fit.
    SNR : float
        Signal-to-noise ratio of the estimated variability.
    """
    
    # Define bounds for the DRW MLE fit
    min_precision = np.min(yerr)
    amplitude = np.max(y+yerr)-np.min(y-yerr)
    amin = np.log(0.001*min_precision)
    amax = np.log(1000*amplitude)
    min_cadence = np.clip(np.min(np.diff(x)), 1e-8, None)
    cmin = np.log(1/(1e9))
    cmax = np.log(1/(1e-3))
    smin = -10
    smax = np.log(amplitude)
    bounds = [amin, amax, cmin, cmax, smin, smax] 

    # Perform DRW MLE fitting
    gp, initial, _ = drw_mle(x,y,yerr, bounds=bounds, verbose=False)
    AIC_best = -gp.log_likelihood(y) +2*3
    log_tau_out = np.log10(1/np.exp(initial[1]))
    sigma_out = np.sqrt(np.exp(initial[0])/2)
    SNR = np.sqrt(np.exp(initial[0])/2)/(np.sqrt(np.mean(yerr)**2 + np.exp(initial[2])**2))

    # Compute AIC when forcing a short timescale fit    
    min_precision = np.min(yerr)
    cmin = np.log(100/min_cadence)
    cmax = np.log(100/min_cadence)
    bounds = [amin, amax, cmin, cmax, smin,smax]
    gp, _, _ = drw_mle(x,y,yerr, bounds=bounds, verbose=False)
    AIC_low = -gp.log_likelihood(y) +2*3
    Delta_AIC_low = AIC_low - AIC_best

    # Compute AIC when forcing a long timescale fit    
    min_precision = np.min(yerr)
    cmin = np.log(1/(100*np.ptp(x)))
    cmax = np.log(1/(100*np.ptp(x)))
    bounds = [amin, amax, cmin, cmax, smin,smax]
    gp, _, _ = drw_mle(x,y,yerr, bounds=bounds, verbose=False)
    AIC_hi = -gp.log_likelihood(y) +2*3
    Delta_AIC_hi = AIC_hi- AIC_best
    
    return log_tau_out, sigma_out, Delta_AIC_low, Delta_AIC_hi, SNR


def simu_fit_mle(x, log_tau_drw, sigma, yerr=0.2, ymean=18., verbose=False):
    """
    Simulates a Damped Random Walk (DRW) light curve and fits it using a maximum likelihood estimator (MLE).

    Parameters:
    -----------
    x : array-like
        Time values of a light curve.
    log_tau_drw : float
        Logarithm of the DRW characteristic timescale.
    sigma : float
        DRW amplitude.
    yerr : float or array-like, optional (default=0.2)
        Measurement uncertainties (errors). If a single float is provided, it is applied to all points.
    ymean : float, optional (default=18.)
        Mean flux value of the simulated light curve.
    verbose : bool, optional (default=False)
        Whether to print additional information.

    Returns:
    --------
    log_tau_out : float
        Estimated log timescale from the DRW MLE fit.
    Delta_AIC_low : float
        Change in AIC when forcing a short timescale fit.
    Delta_AIC_hi : float
        Change in AIC when forcing a long timescale fit.
    SNR : float
        Signal-to-noise ratio of the estimated variability.
    """

    # Convert log_tau_drw back to linear scale    
    tau = 10**log_tau_drw

    # Convert yerr to an array if it's a scalar
    if type(yerr) == float:
        yerr = yerr*np.ones_like(x)

    # Ensure input arrays are numpy arrays
    x = np.array(x)
    yerr = np.array(yerr)
    # Sort x values and apply the same ordering to yerr
    ind = np.argsort(x)
    x = x[ind]
    yerr = np.clip(yerr[ind],1e-5,np.inf) # Clip yerr to avoid zero errors
    # Remove duplicate x values while keeping the first occurrence    
    x, ind = np.unique(x, return_index=True)
    yerr = yerr[ind]
    # Simulate the DRW light curve with given parameters    
    y = simulate_drw(x, tau=tau, sigma=sigma, ymean=ymean, size=1)
    y = y[0, :] # The first dimension is each sample
    
    # Add Gaussian noise based on the measurement uncertainty
    y += np.random.normal(0, yerr)
    # Apply Hampel filter to remove outliers
    x, y, outlier_mask = hampel_filter(x, y, 150, 3)
    yerr = yerr[~outlier_mask]
    log_tau_out, sigma_out, Delta_AIC_low, Delta_AIC_hi, SNR = drw_fit_mle(x, y, yerr)
    return log_tau_out, sigma_out, Delta_AIC_low, Delta_AIC_hi, SNR


