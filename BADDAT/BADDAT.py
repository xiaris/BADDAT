import numpy as np
from sklearn.linear_model import BayesianRidge
from multiprocessing import Pool
import emcee
from scipy.ndimage import gaussian_filter
import autograd
import os
"""
tau values used in this project are in logarithmic scale in default.

This project includes code from the repository:
https://github.com/dfm/emcee, authored by Dan Foreman-Mackey.
"""


def HXY(tau1, tau0):
    """
    Generate a 2D histogram of input data and apply Gaussian filtering. 

    Parameters:
    tau1 (np.ndarray): First input data array for the histogram
    tau0 (np.ndarray): Second input data array for the histogram

    Returns:
    tuple: 
        - X2 (np.ndarray): Extended bin centers for the x-axis
        - Y2 (np.ndarray): Extended bin centers for the y-axis
        - H2 (np.ndarray): 2D histogram data with Gaussian filtering applied
    """

    # Create a 2D histogram from the input data
    H, X, Y = np.histogram2d(
        tau1,
        tau0,
        bins=[np.arange(-3, 9, 0.01),np.arange(-3, 9, 0.01)]
    )
    # Apply a Gaussian filter to smooth the histogram data
    H = gaussian_filter(H, 10)
    
    # Compute the density levels.
    Hflat = H.flatten()
    inds = np.argsort(Hflat)[::-1]
    Hflat = Hflat[inds]
    sm = np.cumsum(Hflat)
    sm /= sm[-1]
    
    # Compute the bin centers.
    X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])
    
    # Extend the array for the sake of the contours at the plot edges.
    H2 = H.min() + np.zeros((H.shape[0] + 4, H.shape[1] + 4))
    H2[2:-2, 2:-2] = H
    H2[2:-2, 1] = H[:, 0]
    H2[2:-2, -2] = H[:, -1]
    H2[1, 2:-2] = H[0]
    H2[-2, 2:-2] = H[-1]
    H2[1, 1] = H[0, 0]
    H2[1, -2] = H[0, -1]
    H2[-2, 1] = H[-1, 0]
    H2[-2, -2] = H[-1, -1]
    X2 = np.concatenate(
        [
            X1[0] + np.array([-2, -1]) * np.diff(X1[:2]),
            X1,
            X1[-1] + np.array([1, 2]) * np.diff(X1[-2:]),
        ]
    )
    Y2 = np.concatenate(
        [
            Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]),
            Y1,
            Y1[-1] + np.array([1, 2]) * np .diff(Y1[-2:]),
        ]
    )
    return X2, Y2, H2




def median_and_error(flat_samples,axis):
    """
    Compute the median and symmetric error bounds (68% confidence interval) 
    for a given set of samples along a specified axis.

    Parameters:
    flat_samples (np.ndarray): nD array of samples.
    axis (int): The axis along which to compute the median and errors.

    Returns:
    tuple: A tuple containing:
        - mid (float): The median value of the samples along the specified axis.
        - high_error (float): The positive error (84th percentile - median).
        - low_error (float): The negative error (median - 16th percentile).
    
    The function computes the median and the symmetric errors (68% confidence interval) 
    by calculating the 16th and 84th percentiles of the samples along the given axis.
    """
    lo, mid, hi = np.percentile(flat_samples[:,axis], [16,50,84])
    return mid, hi-mid, mid-lo


# Global variables for storing Joint Probability Map and rho iteration values
_JPM = None
_rho_iter = None  

def _compute_jpm():
    """
    Compute the Joint Probability Map (JPM) for a range of cadence values using data from simulated MLE runs.
    
    The function iterates over a range of ncadence (sampling count of light curves) values, loads simulation results, 
    calculates a joint probability distribution, and stores the result in the JPM array.

    Returns:
    tuple: A tuple containing:
        - JPM (np.ndarray): The computed Joint Probability Map (JPM) as a 3D array.
        - rho_iter (np.ndarray): The array representing the iteration values based on the X2 bin centers.
        
    The Joint Probability Map is computed for each ncadence, where the map is constructed
    by calculating the histogram of two parameter values (tau1 and tau0) from the simulation data.
    The method uses AIC to filter out bad data points.
    """
    print("Computing JPM")
    
    # Constants 
    base = 10000
    N1 = 1201
    N2 = 101
    JPM = []
    dir_path = os.path.dirname(os.path.abspath(__file__))
    for ncadence in np.logspace(1, 4, 20).astype('int'):
        params = np.load(f"{dir_path}/simulate_mle_nc{ncadence}.npy")
        log_tau_drw_true = np.linspace(-3, 9, N1)
        ind_array = (np.abs( np.log10(params[:, 1]) - np.log10(0.3) ) < 0.1)
        Delta_AIC_low = params[ind_array, 2]
        Delta_AIC_hi = params[ind_array, 3]
        tau1 =  params[ind_array, 0][(Delta_AIC_hi>1)&(Delta_AIC_low>1)]
        tau0 =  np.repeat(np.linspace(-3, 9, N1), N2)[ind_array][(Delta_AIC_hi>1)&(Delta_AIC_low>1)]
    
        X2,Y2,H2=HXY(tau1,tau0)
        JPM.append(H2)
    JPM = np.array(JPM)
    rho_iter = X2 - np.log10(base)    
    return JPM, rho_iter

def get_computed_jpm():
    """
    Retrieve the computed Joint Probability Map (JPM).

    This function checks whether the global variables _JPM and _rho_iter have been computed.
    If they have not been computed (i.e., they are None), it calls _compute_jpm()
    to compute them. The computed values are then returned.

    Returns:
    tuple: A tuple containing:
        - _JPM (np.ndarray): The computed Joint Probability Map (JPM).
        - _rho_iter (np.ndarray): The iteration values used for computing the JPM.
    """
    global _JPM
    global _rho_iter
    # Check if the Joint Probability Map and rho iteration values are already computed
    if _JPM is None or _rho_iter is None:
    # If not computed, calculate them
        _JPM, _rho_iter = _compute_jpm()
    # Return the computed JPM and rho_iter
    return _JPM, _rho_iter

def tau0_to_tau1(tau0_array, baseline_array, ncadence_array):
    """
    Convert tau0 values to tau1 using a Joint Probability Map (JPM).

    This function uses the computed JPM and rho iteration values to calculate tau1 
    from given tau0 values. 

    Parameters:
    tau0_array (np.ndarray): Array of tau values to be converted.
    baseline_array (np.ndarray): Array of baseline values used for scaling tau0.
    ncadence_array (np.ndarray): Array of sampling count values for matching the computed JPM.

    Returns:
    tuple: A tuple containing:
        - tau_mean (np.ndarray): The mean of the tau1 values computed from JPM.
        - rho_std (np.ndarray): The standard deviation of the tau1 values computed from JPM (tau_std).
    """
    # Baseline set to simulate    
    base = 10000
    # Retrieve the computed Joint Probability Map (JPM) and rho iteration values    
    JPM, rho_iter = get_computed_jpm()
    # Calculate rho0 for each tau0 value and baseline
    rho0_array = tau0_array - np.log10(baseline_array)
    # Find the closest indices for rho0 values in the rho_iter array
    ind_array = np.argmin(np.abs(rho0_array - rho_iter.reshape(-1,1)), axis=0)
    # Find the closest indices for ncadence values in the ncadence array
    ind_map = np.argmin(np.abs(ncadence_array - np.logspace(1, 4, 20).astype('int').reshape(-1,1)), axis=0)
    # Extract the probability distribution (PDF) from JPM based on the indices
    pdf_array = (JPM[ind_map, :, ind_array]).T
    # Calculate the mean of the rho values weighted by the PDF
    rho_mean = np.sum(pdf_array * rho_iter.reshape(-1,1) / np.sum(pdf_array,axis=0), axis=0)
    # Calculate the standard deviation of the rho values weighted by the PDF
    rho_std = np.sqrt(np.sum(pdf_array * (rho_iter.reshape(-1,1) - rho_mean)**2 / np.sum(pdf_array,axis=0), axis=0))
    # Return the computed mean and standard deviation, converted back to the scale of tau
    return rho_mean+np.log10(baseline_array), rho_std

def softplus(x,s_k=1):
    return x - s_k * np.log(1 + np.exp(x / s_k))

def softplus_derivative(x, k_s=1):
    return 1 - 1 / (1 + np.exp(-x / k_s))
    
class DependenceFitter1:
    """
    A model to fit the dependence of tau on param1, incorporating the effects of 
    observation baseline and cadence. The fitting is performed using MCMC sampling.

    Parameters:
    -----------
    tau : array-like
        Observed tau values (logarithmic scale).
    baseline : array-like
        Observation baseline for each light curve.
    n_cadence : array-like
        Number of data point for each light curve.
    param1 : array-like
        Physical parameter of the sources.
    param1_e : float or array-like, optional (default=0.)
        Uncertainty in param1.
    redshift : float, optional (default=0.)
        Redshift of the sources.
    """
    def __init__(self, tau, baseline, n_cadence, param1, param1_e=0., redshift=0.):
        self.tau = tau
        self.baseline = baseline
        self.n_cadence = n_cadence
        self.param1 = param1
        self.param1_e = param1_e
        self.redshift = redshift
            
    def log_likelihood(self, theta):
        """
        Compute the log-likelihood of the model given the parameters.

        Parameters:
        -----------
        theta : tuple
            Model parameters (slope, intercept, noise).

        Returns:
        --------
        float
            Log-likelihood value.
        """
        slope, intercept, noise = theta
        # Compute model tau values with redshift correction
        tau_m = slope * self.param1 + intercept + np.log10(1+self.redshift)
        # Convert tau0 to tau1 using the Joint Probability Map (JPM)
        tau1_me, tau1_err = tau0_to_tau1(tau_m, self.baseline, self.n_cadence)
        # Check for invalid values
        if np.any(np.isnan(tau1_me)):
            return -np.inf
        # Compute total variance
        sigma2 = tau1_err**2 + (noise**2 + (slope*self.param1_e)**2) * softplus_derivative(tau_m- np.log10(self.baseline) + 0.511636601229464, 0.4239987587065787) ** 2 

        # Compute the log-likelihood
        return -0.5 * np.sum((self.tau - tau1_me) ** 2 / sigma2 + np.log(sigma2))
    def log_prior(self, theta):
        """
        Define the prior probability distribution for the parameters.

        Parameters:
        -----------
        theta : tuple
            Model parameters (slope, intercept, noise).

        Returns:
        --------
        float
            Log-prior value.
        """
        slope, intercept, noise = theta
        if -2 < slope < 2 and 0 < noise:
            return 0.0
        return -np.inf
    
    def log_probability(self, theta):
        """
        Compute the total log-probability, combining prior and likelihood.

        Parameters:
        -----------
        theta : tuple
            Model parameters (slope, intercept, noise).

        Returns:
        --------
        float
            Log-probability value.
        """
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta)

    def fit(self):
        """
        Fit the model using MCMC sampling.

        Returns:
        --------
        flat_samples : array
            Posterior samples of model parameters.
        """
        initial = np.array([*np.polyfit(self.param1, self.tau, deg=1), 0.1])+ 0.01 * np.random.randn(3)
        pos = initial + 1e-4 * np.random.randn(16, 3)
        nwalkers, ndim = pos.shape
        # Run MCMC sampling using multiprocessing
        with Pool(8) as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, self.log_probability, args=(), pool=pool
            )
            sampler.run_mcmc(pos, 1000, progress=True);
        flat_samples = sampler.get_chain(discard=200, thin=15, flat=True)
        return flat_samples

class DependenceFitter2:
    """
    A model to fit the dependence of tau on param1 and param2, incorporating the effects of 
    observation baseline and cadence. The fitting is performed using MCMC sampling.


    Parameters:
    -----------
    tau : array-like
        Observed tau values (logarithmic scale).
    baseline : array-like
        Observation baseline for each light curve.
    n_cadence : array-like
        Number of data point for each light curve.
    param1 : array-like
        First physical parameter of the sources.
    param2 : array-like
        Second physical parameter of the sources.
    param1_e : float or array-like, optional (default=0.)
        Uncertainty in param1.
    param2_e : float or array-like, optional (default=0.)
        Uncertainty in param2.
    redshift : float, optional (default=0.)
        Redshift of the sources.
    """
    def __init__(self, tau, baseline, n_cadence, param1, param2, param1_e=0., param2_e=0., redshift=0.):
        self.tau = tau
        self.baseline = baseline
        self.n_cadence = n_cadence
        self.param1 = param1
        self.param1_e = param1_e
        self.param2 = param2
        self.param2_e = param2_e        
        self.redshift = redshift
    def log_likelihood(self, theta):
        """
        Compute the log-likelihood of the model given the parameters.

        Parameters:
        -----------
        theta : tuple
            Model parameters (slope, intercept, noise).

        Returns:
        --------
        float
            Log-likelihood value.
        """
        slope1, slope2, intercept, noise = theta
        tau_m = slope1* self.param1 + slope2*self.param2 +intercept + np.log10(1+self.redshift)
        tau1_me, tau1_err = tau0_to_tau1(tau_m, self.baseline, self.n_cadence)
        if np.any(np.isnan(tau1_me)):
            return -np.inf
        sigma2 = tau1_err**2 + (noise**2 + (slope1*self.param1_e)**2 + (slope2*self.param2_e)**2) * softplus_derivative(tau_m- np.log10(self.baseline) + 0.511636601229464, 0.4239987587065787) ** 2 
        return -0.5 * np.sum((self.tau - tau1_me) ** 2 / sigma2 + np.log(sigma2))
    def log_prior(self, theta):
        """
        Define the prior probability distribution for the parameters.

        Parameters:
        -----------
        theta : tuple
            Model parameters (slope, intercept, noise).

        Returns:
        --------
        float
            Log-prior value.
        """
        slope1, slope2, intercept, noise = theta
        if -2 < slope1 < 2 and -2 < slope2 < 4 and 0 < noise:
            return 0.0
        return -np.inf

    def log_probability(self, theta):
        """
        Compute the total log-probability, combining prior and likelihood.

        Parameters:
        -----------
        theta : tuple
            Model parameters (slope, intercept, noise).

        Returns:
        --------
        float
            Log-probability value.
        """
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta)

    def fit(self):
        """
        Fit the model using MCMC sampling.

        Returns:
        --------
        flat_samples : array
            Posterior samples of model parameters.
        """
        # Use Bayesian Ridge Regression to obtain an initial guess for parameters
        X = np.column_stack((self.param1, self.param2))
        y = self.tau
        model = BayesianRidge();
        model.fit(X, y)
        initial = np.array([*model.coef_, model.intercept_, 0.1])+ 0.01 * np.random.randn(4)
        pos = initial + 1e-4 * np.random.randn(16, 4)
        nwalkers, ndim = pos.shape
        # Run MCMC sampling using multiprocessing
        with Pool(8) as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, self.log_probability, args=(), pool=pool
            )
            sampler.run_mcmc(pos, 1000, progress=True);
        flat_samples = sampler.get_chain(discard=200, thin=15, flat=True)
        return flat_samples

