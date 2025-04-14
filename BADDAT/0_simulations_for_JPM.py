from multiprocessing import Pool
import numpy as np
from drw_toolkit import simu_fit_mle
noise = 0.1
base = 1e4
for ncadence in np.logspace(1, 4, 20).astype('int'):
    N1 = 1201
    N2 = 101
    N = N1*N2
    t= np.array([np.sort(np.array([0,*np.random.uniform(0, base, ncadence-2),base])) for _ in range(N)])
    log_tau_drw_true = np.linspace(-3, 9, N1)
    sigma_drw_true = 10**np.linspace(-2, 2, N2)
    pool = Pool(6)
    args = zip(t, np.repeat(log_tau_drw_true, N2), np.tile(sigma_drw_true,N1), noise*np.ones_like(t), [0.]*N)
    log_pars = pool.starmap(simu_fit_mle, args)
    pool.close()
    pool.join()
    pars0 = np.array(log_pars)
    np.save(f"simulate_mle_nc{ncadence}.npy", np.array(pars0))