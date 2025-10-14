import numpy as np
from numpy.linalg import cholesky, solve
from distributions import second_moment, make_family

# ---------- helpers ----------
def chol_solve(A,b):
    L = cholesky(A)
    return solve(L.T, solve(L, b))

# ---------- ridge population coefficients ----------
def pop_ridge_target(beta_star, alpha, p, specified_mode, rng, eps, k, family, params, pop_size = 200000, prefer_closed_form = True, verbose = False):
     """
    Population ridge minimizer under general X-distributions.

    Args (key):
      - family: 'gaussian_AR1', 'gaussian_indep', 'laplace_indep', 'student_t_indep'
      - params: dicts with keys depending on family
      - k: shift dimension
        if iid family, can use first k and append (p-k) iid of same family
        if non-iid, ignore split and set k=p

    Returns:
      - beta_pop: population ridge coefficient in R^p.
    """
     # check some args
     if params is None: params = {}
     if not (0 < k <= p):
        raise ValueError("k must be in 1..p")
     if specified_mode not in ["well", "mis"]:
        raise ValueError("specified_mode must be 'well' or 'mis'")
     
    # if family is not iid, set k = p
     if family not in ["gaussian_indep", "laplace_indep", "student_t_indep"] and k < p:
        k = p
     
    # get second moments if possible
     E_xx_A, muA = second_moment(family, k, params)
     E_xx_R, muR = second_moment(family, p - k, params) if p > k else (None, None)

     have_closed_form = (E_xx_A is not None) and ((p == k) or (E_xx_R is not None))

     if prefer_closed_form and have_closed_form:
         
         if p == k:
             mu_vec = muA
             Cov_A = E_xx_A - np.outer(muA, muA)
             Cov = Cov_A
         else:
             mu_vec = np.concatenate([muA, muR])
             Cov_A = E_xx_A - np.outer(muA, muA)
             Cov_R = (E_xx_R - np.outer(muR, muR)) 
             Cov = np.block([
                [Cov_A,                     np.zeros((k, p - k))],
                [np.zeros((p - k, k)),      Cov_R]
            ])
             
         E_xx = Cov + np.outer(mu_vec, mu_vec)

         if specified_mode == "well":
            if verbose:
                print("Getting true population ridge for well-specified case.")
                return chol_solve(E_xx + 2*alpha*np.eye(p), E_xx @ beta_star)
    
    # need monte carlo
     if verbose:
        if specified_mode == "well" and not have_closed_form:
            print("Closed-form moments unavailable; using MC for well-specified case.")
        if specified_mode != "well":
            print("Estimating E[xx^T] and/or E[xy] by MC for mis-specified case.")

        sample_func, _ = make_family(family, **params)

        X_A_pop = sample_func(pop_size, k, rng)

        if p > k:
            X_R_pop = sample_func(pop_size, p - k, rng)
            X_pop = np.concatenate([X_A_pop, X_R_pop], axis=1)
        else:
            X_pop = X_A_pop
        
        if specified_mode == "well":
            y_pop = X_pop @ beta_star + rng.normal(0, eps, size=pop_size)
        else:
            y_pop = mis_specficied_model(X_pop, beta_star, eps, rng, pop_size)

        XtX = (X_pop.T @ X_pop) / pop_size
        XtY = (X_pop.T @ y_pop) / pop_size

        return chol_solve(XtX + 2*alpha*np.eye(p), XtY)
     
def pop_ridge_source(beta_star, alpha, p, specified_mode, rng, eps, k, family_T, family_S, params_T, params_S, pop_size = 200000, prefer_closed_form = True, verbose = False):
    """
Population ridge minimizer under general X-distributions.

Args (key):
    - family: 'gaussian_AR1', 'gaussian_indep', 'laplace_indep', 'student_t_indep'
    - params: dicts with keys depending on family
    - k: shift dimension
    if iid family, can use first k and append (p-k) iid of same family
    if non-iid, ignore split and set k=p

Returns:
    - beta_pop: population ridge coefficient in R^p.
"""
    # check some args
    if not (0 < k <= p):
        raise ValueError("k must be in 1..p")
    if specified_mode not in ["well", "mis"]:
        raise ValueError("specified_mode must be 'well' or 'mis'")
    
    # if family is not iid, set k = p
    if family_S not in ["gaussian_indep", "laplace_indep", "student_t_indep"] and k < p:
        k = p

    sample_func_S, _ = make_family(family_S, **params_S)
    sample_func_T, _ = make_family(family_T, **params_T)

    X_A_pop = sample_func_S(pop_size, k, rng)

    if p > k:
        X_R_pop = sample_func_T(pop_size, p - k, rng)
        X_pop = np.concatenate([X_A_pop, X_R_pop], axis=1)
    else:
        X_pop = X_A_pop
    
    if specified_mode == "well":
        y_pop = X_pop @ beta_star + rng.normal(0, eps, size=pop_size)
    else:
        y_pop = mis_specficied_model(X_pop, beta_star, eps, rng, pop_size)

    XtX = (X_pop.T @ X_pop) / pop_size
    XtY = (X_pop.T @ y_pop) / pop_size

    return chol_solve(XtX + 2*alpha*np.eye(p), XtY)

# ---------- more helpers ----------
def pred_mse(X, y, b):
    r = y - X @ b
    return float(np.mean(r*r))

def mis_specficied_model(X, beta_star, eps, rng, n):
    f_of_X = np.column_stack([
        X[:, 0]**2,            # square of 1st covariate
        X[:, 1]**2,            # square of 2nd covariate
        X[:, 2]**2,            # square of 3rd covariate
        # X[:, 3]**2,            # square of 4th covariate
        # 5* X[:, 4]**2,            # square of 5th covariate
        # X[:, 5]**2,            # square of 6th covariate
        # 0.1* X[:, 6]**2,            # square of 7th covariate
        # X[:, 7]**2,            # square of 8th covariate
        X[:, 3:X.shape[1]]     # the rest are used linearly
    ])
    return np.dot(beta_star, f_of_X.T) + rng.normal(0, eps, size=n)