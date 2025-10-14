import numpy as np
from numpy.linalg import cholesky, solve


# ---------- distribution property functions ----------


def ar1_cov(k, rho, eps=1e-8):
    """
    AR(1) covariance with |rho|<1. Adds small eps to the diagonal for SPD safety.
    """
    idx = np.arange(k)
    S = rho ** np.abs(idx[:, None] - idx[None, :])
    S.flat[::k+1] += eps
    return S

def second_moment(family, p, params):
    """
    Return (E[xx^T], mu) for a d-dimensional block, if closed-form exists.
    Otherwise return (None, mu) to signal Monte Carlo needed.
    family options: 'gaussian_AR1', 'gaussian_indep', 'laplace_indep', 'student_t_indep'
    params is a dict; for Gaussian, needs 'rho' and 'mu'; for Laplace, needs 'b' and 'mu'; for t, needs 'df' and 'mu'.
    """

    fam = family.lower()
    if fam == "gaussian_indep":
        mu  = params.get("mu", 0.0) # default mean 0
        sig = params.get("sigma", 1.0) # default std 1
        mu_vec = np.full(p, mu) if np.isscalar(mu) else np.asarray(mu)
        var = (sig * np.ones(p))**2 if np.isscalar(sig) else (np.asarray(sig)**2)
        Cov = np.diag(var)
        E_xx = Cov + np.outer(mu_vec, mu_vec)
        return E_xx, mu_vec


    if fam == "gaussian_ar1":
        rho = params.get("rho", 0.0)
        mu  = params.get("mu", 0.0)
        mu_vec = np.full(p, mu)
        S = ar1_cov(p, rho)
        E_xx = S + np.outer(mu_vec, mu_vec)
        return E_xx, mu_vec

   

    if fam == "laplace_indep":
        # X_i ~ Laplace(mu, b) indep => Var = 2 b^2 (exists)
        mu  = params.get("mu", 0.0)
        b   = params.get("b", 1.0)
        mu_vec = np.full(p, mu) if np.isscalar(mu) else np.asarray(mu)
        b_vec  = np.full(p, b)  if np.isscalar(b) else np.asarray(b)
        var = 2.0 * (b_vec**2)
        Cov = np.diag(var)
        E_xx = Cov + np.outer(mu_vec, mu_vec)
        return E_xx, mu_vec

    if fam == "student_t_indep":
        # X_i = mu + scale * t_df ; Var exists only for df>2: df/(df-2) * scale^2
        mu    = params.get("mu", 0.0)
        df    = params.get("df", 5.0)
        scale = params.get("scale", 1.0)
        mu_vec = np.full(p, mu) if np.isscalar(mu) else np.asarray(mu)
        if df is not None and df > 2:
            sc  = np.full(p, scale) if np.isscalar(scale) else np.asarray(scale)
            var = (df / (df - 2.0)) * (sc**2)
            Cov = np.diag(var)
            E_xx = Cov + np.outer(mu_vec, mu_vec)
            return E_xx, mu_vec
        else:
            # no finite second moment -> need Monte Carlo
            return None, mu_vec

    raise ValueError(f"Unknown family '{family}'")

# ---------- IID families ----------
def sample_laplace(rng, n, loc, b, d):
    loc = np.full(d, loc) if np.isscalar(loc) else loc
    return rng.laplace(loc=loc, scale=b, size=(n, d))

def logpdf_laplace(X, loc, b):
    loc = np.full(X.shape[1], loc) if np.isscalar(loc) else loc
    d = X.shape[1]
    return -d*np.log(2*b) - np.sum(np.abs(X - loc)/b, axis=1)

def sample_student_t(rng, n, df, loc, scale, d):
    loc = np.full(d, loc) if np.isscalar(loc) else loc
    Z = rng.standard_t(df=df, size=(n, d))
    return loc + scale*Z

def logpdf_student_t(X, df, loc, scale):
    loc = np.full(X.shape[1], loc) if np.isscalar(loc) else loc
    d = X.shape[1]
    from math import lgamma
    c = lgamma((df+1)/2) - lgamma(df/2) - 0.5*np.log(df*np.pi) - np.log(scale)
    quad = np.sum(np.log1p(((X - loc)/scale)**2 / df), axis=1)
    return d*c - ((df+1)/2)*quad


# ---------- MVN (could be non-IID) ----------
def logpdf_mvn(X, mu, Sigma, L=None):
    if L is None:
        L = cholesky(Sigma)
    Z = solve(L, (X - mu).T)
    quad = np.sum(Z**2, axis=0)
    logdet = 2*np.sum(np.log(np.diag(L)))
    d = Sigma.shape[0]
    return -0.5*(d*np.log(2*np.pi) + logdet + quad)

# ---------- make family ----------
def make_family(name, **params):
    """
    Returns two callables: sample(n, p, rng) and logpdf(X).
    Supported: 'gaussian_indep', 'gaussian_AR1', 'laplace_indep', 'student_t_indep'.
    """
    name = name.lower()
    if name == "gaussian_indep":
        mu = params.get("mu", 0.0)
        sigma = params.get("sigma", 1.0)

        def sample(n, p, rng):
            mu_vec = np.full(p, mu)
            return rng.normal(loc=mu_vec, scale=sigma, size=(n, p))
        def lpdf(X):
            mu_vec = np.full(X.shape[1], mu) if np.isscalar(mu) else np.asarray(mu)
            Sigma = np.eye(X.shape[1]) * (sigma**2)
            return logpdf_mvn(X, mu_vec, Sigma)
        return sample, lpdf
    
    elif name == "gaussian_ar1":
        mu = params.get("mu", 0.0)
        rho = params.get("rho", 0.0)

        def sample(n, p, rng):
            mu_vec = np.full(p, mu) if np.isscalar(mu) else np.asarray(mu)
            Sigma = ar1_cov(p, rho)
            return rng.multivariate_normal(mu_vec, Sigma, size=n)
        def lpdf(X):
            mu_vec = np.full(X.shape[1], mu) if np.isscalar(mu) else np.asarray(mu)
            Sigma = ar1_cov(X.shape[1], rho)
            L = cholesky(Sigma)
            return logpdf_mvn(X, mu_vec, Sigma, L)
        return sample, lpdf
    
    elif name == "laplace_indep":
        mu  = params.get("mu", 0.0)
        b   = params.get("b", 1.0)

        def sample(n, p, rng):
            return sample_laplace(rng, n, mu, b, p)
        def lpdf(X):
            return logpdf_laplace(X, mu, b)
        return sample, lpdf
    
    elif name == "student_t_indep":
        mu  = params.get("mu", 0.0)
        df  = params.get("df", 5.0)
        scl = params.get("scale", 1.0)
        def sample(n, p, rng):
            return sample_student_t(rng, n, df, mu, scl, p)
        def lpdf(X):
            return logpdf_student_t(X, df, mu, scl)
        return sample, lpdf
    else:
        raise ValueError(f"Unknown family '{name}'")
    