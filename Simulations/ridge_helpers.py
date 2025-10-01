import numpy as np
from numpy.linalg import cholesky, solve
from dataclasses import dataclass

# ---------- helpers ----------
def ar1_cov(k, rho, eps=1e-8):
    idx = np.arange(k)
    S = rho ** np.abs(idx[:, None] - idx[None, :])
    S.flat[::k+1] += eps
    return S

def logpdf(X, mu, Sigma, L=None):
    if L is None:
        L = cholesky(Sigma)
    Z = solve(L, (X - mu).T)
    quad = np.sum(Z**2, axis=0)
    logdet = 2*np.sum(np.log(np.diag(L)))
    d = Sigma.shape[0]
    return -0.5*(d*np.log(2*np.pi) + logdet + quad)

def pop_ridge(beta_star, rho, mu, k, alpha, p, mode, rng, eps):
    S_A = ar1_cov(k, rho)
    mu_vec = np.full(k, mu)
    E_xx_A = S_A + np.outer(mu_vec, mu_vec)
    pop_size = 200000
    if mode == "well":
        print("Getting true population ridge for well-specified case.")
        bA = solve(E_xx_A + 2*alpha*np.eye(k), E_xx_A @ beta_star[:k])
        if p > k:
            bR = beta_star[k:] / (1 + 2*alpha)
        else:
            bR = np.array([])  # no remaining coordinates
        return np.concatenate([bA, bR])
    else: # mode == "mis"
        print("Getting population ridge for very big sample for mis-specified case.")
        X_A_pop = rng.multivariate_normal(mu_vec, S_A, size=pop_size)
        if p > k:
            X_R_pop = rng.standard_normal(size=(pop_size, p-k))
            X_pop = np.concatenate([X_A_pop, X_R_pop], axis=1)
        else:
            X_pop = X_A_pop
        y_pop = mis_specficied_model(X_pop, beta_star, eps, rng, pop_size)
        return solve(X_pop.T @ X_pop / pop_size + 2*alpha*np.eye(p), X_pop.T @ y_pop / pop_size)

def n_eff(weights):
    s1 = np.sum(weights)
    s2 = np.sum(weights**2)
    return (s1*s1) / (s2 + 1e-12)

def pred_mse(X, y, b):
    r = y - X @ b
    return float(np.mean(r*r))

def mis_specficied_model(X, beta_star, eps, rng, n):
    f_of_X = np.column_stack([
        X[:, 0]**2,            # square of 1st covariate
        X[:, 1]**2,            # square of 2nd covariate
        X[:, 2:X.shape[1]]     # the rest are used linearly
    ])
    return np.dot(beta_star, f_of_X.T) + rng.normal(0, eps, size=n)


# import numpy as np
# from sklearn.linear_model import LinearRegression, Lasso, Ridge
# import matplotlib.pyplot as plt


# gamma = 2

# for n_exponent in range(6):
#     n = 50 ** n_exponent
#     p = gamma * n
#     print(f"n: {n}, p: {p}")

#     lambda_n = (1/2)*np.sqrt(np.log(p)/n)

# #     #get true beta_ridge for p,n on target domain

# #     for r in range(R):
# #         #generate source X_S data from a distribution
# #         X_S = 

# #         #generate target X_T data from a different distribution
# #         X_T = 

# #         # generate Y_S using a linear model (well-specified)

# #         #or 
# #         # generate Y_S and Y_T using a mis-specified model

# #         #fit ridge on source data
# #         hbeta_ridge_S = Ridge(alpha=lambda_n).fit(X_S, Y_S).coef_

# #         #compute difference between hbeta_ridge_S and true beta_ridge_T (Scalar)
        

# #         #get weights on source data for weighted ridge
# #         weights = pdf_X_T(X_S)/pdf_X_S(X_S)

# #         #fit weighted ridge on source data
# #         hbeta_weighted_ridge_S = WeightedRidge(alpha=lambda_n, weights=weights
                                               
# #         #compute difference between hbeta_weighted_ridge_S and true beta_ridge_T (Scalar)

# #     #average differences over R replications
# #     avg_diff_ridge = np.mean(diff_ridge)
# #     avg_diff_weighted_ridge = np.mean(diff_weighted_ridge)

# #     #store avg differences for plotting
# #     avg_diffs_ridge.append(avg_diff_ridge)
# #     avg_diffs_weighted_ridge.append(avg_diff_weighted_ridge)

# # #plot avg differences vs n
# # plt.plot(n_values, avg_diffs_ridge, label='Ridge')



# import numpy as np
# from numpy.linalg import cholesky, solve
# from sklearn.linear_model import Ridge
# import matplotlib.pyplot as plt

# rng = np.random.default_rng(0)

# # ---------- helpers ----------
# def ar1_cov(p, rho):
#     idx = np.arange(p)
#     return rho ** np.abs(idx[:, None] - idx[None, :])

# # def logpdf_mvn(X, mu, Sigma):
# #     # X: (n,d), mu: (d,), Sigma: (d,d)
# #     d = Sigma.shape[0]
# #     L = cholesky(Sigma)                          # Sigma = L L^T
# #     Z = solve(L, (X - mu).T)                     # d x n
# #     quad = np.sum(Z**2, axis=0)                  # length n
# #     logdet = 2.0 * np.sum(np.log(np.diag(L)))
# #     return -0.5 * (d*np.log(2*np.pi) + logdet + quad)

# # def true_ridge_beta(Sigma, beta_star, alpha):
# #     # Population minimizer for sklearn's ridge objective
# #     d = Sigma.shape[0]
# #     return solve(Sigma + 2*alpha*np.eye(d), Sigma @ beta_star)

# # ---------- settings ----------
# gamma = 1.1
# R = 10                          # replications
# sig_eps = 1.0                   # noise sd
# n_values = [200, 400, 800, 1600]  # grow n; p = gamma*n
# rho_S, rho_T = 0.2, 0.8         # AR(1) correlation, source vs target

# avg_diffs_ridge = []
# avg_diffs_weighted_ridge = []

# for n in n_values:
#     p = int(gamma * n)
#     print(f"n={n}, p={p}")

#     # tuning; feel free to swap for CV if you like
#     alpha = 0.5 * np.sqrt(np.log(p) / n)

#     # target covariance and population ridge target
#     Sigma_T = ar1_cov(p, rho_T)
#     # sparse beta*
#     s = min(20, p)
#     beta_star = np.zeros(p); beta_star[:s] = 1/np.sqrt(s)
#     beta_true_T = true_ridge_beta(Sigma_T, beta_star, alpha)

#     diffs_ridge = []
#     diffs_weighted = []

#     # fixed means at zero to avoid intercept complications
#     mu_S = np.zeros(p); mu_T = np.zeros(p)
#     Sigma_S = ar1_cov(p, rho_S)

#     for r in range(R):
#         # source / target draws
#         X_S = rng.multivariate_normal(mu_S, Sigma_S, size=n)
#         X_T = rng.multivariate_normal(mu_T, Sigma_T, size=n)

#         # well-specified linear responses on source/target
#         y_S = X_S @ beta_star + rng.normal(0, sig_eps, size=n)
#         y_T = X_T @ beta_star + rng.normal(0, sig_eps, size=n)

#         # plain ridge on source
#         ridge = Ridge(alpha=alpha, fit_intercept=False, solver="auto", random_state=0)
#         ridge.fit(X_S, y_S)
#         bhat_S = ridge.coef_.copy()

#         # distance to population target
#         diffs_ridge.append(np.linalg.norm(bhat_S - beta_true_T))

#         # importance weights w(x) = p_T(x) / p_S(x) evaluated at source samples
#         logw = logpdf_mvn(X_S, mu_T, Sigma_T) - logpdf_mvn(X_S, mu_S, Sigma_S)
#         w = np.exp(np.clip(logw, -700, 700))     # stabilize

#         # weighted ridge (sklearn supports sample_weight)
#         ridge_w = Ridge(alpha=alpha, fit_intercept=False, solver="auto", random_state=0)
#         ridge_w.fit(X_S, y_S, sample_weight=w)
#         bhat_w = ridge_w.coef_.copy()

#         diffs_weighted.append(np.linalg.norm(bhat_w - beta_true_T))

#     avg_diffs_ridge.append(np.mean(diffs_ridge))
#     avg_diffs_weighted_ridge.append(np.mean(diffs_weighted))

# # ---------- plot ----------
# plt.figure()
# plt.plot(n_values, avg_diffs_ridge, marker='o', label='Ridge (source)')
# plt.plot(n_values, avg_diffs_weighted_ridge, marker='o', label='IW-Ridge (source→target)')
# plt.xlabel('n')
# plt.ylabel(r'$\|\hat\beta - \beta_{\mathrm{ridge},T}^{\mathrm{pop}}\|_2$')
# plt.title('Distance to Population Target on Target Domain')
# plt.legend()
# plt.tight_layout()
# plt.show()