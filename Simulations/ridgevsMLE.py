import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal 

# def generate_data(n,d, x,verbose = False):
#     x = np.random.randn(n, d)

#     # Resample 50 points with covariate shift in the first feature
#     x_shifted, w = resample_shift(x, covariate_index=0, alpha=1, n_sample=n)

#     if verbose:
#         # Plot histograms of the first covariate
#         plt.figure(figsize=(10, 6))
#         plt.hist(x[:, 0], bins=20, alpha=0.5, label='Original x[:, 0]')
#         plt.hist(x_shifted[:, 0], bins=20, alpha=0.5, label='Shifted x[:, 0]')
#         plt.xlabel('Value of x[:, 0]')
#         plt.ylabel('Frequency')
#         plt.title('Histogram of First Covariate: Original vs. Shifted')
#         plt.legend()
#         plt.grid(True)
#         plt.show()

#     return x, w, x_shifted

def getEXY(mu, sigma,d, gamma):
    moment3 = mu**3 + 3 * mu * sigma**2
    moment2 = mu**2 + sigma**2

    if d == 1:
        return moment3


    #squares
    index1 = np.concatenate([[moment3], np.full(d-1, mu * moment2)])
    index2 = np.concatenate([[mu*moment2], [moment3], np.full(d-2, mu*moment2)])
    indexs = []
    for i in range(2, d):
        indexs.append(np.concatenate([np.full(i, mu**2), [moment2], np.full(d-i-1, mu**2)]))

    #return [moment3+ mu* moment2+ 1.5 * mu**2, mu*moment2 +moment3+1.5*mu**2, mu*moment2+mu*moment2+1.5*moment2, mu*moment2+mu*moment2 +1.5*mu**2]
    return np.dot(gamma, np.column_stack([index1,index2, *indexs]).T)

#np.random.seed(1)

d = 100
# delta = 0.01
# c = 1
model_mode = 'well-specified'  # 'misspecified' or 'well-specified'
regularize = True

#n_vals = np.linspace(d/2, 10*d, 100, dtype=int)
n_vals = np.arange(105, 150, 5)  # Sample sizes from 100 to 1000 in steps of 100
#n_vals = [10000, 100000]
excess_risk_MLE = []
excess_risk_MWLE = []
upper_bounds = []

mu_S = 0
mu_T = 2

mu_S_vec = np.full(d, mu_S)
mu_T_vec = np.full(d, mu_T)


# rho_S = 0.1
# rho_T = 0.2

# Sigma_S = rho_S ** np.abs(np.subtract.outer(np.arange(d), np.arange(d)))
# Sigma_T = rho_T ** np.abs(np.subtract.outer(np.arange(d), np.arange(d)))
Sigma_S = np.identity(d)

sigma_T = 2
sigma_S = 1
Sigma_T = sigma_T * np.identity(d) 
Sigma_S = sigma_S * np.identity(d)
# Sigma_T = np.identity(d) * sigma_2

# trace = np.linalg.norm(mu_T_vec)**2 + sigma_T * d
# log_term = np.log(d/delta)
mc_iters = 10000
# regularization parameter
# shift_vec = np.zeros(d, dtype = np.float64)
# shift_vec[:3] = [1,0,0]

X_T_oracle = np.random.multivariate_normal(mean=mu_T_vec, cov=Sigma_T, size=100000)

if model_mode == 'well-specified':
    #  beta_star = np.zeros(d, dtype = np.float64)
    #  # beta_star[:5] = [1,1, 1.5, 0.5,1]
    #  beta_star[:3] = [1,1,1.5]
    beta_star = np.random.randn(d)
     

elif model_mode == 'misspecified':
    gamma = np.zeros(d, dtype = np.float64)
    gamma[:3] = [1, 1, 1.5]

    #analytical MLE
    # beta_analytical_T = np.linalg.inv(mu_T_vec.reshape(-1,1)@mu_T_vec.reshape(-1,1).T+Sigma_T) @ getEXY(mu_T,np.sqrt(sigma_T),d, gamma)

    #source
    # beta_analytical_S = np.linalg.inv(mu_S_vec.reshape(-1,1)@mu_S_vec.reshape(-1,1).T+Sigma_S) @ getEXY(mu_S,np.sqrt(sigma_S),d, gamma)
    
    

# oracle_model_T = LinearRegression().fit(X_T_oracle, y_T_oracle)
# beta_oracle_T = oracle_model_T.coef_

# oracle_model_S = LinearRegression().fit(X_S_oracle, y_S_oracle)
# beta_oracle_S = oracle_model_S.coef_
# alphas = np.linspace(0.01, 0.7, 12) # Regularization parameters for Lasso

# #fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(12, 12))
# for i in range(4):
#     for j in range(4):
#         ax = axes[i, j]
#         alpha = alphas[i+j]

all_mc_excess_risks = []
all_mc_excess_risks_ridge = []
all_mc_excess_risks_MWLE = []

for n in n_vals:
    alpha = np.sqrt(np.log(d) / n)
    mc_excess_risks = []
    mc_excess_risks_ridge = []
    mc_excess_risks_MWLE = []
    
    for _ in range(mc_iters):
        X_S = np.random.multivariate_normal(mean = mu_S_vec, cov = Sigma_S, size=n)
        X_T = np.random.multivariate_normal(mean = mu_S_vec, cov = Sigma_S, size=n)

        # weights = np.exp(np.dot(X_S, shift_vec))
        # weights = weights / np.sum(weights)

        # X_T_for_shift = np.random.multivariate_normal(mean = mu_S_vec, cov = Sigma_S, size=n)
        # X_T = resample_shift(X_T_for_shift, weights, n)
        # #X_T = np.random.multivariate_normal(mean = mu_T_vec, cov = Sigma_T, size=n)


        if model_mode == 'well-specified':
            y_S = np.dot(beta_star,X_S.T) + np.random.normal(0, 1, size=n)
            y_T = np.dot(beta_star,X_T.T) + np.random.normal(0, 1, size=n)

        elif model_mode == 'misspecified':
        
            f_of_X_S = np.column_stack([
                X_S[:, 0]**2,            # square of 1st covariate
                X_S[:, 1]**2,            # square of 2nd covariate
                X_S[:, 2:d]              # the rest are used linearly
            ])
            f_of_X_T = np.column_stack([
                X_T[:, 0]**2,            # square of 1st covariate
                X_T[:, 1]**2,            # square of 2nd covariate
                X_T[:, 2:d]              # the rest are used linearly
            ])
            y_S = np.dot(gamma, f_of_X_S.T) + np.random.normal(0, 1, size=n)
            y_T = np.dot(gamma, f_of_X_T.T) + np.random.normal(0, 1, size=n)

            #y_S = X_S**2+ np.random.normal(0, 1, size=n).reshape(-1, 1)
            # y_T = X_T**2 + np.random.normal(0, 1, size=n).reshape(-1, 1)

        #X_S, weights, X_T = generate_data(n, d, verbose=False)

        # pdf_T = multivariate_normal(mean=mu_T_vec, cov=Sigma_T).pdf(X_S)
        # pdf_S = multivariate_normal(mean=mu_S_vec, cov=Sigma_S).pdf(X_S)


        # weights = pdf_T / pdf_S

        # # clip_threshold = np.percentile(weights, 95)
        # # weights = np.minimum(weights, clip_threshold)

        # weights /= np.sum(weights)  # Normalize weights
        # log_pdf_T = multivariate_normal(mean=mu_T_vec, cov=Sigma_T).logpdf(X_S)
        # log_pdf_S = multivariate_normal(mean=mu_S_vec, cov=Sigma_S).logpdf(X_S)
        # log_weights = log_pdf_T - log_pdf_S
        # weights = np.exp(log_weights)
        # weights /= np.sum(weights)


        #if regularize == False:
        model = LinearRegression().fit(X_S, y_S)
        beta_hat = model.coef_

            # model_weighted = LinearRegression().fit(X_S, y_S, sample_weight=weights)
            # beta_hat_weighted = model_weighted.coef_
        
        if regularize:
            # MLE vs Ridge
            model_ridge = Ridge(alpha=alpha, fit_intercept=False, max_iter=10000)
            model_ridge.fit(X_S, y_S)
            beta_hat_ridge = model_ridge.coef_

            # # # MWLE via Lasso
            # model_weighted = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)
            # model_weighted.fit(X_S, y_S, sample_weight=weights)
            # beta_hat_weighted = model_weighted.coef_


        #X_T = resample_shift_one_covariate(X_S, covariate_index=0, alpha=1.0, n_sample=n)
        # if model_mode == 'misspecified':
        #     oracle_mse = np.mean((y_S - np.dot(beta_analytical_T, X_T.T))**2)

        # if model_mode == 'well-specified':
        #     oracle_mse = np.mean((y_S - np.dot(beta_star, X_T.T))**2)

        mse = np.mean((y_T - np.dot(beta_hat, X_T.T))**2)
        excess_risk_unweighted = mse #- oracle_mse    # since Var(epsilon) = 1

        mse_ridge = np.mean((y_T - np.dot(beta_hat_ridge, X_T.T))**2)
        excess_risk_ridge = mse_ridge #- oracle_mse    # since Var(epsilon) = 1

        # mse_weighted = np.mean((y_T - np.dot(beta_hat_weighted, X_T.T))**2)
        # excess_risk_weighted = mse_weighted - oracle_mse    # since Var(epsilon) = 1


        
        mc_excess_risks.append(excess_risk_unweighted)
        mc_excess_risks_ridge.append(excess_risk_ridge)

        # mc_excess_risks_MWLE.append(excess_risk_weighted)

    all_mc_excess_risks.append(mc_excess_risks)
    all_mc_excess_risks_ridge.append(mc_excess_risks_ridge)

    # all_mc_excess_risks_MWLE.append(mc_excess_risks_MWLE)

    # excess_risk_MLE.append(np.mean(mc_excess_risks))
    # # excess_risk_MWLE.append(np.mean(mc_excess_risks_MWLE))
    # upper_bound = trace * log_term / n
    # upper_bounds.append(upper_bound)

# # Theoretical threshold
# n_thresh = d * np.log(1/delta)
# # bound

# print(beta_hat_weighted)
excess_risk_MLE_mean = [np.mean(risks) for risks in all_mc_excess_risks]
excess_risk_MLE_std = [np.std(risks)/np.sqrt(mc_iters) for risks in all_mc_excess_risks]

excess_risk_ridge_mean = [np.mean(risks) for risks in all_mc_excess_risks_ridge]
excess_risk_ridge_std = [np.std(risks)/np.sqrt(mc_iters) for risks in all_mc_excess_risks_ridge]

# excess_risk_MWLE_mean = [np.mean(risks) for risks in all_mc_excess_risks_MWLE]
# excess_risk_MWLE_std = [np.std(risks)/np.sqrt(mc_iters) for risks in all_mc_excess_risks_MWLE]

plt.plot(n_vals, excess_risk_MLE_mean, label='Excess Risk (MLE)')
# plt.fill_between(n_vals,
#                  np.array(excess_risk_MLE_mean) - np.array(excess_risk_MLE_std),
#                  np.array(excess_risk_MLE_mean) + np.array(excess_risk_MLE_std),
#                  alpha=0.3, color='blue')

plt.plot(n_vals, excess_risk_ridge_mean, label='Excess Risk (Ridge)')

#plt.plot(n_vals, excess_risk_MWLE_mean, label='Excess Risk (MWLE)')
# plt.fill_between(n_vals,
#                 np.array(excess_risk_MWLE_mean) - np.array(excess_risk_MWLE_std),
#                 np.array(excess_risk_MWLE_mean) + np.array(excess_risk_MWLE_std),
#                 alpha=0.3, color='orange')
#plt.plot(n_vals, upper_bounds, linestyle = '--', color = 'red', label='Theoretical Upper Bound')
#plt.axvline(n_thresh, linestyle='--', color='gray', label='Sample Size Threshold')
plt.xlabel('Sample Size n')
plt.ylabel('Excess Risk')
plt.legend()
# ax.set_title(f'λ = {alpha}')
#plt.title('Simulation of Theorem 8.1')
plt.title('Excess Risk of MLE and Ridge with Covariate Shift, d = {}'.format(d))

#plt.tight_layout()
plt.show()