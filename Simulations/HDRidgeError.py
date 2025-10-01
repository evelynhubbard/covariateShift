import numpy as np
from numpy.linalg import cholesky, solve
from dataclasses import dataclass
import matplotlib.pyplot as plt

from ridge_helpers import ar1_cov, pop_ridge, logpdf, pred_mse, mis_specficied_model


@dataclass
class Cfg:
    regime: str = "highdim"       # "highdim" or "fixedp"
    gamma: float = 1.5            # p = ceil(gamma*n) if highdim
    p_fixed: int = 50            # used if regime="fixedp"
    n_list: tuple = (1000, 1200)
    R: int = 1000                   # repetitions per n

    specified: str = "well"        # "well" or "mis"
    model_style: str = "sparse"       # "sparse" or "dense"
    sig_shift: int = 5             # shifted coordinates
    sig_beta: int = 3              # beta* supports

    rho_S: float = 0            # params for covariate distributions
    rho_T: float = 0
    mu_S: float = 2.0
    mu_T: float = -2.0
    sigma_eps: float = 1.0        # noise level

    # weighting controls
    use_subset_weights: bool = True
    tau: float = 0.6              # temper weights: w^tau, tau∈[0,1]
    clip_q: float = 0.95          # clip at quantile
    normalize_w: bool = True

    # alpha rules
    alpha_rule: str = "fixed"      # "fixed", "sqrtlog", or "neff"
    alpha_fixed: float = 1.0
    c_sqrtlog: float = 0.25       # α = c*sqrt(log p / n)
    # c_neff: float = 0.8           # α = c*sqrt(log p / n_eff)
    
    seed: int = 0
    n_test: int = 1000              # test set size 

def run(cfg: Cfg):
    rng = np.random.default_rng(cfg.seed)
    results = []

    if cfg.regime == "fixedp":
        p = cfg.p_fixed

        # choose alpha
        if cfg.alpha_rule == "fixed":
            alpha = cfg.alpha_fixed
        else: #cfg.alpha_rule == "sqrtlog":
            alpha = cfg.c_sqrtlog * np.sqrt(np.log(p) / 20000)
        
        k = min(cfg.sig_shift, p)
        if cfg.model_style == "sparse":
            # beta*
            s = min(cfg.sig_beta, p)
        else :  # dense
            s = p
        beta_star = np.zeros(p)
        beta_star[:s] = 1/np.sqrt(s)
        b_pop_T = pop_ridge(beta_star, cfg.rho_T, cfg.mu_T, k, alpha, p, cfg.specified, rng, cfg.sigma_eps)
        b_pop_S = pop_ridge(beta_star, cfg.rho_S, cfg.mu_S, k, alpha, p, cfg.specified, rng, cfg.sigma_eps)
        pop_gap = np.linalg.norm(b_pop_S - b_pop_T)

    for n in cfg.n_list:
        if cfg.regime == "fixedp":
            if cfg.alpha_rule == "fixed":
                    alpha = cfg.alpha_fixed
            else: #cfg.alpha_rule == "sqrtlog":
                alpha = cfg.c_sqrtlog * np.sqrt(np.log(cfg.p_fixed) / n)

        if cfg.regime == "highdim":
            p = int(np.ceil(cfg.gamma * n))
            if cfg.alpha_rule == "fixed":
                    alpha = cfg.alpha_fixed
            else: #cfg.alpha_rule == "sqrtlog":
                alpha = cfg.c_sqrtlog * np.sqrt(np.log(p) / n)

            #have to do this for every p:
            k = min(cfg.sig_shift, p)
            if cfg.model_style == "sparse":
                # beta*
                s = min(cfg.sig_beta, p)
            else :  # dense
                s = p
            beta_star = np.zeros(p)
            beta_star[:s] = 1/np.sqrt(s)
            
            # population targets for this (n,p)
            b_pop_T = pop_ridge(beta_star, cfg.rho_T, cfg.mu_T, k, alpha, p, cfg.specified, rng, cfg.sigma_eps)
            b_pop_S = pop_ridge(beta_star, cfg.rho_S, cfg.mu_S, k, alpha, p, cfg.specified, rng, cfg.sigma_eps)
           

            

            pop_gap = np.linalg.norm(b_pop_S - b_pop_T)
            if pop_gap < 1e-6:
                print("Warning: No shift detected.")

        # test set from target
        S_A_T = ar1_cov(k, cfg.rho_T)
        mu_vec_T = np.full(k, cfg.mu_T)
        X_A_T = rng.multivariate_normal(mu_vec_T, S_A_T, size=cfg.n_test)
        if p > k:
            X_R_T = rng.standard_normal(size=(cfg.n_test, p-k))
            X_Ttest = np.concatenate([X_A_T, X_R_T], axis=1)
        else:
            X_Ttest = X_A_T

        if cfg.specified == "well":
            y_Ttest = X_Ttest @ beta_star + rng.normal(0, cfg.sigma_eps, size=cfg.n_test)
        else: 
            y_Ttest = mis_specficied_model(X_Ttest, beta_star, cfg.sigma_eps, rng, cfg.n_test)

        d_unw, d_iw = [], []
        r_unw, r_iw, r_pop = [], [], []
        neffs = []

        oracle_risk = pred_mse(X_Ttest, y_Ttest, b_pop_T)

        for _ in range(cfg.R):
            
            # source draws
            S_A = ar1_cov(k, cfg.rho_S)
            mu_vec_S = np.full(k, cfg.mu_S)
            X_A_S = rng.multivariate_normal(mu_vec_S, S_A, size=n)
            if p > k:
                X_R_S = rng.standard_normal(size=(n, p-k))
                X_S = np.concatenate([X_A_S, X_R_S], axis=1)
            else:
                X_S = X_A_S

            if cfg.specified == "well":
                y_S = X_S @ beta_star + rng.normal(0, cfg.sigma_eps, size=n)
            else:
                y_S = mis_specficied_model(X_S, beta_star, cfg.sigma_eps, rng, n)
            
            # X_S_ = np.c_[np.ones(n), X_S]          # (n, p+1)
            # X_Ttest_ = np.c_[np.ones(cfg.n_test), X_Ttest]
            # p_plus = X_S_.shape[1]
            
            # weights on shifted block or full
            if cfg.use_subset_weights:
                logpT = logpdf(X_A_S, np.zeros(k)+cfg.mu_T, ar1_cov(k, cfg.rho_T))
                logpS = logpdf(X_A_S, np.zeros(k)+cfg.mu_S, ar1_cov(k, cfg.rho_S))
            else:
                # full densities (heavier variance)
                S_full = np.block([[ar1_cov(k, cfg.rho_S), np.zeros((k, p-k))],
                                   [np.zeros((p-k, k)), np.eye(p-k)]])
                T_full = np.block([[ar1_cov(k, cfg.rho_T), np.zeros((k, p-k))],
                                   [np.zeros((p-k, k)), np.eye(p-k)]])
                logpT = logpdf(X_S, np.zeros(p), T_full)
                logpS = logpdf(X_S, np.zeros(p), S_full)

         
            logw = logpT - logpS
            w = np.exp(logw)
            # if cfg.clip_q is not None:
            #     cap = np.quantile(w, cfg.clip_q)        # e.g., 0.90–0.95
            #     w = np.minimum(w, cap)
            w = w / (w.mean() + 1e-12)
            sum_w = w.sum()

            # # temper and clip
            # if cfg.tau < 1.0:
            #     w = w**cfg.tau
            # if cfg.clip_q is not None:
            #     cap = np.quantile(w, cfg.clip_q)
            #     w = np.minimum(w, cap)
            # if cfg.normalize_w:
            #     w /= (w.mean() + 1e-12)

            #neff = n_eff(w); neffs.append(neff)
            Pen = 2*alpha * np.eye(p)
            # Pen[0, 0] = 0.0

            # ridge via normal equations (aligned with population form)
            XtX = (X_S.T @ X_S) / n
            Xty = (X_S.T @ y_S) / n
            b_unw = solve(XtX + Pen, Xty)

            XtWX = (X_S.T * w) @ X_S / sum_w
            XtWy = (X_S.T * w) @ y_S / sum_w
            b_iw = solve(XtWX + Pen, XtWy)

            # b_unw = b_unw_full[1:]
            # b_iw  = b_iw_full[1:]

            # metrics
            d_unw.append(np.linalg.norm(np.abs(b_unw - b_pop_T)))
            d_iw.append(np.linalg.norm(np.abs(b_iw - b_pop_T)))
            r_unw.append((pred_mse(X_Ttest, y_Ttest, b_unw) - oracle_risk)/np.sqrt(p))
            r_iw.append((pred_mse(X_Ttest, y_Ttest, b_iw) - oracle_risk)/np.sqrt(p))
            r_pop.append((pred_mse(X_Ttest, y_Ttest, b_pop_T) - oracle_risk)/np.sqrt(p))

        results.append({
            "n": n, "p": p, "k": k,
            # "mean_neff": float(np.mean(neffs)),
            "pop_gap": float(pop_gap),
            "dist_unw": (float(np.mean(d_unw)), float(np.std(d_unw, ddof=1)/np.sqrt(cfg.R))),
            "dist_iw":  (float(np.mean(d_iw)), float(np.std(d_iw, ddof=1)/np.sqrt(cfg.R))),
            "risk_unw": (float(np.mean(r_unw)), float(np.std(r_unw, ddof=1)/np.sqrt(cfg.R))),
            "risk_iw":  (float(np.mean(r_iw)), float(np.std(r_iw, ddof=1)/np.sqrt(cfg.R))),
            "risk_pop": (float(np.mean(r_pop)), float(np.std(r_pop, ddof=1)/np.sqrt(cfg.R))),
        })
    return results
def plot_all(res):
    nvals = np.array([r["n"] for r in res])
    # distances
    mu_u = np.array([r["dist_unw"][0] for r in res])
    se_u = np.array([r["dist_unw"][1] for r in res])
    mu_w = np.array([r["dist_iw"][0] for r in res])
    se_w = np.array([r["dist_iw"][1] for r in res])
    cvals = np.array([r["pop_gap"] for r in res])

    plt.figure()
    plt.plot(nvals, mu_u, marker='o', label='Ridge (unweighted)')
    plt.fill_between(nvals, mu_u-se_u, mu_u+se_u, alpha=0.2)
    # plt.plot(nvals, mu_w, marker='o', label='IW-Ridge (tempered/clip)')
    plt.plot(nvals, mu_w, marker='o', label='IW-Ridge')
    plt.fill_between(nvals, mu_w-se_w, mu_w+se_w, alpha=0.2)
    # plt.plot(nvals, cvals, '--', label='Population gap')
    plt.plot(nvals, cvals, '--', label='Population gap '+ r'$||β_S^{pop}-β_T^{pop}||$')
    plt.xlabel('n'); plt.ylabel(r'$||\hat\beta-\beta^{pop}_{T}||_2$')
    plt.title('Parameter distance in high dimension')
    plt.legend(); plt.tight_layout(); plt.show()

    # risks on target
    ru = np.array([r["risk_unw"][0] for r in res])
    su = np.array([r["risk_unw"][1] for r in res])
    rw = np.array([r["risk_iw"][0] for r in res])
    sw = np.array([r["risk_iw"][1] for r in res])
    rp = np.array([r["risk_pop"][0] for r in res])

    plt.figure()
    plt.plot(nvals, ru, marker='o', label='Ridge risk (unweighted)')
    plt.fill_between(nvals, ru-su, ru+su, alpha=0.2)
    plt.plot(nvals, rw, marker='o', label='IW-Ridge risk')
    plt.fill_between(nvals, rw-sw, rw+sw, alpha=0.2)
    plt.plot(nvals, rp, '--', label='Risk at population ridge (target)')
    plt.xlabel('n'); plt.ylabel('Target MSE')
    plt.title('Prediction risk on target')
    plt.legend(); plt.tight_layout(); plt.show()

    # n_eff
    # ne = np.array([r["mean_neff"] for r in res])
    # plt.figure()
    # plt.plot(nvals, ne, marker='o')
    # plt.xlabel('n'); plt.ylabel('Mean n_eff')
    # plt.title('Effective sample size under weighting')
    # plt.tight_layout(); plt.show()

if __name__ == "__main__":
    cfg = Cfg(
        regime="highdim",       
        gamma=1.5,
        specified="mis",
        model_style="sparse",
        sig_shift=10,
        sig_beta=3,
        rho_S=0.0,
        rho_T=0.0,
        mu_S=-2.0,
        mu_T=1.0,
        sigma_eps=1.0,
        # key knobs:
        tau=1.0,                 # tempering lowers variance
        clip_q=None,             # clip heavy tails
        alpha_rule="sqrtlog",       # tune α to n_eff
        #c_neff=0.8,
        seed=12
    )
    res = run(cfg)
    for r in res:
        print(f"n={r['n']:4d}, p={r['p']:4d}, "#n_eff≈{r['mean_neff']:.1f}, "
              f"dist_unw={r['dist_unw'][0]:.3f}, dist_iw={r['dist_iw'][0]:.3f}, "
              f"risk_unw={r['risk_unw'][0]:.3f}, risk_iw={r['risk_iw'][0]:.3f}")
    plot_all(res)