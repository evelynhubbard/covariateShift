import numpy as np
from numpy.linalg import solve
import math
from dataclasses import dataclass


from ridge_helpers import pop_ridge_target, pop_ridge_source, pred_mse, mis_specficied_model
from distributions import make_family
from show_results import present_results

#---------- config ---------------------------------
@dataclass
class Cfg:
    # Simulation regime
    dimension_regime: str = "highdim"       # "highdim" or "fixedp"
    gamma: float = 1.5                      # p = ceil(gamma*n) if highdim
    p_fixed: int = 50                       # used if regime="fixedp"
    n_list: tuple = (100,200,300,400,500,600,700,800,900,1000)  # training sizes
    R: int = 1000                   # repetitions per n
    seed: int = 0
    n_test: int = 1000              # test set size

    # Model params
    specified: str = "mis"        # "well" or "mis"
    model_style: str = "sparse"       # "sparse" or "dense"
    sig_beta: int = 5              # beta* supports
    noise_eps: float = 1.0          # noise level

    # Params for covariate distributions
    sig_shift: int = 10             # shifted coordinates

    family_S: str = "gaussian_AR1"   # source distribution family
    family_T: str = "gaussian_AR1"   # target distribution family
    params_S: dict = None            # source distribution params
    params_T: dict = None            # target distribution params
    
    
    # weighting controls
    use_subset_weights: bool = True # weights computed on only significant and shifted covariates
    normalize_weights: bool = True
    # for clipping weights
    stabilize_weights: bool = False
    tau: float = 0.6              # temper weights: w^tau, tau∈[0,1]
    clip_q: float = 0.95          # clip at quantile

    # Regularization params
    alpha_rule: str = "sqrtlog"      # "fixed", "sqrtlog", or "neff"
    alpha_fixed: float = 1.0
    c_sqrtlog: float = 0.5         # α = c*sqrt(log p / n)
    
    # report output
    out_dir: str = "iw_ridge"

# ---------- utils ----------
def get_alpha(n,p, cfg: Cfg):
    if cfg.alpha_rule == "fixed":
        return cfg.alpha_fixed
    elif cfg.alpha_rule == "sqrtlog":
        return cfg.c_sqrtlog * np.sqrt(np.log(p) / n)
    else:
        raise ValueError("alpha_rule must be 'fixed' or 'sqrtlog'")
    
def treat_weights(w, cfg: Cfg):
    if cfg.stabilize_weights:
        # temper and clip
        if cfg.tau < 1.0:
            w = w**cfg.tau
        if cfg.clip_q is not None:
            cap = np.quantile(w, cfg.clip_q)
            w = np.minimum(w, cap)
    if cfg.normalize_weights:
        w /= (w.mean() + 1e-12)
    return w



# ---------- run ----------

def run(cfg: Cfg):
    # set up
    rng = np.random.default_rng(cfg.seed)
    results = []
    plot_dists = []

    fam_S, fam_T = cfg.family_S.lower(), cfg.family_T.lower()
    params_S = {} if cfg.params_S is None else dict(cfg.params_S)
    params_T = {} if cfg.params_T is None else dict(cfg.params_T)

    # check some args
    for params in (params_S, params_T):
        if "rho" in params and abs(params["rho"]) >= 1:
            raise ValueError("For gaussian_ar1, |rho| must be < 1 for SPD covariance.")


    for n in cfg.n_list:
        p = cfg.p_fixed if cfg.dimension_regime == "fixedp" else int(np.ceil(cfg.gamma * n))
        k = min(cfg.sig_shift, p)  # shifted coordinates
     

        alpha = get_alpha(n, p, cfg)

        # beta*
        if cfg.model_style == "sparse":
            s = min(cfg.sig_beta, p) # beta* supports
        else :  # dense
            s = p

        beta_star = np.zeros(p)
        beta_star[:s] = 1/math.sqrt(s)

        # population targets for this (n,p)
        b_pop_T = pop_ridge_target(beta_star, alpha, p, cfg.specified, rng, cfg.noise_eps, k, fam_T, params_T, prefer_closed_form= True, verbose=True)
        b_pop_S = pop_ridge_source(beta_star, alpha, p, cfg.specified, rng, cfg.noise_eps, k, fam_T, fam_S, params_T, params_S, prefer_closed_form= True, verbose=True)

        pop_gap = float(np.linalg.norm(b_pop_S - b_pop_T))

        if pop_gap < 1e-6:
            print("Warning: No shift detected.")

        # target test set (for risk) — use chosen target family
        sample_func_T, lpdf_T = make_family(fam_T, **params_T)
        X_Ttest = sample_func_T(cfg.n_test, p, rng)

        if cfg.specified == "well":
            y_Ttest = X_Ttest @ beta_star + rng.normal(0, cfg.noise_eps, size=cfg.n_test)
        else: 
            y_Ttest = mis_specficied_model(X_Ttest, beta_star, cfg.noise_eps, rng, cfg.n_test)

        oracle_risk = pred_mse(X_Ttest, y_Ttest, b_pop_T)

        d_unw, d_iw = [], []
        r_unw, r_iw, r_pop = [], [], []

        sample_func_S, lpdf_S = make_family(fam_S, **params_S)

        # for plotting distributions:
        X_S_for_plot = None
        X_T_for_plot = None
        W_for_plot   = None

        for r in range(cfg.R):
            
            # source draws
            X_S_A = sample_func_S(n, k, rng)      # (n, k)
            X_S_B = sample_func_T(n, p-k, rng)          # (n, p-k)
            X_S = np.concatenate([X_S_A, X_S_B], axis=1)  # (n, p)

            if cfg.specified == "well":
                y_S = X_S @ beta_star + rng.normal(0, cfg.noise_eps, size=n)
            else:
                y_S = mis_specficied_model(X_S, beta_star, cfg.noise_eps, rng, n)


            # weights on shifted block or full
            if cfg.use_subset_weights and k < p:
                X_shiftfeatures = X_S[:, :k]  # (n, k)
            else:
                X_shiftfeatures = X_S 
                      # (n, p)

            logw = lpdf_T(X_shiftfeatures) - lpdf_S(X_shiftfeatures)
            #logw = np.clip(logw, -700, 700)
            w = np.exp(logw)
            w = treat_weights(w, cfg)
            sum_w = w.sum()


            # ridge fits
            penalty = 2*alpha * np.eye(p)
            # penalty[0, 0] = 0.0

            # ridge via normal equations (aligned with population form)
            XtX = (X_S.T @ X_S) / n
            Xty = (X_S.T @ y_S) / n
            b_unw = solve(XtX + penalty, Xty)

            XtWX = (X_S.T * w) @ X_S / sum_w
            XtWy = (X_S.T * w) @ y_S / sum_w
            b_iw = solve(XtWX + penalty, XtWy)

            # b_unw = b_unw_full[1:]
            # b_iw  = b_iw_full[1:]

            # metrics
            d_unw.append(np.linalg.norm(b_unw - b_pop_T))
            d_iw.append(np.linalg.norm(b_iw - b_pop_T))
            r_unw.append((pred_mse(X_Ttest, y_Ttest, b_unw) - oracle_risk)/p)
            r_iw.append((pred_mse(X_Ttest, y_Ttest, b_iw) - oracle_risk)/p)
            r_pop.append((pred_mse(X_Ttest, y_Ttest, b_pop_T) - oracle_risk)/p)

            #save one draw for plotting distributions
            if r == 0 and n == cfg.n_list[-1]:
                X_S_for_plot = X_S.copy()
                X_T_for_plot = X_Ttest[:n].copy()
                W_for_plot   = w.copy()
                plot_dists.append({
                    "plot_dist":{
                        "X_S": X_S_for_plot[:,:2],
                        "X_T": X_T_for_plot[:,:2],
                        "W": W_for_plot
                    }
                })

        results.append({
            "n": n, "p": p, "k": k,
            "alpha": alpha,
            "pop_gap": pop_gap,
            "dist_unw": (float(np.mean(d_unw)), float(np.std(d_unw, ddof=1)/np.sqrt(cfg.R))),
            "dist_iw":  (float(np.mean(d_iw)), float(np.std(d_iw, ddof=1)/np.sqrt(cfg.R))),
            "risk_unw": (float(np.mean(r_unw)), float(np.std(r_unw, ddof=1)/np.sqrt(cfg.R))),
            "risk_iw":  (float(np.mean(r_iw)), float(np.std(r_iw, ddof=1)/np.sqrt(cfg.R))),
            "risk_pop": (float(np.mean(r_pop)), float(np.std(r_pop, ddof=1)/np.sqrt(cfg.R)))
        })

    return results, plot_dists

# ---------- Main ----------

if __name__ == "__main__":
    cfg = Cfg(
        # simulation controls
        dimension_regime="highdim",
        p_fixed =20,
        gamma = 1.5,
        n_list = (50, 100, 150, 200, 300, 500, 1000, 1500, 2000),
        R = 500,
        seed = 11,

        # model controls
        specified="mis",
        model_style="sparse",
        sig_shift=10,
        sig_beta=5,
        noise_eps=1.0,
        alpha_rule = "sqrtlog",
        c_sqrtlog = 1,

        # distribution controls
        family_S = "gaussian_indep", params_S={"mu": -1.0, "rho": 0},
        family_T = "gaussian_indep", params_T={"mu":  1.5, "rho": 0},
        # family_T = "laplace_indep", params_S={"mu": 2.0, "b": 4},
        # family_S = "gaussian_indep", params_T={"mu":  1.0, "rho": 0},

        # weighting controls
        use_subset_weights=True,
        normalize_weights=True,
        stabilize_weights=False,

        # output
        out_dir = "iw_ridge"
    )
    res, plot_dists = run(cfg)
    for r in res:
        print(f"n={r['n']:4d}, p={r['p']:4d},"
              f"dist_unw={r['dist_unw'][0]:.3f}, dist_iw={r['dist_iw'][0]:.3f}, "
              f"risk_unw={r['risk_unw'][0]:.3f}, risk_iw={r['risk_iw'][0]:.3f}")
        
    present_results(cfg, res, plot_dists, outdir=cfg.out_dir)
    print(f"Results and plots saved at: {cfg.out_dir}/index.html")