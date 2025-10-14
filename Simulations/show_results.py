import os, json
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import asdict
import html

def make_output_dir(dir):
    os.makedirs(dir, exist_ok=True)

# ---------- plotting ----------
def plot_samples(ax, X_S_2d, X_T_2d):
    ax.scatter(X_S_2d[:,0], X_S_2d[:,1], s=8, alpha=0.5, label="Source")
    ax.scatter(X_T_2d[:,0], X_T_2d[:,1], s=8, alpha=0.5, label="Target")
    ax.set_title("Source vs Target (first 2 dims)")
    ax.legend(frameon=False)

def plot_weightmap(ax, X_S_2d, W):  # <-- NEW
    sc = ax.scatter(X_S_2d[:,0], X_S_2d[:,1], c=np.log(W+1e-12), s=8, alpha=0.7)
    ax.set_title("Weight map (log w) on source samples")
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

def plot_weights(ax_scatter, ax_hist, X_S_2d, W):
    # map + histogram (two axes)
    plot_weightmap(ax_scatter, X_S_2d, W)
    ax_hist.hist(np.log(W+1e-12), bins=40, density=True, alpha=0.9)
    ax_hist.set_title("Histogram of log weights")

def plot_curves(outdir, res):
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
    plt.plot(nvals, cvals, '--', label='Population gap '+ r'$||β_S^{pop}-β_T^{pop}||$')
    plt.xlabel('n'); plt.ylabel(r'$||\hat\beta-\beta^{pop}_{T}||_2$')
    plt.title('Parameter distance vs n')
    plt.legend(); plt.tight_layout(); 
    f1 = os.path.join(outdir, "dist_vs_n.png")
    plt.savefig(f1, dpi = 140); plt.close()

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
    plt.xlabel('n'); plt.ylabel('Excess Target Risk scaled by p')
    plt.title('Prediction risk vs n')
    plt.legend(); plt.tight_layout(); 
    f2 = os.path.join(outdir, "risk_vs_n.png")
    plt.savefig(f2, dpi = 140); plt.close()

    return {"dist_plot": f1, "risk_plot": f2}


# ---------- main presentation function ----------
def present_results(cfg, results, plot_dists, outdir = "iw_ridge"):
    make_output_dir(outdir)
    
     # distribution plots
    r0 = plot_dists[0]["plot_dist"]
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    plot_samples(axs[0], r0["X_S"], r0["X_T"])
    plot_weightmap(axs[1], r0["X_S"], r0["W"])   # <-- FIX: map-only on second axis
    f_dists = os.path.join(outdir, "distributions_weights.png")
    plt.tight_layout(); plt.savefig(f_dists, dpi=140); plt.close(fig)

    # # separate weight map + histogram (two axes)
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    # plot_weights(ax1, ax2, r0["X_S"], r0["W"])
    # f_weights = os.path.join(outdir, "weights_map_hist.png")
    # plt.tight_layout(); plt.savefig(f_weights, dpi=140); plt.close(fig)

    result_files = plot_curves(outdir, results)

    # params, families, results summaries
    meta = asdict(cfg)
    meta["family_S"] = cfg.family_S
    meta["family_T"] = cfg.family_T
    meta["results"] = [
        {
            "n": r["n"], "p": r["p"], "k": r["k"], "alpha": r["alpha"],
            "pop_gap": r["pop_gap"],
            "dist_unw_mean": r["dist_unw"][0], "dist_unw_se": r["dist_unw"][1],
            "dist_iw_mean":  r["dist_iw"][0], "dist_iw_se":  r["dist_iw"][1],
            "risk_unw_mean": r["risk_unw"][0], "risk_unw_se": r["risk_unw"][1],
            "risk_iw_mean":  r["risk_iw"][0], "risk_iw_se":  r["risk_iw"][1],
            "risk_pop_mean": r["risk_pop"][0], "risk_pop_se": r["risk_pop"][1],
        }
        for r in results
    ]
    meta["figures"] = {
        "distributions_weights": f_dists,
        #"weights_map_hist": f_weights,
        **result_files
    }
    with open(os.path.join(outdir, "report.json"), "w") as f:
        json.dump(meta, f, indent=2)

    cfg_block = render_cfg_html(cfg)

    html_doc = f"""
<!doctype html><html><head><meta charset="utf-8"><title>IW-Ridge Results</title>
<style>
body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; }}
h1,h2 {{ margin: 0 0 8px 0; }}
.card {{ border: 1px solid #ddd; border-radius: 10px; padding: 16px; margin: 16px 0; }}
img {{ max-width: 100%; height: auto; border: 1px solid #eee; border-radius: 8px; }}
code, pre {{ background: #f6f8fa; padding: 6px 8px; border-radius: 6px; }}
.kv {{ display:grid; grid-template-columns: 220px 1fr; gap:6px 16px; }}
.kv dt {{ font-weight:600; color:#333; }}
.kv dd {{ margin:0; color:#111; }}
</style></head><body>
<h1>Importance-Weighted Ridge Results</h1>
<div class="card"><h2>Configuration</h2>
{cfg_block}
</div>
<div class="card"><h2>Figures</h2>
<h3>Parameter distance from target population coefficient</h3>
<img src="dist_vs_n.png" />
<h3>Excess Prediction risk scaled by p</h3>
<img src="risk_vs_n.png" />
<h3>Source vs Target (first 2 dims) & Weight Map</h3>
<p>This is a bit hard to interpret since it is just the first 2 coordinates. Sometimes helps to see if the weights look like they are doing the right thing.</p>
<img src="distributions_weights.png" />
</div>
<div class="card"><h2>Raw summary</h2>
<pre>{json.dumps(meta["results"], indent=2)}</pre></div>
</body></html>
"""
    with open(os.path.join(outdir, "index.html"), "w") as f:
        f.write(html_doc)

# ---------- pretty config renderer ----------
def _fmt_val(v):
    if isinstance(v, (list, tuple)):
        return ", ".join(map(str, v))
    if isinstance(v, dict):
        return "<code>"+html.escape(json.dumps(v, indent=0))+"</code>"
    if isinstance(v, float):
        return f"{v:.4g}"
    return html.escape(str(v))

def render_cfg_html(cfg) -> str:
    d = asdict(cfg)
    groups = [
        ("Simulation", [
            ("dimension_regime", d.get("dimension_regime")),
            ("gamma = p/n",            d.get("gamma")),
            #("p_fixed (for fixed regime)",          d.get("p_fixed")),
            ("sample sizes (n)",           d.get("n_list")),
            ("R (MC replications)", d.get("R")),
            #("seed",             d.get("seed")),
            ("Target test set size",           d.get("n_test")),
        ]),
        ("Model", [
            ("specified",   d.get("specified")),
            ("model_style", d.get("model_style")),
            ("Significant beta covariates for model",    d.get("sig_beta")),
            ("Significant covariates for shift",   d.get("sig_shift")),
            ("Noise variance",   d.get("sigma_eps", d.get("noise_eps"))),
        ]),
        ("Feature families", [
            ("Source distribution", d.get("family_S")),
            ("Source distribution parameters", d.get("params_S")),
            ("Target distribution", d.get("family_T")),
            ("Target distribution parameters", d.get("params_T")),
        ]),
        ("Weighting", [
            ("Weight using only significant covariates (only possible when IID)", d.get("use_subset_weights")),
            ("normalize_weights",        d.get("normalize_weights")),
            ("stabilize_weights",  d.get("stabilize_weights")),
            # ("tau",                d.get("tau")),
            # ("clip_q",             d.get("clip_q")),
        ]),
        ("Regularization", [
            ("Penalty parameter method", d.get("alpha_rule")),
            ("constant (in front of √[log(p)/n])",  d.get("c_sqrtlog")),
            #("alpha_fixed (only for fixed penalty)",d.get("alpha_fixed")),
        ]),
    ]
    parts = []
    for title, rows in groups:
        parts.append(f'<div class="card"><h3>{html.escape(title)}</h3><dl class="kv">')
        for k, v in rows:
            parts.append(f"<dt>{html.escape(str(k))}</dt><dd>{_fmt_val(v)}</dd>")
        parts.append("</dl></div>")
    return "\n".join(parts)
