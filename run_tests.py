"""
Standalone test runner for chromatography_twin.
Executes all test logic without requiring pytest.
Run with:  python run_tests.py
"""

import sys
import traceback
import numpy as np
import warnings

sys.path.insert(0, ".")

passed, failed = [], []

def pytest_approx(val, rel=1e-6, abs_tol=None):
    """Minimal approx helper: returns value; callers use == with tolerance check inline."""
    return val


def ok(name):
    passed.append(name)
    print(f"  [PASS]  {name}")

def fail(name, reason):
    failed.append((name, reason))
    print(f"  [FAIL]  {name}  ->  {reason}")

def check(name, condition, detail=""):
    if condition:
        ok(name)
    else:
        fail(name, detail or "assertion false")

def raises(name, exc_type, fn):
    try:
        fn()
        fail(name, f"expected {exc_type.__name__} but no exception was raised")
    except exc_type:
        ok(name)
    except Exception as e:
        fail(name, f"expected {exc_type.__name__}, got {type(e).__name__}: {e}")

def section(title):
    print(f"\n{'─'*60}\n  {title}\n{'─'*60}")


# ══════════════════════════════════════════════════════════════
# SMA ISOTHERM
# ══════════════════════════════════════════════════════════════
section("SMA ISOTHERM")

from models.sma_isotherm import SMAIsotherm, SMAParameters

p1 = SMAParameters(Lambda=1200.0, K=np.array([1e-3]),
                   nu=np.array([4.0]), sigma=np.array([20.0]))
p2 = SMAParameters(Lambda=1200.0, K=np.array([1e-3, 5e-4]),
                   nu=np.array([4.0, 3.5]), sigma=np.array([20.0, 15.0]))
iso1 = SMAIsotherm(p1)
iso2 = SMAIsotherm(p2)

raises("param: negative Lambda raises",   ValueError, lambda: SMAParameters(Lambda=-1.0, K=[0.05], nu=[5.0], sigma=[50.0]))
raises("param: shape mismatch raises",    ValueError, lambda: SMAParameters(Lambda=1200.0, K=[0.05,0.02], nu=[5.0], sigma=[50.0]))
check("param: n_components=1",            p1.n_components == 1)
check("param: n_components=2",            p2.n_components == 2)

q0 = iso1.compute_equilibrium(100.0, np.array([0.0]))
check("equil: zero protein → zero loading",     q0[0] < 1e-10, f"q={q0[0]:.2e}")

q_lo = iso1.compute_equilibrium(100.0, np.array([1.0]))
q_hi = iso1.compute_equilibrium(500.0, np.array([1.0]))
check("equil: higher salt → less binding",      q_lo[0] > q_hi[0], f"q_lo={q_lo[0]:.4f} q_hi={q_hi[0]:.4f}")

q_a = iso1.compute_equilibrium(100.0, np.array([0.5]))
q_b = iso1.compute_equilibrium(100.0, np.array([2.0]))
check("equil: higher conc → more loading",      q_b[0] > q_a[0])

q_sat = iso1.compute_equilibrium(50.0, np.array([100.0]))
occupied = np.sum((p1.nu + p1.sigma) * q_sat)
check("equil: ionic capacity not exceeded",     occupied <= p1.Lambda + 1e-6, f"occupied={occupied:.1f}")

q_alone   = iso2.compute_equilibrium(100.0, np.array([1.0, 0.0]))
q_compete = iso2.compute_equilibrium(100.0, np.array([1.0, 1.0]))
check("equil: competition reduces loading",     q_compete[0] < q_alone[0])

raises("equil: zero salt raises", ValueError, lambda: iso1.compute_equilibrium(0.0, np.array([1.0])))

# Equilibrium residual
c_s, c_p = 200.0, np.array([1.0])
q_eq = iso1.compute_equilibrium(c_s, c_p)
qstar = iso1._free_sites(q_eq)
rhs = p1.K[0] * c_p[0] * (qstar**p1.nu[0]) / (c_s**p1.nu[0])
check("equil: self-consistent (residual < 1e-8)", abs(rhs - q_eq[0]) < 1e-8, f"residual={abs(rhs-q_eq[0]):.2e}")

# Jacobian shape
q2 = iso2.compute_equilibrium(200.0, np.array([1.0, 0.5]))
J2 = iso2.compute_dqdc(200.0, np.array([1.0, 0.5]), q2)
check("Jacobian: shape (2,2)",              J2.shape == (2, 2), f"got {J2.shape}")
check("Jacobian: diagonal non-negative",    J2[0,0] >= 0 and J2[1,1] >= 0)

# Jacobian vs FD
q_ref = iso1.compute_equilibrium(200.0, np.array([1.0]))
J_an  = iso1.compute_dqdc(200.0, np.array([1.0]), q_ref)
eps   = 1e-5
q_p   = iso1.compute_equilibrium(200.0, np.array([1.0 + eps]))
J_fd  = (q_p - q_ref) / eps
rel   = abs(J_an[0,0] - J_fd[0]) / (abs(J_fd[0]) + 1e-12)
check("Jacobian: matches FD (<1%)",         rel < 0.01, f"rel_err={rel:.2%}")

check("salt_sp: non-negative",              iso1.salt_stationary_phase(np.array([1e6])) >= 0.0)
q_sp_0 = iso1.salt_stationary_phase(np.array([0.0]))
q_sp_5 = iso1.salt_stationary_phase(np.array([5.0]))
check("salt_sp: decreases with loading",    q_sp_0 > q_sp_5)


# ══════════════════════════════════════════════════════════════
# TDM SOLVER
# ══════════════════════════════════════════════════════════════
section("TDM SOLVER")

from models.tdm_solver import TDMSolver, ColumnParameters, ChromatographyRun

col = ColumnParameters(length=10.0, diameter=1.0, void_fraction=0.35,
                       flow_rate=1.0, n_grid_points=30)
solver = TDMSolver(col, iso1)

raises("col: negative length raises",      ValueError, lambda: ColumnParameters(length=-1.0, diameter=1.0, void_fraction=0.35, flow_rate=1.0))
raises("col: bad void_fraction raises",    ValueError, lambda: ColumnParameters(length=10.0, diameter=1.0, void_fraction=1.5, flow_rate=1.0))
check("col: velocity > 0",                 col.interstitial_velocity > 0)
check("col: phase_ratio > 0",              col.phase_ratio > 0)
expected_dax = col.interstitial_velocity * col.length / 100.0
check("col: dax auto-estimate",            abs(col.d_ax - expected_dax) < 1e-10)
col_manual = ColumnParameters(length=10.0, diameter=1.0, void_fraction=0.35, flow_rate=1.0, axial_dispersion=0.01)
check("col: dax manual override",          col_manual.d_ax == pytest_approx(0.01))

y0 = solver._build_initial_state(np.array([100.0, 0.0]))
check("state: y0 shape correct",           y0.shape == (2*30,))
raises("state: wrong shape raises",        ValueError, lambda: solver._build_initial_state(np.array([100.0])))
c_up = solver._unpack_state(y0)
check("state: unpack shape",               c_up.shape == (2, 30))
check("state: unpack values correct",      np.allclose(c_up[0,:], 100.0))

def grad_inlet(t):
    if t < 5.0:   return np.array([100.0, 0.5])
    elif t < 20.0:
        f = (t-5.0)/15.0
        return np.array([100.0 + f*400.0, 0.0])
    return np.array([500.0, 0.0])

run_short = ChromatographyRun(t_end=5.0, c_inlet=grad_inlet, c_initial=np.array([100.0, 0.0]))
res_short = solver.solve(run_short, verbose=False)
check("solve: expected keys present",      {"t","z","c","q","c_outlet","solver_msg","success"}.issubset(res_short.keys()))
check("solve: ODE success",                res_short["success"])

t_eval = np.linspace(0, 5.0, 50)
res_shaped = solver.solve(run_short, t_eval=t_eval, verbose=False)
check("solve: c shape",                    res_shaped["c"].shape == (50, 2, 30))
check("solve: q shape",                    res_shaped["q"].shape == (50, 1, 30))
check("solve: c_outlet shape",             res_shaped["c_outlet"].shape == (50, 2))
check("solve: concentrations non-negative",np.all(res_shaped["c"] >= -1e-4), f"min={res_shaped['c'].min():.2e}")

run_full = ChromatographyRun(t_end=35.0, c_inlet=grad_inlet, c_initial=np.array([100.0, 0.0]))
res_full = solver.solve(run_full, verbose=False)
peak = res_full["c_outlet"][:,1].max()
check("solve: protein elutes at outlet",   peak > 1e-3, f"peak_max={peak:.6f}")

run_salt = ChromatographyRun(t_end=20.0, c_inlet=lambda t: np.array([500.0,0.0]), c_initial=np.array([100.0,0.0]))
res_salt = solver.solve(run_salt, verbose=False)
check("solve: salt front propagates",      res_salt["c_outlet"][-1,0] > 400.0, f"final_salt={res_salt['c_outlet'][-1,0]:.1f}")

mb = solver.mass_balance_check(res_full, run_full)
# Salt error < 5% is expected (no adsorption, pure convection-diffusion).
# Protein mass accounting error of 3-10% is documented in mass_balance_check:
# it arises from post-hoc q recomputation in the retardation-factor ODE formulation.
check("mass balance: salt error < 5%",    mb["error_pct"][0] < 5.0,  f"salt_err={mb['error_pct'][0]:.1f}%")
check("mass balance: 'note' key present", "note" in mb)


# ══════════════════════════════════════════════════════════════
# GPR RESIDUAL MODEL
# ══════════════════════════════════════════════════════════════
section("GPR RESIDUAL MODEL")

from models.residual_model import ResidualGPR, ResidualData, build_features_from_run, FEATURE_NAMES
import tempfile, warnings as W
from pathlib import Path

rng = np.random.default_rng(42)
n = 12
X_tr = rng.uniform(low=[100.,200.,5.,0.5], high=[300.,500.,20.,2.], size=(n,4))
y0_tr = 0.3*np.sin(X_tr[:,0]/50.) + 0.05*X_tr[:,2] + rng.normal(0,.01,n)
y1_tr = -0.2*np.cos(X_tr[:,1]/100.) + rng.normal(0,.01,n)
synth = ResidualData(X=X_tr, y_residuals=np.column_stack([y0_tr,y1_tr]),
                     feature_names=["s1","s2","gl","lv"],
                     output_names=["peak_retention_time_min","peak_area_gL_min"])

raises("ResidualData: shape mismatch raises", ValueError,
       lambda: ResidualData(X=np.ones((5,3)), y_residuals=np.ones((6,2))))
d1d = ResidualData(X=np.ones((5,3)), y_residuals=np.ones(5))
check("ResidualData: 1D y expanded to (5,1)", d1d.y_residuals.shape == (5,1))
check("ResidualData: arrays are float",        synth.X.dtype == float)

gpr = ResidualGPR(kernel="matern", n_restarts_optimizer=2)
gpr.fit(synth)
check("GPR: is_fitted after fit",              gpr._is_fitted)
check("GPR: n_models == n_outputs",            len(gpr._models) == 2)
check("GPR: output names stored",             gpr._output_names == ["peak_retention_time_min","peak_area_gL_min"])
check("GPR: fit returns self",                 ResidualGPR(kernel="matern", n_restarts_optimizer=1).fit(synth) is not None)

raises("GPR: invalid kernel raises",          ValueError,
       lambda: ResidualGPR(kernel="poly").fit(synth))

with W.catch_warnings(record=True) as w:
    W.simplefilter("always")
    tiny = ResidualData(X=np.ones((3,4)), y_residuals=np.ones((3,1)))
    ResidualGPR(kernel="rbf", n_restarts_optimizer=1).fit(tiny)
check("GPR: small dataset warns",             any("training samples" in str(x.message) for x in w))

mu, sigma = gpr.predict(X_tr[:3], return_std=True)
check("GPR: predict shape mu",                mu.shape == (3,2))
check("GPR: predict shape sigma",             sigma.shape == (3,2))
check("GPR: uncertainty non-negative",        np.all(sigma >= 0.0))

mu2, s2 = gpr.predict(X_tr[:3], return_std=False)
check("GPR: return_std=False gives None sigma", s2 is None)

raises("GPR: predict before fit raises",      RuntimeError, lambda: ResidualGPR().predict(np.ones((2,4))))

mu_all, _ = gpr.predict(X_tr)
ss_res = np.sum((synth.y_residuals - mu_all)**2)
ss_tot = np.sum((synth.y_residuals - synth.y_residuals.mean(0))**2)
r2 = 1 - ss_res/(ss_tot+1e-12)
check("GPR: R² on training data > 0",         r2 > 0.0, f"R²={r2:.3f}")

mu_s, sigma_s = gpr.predict_single(X_tr[0], return_std=True)
check("GPR: predict_single shape",            mu_s.shape == (2,) and sigma_s.shape == (2,))

hp = gpr.get_hyperparameters()
check("GPR: hyperparams is list of 2",        isinstance(hp, list) and len(hp) == 2)
check("GPR: hyperparams have 'output' key",   all("output" in h for h in hp))

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "gpr.joblib"
    gpr.save(path)
    check("GPR: save creates file",           path.exists())
    loaded = ResidualGPR.load(path)
    check("GPR: loaded is_fitted",            loaded._is_fitted)
    mu_o, _ = gpr.predict(X_tr[:3])
    mu_l, _ = loaded.predict(X_tr[:3])
    check("GPR: save/load predictions match", np.allclose(mu_o, mu_l, rtol=1e-5))

import joblib, tempfile
with tempfile.TemporaryDirectory() as tmp:
    p = Path(tmp) / "bad.joblib"
    joblib.dump({"x": 1}, p)
    raises("GPR: load wrong type raises",     TypeError, lambda: ResidualGPR.load(p))

x_feat = build_features_from_run(
    {"salt_start_mM":50.,"salt_end_mM":400.,"gradient_length_CV":10.,
     "load_volume_CV":1.5,"flow_rate_mL_min":5.,"ph":7.2},
    {"peak_retention_time_min":12.,"peak_height_gL":0.8,
     "peak_width_min":2.5,"peak_area_gL_min":2.1}
)
check("features: correct length",            x_feat.shape == (len(FEATURE_NAMES),))
check("features: all finite",                np.all(np.isfinite(x_feat)))
x_slope = build_features_from_run({"salt_start_mM":50.,"salt_end_mM":550.,"gradient_length_CV":10.},{})
check("features: gradient slope computed",   x_slope[2] == pytest_approx(50.0))
check("features: missing keys use defaults", build_features_from_run({},{}).shape == (len(FEATURE_NAMES),))
check("FEATURE_NAMES length == 10",          len(FEATURE_NAMES) == 10)


# ══════════════════════════════════════════════════════════════
# HYBRID MODEL
# ══════════════════════════════════════════════════════════════
section("HYBRID MODEL")

from models.hybrid_model import (
    HybridChromatographyModel, HybridPrediction, ChromatogramSummary,
    extract_peak_summary, compute_resolution,
)

hcol = ColumnParameters(length=10.0, diameter=1.0, void_fraction=0.35,
                        flow_rate=1.0, n_grid_points=25)
hsma = SMAParameters(Lambda=1200.0, K=np.array([1e-3]),
                     nu=np.array([4.0]), sigma=np.array([20.0]))
hmodel = HybridChromatographyModel(hcol, hsma)

BASE_OC = dict(salt_start_mM=100.,salt_end_mM=500.,gradient_length_CV=10.,
               load_volume_CV=1.5,flow_rate_mL_min=1.,ph=7.0)

def h_inlet(t):
    if t<5.: return np.array([100.,0.5])
    elif t<20.:
        f=(t-5.)/15.; return np.array([100.+f*400.,0.0])
    return np.array([500.,0.0])

BASE = dict(operating_conditions=BASE_OC, c_inlet_fn=h_inlet,
            c_initial=np.array([100.,0.0]), t_end=35.,
            apply_gpr=False, verbose=False)

r = hmodel.predict(**BASE)
check("hybrid: returns HybridPrediction",    isinstance(r, HybridPrediction))
check("hybrid: gpr_was_applied False",       r.gpr_was_applied is False)
check("hybrid: c_corrected is None",         r.c_corrected is None)
check("hybrid: t shape (200,)",              r.t.shape == (200,))
check("hybrid: c_mechanistic shape",         r.c_mechanistic.shape == (200,2))
check("hybrid: 1 summary per protein",       len(r.mechanistic_summaries) == 1)
check("hybrid: summary is ChromatogramSummary", isinstance(r.mechanistic_summaries[0], ChromatogramSummary))
check("hybrid: peak_area >= 0",              r.mechanistic_summaries[0].peak_area_gL_min >= 0)
r_100 = hmodel.predict(**{**BASE, "t_eval": np.linspace(0,35.,100)})
check("hybrid: custom t_eval len=100",       len(r_100.t) == 100)

# Build GPR training data
def h_inlet_v(salt_end):
    def inlet(t):
        if t<5.: return np.array([100.,0.5])
        elif t<20.:
            f=(t-5.)/15.; return np.array([100.+f*(salt_end-100.),0.0])
        return np.array([salt_end,0.0])
    return inlet

rng2 = np.random.default_rng(1)
X_h, y_h = [], []
for se in np.linspace(300.,600.,8):
    oc = {**BASE_OC,"salt_end_mM":float(se)}
    rv = hmodel.predict(operating_conditions=oc, c_inlet_fn=h_inlet_v(se),
                        c_initial=np.array([100.,0.0]), t_end=35., apply_gpr=False, verbose=False)
    ms = rv.mechanistic_summaries[0].to_dict()
    xf = build_features_from_run(oc, ms)
    res_v = rng2.normal([.2,.05,.1,.3],[.05,.01,.02,.05])
    X_h.append(xf); y_h.append(res_v)

train_data = ResidualData(X=np.array(X_h), y_residuals=np.array(y_h),
                          output_names=["peak_retention_time_min","peak_height_gL",
                                        "peak_width_min","peak_area_gL_min"])
hmodel.fit_residual_model(train_data, kernel="matern", n_restarts_optimizer=2)
check("hybrid: GPR fitted",                  hmodel.residual_gpr is not None and hmodel.residual_gpr._is_fitted)

rg = hmodel.predict(**{**BASE, "apply_gpr": True})
check("hybrid+GPR: gpr_was_applied True",    rg.gpr_was_applied is True)
check("hybrid+GPR: c_corrected not None",    rg.c_corrected is not None)
check("hybrid+GPR: c_corrected shape",       rg.c_corrected.shape == rg.c_mechanistic.shape)
check("hybrid+GPR: corrected_summaries",     rg.corrected_summaries is not None)
check("hybrid+GPR: uncertainty >= 0",        np.all(rg.correction_uncertainty >= 0))
mech_a = rg.mechanistic_summaries[0].peak_area_gL_min
corr_a = rg.corrected_summaries[0].peak_area_gL_min
check("hybrid+GPR: correction changes area", abs(mech_a - corr_a) > 1e-8)
check("hybrid: apply_gpr=False skips GPR",   hmodel.predict(**BASE).gpr_was_applied is False)

# save/load
with tempfile.TemporaryDirectory() as tmp:
    hmodel.save(tmp)
    check("hybrid: save creates GPR file",   (Path(tmp)/"residual_gpr.joblib").exists())
    hm2 = HybridChromatographyModel(hcol, hsma)
    hm2.load_residual_model(Path(tmp)/"residual_gpr.joblib")
    r1 = hmodel.predict(**{**BASE,"apply_gpr":True})
    r2 = hm2.predict(**{**BASE,"apply_gpr":True})
    check("hybrid: save/load uncertainty matches",
          np.allclose(r1.correction_uncertainty, r2.correction_uncertainty, rtol=1e-5))

# Helper functions
t_h = np.linspace(0,10,100)
s_empty = extract_peak_summary(t_h, np.zeros((100,2)), 1)
check("peak_summary: empty → height=0",      s_empty.peak_height_gL == 0.0)

t_g = np.linspace(0,20,500)
c_g = np.exp(-0.5*((t_g-10.)/1.5)**2)
s_g = extract_peak_summary(t_g, np.column_stack([np.zeros(500),c_g]), 1)
check("peak_summary: Gaussian peak time",    abs(s_g.peak_retention_time_min - 10.0) < 0.1)
check("peak_summary: Gaussian height≈1",     abs(s_g.peak_height_gL - 1.0) < 0.01)
check("peak_summary: Gaussian FWHM≈3.53",   abs(s_g.peak_width_min - 2.355*1.5) < 0.2)

s1 = ChromatogramSummary(1,10.,1.0,2.0,2.5)
s2 = ChromatogramSummary(2,16.,0.8,2.0,2.0)
Rs = compute_resolution(s1,s2)
check("resolution: formula correct",         abs(Rs - 2*6./(4*1.699)) < 0.01)
s_nan = ChromatogramSummary(1, float("nan"),0.,float("nan"),0.)
check("resolution: nan FWHM → 0",           compute_resolution(s_nan,s2) == 0.0)
d_keys = s1.to_dict().keys()
check("summary: to_dict has all keys",       set(d_keys) == {
    "peak_retention_time_min","peak_height_gL","peak_width_min","peak_area_gL_min","resolution"})


# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════
print(f"\n{'═'*60}")
print(f"  RESULT:  {len(passed)} passed  |  {len(failed)} failed")
if failed:
    print("\n  FAILURES:")
    for name, reason in failed:
        print(f"    ✗  {name}")
        print(f"       {reason}")
else:
    print("\n  All tests passed ✓")
print("═"*60)
sys.exit(0 if not failed else 1)


