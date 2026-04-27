"""
run_figure2.py
==============
Figure 2 — Effect of ion temperature anisotropy (alpha_s = T_perp/T_par)
on the ion-to-electron heating ratio Qi/Qe.

Three panels side by side, one per anisotropy value:
    alpha_s = 0.5  (T_perp < T_par  ->  firehose-prone)
    alpha_s = 1.0  (isotropic       ->  baseline, same as Figure 1)
    alpha_s = 2.0  (T_perp > T_par  ->  ion-cyclotron-prone)

"""

import os
import subprocess
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import cumulative_trapezoid
from datetime import datetime

#Timer
t_total_start = time.time()

def elapsed(start):
    s = time.time() - start
    m, sec = divmod(int(s), 60)
    return f"{m}m {sec}s" if m else f"{sec}s"

#Kolmogorov constants (Howes 2010)
C1 = 1.96
C2 = 1.09

#Parameter grids 150 points each
N_BETA   = 15
N_TRATIO = 10

beta_i_arr = np.logspace(-2, 2, N_BETA)                      # 0.01 → 100
tratio_arr = np.logspace(np.log10(0.2), 2, N_TRATIO)         # 0.2  → 100

# Anisotropy values to scan
# alpha_s = T_perp / T_par  for the ION species only.
# Electrons are kept isotropic (alphS=1) to isolate the ion effect.
ALPHA_VALUES = [0.5, 1.0, 2.0]
ALPHA_LABELS = [r'$\alpha_i = T_{{\perp}}/T_{{\parallel}} = 0.5$  (firehose-prone)',
                r'$\alpha_i = 1.0$  (isotropic, Fig. 1 baseline)',
                r'$\alpha_i = 2.0$  (ion-cyclotron-prone)']

#Wave-vector scan
KPERP_MIN = 1e-3
KPERP_MAX = 10.0
N_KPERP   = 150
PLUME_FID = "10000"

#Output directories
INPUT_DIR = "inputs/figure2"
DATA_DIR  = "data/figure2"
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR,  exist_ok=True)

#Column indices in PLUME output
COL_KPERP  = 0
COL_OMEGA  = 4
COL_GAMMA  = 5
COL_P_ION  = 6
COL_P_ELEC = 7


# Input-file generator
def make_input_file(beta_i, Ti_Te, alpha_ion, label, map_mult=10.0):
    """
    Write a PLUME namelist input file.

    Parameters
    ----------
    beta_i    : float   ion plasma beta (= 8pi n Ti / B^2)
    Ti_Te     : float   ion-to-electron temperature ratio
    alpha_ion : float   ion temperature anisotropy T_perp/T_par  ← NEW
    label     : str     unique run identifier
    map_mult  : float   sets the omega search box size
    """
    om_alfven = KPERP_MIN / max(np.sqrt(beta_i), 1e-3)
    om_box    = max(om_alfven * map_mult, 1e-4)
    gam_box   = max(om_box * 0.5, 5e-5)

    content = f"""!-=-=- Figure 2 Howes (2010) extension: beta_i={beta_i:.4e}, Ti/Te={Ti_Te:.4e}, alpha_i={alpha_ion:.2f}
&params
betap={beta_i:.6e}
kperp={KPERP_MIN:.3e}
kpar={KPERP_MIN:.3e}
vtp=1.E-4
nspec=2
nscan=1
option=1
nroot_max=4
use_map=.true.
writeOut=.false.
dataName='figure2'
outputName='{label}'
/
&species_1
tauS=1.0
muS=1.0
alphS=1.0
Qs=1.0
Ds=1.0
vvS=0.0
/
&species_2
tauS={Ti_Te:.6e}
muS=1836.0
alphS={alpha_ion:.6f}
Qs=-1.0
Ds=1.0
vvS=0.0
/
&maps
loggridw=.false.
loggridg=.false.
omi=0.0
omf={om_box:.6e}
gami=-{gam_box:.6e}
gamf={gam_box*0.01:.6e}
positive_roots=.true.
nr=128
ni=128
/
&scan_input_1
scan_type=0
scan_style=0
swi={KPERP_MIN:.3e}
swf={KPERP_MAX:.3e}
swlog=.true.
ns={N_KPERP}
nres=1
heating=.true.
eigen=.false.
tensor=.false.
/
"""
    path = os.path.join(INPUT_DIR, f"{label}.in")
    with open(path, "w") as f:
        f.write(content)
    return path


# Mode identification  (unchanged from Figure 1)
def identify_alfven_mode(modes_data):
    candidates = []
    for data in modes_data:
        omega_r = data[:, COL_OMEGA]
        gamma   = data[:, COL_GAMMA]
        P_ion   = data[:, COL_P_ION]
        P_elec  = data[:, COL_P_ELEC]

        if np.mean(np.abs(omega_r)) < 1e-10:        continue
        if np.abs(np.mean(omega_r)) > 100:           continue
        if np.mean(omega_r > 0) < 0.7:              continue
        if np.max(np.abs(P_ion))  < 1e-40 and \
           np.max(np.abs(P_elec)) < 1e-40:          continue

        n_low     = len(omega_r) // 3
        om_low    = np.where(np.abs(omega_r[:n_low]) < 1e-30, 1e-30,
                             np.abs(omega_r[:n_low]))
        gam_ratio = np.mean(np.abs(gamma[:n_low]) / om_low)
        candidates.append((gam_ratio, data))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def read_alfven_mode(label):
    modes = []
    for m in range(1, 5):
        path = os.path.join(DATA_DIR, f"{label}_kperp_1_{PLUME_FID}.mode{m}")
        if not os.path.exists(path):
            continue
        try:
            data = np.loadtxt(path)
        except Exception:
            continue
        if data.ndim < 2 or data.shape[0] < 10 or data.shape[1] <= COL_P_ELEC:
            continue
        modes.append(data)
    if not modes:
        return None
    best = identify_alfven_mode(modes)
    if best is None:
        return None
    idx = np.argsort(best[:, COL_KPERP])
    return (best[idx, COL_KPERP], np.abs(best[idx, COL_OMEGA]),
            best[idx, COL_P_ION], best[idx, COL_P_ELEC])


# Heating-ratio integrator  (unchanged from Figure 1)
def compute_heating_ratio(kperp, omega, P_ion, P_elec):
    prefactor = 2.0 * C1**1.5 * C2
    log_k     = np.log(kperp)
    P_tot     = np.abs(P_ion) + np.abs(P_elec)
    omega     = np.where(omega < 1e-30, 1e-30, omega)
    exponent  = cumulative_trapezoid(prefactor * P_tot, log_k, initial=0)
    E_kperp   = np.exp(-exponent)
    Q_i = np.trapezoid(prefactor * np.abs(P_ion)  * E_kperp, log_k)
    Q_e = np.trapezoid(prefactor * np.abs(P_elec) * E_kperp, log_k)
    return Q_i / Q_e if Q_e > 1e-30 else np.nan


print("=" * 65)
print("  Figure 2 — Ion Temperature Anisotropy  (T_perp/T_par scan)")
print("=" * 65)

#1. Generate all input files
t_step = time.time()
n_total = len(ALPHA_VALUES) * N_BETA * N_TRATIO
print(f"\n[1/4] Generating {n_total} input files...", flush=True)

input_files = {}   # key: (alpha_idx, i, j) -> (label, path)
for ai, alpha in enumerate(ALPHA_VALUES):
    for i, beta_i in enumerate(beta_i_arr):
        for j, Ti_Te in enumerate(tratio_arr):
            label = f"fig2_a{ai}_b{i:02d}_t{j:02d}"
            mult  = 50.0 if beta_i < 0.05 else (20.0 if beta_i < 0.2 else 10.0)
            path  = make_input_file(beta_i, Ti_Te, alpha, label, map_mult=mult)
            input_files[(ai, i, j)] = (label, path)

print(f"    Done in {elapsed(t_step)}.")

#2. Run PLUME
t_step = time.time()
print(f"\n[2/4] Running PLUME ({n_total} runs)...", flush=True)
failed = []
for idx, ((ai, i, j), (label, inpath)) in enumerate(input_files.items()):
    out_check = os.path.join(DATA_DIR, f"{label}_kperp_1_{PLUME_FID}.mode1")
    if os.path.exists(out_check):
        continue
    result = subprocess.run(["./plume.e", inpath], capture_output=True, text=True)
    if result.returncode != 0:
        failed.append((ai, i, j))
    if (idx + 1) % 50 == 0:
        frac  = (idx + 1) / n_total
        spent = time.time() - t_step
        eta   = (spent / frac) * (1 - frac) if frac > 0 else 0
        em, es = divmod(int(eta), 60)
        print(f"    {idx+1:4d}/{n_total}  |  elapsed {elapsed(t_step)}"
              f"  |  ETA {em}m {es}s", flush=True)
print(f"    Finished in {elapsed(t_step)}. Failures: {len(failed)}")

#3. Compute heating ratios for each alpha
t_step = time.time()
print(f"\n[3/4] Computing heating ratios...", flush=True)

# One grid per anisotropy value
ratio_grids = [np.full((N_BETA, N_TRATIO), np.nan) for _ in ALPHA_VALUES]

for (ai, i, j), (label, _) in input_files.items():
    result = read_alfven_mode(label)
    if result is None:
        continue
    kperp, omega, P_ion, P_elec = result
    if np.max(np.abs(P_ion)) < 1e-40:
        continue
    ratio = compute_heating_ratio(kperp, omega, P_ion, P_elec)
    ratio_grids[ai][i, j] = ratio

n_valid = sum(np.sum(~np.isnan(g)) for g in ratio_grids)
print(f"    Valid: {n_valid}/{n_total}  |  Done in {elapsed(t_step)}")

#Print physics summary table
print(f"\n{'='*65}")
print("  ANISOTROPY EFFECT ON HEATING — REPRESENTATIVE POINTS")
print(f"{'='*65}")
print(f"\n  {'alpha_i':<10} {'beta_i':<10} {'Ti/Te':<8} "
      f"{'Qi/Qe':<12} {'Physics regime'}")
print(f"  {'-'*60}")

# Sample at a few (beta_i, Ti/Te) points
sample_points = [(1.0, 1.0, "mid-beta, isotropic T"),
                 (0.1, 1.0, "low-beta"),
                 (10., 1.0, "high-beta"),
                 (1.0, 4.0, "fast-wind-like")]

for b_target, t_target, regime in sample_points:
    ib = np.argmin(np.abs(np.log10(beta_i_arr) - np.log10(b_target)))
    it = np.argmin(np.abs(np.log10(tratio_arr)  - np.log10(t_target)))
    for ai, alpha in enumerate(ALPHA_VALUES):
        r = ratio_grids[ai][ib, it]
        rstr = f"{r:.3f}" if not np.isnan(r) else "NaN"
        print(f"  α={alpha:<7.1f}  β_i={beta_i_arr[ib]:.2f}  "
              f"Ti/Te={tratio_arr[it]:.1f}  Qi/Qe={rstr:<12}  {regime}")
    print()

#Instability threshold annotation helper
def firehose_threshold(beta_i, alpha):
    """
    Approximate firehose threshold: alpha < 1 - 2/beta_par.
    For a rough check with beta_par ~ beta_i / alpha.
    Returns True if this (beta_i, alpha) is in the unstable region.
    """
    beta_par = beta_i / alpha if alpha > 0 else np.inf
    return alpha < 1.0 - 2.0 / beta_par if beta_par > 2 else False

def ioncyclotron_threshold(beta_i, alpha):
    """
    Approximate ion-cyclotron threshold: alpha > 1 + 2/beta_perp.
    beta_perp ~ beta_i * alpha.
    """
    beta_perp = beta_i * alpha
    return alpha > 1.0 + 2.0 / beta_perp if beta_perp > 0 else False

#4. Plot
t_step = time.time()
print(f"\n[4/4] Plotting...", flush=True)

fig = plt.figure(figsize=(25, 10))
gs  = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.06],
                        wspace=0.12, left=0.06, right=0.92,
                        top=0.88, bottom=0.12)

axes  = [fig.add_subplot(gs[0, k]) for k in range(3)]
cax   = fig.add_subplot(gs[0, 3])

levels_fill    = np.linspace(-2, 3, 101)
contour_levels = [-3, -2, -1, 0, 1, 2, 3]

# Shared colormap range so panels are directly comparable
vmin, vmax = -2, 3

# Instability-region overlay colours
firehose_color   = 'royalblue'
cyclotron_color  = 'tomato'

for ai, (ax, alpha, alabel) in enumerate(zip(axes, ALPHA_VALUES, ALPHA_LABELS)):
    log_ratio_plot   = np.log10(ratio_grids[ai].T)
    log_ratio_masked = np.ma.masked_invalid(log_ratio_plot)

    # Filled contours
    cf = ax.contourf(beta_i_arr, tratio_arr, log_ratio_masked,
                     levels=levels_fill, cmap='plasma',
                     extend='both', vmin=vmin, vmax=vmax)

    # Black contour lines
    cl = ax.contour(beta_i_arr, tratio_arr, log_ratio_masked,
                    levels=contour_levels,
                    colors='k', linewidths=0.9, alpha=0.6)
    fmt = {lv: str(lv) for lv in contour_levels}
    ax.clabel(cl, fmt=fmt, fontsize=7, inline=False)

    #Instability thresholds
    if alpha != 1.0:
        beta_line = np.logspace(-2, 2, 500)
        if alpha < 1.0:
            # Firehose: alpha = 1 - 2/beta_par, beta_par ~ beta_i/alpha
            # Rearranged: beta_i = 2*alpha / (1 - alpha), (vertical line approx)
            beta_fh = 2.0 * alpha / (1.0 - alpha) if alpha < 1.0 else np.inf
            if 0.01 < beta_fh < 100:
                ax.axvline(beta_fh, color=firehose_color, lw=1.5,
                           ls='--', label='Firehose threshold')
                ax.text(beta_fh * 1.08, 0.25, 'Firehose', color=firehose_color,
                        fontsize=7, va='bottom', rotation=90)
        else:
            # Ion-cyclotron: alpha = 1 + 2/beta_perp, beta_perp ~ beta_i*alpha
            beta_ic = 2.0 / (alpha - 1.0) / alpha if alpha > 1.0 else np.inf
            if 0.01 < beta_ic < 100:
                ax.axvline(beta_ic, color=cyclotron_color, lw=1.5,
                           ls='--', label='Ion-cyclotron threshold')
                ax.text(beta_ic * 1.08, 0.25, 'IC unstable', color=cyclotron_color,
                        fontsize=7, va='bottom', rotation=90)

    # Solar wind band
    ax.axvspan(0.3, 3.0, alpha=0.07, color='limegreen')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\beta_i$', fontsize=12)
    ax.set_xlim(0.01, 100)
    ax.set_ylim(0.2, 100)

    if ai == 0:
        ax.set_ylabel(r'$T_i / T_e$', fontsize=12)
    else:
        ax.set_yticklabels([])

    # Panel title alpha value prominently
    sign = ">" if alpha > 1 else ("<" if alpha < 1 else "=")
    ax.set_title(f'$\\alpha_i = T_{{\\perp}}/T_{{\\parallel}} = {alpha}$\n'
                 f'$T_\\perp {sign} T_\\parallel$', fontsize=11)

    # Panel letter
    ax.text(0.03, 0.97, f'({chr(97+ai)})', transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top')

# Shared colourbar
sm = plt.cm.ScalarMappable(cmap='plasma',
                            norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm.set_array([])
cbar = fig.colorbar(sm, cax=cax)
cbar.set_label(r'$\log_{10}(Q_i / Q_e)$', fontsize=11)
cbar.set_ticks(contour_levels)

# Figure-level annotations
fig.text(0.50, 0.97,
         r'Ion-to-Electron Heating Ratio $Q_i/Q_e$ vs Ion Temperature Anisotropy',
         ha='center', va='top', fontsize=13, fontweight='bold')

# Runtime stamp
t_end  = time.time()
tm, ts = divmod(int(t_end - t_total_start), 60)
fig.text(0.01, 0.01, f"Runtime: {tm}m {ts}s  |  Valid: {n_valid}/{n_total}",
         fontsize=7, color='gray')

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
outpath   = f"figure2_anisotropy_{timestamp}.png"
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"    Saved: {outpath}")

print(f"\n{'='*65}")
print(f"  Total runtime: {tm}m {ts}s  |  Valid: {n_valid}/{n_total}")
print(f"{'='*65}")

plt.show()
