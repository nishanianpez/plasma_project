"""
run_figure1.py
==============
Reproduces Figure 1 from Howes (2010) — log(Qi/Qe) contour plot.

"""

import os
import subprocess
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import RegularGridInterpolator
from datetime import datetime

#Timer to make sure computing time seems realisitc
t_total_start = time.time()

def elapsed(start):
    s = time.time() - start
    m, sec = divmod(int(s), 60)
    return f"{m}m {sec}s" if m else f"{sec}s"

#Kologorov constants derived by Howes
C1 = 1.96
C2 = 1.09

# Grid 300 points
N_BETA   = 20
N_TRATIO = 15

beta_i_arr = np.logspace(-2, 2, N_BETA)
tratio_arr = np.logspace(np.log10(0.2), 2, N_TRATIO)

KPERP_MIN = 1e-3
KPERP_MAX = 10.0
N_KPERP   = 300
PLUME_FID = "10000"

INPUT_DIR = "inputs/figure1"
DATA_DIR  = "data/figure1"
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR,  exist_ok=True)

COL_KPERP  = 0
COL_OMEGA  = 4
COL_GAMMA  = 5
COL_P_ION  = 6
COL_P_ELEC = 7


#Input file generator
def make_input_file(beta_i, Ti_Te, label, map_mult=10.0):
    om_alfven = KPERP_MIN / max(np.sqrt(beta_i), 1e-3)
    om_box    = max(om_alfven * map_mult, 1e-4)
    gam_box   = max(om_box * 0.5, 5e-5)

    content = f"""!-=-=- Figure 1 Howes (2010): beta_i={beta_i:.4e}, Ti/Te={Ti_Te:.4e}
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
dataName='figure1'
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
alphS=1.0
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


#Mode identification
def identify_alfven_mode(modes_data):
    candidates = []
    for data in modes_data:
        omega_r = data[:, COL_OMEGA]
        gamma   = data[:, COL_GAMMA]
        P_ion   = data[:, COL_P_ION]
        P_elec  = data[:, COL_P_ELEC]

        if np.mean(np.abs(omega_r)) < 1e-10:        continue  # entropy mode
        if np.abs(np.mean(omega_r)) > 100:           continue  # failed solve
        if np.mean(omega_r > 0) < 0.7:              continue  # not propagating
        if np.max(np.abs(P_ion))  < 1e-40 and \
           np.max(np.abs(P_elec)) < 1e-40:          continue  # no heating

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
    idx    = np.argsort(best[:, COL_KPERP])
    return (best[idx, COL_KPERP], np.abs(best[idx, COL_OMEGA]),
            best[idx, COL_P_ION], best[idx, COL_P_ELEC])


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


# =========================================================================
print("="*60)
print("  Howes (2010) Figure 1 — PLUME reproduction script")
print("="*60)

# 1. Generate inputs
t_step = time.time()
print(f"\n[1/4] Generating {N_BETA*N_TRATIO} input files...", flush=True)
input_files = {}
for i, beta_i in enumerate(beta_i_arr):
    for j, Ti_Te in enumerate(tratio_arr):
        label = f"b{i:02d}_t{j:02d}"
        mult  = 50.0 if beta_i < 0.05 else (20.0 if beta_i < 0.2 else 10.0)
        path  = make_input_file(beta_i, Ti_Te, label, map_mult=mult)
        input_files[(i, j)] = (label, path)
print(f"    Done in {elapsed(t_step)}.")


#2. Run PLUME
t_step = time.time()
print(f"\n[2/4] Running PLUME ({N_BETA*N_TRATIO} runs)...", flush=True)
failed = []
for idx, ((i, j), (label, inpath)) in enumerate(input_files.items()):
    out_check = os.path.join(DATA_DIR, f"{label}_kperp_1_{PLUME_FID}.mode1")
    if os.path.exists(out_check):
        continue
    result = subprocess.run(["./plume.e", inpath], capture_output=True, text=True)
    if result.returncode != 0:
        failed.append((i, j))
    if (idx + 1) % 20 == 0:
        frac  = (idx + 1) / len(input_files)
        spent = time.time() - t_step
        eta   = (spent / frac) * (1 - frac) if frac > 0 else 0
        em, es = divmod(int(eta), 60)
        print(f"    {idx+1:3d}/{len(input_files)}  |  "
              f"elapsed {elapsed(t_step)}  |  ETA {em}m {es}s", flush=True)
print(f"    Finished in {elapsed(t_step)}. Failures: {len(failed)}")


#3. Compute heating ratios
t_step = time.time()
print(f"\n[3/4] Computing heating ratios...", flush=True)
ratio_grid = np.full((N_BETA, N_TRATIO), np.nan)

for (i, j), (label, _) in input_files.items():
    result = read_alfven_mode(label)
    if result is None:
        print(f"  SKIP {label}: no valid Alfven mode")
        continue
    kperp, omega, P_ion, P_elec = result
    if np.max(np.abs(P_ion)) < 1e-40:
        print(f"  SKIP {label}: P_ion~0")
        continue
    ratio = compute_heating_ratio(kperp, omega, P_ion, P_elec)
    ratio_grid[i, j] = ratio
    print(f"  {label} b={beta_i_arr[i]:.3f} T={tratio_arr[j]:.3f} "
          f"Qi/Qe={ratio:.3e}")

n_valid = np.sum(~np.isnan(ratio_grid))
print(f"\n    Valid: {n_valid}/{N_BETA*N_TRATIO}  |  Done in {elapsed(t_step)}")


#4. Extract parameters along integer contour lines
print(f"\n{'='*60}")
print("  PLASMA PARAMETERS ALONG INTEGER LOG10(Qi/Qe) CONTOURS")
print(f"{'='*60}")

#Integer contour levels to report
report_levels = [-20, -10, -5,-4,-3,-2,-1,0, 1, 2, 3]

#Build an interpolator on the log-log grid
log_beta   = np.log10(beta_i_arr)
log_tratio = np.log10(tratio_arr)
log_ratio  = np.log10(np.where(ratio_grid > 0, ratio_grid, np.nan))

# For each contour level, find (beta_i, Ti/Te) pairs along it by scanning across Ti/Te rows and interpolating in beta_i
print(f"\n  {'Level':<8} {'Qi/Qe':<10} {'beta_i':<10} {'Ti/Te':<10} "
      f"{'v_ti/v_A':<12} {'Physics note'}")
print(f"  {'-'*80}")

for level in report_levels:
    Qi_Qe = 10**level
    beta_samples  = []
    tratio_samples = []

    # Scan each Ti/Te row, find beta where log_ratio crosses this level
    for j in range(N_TRATIO):
        col = log_ratio[:, j]  # ratio as function of beta at fixed Ti/Te
        # Find crossings
        for ii in range(N_BETA - 1):
            v0, v1 = col[ii], col[ii+1]
            if np.isnan(v0) or np.isnan(v1):
                continue
            if (v0 - level) * (v1 - level) < 0:  # sign change = crossing
                # Linear interpolation in log space
                frac = (level - v0) / (v1 - v0)
                beta_cross = 10**(log_beta[ii] + frac * (log_beta[ii+1] - log_beta[ii]))
                beta_samples.append(beta_cross)
                tratio_samples.append(tratio_arr[j])

    if not beta_samples:
        print(f"  {level:<8} {Qi_Qe:<10.4f}  -- no crossing found in grid --")
        continue

    # Report a few representative points along this contour
    # Pick points
    target_tratios = [0.2, 1.0, 10.0, 100.0]
    # Sort by tratio
    pairs = sorted(zip(tratio_samples, beta_samples))

    printed = set()
    for target_t in target_tratios:
        # Find closest point
        if not pairs:
            continue
        closest = min(pairs, key=lambda x: abs(np.log10(x[0]) - np.log10(target_t)))
        t_val, b_val = closest

        # Avoid duplicates
        key = (round(np.log10(b_val), 1), round(np.log10(t_val), 1))
        if key in printed:
            continue
        printed.add(key)

        # Derived quantities
        # v_ti/v_A = sqrt(beta_i)  (from definition of beta_i = 8pi n T_i/B^2
        #                            and v_ti = sqrt(2Ti/mi), v_A = B/sqrt(4pi n mi))
        # beta_i = v_ti^2 / (2 v_A^2)  =>  v_ti/v_A = sqrt(2 * beta_i)
        vti_over_vA = np.sqrt(2.0 * b_val)

        # Ion Larmor radius relative to electron: rho_i/rho_e = sqrt(mi*Ti/(me*Te))
        #                                                     = sqrt(1836 * t_val)
        rho_ratio = np.sqrt(1836.0 * t_val)

        # Qualitative physics notes
        if b_val < 0.1:
            note = "few ions at resonance, e-dominated"
        elif b_val < 1.0:
            note = "transition regime"
        elif b_val < 10.0:
            note = "strong ion Landau damping"
        else:
            note = "ion heating dominant"

        print(f"  {level:<8} {Qi_Qe:<10.3g} "
              f"β_i={b_val:<8.3f} Ti/Te={t_val:<8.3f} "
              f"v_ti/v_A={vti_over_vA:<8.3f}  {note}")

print()

# Also print a summary table of where solar wind sits
print(f"  SOLAR WIND REFERENCE POINTS:")
print(f"  {'Region':<20} {'beta_i':<10} {'Ti/Te':<10} {'Qi/Qe (approx)'}")
print(f"  {'-'*55}")
sw_points = [
    ("Slow wind (~1 AU)",     1.0,  2.0),
    ("Fast wind (~1 AU)",     0.5,  4.0),
    ("High-beta wind",        3.0,  1.0),
]
for name, b, t in sw_points:
    # Interpolate ratio_grid
    ib = np.argmin(np.abs(np.log10(beta_i_arr) - np.log10(b)))
    it = np.argmin(np.abs(np.log10(tratio_arr)  - np.log10(t)))
    r  = ratio_grid[ib, it]
    rstr = f"{r:.2f}" if not np.isnan(r) else "NaN"
    print(f"  {name:<20} {b:<10.2f} {t:<10.2f} {rstr}")


# 5. Plot 
t_step = time.time()
print(f"\n[4/4] Plotting...", flush=True)

fig, ax = plt.subplots(figsize=(7.5, 6))

log_ratio_plot   = np.log10(ratio_grid.T)
log_ratio_masked = np.ma.masked_invalid(log_ratio_plot)

# Filled contour background
levels_fill = np.linspace(-2, 3, 101)
cf = ax.contourf(beta_i_arr, tratio_arr, log_ratio_masked,
                 levels=levels_fill, cmap='plasma', extend='both')

# Integer contour lines as
contour_levels =[-20, -10, -5,-4,-3,-2,-1,0, 1, 2, 3]
cl = ax.contour(beta_i_arr, tratio_arr, log_ratio_masked,
                levels=contour_levels,
                colors='k', linewidths=1, alpha=0.5)

# Label the contour lines with Qi/Qe values
fmt = {lv: f'{lv}' for lv in contour_levels}
ax.clabel(cl, fmt=fmt, fontsize=7.5, inline=False, inline_spacing=0)

cbar = fig.colorbar(cf, ax=ax, pad=0.02)
cbar.set_label(r'$\log_{10}(Q_i/Q_e)$', fontsize=12)
cbar.set_ticks([-20, -10, -5,-4,-3,-2,-1,0, 1, 2, 3])

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$\beta_i$', fontsize=13)
ax.set_ylabel(r'$T_i / T_e$', fontsize=13)
ax.set_title(r'Ion-to-electron heating ratio $Q_i/Q_e$' + '\n' +
             r'(Howes 2010, Fig. 1 reproduction)', fontsize=11)
ax.set_xlim(0.01, 100)
ax.set_ylim(0.2, 100)

# Mark solar wind region
ax.axvspan(0.3, 3.0, alpha=0.08, color='green', label='Typical solar wind')
ax.legend(fontsize=8, loc='lower right')

t_end    = time.time()
tm, ts   = divmod(int(t_end - t_total_start), 60)
time_str = f"{tm}m {ts}s"
ax.annotate(f"Runtime: {time_str}  |  Valid: {n_valid}/{N_BETA*N_TRATIO}",
            xy=(0.02, 0.02), xycoords='axes fraction', fontsize=8, color='gray')

plt.tight_layout()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
outpath   = f"figure1_Howes2010_{timestamp}.png"
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"    Saved: {outpath}")

print(f"\n{'='*60}")
print(f"  Total runtime: {time_str}")
print(f"  Valid: {n_valid}/{N_BETA*N_TRATIO}")
print(f"{'='*60}")

plt.show()