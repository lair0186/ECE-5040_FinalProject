# -*- coding: utf-8 -*-
"""
Problem 7: 2D Radiation from a PEC Slit Aperture

Created on Fri May 15 2026

@author: nsomb
"""

import numpy as np
import matplotlib.pyplot as plt
import fdtd_constants as fdtd

# 1. Physical Constants
# ============================================================================

c0 = fdtd.c0
mu0 = fdtd.mu0
eps0 = fdtd.eps0

# 2. Source and Waveguide Parameters
# ============================================================================

fc = 9e9                         # Reference cutoff frequency from Problem 2
f_source = 11e9                   # Propagating excitation frequency

a = c0 / (2 * fc)                 # Parallel-plate separation
print(f"Plate Separation = {a*1e3:.3f} mm")

omega = 2 * np.pi * f_source

# 3. Grid Parameters
# ============================================================================

fmax = f_source
lambda_min = c0 / fmax

dl = lambda_min / 20              # Spatial resolution
dt = 0.99 * dl / (c0 * np.sqrt(2))

Nx, Ny = 700, 180   # Larger y-domain for open-space radiation

guide_height = int(round(a / dl))
a_eff = guide_height * dl

y_center = Ny // 2
y1 = y_center - guide_height // 2
y2 = y1 + guide_height

Ch = dt / (mu0 * dl)
Ce = dt / (eps0 * dl)

print(f"Grid Resolution = {dl*1e3:.3f} mm")
print(f"Guide Height = {guide_height} cells")
print(f"Effective Plate Separation = {a_eff*1e3:.3f} mm")

# 4. Aperture and Source Parameters
# ============================================================================

wall_x = 250                      # PEC aperture wall location
src_x = wall_x - 150               # Source location inside waveguide feed

slit_y1 = y1
slit_y2 = y2

src_yvals = np.arange(y1, y2 + 1)

mode_profile = np.cos(
    np.pi * (src_yvals - y1) / guide_height
)

mode_profile = mode_profile / np.max(np.abs(mode_profile))

# 5. Mur ABC Parameters
# ============================================================================

mur_coef = (c0 * dt - dl) / (c0 * dt + dl)

# 6. Right-Side UPML Parameters
# ============================================================================

pml_cells = 35
pml_order = 3
R_err = 1e-8

sigma_x = np.zeros(Nx)
eta0 = np.sqrt(mu0 / eps0)

sigma_max = -(pml_order + 1) * np.log(R_err) / (
    2 * eta0 * pml_cells * dl
)

for i in range(Nx - pml_cells, Nx):

    d_i = i - (Nx - pml_cells - 1)

    sigma_x[i] = sigma_max * (d_i / pml_cells) ** pml_order

sigma_e = np.tile(sigma_x[:, None], (1, Ny))
sigma_m = sigma_e * mu0 / eps0

Cay = (1.0 - sigma_e * dt / (2.0 * eps0)) / (
      1.0 + sigma_e * dt / (2.0 * eps0)
)

Cby = (dt / (eps0 * dl)) / (
      1.0 + sigma_e * dt / (2.0 * eps0)
)

Da = (1.0 - sigma_m * dt / (2.0 * mu0)) / (
     1.0 + sigma_m * dt / (2.0 * mu0)
)

Db = (dt / (mu0 * dl)) / (
     1.0 + sigma_m * dt / (2.0 * mu0)
)

pml_start = Nx - pml_cells

# 7. PEC Geometry Mask
# ============================================================================

pec_mask = np.zeros((Nx, Ny), dtype=bool)

# Left-side waveguide feed plates
pec_mask[:wall_x + 1, :y1] = True
pec_mask[:wall_x + 1, y2 + 1:] = True

# Vertical PEC wall with central aperture
pec_mask[wall_x, :] = True
pec_mask[wall_x, slit_y1:slit_y2 + 1] = False

# 8. Probe and Sampling Parameters
# ============================================================================

aperture_x_sample = wall_x + 2
aperture_y_sample = np.arange(slit_y1, slit_y2 + 1)

pml_probe_x = Nx - pml_cells // 2
pml_probe_y = y_center + guide_height//4

theta = np.linspace(-np.pi / 2, np.pi / 2, 181)
r_sample = min(
    pml_start - wall_x - 25,
    y_center - 10,
    Ny - y_center - 10
)

x_far = wall_x + r_sample * np.cos(theta)
y_far = y_center + r_sample * np.sin(theta)

x_far_idx = np.clip(np.round(x_far).astype(int), 0, Nx - 1)
y_far_idx = np.clip(np.round(y_far).astype(int), 0, Ny - 1)

# 9. Time-Stepping Parameters
# ============================================================================

n_steps = 5500
steps_per_frame = 50

tau = 0.55e-9
t0 = 3 * tau
source_amplitude = 1.0

# 10. TEz FDTD Aperture Solver
# ============================================================================

def run_tez_aperture():

    Hz = np.zeros((Nx, Ny))
    Ex = np.zeros((Nx, Ny))
    Ey = np.zeros((Nx, Ny))

    pml_probe_td = []
    source_td = []

    E_frames = []
    Hz_frames = []

    best_E_mag = np.zeros((Nx, Ny))
    best_score = 0.0

    global_max = 0.0

    print(f"Running TEz Aperture Radiation Simulation at {f_source/1e9:.1f} GHz...")

    for n in range(n_steps):

        Hz_left_old = Hz[0, :].copy()
        Hz_left_inner = Hz[1, :].copy()

        Hz_bottom_old = Hz[:, 0].copy()
        Hz_bottom_inner = Hz[:, 1].copy()

        Hz_top_old = Hz[:, -1].copy()
        Hz_top_inner = Hz[:, -2].copy()

        # Ex Update
        # ====================================================================

        Ex[:, 1:-1] += Ce * (
            Hz[:, 1:-1] - Hz[:, :-2]
        )

        # Ey Update with Right-Side PML Conductivity
        # ====================================================================

        Ey[1:-1, :] = (
            Cay[1:-1, :] * Ey[1:-1, :]
            -
            Cby[1:-1, :] * (
                Hz[1:-1, :] - Hz[:-2, :]
            )
        )

        # PEC Electric Field Enforcement
        # ====================================================================

        Ex[pec_mask] = 0.0
        Ey[pec_mask] = 0.0

        # Hz Update with Right-Side PML Conductivity
        # ====================================================================

        Hz[1:-1, 1:-1] = (
            Da[1:-1, 1:-1] * Hz[1:-1, 1:-1]
            +
            Db[1:-1, 1:-1] * (
                (Ex[1:-1, 2:] - Ex[1:-1, 1:-1]) -
                (Ey[2:, 1:-1] - Ey[1:-1, 1:-1])
            )
        )

        # PEC Magnetic Field Blocking Inside Metal Regions
        # ====================================================================

        Hz[pec_mask] = 0.0

        # Mur ABC on Left, Bottom, and Top Boundaries Only
        # ====================================================================

        Hz[0, 1:-1] = Hz_left_inner[1:-1] + mur_coef * (
            Hz[1, 1:-1] - Hz_left_old[1:-1]
        )

        Hz[1:-1, 0] = Hz_bottom_inner[1:-1] + mur_coef * (
            Hz[1:-1, 1] - Hz_bottom_old[1:-1]
        )

        Hz[1:-1, -1] = Hz_top_inner[1:-1] + mur_coef * (
            Hz[1:-1, -2] - Hz_top_old[1:-1]
        )

        Hz[0, 0] = 0.5 * (Hz[1, 0] + Hz[0, 1])
        Hz[0, -1] = 0.5 * (Hz[1, -1] + Hz[0, -2])
        Hz[-1, 0] = 0.0
        Hz[-1, -1] = 0.0

        # Gaussian-Modulated Modal Source
        # ====================================================================

        t = n * dt

        source = source_amplitude * np.sin(omega * t) * np.exp(
            -((t - t0) / tau) ** 2
        )

        Hz[src_x, y1:y2 + 1] += source * mode_profile

        source_td.append(source)

        # Field Magnitude and Probe Collection
        # ====================================================================

        E_mag = np.sqrt(Ex ** 2 + Ey ** 2)

        pml_probe_td.append(E_mag[pml_probe_x, pml_probe_y])

        radiation_region = E_mag[
            wall_x + 10:pml_start - 10,
            10:Ny - 10
        ]

        score = np.max(radiation_region)

        if score > best_score and n > int(t0 / dt):

            best_score = score
            best_E_mag = E_mag.copy()

        # Frame Storage
        # ====================================================================

        if n % steps_per_frame == 0:

            E_frames.append(E_mag.copy())
            Hz_frames.append(Hz.copy())

            current_max = np.max(E_mag)

            if current_max > global_max:
                global_max = current_max

        if n % 500 == 0:
            print(f"Computed step {n}/{n_steps}")

    return {
        "Hz": Hz.copy(),
        "Ex": Ex.copy(),
        "Ey": Ey.copy(),
        "E_mag": np.sqrt(Ex ** 2 + Ey ** 2),
        "best_E_mag": best_E_mag,
        "E_frames": E_frames,
        "Hz_frames": Hz_frames,
        "pml_probe_td": np.array(pml_probe_td),
        "source_td": np.array(source_td),
        "global_max": global_max
    }

# 11. Run Simulation
# ============================================================================

result = run_tez_aperture()

t_axis = np.arange(n_steps) * dt

E_mag = result["E_mag"]
best_E_mag = result["best_E_mag"]
pml_probe_td = result["pml_probe_td"]

# 12. Plot 1: 2D Near-Field Electric Field Snapshot
# ============================================================================

# vmax_E = max(np.max(best_E_mag) * 0.6, 1e-12)
vmax_E = max(np.max(best_E_mag[wall_x+5:pml_start-5, :]) * 0.8, 1e-12)

plt.figure(figsize=(11, 5))

im = plt.imshow(
    best_E_mag.T,
    cmap='inferno',
    origin='lower',
    vmin=0,
    vmax=vmax_E,
    aspect='auto'
)

plt.vlines(wall_x, 0, slit_y1, colors='cyan', linewidth=2)
plt.vlines(wall_x, slit_y2, Ny - 1, colors='cyan', linewidth=2)

plt.hlines(y1, 0, wall_x, colors='cyan', linewidth=2)
plt.hlines(y2, 0, wall_x, colors='cyan', linewidth=2)

plt.axvspan(pml_start, Nx - 1, color='gray', alpha=0.25, label='Right PML')

plt.xlabel('x (cells)')
plt.ylabel('y (cells)')
plt.title('Problem 7 Plot 1: 2D Near-Field |E| Snapshot from PEC Slit Aperture')

cbar = plt.colorbar(im, pad=0.01)
cbar.set_label('|E| = sqrt(Ex² + Ey²)')

plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('Final-Prob7-Plot1-NearField.png', dpi=300, bbox_inches='tight')
plt.show()

# 13. Plot 2: Electric Field Amplitude Across Aperture Plane
# ============================================================================

aperture_E = best_E_mag[aperture_x_sample, aperture_y_sample]
aperture_y_mm = (aperture_y_sample - y_center) * dl * 1e3

plt.figure(figsize=(6, 5))

plt.plot(aperture_y_mm, aperture_E, linewidth=2)

plt.xlabel('Position Across Aperture Relative to Center (mm)')
plt.ylabel('|E| Amplitude')
plt.title('Problem 7 Plot 2: E-Field Amplitude Across Aperture Plane')

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Final-Prob7-Plot2-AperturePlane.png', dpi=300, bbox_inches='tight')
plt.show()

# 14. Plot 3: Uncalibrated Far-Field Radiation Pattern
# ============================================================================

far_field = best_E_mag[x_far_idx, y_far_idx]

if np.max(far_field) > 0:
    far_field_norm = far_field / np.max(far_field)
else:
    far_field_norm = far_field

plt.figure(figsize=(6, 6))

ax = plt.subplot(111, projection='polar')

ax.plot(theta, far_field_norm, linewidth=2)

ax.set_theta_zero_location('E')
ax.set_theta_direction(1)
ax.set_title('Problem 7 Plot 3: Uncalibrated Far-Field Radiation Pattern')

plt.tight_layout()
plt.savefig('Final-Prob7-Plot3-FarFieldPolar.png', dpi=300, bbox_inches='tight')
plt.show()

# 15. Plot 4: Electric Field Sampled Deep Inside the PML
# ============================================================================

plt.figure(figsize=(8, 5))

plt.plot(t_axis * 1e9, pml_probe_td, linewidth=1.5)

plt.xlabel('Time (ns)')
plt.ylabel('|E| at PML Probe')
plt.title('Problem 7 Plot 4: Electric Field Sampled Deep Inside Right PML')

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Final-Prob7-Plot4-PMLProbe.png', dpi=300, bbox_inches='tight')
plt.show()

# 16. Optional Diagnostic Prints
# ============================================================================

print(f"\nSource Frequency = {f_source/1e9:.3f} GHz")
print(f"Free-Space Wavelength = {(c0/f_source)*1e3:.3f} mm")
print(f"Grid Resolution = {dl*1e3:.3f} mm")
print(f"Wall x-index = {wall_x}")
print(f"Slit y-range = {slit_y1} to {slit_y2}")
print(f"Right PML starts at x-index = {pml_start}")
print(f"PML probe location = ({pml_probe_x}, {pml_probe_y})")
print(f"PML probe offset from aperture center = {pml_probe_y - y_center} cells")
print(f"Far-field sample radius = {r_sample} cells")
print("Simulation complete.")