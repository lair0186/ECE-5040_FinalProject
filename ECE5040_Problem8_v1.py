# -*- coding: utf-8 -*-
"""
Problem 8: 2D H-Plane Flared Horn Antenna

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
print(f"Input Plate Separation = {a*1e3:.3f} mm")

omega = 2 * np.pi * f_source

# 3. Grid Parameters
# ============================================================================

fmax = f_source
lambda_min = c0 / fmax

dl = lambda_min / 20              # Spatial resolution
dt = 0.99 * dl / (c0 * np.sqrt(2))

# Nx, Ny = 760, 320                 # Larger y-domain reduces top/bottom reflections

guide_height = int(round(a / dl))
# aperture_height = 2 * guide_height

Nx, Ny = 900, 420
aperture_height = 3 * guide_height
flare_length = 80

a_eff = guide_height * dl
aperture_eff = aperture_height * dl

y_center = Ny // 2

y1 = y_center - guide_height // 2
y2 = y1 + guide_height

ap_y1 = y_center - aperture_height // 2
ap_y2 = ap_y1 + aperture_height

Ch = dt / (mu0 * dl)
Ce = dt / (eps0 * dl)

print(f"Grid Resolution = {dl*1e3:.3f} mm")
print(f"Input Guide Height = {guide_height} cells")
print(f"Output Aperture Height = {aperture_height} cells")
print(f"Effective Input Plate Separation = {a_eff*1e3:.3f} mm")
print(f"Effective Horn Aperture Height = {aperture_eff*1e3:.3f} mm")

# 4. Horn and Source Parameters
# ============================================================================

horn_start_x = 250
# flare_length = 30
aperture_x = horn_start_x + flare_length

src_x = horn_start_x - 150

src_yvals = np.arange(y1, y2 + 1)

mode_profile = np.sin(
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

# 7. PEC Horn Geometry Mask
# ============================================================================

pec_mask = np.zeros((Nx, Ny), dtype=bool)

lower_plate = np.full(Nx, -1, dtype=int)
upper_plate = np.full(Nx, -1, dtype=int)

for i in range(aperture_x + 1):

    if i < horn_start_x:

        local_height = guide_height

    else:

        flare_frac = (i - horn_start_x) / flare_length
        local_height = guide_height + flare_frac * (
            aperture_height - guide_height
        )

    local_height = int(round(local_height))

    y_low = y_center - local_height // 2
    y_high = y_low + local_height

    lower_plate[i] = y_low
    upper_plate[i] = y_high

    pec_mask[i, :y_low] = True
    pec_mask[i, y_high + 1:] = True

pec_mask[pml_start:, :] = False

# 8. Probe and Sampling Parameters
# ============================================================================

aperture_x_sample = aperture_x + 10
aperture_y_sample = np.arange(ap_y1, ap_y2 + 1)

theta = np.linspace(-np.pi / 2, np.pi / 2, 181)

r_sample = min(
    pml_start - aperture_x - 25,
    y_center - 10,
    Ny - y_center - 10
)

x_far = aperture_x + r_sample * np.cos(theta)
y_far = y_center + r_sample * np.sin(theta)

x_far_idx = np.clip(np.round(x_far).astype(int), 0, Nx - 1)
y_far_idx = np.clip(np.round(y_far).astype(int), 0, Ny - 1)

# 9. Time-Stepping Parameters
# ============================================================================

n_steps = 8000
steps_per_frame = 50

source_amplitude = 1.0
ramp_steps = 600

phasor_start = int(0.55 * n_steps)

# 10. TEz Horn Solver
# ============================================================================

def run_tez_horn():

    Hz = np.zeros((Nx, Ny))
    Ex = np.zeros((Nx, Ny))
    Ey = np.zeros((Nx, Ny))

    source_td = []

    E_frames = []
    Hz_frames = []

    best_transition_E_mag = np.zeros((Nx, Ny))
    best_radiation_E_mag = np.zeros((Nx, Ny))

    best_transition_score = 0.0
    best_radiation_score = 0.0

    aperture_phasor = np.zeros(len(aperture_y_sample), dtype=complex)
    far_phasor = np.zeros(len(theta), dtype=complex)
    phasor_count = 0

    global_max = 0.0

    print(f"Running TEz H-Plane Horn Simulation at {f_source/1e9:.1f} GHz...")

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

        # Ramped Modal Source
        # ====================================================================

        t = n * dt

        ramp = 1.0 - np.exp(-(n / ramp_steps) ** 2)

        source = source_amplitude * ramp * np.sin(omega * t)

        Hz[src_x, y1:y2 + 1] += source * mode_profile

        source_td.append(source)

        # Field Magnitude and Snapshot Selection
        # ====================================================================

        E_mag = np.sqrt(Ex ** 2 + Ey ** 2)

        transition_region = E_mag[
            horn_start_x - 25:aperture_x + 15,
            ap_y1 - 10:ap_y2 + 11
        ]

        radiation_region = E_mag[
            aperture_x + 10:pml_start - 10,
            10:Ny - 10
        ]

        transition_score = np.max(transition_region)
        radiation_score = np.max(radiation_region)

        if transition_score > best_transition_score and n > ramp_steps:

            best_transition_score = transition_score
            best_transition_E_mag = E_mag.copy()

        if radiation_score > best_radiation_score and n > ramp_steps:

            best_radiation_score = radiation_score
            best_radiation_E_mag = E_mag.copy()

        # Steady-State Phasor Sampling
        # ====================================================================

        if n > phasor_start:

            ref = np.exp(-1j * omega * t)

            aperture_phasor[:] += Hz[
                aperture_x_sample,
                aperture_y_sample
            ] * ref

            far_phasor[:] += Hz[x_far_idx, y_far_idx] * ref

            phasor_count += 1

        # Frame Storage
        # ====================================================================

        if n % steps_per_frame == 0:

            E_frames.append(E_mag.copy())
            Hz_frames.append(Hz.copy())

            current_max = np.max(E_mag)

            if current_max > global_max:
                global_max = current_max

        if n % 500 == 0:
            print(f"Computed horn step {n}/{n_steps}")

    if phasor_count > 0:

        aperture_phasor = aperture_phasor / phasor_count
        far_phasor = far_phasor / phasor_count

    return {
        "Hz": Hz.copy(),
        "Ex": Ex.copy(),
        "Ey": Ey.copy(),
        "E_mag": np.sqrt(Ex ** 2 + Ey ** 2),
        "best_transition_E_mag": best_transition_E_mag,
        "best_radiation_E_mag": best_radiation_E_mag,
        "E_frames": E_frames,
        "Hz_frames": Hz_frames,
        "source_td": np.array(source_td),
        "aperture_phasor": aperture_phasor,
        "far_phasor": far_phasor,
        "global_max": global_max
    }

# 10.b TEz Slit Reference Solver for Problem 7 Comparison
# ============================================================================

def run_tez_slit_reference():

    Hz = np.zeros((Nx, Ny))
    Ex = np.zeros((Nx, Ny))
    Ey = np.zeros((Nx, Ny))

    slit_pec_mask = np.zeros((Nx, Ny), dtype=bool)

    wall_x = aperture_x

    slit_y1 = y1
    slit_y2 = y2

    # Left-side parallel-plate waveguide feed
    slit_pec_mask[:wall_x + 1, :y1] = True
    slit_pec_mask[:wall_x + 1, y2 + 1:] = True

    # Vertical PEC wall with central slit
    slit_pec_mask[wall_x, :] = True
    slit_pec_mask[wall_x, slit_y1:slit_y2 + 1] = False

    # Do not place PEC inside the right PML
    slit_pec_mask[pml_start:, :] = False

    far_phasor_slit = np.zeros(len(theta), dtype=complex)
    phasor_count = 0

    print(f"Running TEz Slit Reference Simulation at {f_source/1e9:.1f} GHz...")

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

        Ex[slit_pec_mask] = 0.0
        Ey[slit_pec_mask] = 0.0

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

        Hz[slit_pec_mask] = 0.0

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

        # Same Ramped Modal Source Used for Horn Case
        # ====================================================================

        t = n * dt

        ramp = 1.0 - np.exp(-(n / ramp_steps) ** 2)

        source = source_amplitude * ramp * np.sin(omega * t)

        Hz[src_x, y1:y2 + 1] += source * mode_profile

        # Steady-State Far-Field Phasor Sampling
        # ====================================================================

        if n > phasor_start:

            ref = np.exp(-1j * omega * t)

            far_phasor_slit[:] += Hz[x_far_idx, y_far_idx] * ref

            phasor_count += 1

        if n % 500 == 0:
            print(f"Computed slit reference step {n}/{n_steps}")

    if phasor_count > 0:

        far_phasor_slit = far_phasor_slit / phasor_count

    return {
        "far_phasor": far_phasor_slit
    }

# 11. Run Simulations
# ============================================================================

result = run_tez_horn()
result_slit = run_tez_slit_reference()

t_axis = np.arange(n_steps) * dt

E_mag = result["E_mag"]
best_transition_E_mag = result["best_transition_E_mag"]
best_radiation_E_mag = result["best_radiation_E_mag"]

aperture_phasor = result["aperture_phasor"]
far_phasor = result["far_phasor"]
far_phasor_slit = result_slit["far_phasor"]

# 12. Plot 1: 2D E-Field Snapshot Showing Wave Transition into Flare
# ============================================================================

vmax_E1 = max(
    np.max(best_transition_E_mag[horn_start_x - 40:aperture_x + 40, :]) * 0.8,
    1e-12
)

plt.figure(figsize=(11, 5))

im = plt.imshow(
    best_transition_E_mag.T,
    cmap='inferno',
    origin='lower',
    vmin=0,
    vmax=vmax_E1,
    aspect='auto'
)

x_outline = np.arange(aperture_x + 1)

plt.plot(x_outline, lower_plate[:aperture_x + 1],
         color='cyan', linewidth=2)

plt.plot(x_outline, upper_plate[:aperture_x + 1],
         color='cyan', linewidth=2)

plt.axvline(horn_start_x, color='white', linestyle='--',
            linewidth=1.2, label='Horn Flare Start')

plt.axvline(aperture_x, color='lime', linestyle='--',
            linewidth=1.2, label='Horn Aperture')

plt.axvspan(pml_start, Nx - 1, color='gray', alpha=0.25, label='Right PML')

plt.xlim(src_x - 40, aperture_x + 90)
plt.ylim(ap_y1 - 25, ap_y2 + 25)

plt.xlabel('x (cells)')
plt.ylabel('y (cells)')
plt.title('Problem 8 Plot 1: 2D |E| Snapshot Showing Wave Transition into Horn Flare')

cbar = plt.colorbar(im, pad=0.01)
cbar.set_label('|E| = sqrt(Ex² + Ey²)')

plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('Final-Prob8-Plot1-FlareTransition.png', dpi=300, bbox_inches='tight')
plt.show()

# 13. Plot 2: 2D E-Field Snapshot Showing Radiation from Horn Aperture
# ============================================================================

vmax_E2 = max(
    np.max(best_radiation_E_mag[aperture_x + 5:pml_start - 5, :]) * 0.8,
    1e-12
)

plt.figure(figsize=(11, 5))

im = plt.imshow(
    best_radiation_E_mag.T,
    cmap='inferno',
    origin='lower',
    vmin=0,
    vmax=vmax_E2,
    aspect='auto'
)

plt.plot(x_outline, lower_plate[:aperture_x + 1],
         color='cyan', linewidth=2)

plt.plot(x_outline, upper_plate[:aperture_x + 1],
         color='cyan', linewidth=2)

plt.axvline(aperture_x, color='lime', linestyle='--',
            linewidth=1.2, label='Horn Aperture')

plt.axvspan(pml_start, Nx - 1, color='gray', alpha=0.25, label='Right PML')

plt.xlabel('x (cells)')
plt.ylabel('y (cells)')
plt.title('Problem 8 Plot 2: 2D |E| Snapshot Showing Radiation from Horn Aperture')

cbar = plt.colorbar(im, pad=0.01)
cbar.set_label('|E| = sqrt(Ex² + Ey²)')

plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('Final-Prob8-Plot2-HornRadiation.png', dpi=300, bbox_inches='tight')
plt.show()

# # 14. Plot 3: Normalized Radiation Pattern in dB
# # ============================================================================

# horn_far_field = np.abs(far_phasor)
# slit_far_field = np.abs(far_phasor_slit)

# if np.max(horn_far_field) > 0:
#     horn_far_field_norm = horn_far_field / np.max(horn_far_field)
# else:
#     horn_far_field_norm = horn_far_field

# if np.max(slit_far_field) > 0:
#     slit_far_field_norm = slit_far_field / np.max(slit_far_field)
# else:
#     slit_far_field_norm = slit_far_field

# horn_db = 20 * np.log10(np.maximum(horn_far_field_norm, 1e-6))
# slit_db = 20 * np.log10(np.maximum(slit_far_field_norm, 1e-6))

# plt.figure(figsize=(6, 6))

# ax = plt.subplot(111, projection='polar')

# ax.plot(theta, horn_db, linewidth=2,
#         label='Flared Horn: Problem 8')

# ax.plot(theta, slit_db, '--', linewidth=1.8,
#         label='Simple Slit: Problem 7 Reference')

# ax.set_theta_zero_location('E')
# ax.set_theta_direction(1)
# ax.set_rlim(-40, 0)
# ax.set_title('Problem 8 Plot 3: Normalized Radiation Pattern (dB)')

# plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.18))
# plt.tight_layout()
# plt.savefig('Final-Prob8-Plot3-HornVsSlitFDTD-dB.png',
#             dpi=300, bbox_inches='tight')
# plt.show()

# 14. Plot 3: Normalized Radiation Pattern in dB
# ============================================================================

horn_far_field = np.abs(far_phasor)
slit_far_field = np.abs(far_phasor_slit)

# Use the Problem 7 slit maximum as the common normalization reference.
# This means the slit peak is fixed at 0 dB. The horn is plotted relative
# to the slit peak, not relative to its own maximum.
slit_ref = np.max(slit_far_field)

if slit_ref > 0:

    horn_far_field_norm = horn_far_field / slit_ref
    slit_far_field_norm = slit_far_field / slit_ref

else:

    horn_far_field_norm = horn_far_field
    slit_far_field_norm = slit_far_field

horn_db = 20 * np.log10(np.maximum(horn_far_field_norm, 1e-6))
slit_db = 20 * np.log10(np.maximum(slit_far_field_norm, 1e-6))

plt.figure(figsize=(6, 6))

ax = plt.subplot(111, projection='polar')

ax.plot(theta, horn_db, linewidth=2,
        label='Flared Horn: Problem 8')

ax.plot(theta, slit_db, '--', linewidth=1.8,
        label='Simple Slit: Problem 7 Reference')

ax.set_theta_zero_location('E')
ax.set_theta_direction(1)

# Keep the lower limit fixed at -40 dB, but allow the upper limit to show
# horn gain above the slit reference if it occurs.
upper_db = max(3, np.ceil(np.max([np.max(horn_db), np.max(slit_db)])))
ax.set_rlim(-40, upper_db)

ax.set_title('Problem 8 Plot 3: Radiation Pattern Relative to Problem 7 Slit Peak')

plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.18))
plt.tight_layout()
plt.savefig('Final-Prob8-Plot3-HornVsSlitFDTD-dB-SlitNormalized.png',
            dpi=300, bbox_inches='tight')
plt.show()

print(f"Problem 7 slit normalization reference = {slit_ref:.6e}")
print(f"Horn peak relative to slit peak = {np.max(horn_far_field_norm):.3f}")
print(f"Horn peak relative to slit peak = {20*np.log10(max(np.max(horn_far_field_norm), 1e-12)):.3f} dB")



# 15. Plot 4: Aperture Phase Distribution
# ============================================================================

aperture_phase = np.unwrap(np.angle(aperture_phasor))
aperture_phase = aperture_phase - aperture_phase[len(aperture_phase) // 2]

aperture_y_mm = (aperture_y_sample - y_center) * dl * 1e3
aperture_y_m = (aperture_y_sample - y_center) * dl

quad_fit = np.polyfit(aperture_y_m, aperture_phase, 2)
aperture_phase_fit = np.polyval(quad_fit, aperture_y_m)

plt.figure(figsize=(7, 5))

plt.plot(aperture_y_mm, aperture_phase, 'o-', linewidth=2,
         label='Sampled Aperture Phase')

plt.plot(aperture_y_mm, aperture_phase_fit, '--', linewidth=2,
         label='Quadratic Fit')

plt.xlabel('Position Across Aperture Relative to Center (mm)')
plt.ylabel('Relative Phase (rad)')
plt.title('Problem 8 Plot 4: Phase Distribution Across Horn Aperture')

plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('Final-Prob8-Plot4-AperturePhase.png', dpi=300, bbox_inches='tight')
plt.show()

# 16. Optional Diagnostic Prints
# ============================================================================

print(f"\nSource Frequency = {f_source/1e9:.3f} GHz")
print(f"Free-Space Wavelength = {(c0/f_source)*1e3:.3f} mm")
print(f"Grid Resolution = {dl*1e3:.3f} mm")
print(f"Input Guide Height = {guide_height} cells")
print(f"Horn Aperture Height = {aperture_height} cells")
print(f"Horn Flare Length = {flare_length} cells")
print(f"Horn Flare Start x-index = {horn_start_x}")
print(f"Horn Aperture x-index = {aperture_x}")
print(f"Aperture sample x-index = {aperture_x_sample}")
print(f"Aperture y-range = {ap_y1} to {ap_y2}")
print(f"Right PML starts at x-index = {pml_start}")
print(f"Far-field sample radius = {r_sample} cells")
print(f"Quadratic phase coefficient = {quad_fit[0]:.3e} rad/m^2")
print("Simulation complete.")