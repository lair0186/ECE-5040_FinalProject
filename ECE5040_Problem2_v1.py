# -*- coding: utf-8 -*-
"""
Problem 2: 2D TEz Mode Cutoff in a Parallel-Plate Waveguide

Created on Thu May 14 2026

@author: nsomb
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import fdtd_constants as fdtd

# 1. Physical Constants
# ============================================================================

c0 = fdtd.c0
mu0 = fdtd.mu0
eps0 = fdtd.eps0

# 2. Waveguide Parameters
# ============================================================================

fc = 9e9                         # Designed cutoff frequency (9 GHz)
f_low = 7e9                      # Below cutoff excitation
f_high = 11e9                    # Above cutoff excitation

a = c0 / (2 * fc)                # Parallel-plate separation
print(f"Plate Separation = {a*1e3:.3f} mm")

# 3. Grid Parameters
# ============================================================================

fmax = f_high
lambda_min = c0 / fmax

dl = lambda_min / 20             # Spatial resolution
dt = 0.99 * dl / (c0 * np.sqrt(2))

Nx, Ny = 700, 80                 # Grid dimensions

guide_height = int(round(a / dl))
a_eff = guide_height * dl

y1 = (Ny - guide_height) // 2
y2 = y1 + guide_height

Ch = dt / (mu0 * dl)
Ce = dt / (eps0 * dl)

print(f"Grid Resolution = {dl*1e3:.3f} mm")
print(f"Guide Height = {guide_height} cells")
print(f"Effective Plate Separation = {a_eff*1e3:.3f} mm")

# 4. Source and Probe Parameters
# ============================================================================

src_x = 40

probe_x = 420
probe2_x = 435

# Important: TE1 Hz has a null at the centerline, so do NOT probe at center.
probe_y = y1 + guide_height // 4

# 5. Modal Source Profile
# ============================================================================

src_yvals = np.arange(y1, y2 + 1)

mode_profile = np.cos(
    np.pi * (src_yvals - y1) / guide_height
)

mode_profile = mode_profile / np.max(np.abs(mode_profile))

# 6. Mur ABC Parameters
# ============================================================================

mur_coef = (c0 * dt - dl) / (c0 * dt + dl)

# 7. Time-Stepping Parameters
# ============================================================================

n_steps = 6000
steps_per_frame = 50

# 8. TEz FDTD Solver
# ============================================================================

def run_tez_waveguide(f_source):

    omega = 2 * np.pi * f_source

    Hz = np.zeros((Nx, Ny))
    Ex = np.zeros((Nx, Ny))
    Ey = np.zeros((Nx, Ny))

    Hz_probe1_td = []
    Hz_probe2_td = []

    Hz_frames = []
    global_max = 0.0

    print(f"Running TEz FDTD Simulation at {f_source/1e9:.1f} GHz...")

    for n in range(n_steps):

        Hz_left_old = Hz[0, :].copy()
        Hz_left_inner = Hz[1, :].copy()

        Hz_right_old = Hz[-1, :].copy()
        Hz_right_inner = Hz[-2, :].copy()

        # Ex Update
        # ====================================================================

        Ex[:, 1:-1] += Ce * (
            Hz[:, 1:-1] - Hz[:, :-2]
        )

        # Ey Update
        # ====================================================================

        Ey[1:-1, :] -= Ce * (
            Hz[1:-1, :] - Hz[:-2, :]
        )

        # PEC Parallel Plates
        # ====================================================================

        Ex[:,:y1+1]=0
        Ex[:,y2:]=0
        
        Ey[:,:y1+1]=0
        Ey[:,y2:]=0
        

        # Hz Update
        # ====================================================================

        Hz[1:-1, 1:-1] += Ch * (
            (Ex[1:-1, 2:] - Ex[1:-1, 1:-1]) -
            (Ey[2:, 1:-1] - Ey[1:-1, 1:-1])
        )

        Hz[:,:y1]=0
        Hz[:,y2+1:]=0

        # Mur ABC at Open Ends
        # ====================================================================

        Hz[0, :] = Hz_left_inner + mur_coef * (
            Hz[1, :] - Hz_left_old
        )

        Hz[-1, :] = Hz_right_inner + mur_coef * (
            Hz[-2, :] - Hz_right_old
        )

        # CW Modal Source
        # ====================================================================

        t = n * dt

        source = np.sin(omega * t)

        Hz[src_x, y1:y2+1] += source * mode_profile

        # Probe Collection
        # ====================================================================

        Hz_probe1_td.append(Hz[probe_x, probe_y])
        Hz_probe2_td.append(Hz[probe2_x, probe_y])

        # Frame Storage
        # ====================================================================

        if n % steps_per_frame == 0:

            Hz_frames.append(Hz.copy())

            current_max = np.max(np.abs(Hz))

            if current_max > global_max:
                global_max = current_max

        if n % 500 == 0:
            print(f"Computed step {n}/{n_steps}")

    return {
        "f_source": f_source,
        "Hz": Hz.copy(),
        "Hz_frames": Hz_frames,
        "Hz_probe1_td": np.array(Hz_probe1_td),
        "Hz_probe2_td": np.array(Hz_probe2_td),
        "global_max": global_max
    }

# 9. Run Both Frequency Cases
# ============================================================================

result_7GHz = run_tez_waveguide(f_low)
result_11GHz = run_tez_waveguide(f_high)

t_axis = np.arange(n_steps) * dt

# 10. Plots 1 and 2: 2D Snapshots for 11 GHz and 7 GHz
# ============================================================================

Hz_11 = result_11GHz["Hz"]
Hz_7 = result_7GHz["Hz"]

vmax_11 = max(np.max(np.abs(Hz_11)) * 0.5, 1e-12)
vmax_7 = max(np.max(np.abs(Hz_7)) * 0.5, 1e-12)

fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(10, 5),
    sharex=True,
    constrained_layout=True
)

im1 = ax1.imshow(
    Hz_11.T,
    cmap='RdBu',
    origin='lower',
    vmin=-vmax_11,
    vmax=vmax_11,
    aspect='auto'
)

ax1.hlines(y1, 0, Nx, colors='black', linewidth=2)
ax1.hlines(y2, 0, Nx, colors='black', linewidth=2)

ax1.set_ylabel('y (cells)')
ax1.set_title('TEz Parallel-Plate Waveguide: 11 GHz Propagating Case')
ax1.grid(False)

cbar1 = fig.colorbar(im1, ax=ax1, pad=0.01)
cbar1.set_label('H_z Amplitude')

im2 = ax2.imshow(
    Hz_7.T,
    cmap='RdBu',
    origin='lower',
    vmin=-vmax_7/3,
    vmax=vmax_7/3,
    aspect='auto'
)

ax2.hlines(y1, 0, Nx, colors='black', linewidth=2)
ax2.hlines(y2, 0, Nx, colors='black', linewidth=2)

ax2.set_xlabel('x (cells)')
ax2.set_ylabel('y (cells)')
ax2.set_title('TEz Parallel-Plate Waveguide: 7 GHz Evanescent Case')
ax2.grid(False)

cbar2 = fig.colorbar(im2, ax=ax2, pad=0.01)
cbar2.set_label('H_z Amplitude')

# To save image, remove uncomment next line
plt.savefig('Final-Prob2-Plot1&2.png', dpi=300, bbox_inches='tight')
plt.show()

# 11. Plot 3: Time-Domain Probe Signals
# ============================================================================

Hz_probe_7GHz = result_7GHz["Hz_probe1_td"]
Hz_probe_11GHz = result_11GHz["Hz_probe1_td"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.plot(t_axis * 1e9, Hz_probe_7GHz)
ax1.set_title('7 GHz Probe Signal')
ax1.set_xlabel('Time (ns)')
ax1.set_ylabel('H_z Amplitude')
ax1.grid(True, alpha=0.3)

ax2.plot(t_axis * 1e9, Hz_probe_11GHz)
ax2.set_title('11 GHz Probe Signal')
ax2.set_xlabel('Time (ns)')
ax2.set_ylabel('H_z Amplitude')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
# To save image, remove uncomment next line
plt.savefig('Final-Prob2-Plot3.png', dpi=300, bbox_inches='tight')
plt.show()

# 12. Analytical Phase Velocity
# ============================================================================

vp_analytic = c0 / np.sqrt(
    1 - (fc / f_high)**2
)

print(f"\nAnalytical Phase Velocity = {vp_analytic:.4e} m/s")

# 13. Numerical Phase Velocity from 11 GHz Probe Phase
# ============================================================================

Hz_probe1_td = result_11GHz["Hz_probe1_td"]
Hz_probe2_td = result_11GHz["Hz_probe2_td"]

start_idx = n_steps // 2

sig1 = Hz_probe1_td[start_idx:]
sig2 = Hz_probe2_td[start_idx:]

window = np.hanning(len(sig1))

FFT1 = np.fft.fft(sig1 * window)
FFT2 = np.fft.fft(sig2 * window)

freqs = np.fft.fftfreq(len(FFT1), d=dt)

idx = np.argmin(np.abs(freqs - f_high))

phase1 = np.angle(FFT1[idx])
phase2 = np.angle(FFT2[idx])

delta_phase = np.angle(np.exp(1j * (phase1 - phase2)))

delta_x = (probe2_x - probe_x) * dl

beta_num = np.abs(delta_phase / delta_x)

vp_numerical = 2 * np.pi * f_high / beta_num

print(f"Numerical Phase Velocity = {vp_numerical:.4e} m/s")

# 14. Plot 4: Analytical vs Numerical Phase Velocity
# ============================================================================

plt.figure(figsize=(5, 5))

labels = ['Analytical', 'Numerical']
values = [vp_analytic, vp_numerical]

plt.bar(labels, values)

plt.ylabel('Phase Velocity (m/s)')
plt.title('Analytical vs Numerical Phase Velocity: 11 GHz')

plt.grid(True, alpha=0.3)

plt.tight_layout()
# To save image, remove uncomment next line
plt.savefig('Final-Prob2-Plot4.png', dpi=300, bbox_inches='tight')
plt.show()

# 15. Optional Diagnostic Prints
# ============================================================================

lambda0_11 = c0 / f_high
lambda_g_11 = lambda0_11 / np.sqrt(1 - (fc / f_high)**2)

print(f"\nFree-Space Wavelength at 11 GHz = {lambda0_11*1e3:.3f} mm")
print(f"Analytical Guide Wavelength at 11 GHz = {lambda_g_11*1e3:.3f} mm")
print(f"Probe Separation = {delta_x*1e3:.3f} mm")
print(f"Probe y-index = {probe_y}, guide centerline = {(y1+y2)//2}")