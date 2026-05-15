# -*- coding: utf-8 -*-
"""
Created on Thu May 14 01:29:31 2026

@author: nsomb
"""

import numpy as np
import matplotlib.pyplot as plt
import fdtd_constants as fdtd

# 1. Source Parameters
# ============================================================================

f0 = 10e9                     # Center frequency (10 GHz)
omega0 = 2.0 * np.pi * f0

E0 = 1                        # Normalized Source Amplitude

# Gaussian pulse parameters
n0 = 120
ndecay = 50

# 2. Grid Parameters
# ============================================================================

dz = 1e-3                     # 1 mm spatial resolution
Nz = 2000                     # Number of spatial cells

# CFL condition
dt = 0.99 * dz / fdtd.c0

Nt = 4000                     # Number of time steps

# 3. Yee Grid Fields
# ============================================================================

Ex = np.zeros(Nz)
Hy = np.zeros(Nz)

# 4. Update Coefficients
# ============================================================================

Ce = dt / (fdtd.eps0 * dz)
Ch = dt / (fdtd.mu0 * dz)

# 5. Source Location
# ============================================================================

src = 200

# 6. Data Storage
# ============================================================================

snapshots = {}

source_history = []
pulse_history = []

# 7. Main FDTD Loop
# ============================================================================
plt.ion()
fig_live, ax_live = plt.subplots(figsize=(10,4))
line, = ax_live.plot(Ex, lw=2)

ax_live.set_xlim(0, Nz)
ax_live.set_ylim(-1.2, 1.2)
ax_live.set_xlabel('Spatial Cell Index')
ax_live.set_ylabel('Ex Amplitude')
ax_live.set_title('1D FDTD Propagation')
ax_live.grid(True, alpha=0.3)

# Source and PEC markers
ax_live.axvline(src, color='green', linestyle=':', label='Source')
ax_live.axvline(Nz-1, color='red', linestyle='--', label='PEC')

ax_live.legend()

for n in range(Nt):

    # Magnetic Field Update
    # ------------------------------------------------------------------------

    Hy[:-1] = Hy[:-1] - Ch * (Ex[1:] - Ex[:-1])

    # Electric Field Update
    # ------------------------------------------------------------------------

    Ex[1:] = Ex[1:] - Ce * (Hy[1:] - Hy[:-1])

    # Gaussian-Modulated Source
    # ------------------------------------------------------------------------

    pulse = E0 * np.exp(-((n - n0)/ndecay)**2) * \
                  np.sin(omega0 * (n - n0) * dt)

    Ex[src] += pulse

    # PEC Boundary
    # ------------------------------------------------------------------------

    Ex[-1] = 0.0

    # Store Source Signal
    # ------------------------------------------------------------------------
    pulse_history.append(pulse)
    source_history.append(Ex[src])

    # Save Snapshots
    # ------------------------------------------------------------------------

    if n == 1500:
        snapshots["incident"] = Ex.copy()

    if n == 1950:
        snapshots["interference"] = Ex.copy()

    if n == 2500:
        snapshots["reflected"] = Ex.copy()
        
        # Real-Time Visualization
    # ------------------------------------------------------------------------

    if n % 10 == 0:

        line.set_ydata(Ex)

        ax_live.set_title(
            f'1D FDTD Propagation | Time Step = {n}'
        )

        plt.draw()
        plt.pause(0.05)
        
plt.ioff()

# 8. Snapshot Visualization
# ============================================================================

fig, axes = plt.subplots(3, 1, figsize=(10, 8))

# Incident Wave
axes[0].plot(snapshots["incident"])
axes[0].set_title("Incident Wave Before PEC Reflection")
axes[0].set_ylabel("Ex Amplitude")
axes[0].grid(True, alpha=0.3)

# Interference Region
axes[1].plot(snapshots["interference"])
axes[1].set_title("Incident + Reflected Wave Interference")
axes[1].set_ylabel("Ex Amplitude")
axes[1].grid(True, alpha=0.3)

# Reflected Wave
axes[2].plot(snapshots["reflected"])
axes[2].set_title("Reflected Wave After PEC Reflection")
axes[2].set_xlabel("Spatial Cell Index")
axes[2].set_ylabel("Ex Amplitude")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()

# To save image, remove uncomment next line
# plt.savefig('Final-Prob1-Plot1&2&3.png', dpi=300, bbox_inches='tight')
plt.show()        

# 9. FFT Analysis
# ============================================================================

pulse_history = np.array(pulse_history)

# FFT
pulse_fft = np.fft.fft(pulse_history)

# Frequency axis
freq = np.fft.fftfreq(len(pulse_fft), d=dt)

# Positive frequencies only
pos = freq > 0

freq_pos = freq[pos]
pulse_fft_pos = pulse_fft[pos]

# Magnitude spectrum
pulse_mag = np.abs(pulse_fft_pos)

# Normalize
pulse_mag /= np.max(pulse_mag)

# 10. Source Spectrum Plot
# ============================================================================

plt.figure(figsize=(8,5))

plt.plot(freq_pos / 1e9, pulse_mag)
plt.xlim(0, 20)
plt.xlabel('Frequency (GHz)')
plt.ylabel('Normalized Magnitude')
plt.title('FFT of Gaussian-Modulated Source Pulse')
plt.grid(True, alpha=0.3)

# Highlight X-band
plt.axvspan(8, 12, alpha=0.2)

# To save image, remove uncomment next line
# plt.savefig('Final-Prob1-Plot4.png', dpi=300, bbox_inches='tight')
plt.show()

