# -*- coding: utf-8 -*-
"""
Created on Thu May 14 22:59:20 2026

@author: nsomb
"""

import numpy as np
import matplotlib.pyplot as plt
import fdtd_constants as fdtd
    

# 1. Physical Constants
# =============================================================================
c0, mu0, eps0 = fdtd.c0, fdtd.mu0, fdtd.eps0
eta0 = np.sqrt(mu0/eps0)

# 2. Source and Material Parameters
# =============================================================================

f0 = 192e12
omega0 = 2.0*np.pi*f0
lambda0 = c0/f0
source_amplitude = 1.0

n_core = 3.48               # Silicon Core
n_clad = 1.44               # Silicon Dioxide Cladding
eps_core = n_core**2
eps_clad = n_clad**2

# 3. Waveguide Geometry and Grid
# =============================================================================

height_core = 200e-9
length_core = 5e-6
d = height_core/2.0

dl = 10e-9
dt = 0.99*dl/(np.sqrt(2.0)*c0)

core_length_cells = int(round(length_core/dl))
core_height_cells = int(round(height_core/dl))

pad_x = 120
pad_y = 100

Nx = core_length_cells + 2*pad_x
Ny = core_height_cells + 2*pad_y

x1 = pad_x
x2 = x1 + core_length_cells
y1 = (Ny - core_height_cells)//2
y2 = y1 + core_height_cells

x_um = np.arange(Nx)*dl*1e6
y_um = (np.arange(Ny) - 0.5*(y1+y2-1))*dl*1e6

eps_r = np.full((Nx, Ny), eps_clad)
eps_r[x1:x2, y1:y2] = eps_core

Ch = dt/(mu0*dl)
Ce = dt/(eps0*eps_r*dl)

# 4. Analytical TE0 Mode Profile Using EIM
# =============================================================================

k0 = omega0/c0
V = k0*height_core*np.sqrt(n_core**2 - n_clad**2)

b_vals = np.linspace(1e-8, 1.0-1e-8, 200000)
lhs = V*np.sqrt(1.0 - b_vals)
rhs = 2.0*np.arctan(np.sqrt(b_vals/(1.0 - b_vals)))
b_te0 = b_vals[np.argmin(np.abs(lhs-rhs))]

n_eff = np.sqrt(n_clad**2 + b_te0*(n_core**2 - n_clad**2))
beta = n_eff*k0
kappa = np.sqrt((n_core*k0)**2 - beta**2)
gamma = np.sqrt(beta**2 - (n_clad*k0)**2)

y = (np.arange(Ny) - 0.5*(y1+y2-1))*dl
mode_profile = np.where(np.abs(y) <= d,
                        np.cos(kappa*y),
                        np.cos(kappa*d)*np.exp(-gamma*(np.abs(y)-d)))
mode_profile = mode_profile/np.max(np.abs(mode_profile))

fc_m1 = 1.0/(4.0*d*np.sqrt(mu0*eps0*(n_core**2 - n_clad**2)))
lambda_g = 2*np.pi/beta

print(f"TE0 effective index n_eff = {n_eff:.4f}")
print(f"Guided wavelength lambda_g = {lambda_g*1e6:.4f} um")
print(f"m = 1 cutoff frequency = {fc_m1/1e12:.2f} THz")

# 5. Field Initialization and Simulation Settings
# =============================================================================

Ey = np.zeros((Nx, Ny))
Hx = np.zeros((Nx, Ny))
Hz = np.zeros((Nx, Ny))

src_x = x1 + 8
probe_x = (x1 + x2)//2

n_steps = 9000
ramp_steps = 900
phasor_start = int(0.65*n_steps)

mur_coef = ((c0/n_clad)*dt - dl)/((c0/n_clad)*dt + dl)

Ey_phasor = np.zeros((Nx, Ny), dtype=complex)
Pz_avg = np.zeros((Nx, Ny))
phasor_count = 0

# 6. FDTD Time-Stepping Loop
# =============================================================================

print("Running TE slab waveguide FDTD...")
for n in range(n_steps):
    Ey_right_old, Ey_right_inner_old = Ey[-1, :].copy(), Ey[-2, :].copy()
    Ey_left_old, Ey_left_inner_old = Ey[0, :].copy(), Ey[1, :].copy()
    Ey_top_old, Ey_top_inner_old = Ey[:, -1].copy(), Ey[:, -2].copy()
    Ey_bottom_old, Ey_bottom_inner_old = Ey[:, 0].copy(), Ey[:, 1].copy()

    Hx[1:, :] += Ch*(Ey[1:, :] - Ey[:-1, :])
    Hz[:, 1:] -= Ch*(Ey[:, 1:] - Ey[:, :-1])

    Ey[1:-1, 1:-1] += Ce[1:-1, 1:-1]*((Hx[2:, 1:-1] - Hx[1:-1, 1:-1]) -
                                       (Hz[1:-1, 2:] - Hz[1:-1, 1:-1]))

    Ey[-1, 1:-1] = Ey_right_inner_old[1:-1] + mur_coef*(Ey[-2, 1:-1] - Ey_right_old[1:-1])
    Ey[0, 1:-1] = Ey_left_inner_old[1:-1] + mur_coef*(Ey[1, 1:-1] - Ey_left_old[1:-1])
    Ey[1:-1, -1] = Ey_top_inner_old[1:-1] + mur_coef*(Ey[1:-1, -2] - Ey_top_old[1:-1])
    Ey[1:-1, 0] = Ey_bottom_inner_old[1:-1] + mur_coef*(Ey[1:-1, 1] - Ey_bottom_old[1:-1])

    Ey[0, 0] = 0.5*(Ey[1, 0] + Ey[0, 1])
    Ey[0, -1] = 0.5*(Ey[1, -1] + Ey[0, -2])
    Ey[-1, 0] = 0.5*(Ey[-2, 0] + Ey[-1, 1])
    Ey[-1, -1] = 0.5*(Ey[-2, -1] + Ey[-1, -2])

    t = n*dt
    ramp = 1.0 - np.exp(-(n/ramp_steps)**2)
    Ey[src_x, :] = source_amplitude*ramp*mode_profile*np.cos(omega0*t)

    if n >= phasor_start:
        Ey_phasor += Ey*np.exp(-1j*omega0*t)
        Pz_avg += -Ey*Hx
        phasor_count += 1

    if n % 500 == 0:
        print(f"Step {n}/{n_steps}")

Ey_phasor = 2.0*Ey_phasor/max(phasor_count, 1)
Pz_avg = Pz_avg/max(phasor_count, 1)
if np.mean(Pz_avg[probe_x, y1:y2]) < 0:
    Pz_avg *= -1.0

# 7. Plot 1: Analytical Transverse E-Field Source Profile
# =============================================================================

plt.figure(figsize=(7, 5))
plt.plot(y_um, mode_profile, label='Normalized analytical TE$_0$ profile')
plt.axvspan(y_um[y1], y_um[y2-1], color='lightgray', alpha=0.6, label='Si core')
plt.xlabel('Transverse position y (um)')
plt.ylabel('Normalized $E_y$ amplitude')
plt.title('Analytical TE$_0$ Source Profile at Input Plane')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('Final-Prob5-Plot1.png', dpi=300, bbox_inches='tight')

plt.show()

# 8. Plot 2: 2D Snapshot of Confined Steady-State E-Field
# =============================================================================

vmax = max(np.max(np.abs(Ey))*0.8, 1e-12)
plt.figure(figsize=(10, 5))
plt.imshow(Ey.T, origin='lower', aspect='auto', cmap='bwr',
           extent=[x_um[0], x_um[-1], y_um[0], y_um[-1]], vmin=-vmax, vmax=vmax)
plt.hlines([y_um[y1], y_um[y2-1]], x_um[x1], x_um[x2-1], colors='black', linestyles='--')
plt.vlines([x_um[x1], x_um[x2-1]], y_um[y1], y_um[y2-1], colors='black', linestyles='--')
plt.colorbar(label='$E_y$ amplitude')
plt.xlabel('Propagation distance x (um)')
plt.ylabel('Transverse position y (um)')
plt.title('2D Snapshot of Confined Steady-State Electric Field')
plt.tight_layout()
plt.savefig('Final-Prob5-Plot2.png', dpi=300, bbox_inches='tight')

plt.show()

# 9. Plot 3: Phase Front from Steady-State Phasor
# =============================================================================

phase = np.angle(Ey_phasor)
amp = np.abs(Ey_phasor)
phase_masked = np.ma.masked_where(amp < 0.08*np.max(amp), phase)

plt.figure(figsize=(10, 4.8))
plt.imshow(phase_masked.T, origin='lower', aspect='auto', cmap='twilight',
           extent=[x_um[0], x_um[-1], y_um[0], y_um[-1]], vmin=-np.pi, vmax=np.pi)
plt.hlines([y_um[y1], y_um[y2-1]], x_um[x1], x_um[x2-1], colors='black', linestyles='--')
plt.vlines([x_um[x1], x_um[x2-1]], y_um[y1], y_um[y2-1], colors='black', linestyles='--')
plt.colorbar(label='Phase of $E_y$ phasor (rad)')
plt.xlabel('Propagation distance x (um)')
plt.ylabel('Transverse position y (um)')
plt.title('Phase Front Showing Guided Wavelength Compression in the Core')
plt.tight_layout()
plt.savefig('Final-Prob5-Plot3.png', dpi=300, bbox_inches='tight')

plt.show()

# 10. Plot 4: Cross-Sectional Time-Averaged Poynting Vector
# =============================================================================

Pz_line = Pz_avg[probe_x, :]
Pz_line = Pz_line/np.max(np.abs(Pz_line))

plt.figure(figsize=(5, 5))
plt.plot(y_um, Pz_line, label=f'$P_z$ at x = {x_um[probe_x]:.2f} um')
plt.axvspan(y_um[y1], y_um[y2-1], color='lightgray', alpha=0.6, label='Si core')
plt.xlabel('Transverse position y (um)')
plt.ylabel('Normalized time-averaged $P_z$')
plt.title('Cross-Sectional Time-Averaged Poynting Vector')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('Final-Prob5-Plot4.png', dpi=300, bbox_inches='tight')

plt.show()