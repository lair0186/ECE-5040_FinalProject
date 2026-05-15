# -*- coding: utf-8 -*-
"""
Problem 6: SOI Waveguide Micro-Bend Radiation

Created for ECE 404/504 Project 3

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

tau = 18e-15
t0 = 5.0*tau

n_core = 3.48               # Silicon Core
n_clad = 1.44               # Silicon Dioxide Cladding
eps_core = n_core**2
eps_clad = n_clad**2

# =============================================================================
# 3. Waveguide Geometry and Grid
# =============================================================================

height_core = 500e-9
d = height_core/2.0

bend_radius = 1000e-9
input_length = 3.0e-6
output_length = 3.0e-6

dl = 10e-9
dt = 0.99*dl/(np.sqrt(2.0)*c0)

core_height_cells = int(round(height_core/dl))
half_core_cells = core_height_cells//2

bend_radius_cells = int(round(bend_radius/dl))
input_length_cells = int(round(input_length/dl))
output_length_cells = int(round(output_length/dl))

pad_left = 120
pad_right = 160
pad_bottom = 150
pad_top = 160

x_start = pad_left
x_bend_start = x_start + input_length_cells

y_center = pad_bottom + half_core_cells

x_bend_center = x_bend_start
y_bend_center = y_center + bend_radius_cells

x_out_center = x_bend_center + bend_radius_cells
y_out_start = y_bend_center
y_out_end = y_out_start + output_length_cells

Nx = x_out_center + half_core_cells + pad_right
Ny = y_out_end + half_core_cells + pad_top

x_um = np.arange(Nx)*dl*1e6
y_um = np.arange(Ny)*dl*1e6

X, Y = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='ij')

# ------------------------------------------------------------
# Input straight waveguide arm
# ------------------------------------------------------------

input_mask = ((X >= x_start) &
              (X <= x_bend_start) &
              (np.abs(Y - y_center) <= half_core_cells))

# ------------------------------------------------------------
# Quarter-annular 90-degree bend
# ------------------------------------------------------------

r_cells = np.sqrt((X - x_bend_center)**2 + (Y - y_bend_center)**2)

bend_mask = ((r_cells >= bend_radius_cells - half_core_cells) &
             (r_cells <= bend_radius_cells + half_core_cells) &
             (X >= x_bend_center) &
             (Y <= y_bend_center))

# ------------------------------------------------------------
# Output straight waveguide arm
# ------------------------------------------------------------

output_mask = ((np.abs(X - x_out_center) <= half_core_cells) &
               (Y >= y_out_start) &
               (Y <= y_out_end))

core_mask = input_mask | bend_mask | output_mask

print("Nx, Ny =", Nx, Ny)
print("core_height_cells =", core_height_cells)
print("bend_radius_cells =", bend_radius_cells)
print("Number of core cells =", np.sum(core_mask))
print("Input mask cells =", np.sum(input_mask))
print("Bend mask cells =", np.sum(bend_mask))
print("Output mask cells =", np.sum(output_mask))

eps_r = np.full((Nx, Ny), eps_clad)
eps_r[core_mask] = eps_core

Ch = dt/(mu0*dl)
Ce = dt/(eps0*eps_r*dl)

# 4. Analytical TE0 Source Profile Using EIM
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

y_src = (np.arange(Ny) - y_center)*dl
mode_profile = np.where(np.abs(y_src) <= d,
                        np.cos(kappa*y_src),
                        np.cos(kappa*d)*np.exp(-gamma*(np.abs(y_src)-d)))
mode_profile = mode_profile/np.max(np.abs(mode_profile))

fc_m1 = 1.0/(4.0*d*np.sqrt(mu0*eps0*(n_core**2 - n_clad**2)))
lambda_g = 2*np.pi/beta

print(f"TE0 effective index n_eff = {n_eff:.4f}")
print(f"Guided wavelength lambda_g = {lambda_g*1e6:.4f} um")
print(f"m = 1 cutoff frequency = {fc_m1/1e12:.2f} THz")
print(f"Bend radius = {bend_radius*1e9:.1f} nm")
print(f"Core width = {height_core*1e9:.1f} nm")

# 5. Field Initialization and Simulation Settings
# =============================================================================

Ey = np.zeros((Nx, Ny))
Hx = np.zeros((Nx, Ny))
Hz = np.zeros((Nx, Ny))

src_x = x_start + 8

probe_out_x = x_out_center - core_height_cells//4
probe_out_y = y_out_start + int(0.70*output_length_cells)

input_power_x = x_start + 70
output_power_y = y_out_start + int(0.70*output_length_cells)

power_pad = 3*half_core_cells
input_y1 = max(y_center - power_pad, 0)
input_y2 = min(y_center + power_pad + 1, Ny)

output_x1 = max(x_out_center - power_pad, 0)
output_x2 = min(x_out_center + power_pad + 1, Nx)

n_steps = 9000
steps_per_frame = 20

mur_coef = ((c0/n_clad)*dt - dl)/((c0/n_clad)*dt + dl)

Ey_probe_out_td = []
Ey_input_td = []
Hz_input_td = []
Ey_output_td = []
Hx_output_td = []


u_sum = np.zeros((Nx, Ny))
u_count = 0

distance_to_bend = input_length
energy_start_step = int((distance_to_bend/(c0/n_eff))/dt)
energy_end_step = n_steps

snapshot_Ey = np.zeros((Nx, Ny))
snapshot_metric = 0.0
snapshot_step = 0

global_max_E = 0.0

# 6. FDTD Time-Stepping Loop
# =============================================================================

print("Running SOI 90-degree bend FDTD...")
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
    pulse = np.cos(omega0*t)*np.exp(-((t - t0)/tau)**2)
    Ey[src_x, :] = source_amplitude*pulse*mode_profile

    Ey_probe_out_td.append(Ey[probe_out_x, probe_out_y])

    Ey_input_td.append(Ey[input_power_x, input_y1:input_y2].copy())
    # Hz_input_td.append(Hz[input_power_x, input_y1:input_y2].copy())
    
    Hz_input_td.append(Hx[input_power_x, input_y1:input_y2].copy())

    Ey_output_td.append(Ey[output_x1:output_x2, output_power_y].copy())
    # Hx_output_td.append(Hx[output_x1:output_x2, output_power_y].copy())
    
    Hx_output_td.append(Hz[output_x1:output_x2, output_power_y].copy())

    if energy_start_step <= n < energy_end_step:
        u_inst = 0.5*eps0*eps_r*Ey**2 + 0.5*mu0*(Hx**2 + Hz**2)
        u_sum += u_inst
        u_count += 1

    if n >= energy_start_step and n % steps_per_frame == 0:
        bend_region = np.s_[max(x_bend_center - 40, 0):min(x_out_center + 80, Nx),
                            max(y_center - 40, 0):min(y_bend_center + 80, Ny)]
        current_metric = np.sum(np.abs(Ey[bend_region]))

        if current_metric > snapshot_metric:
            snapshot_metric = current_metric
            snapshot_Ey = Ey.copy()
            snapshot_step = n

    current_max = np.max(np.abs(Ey))
    if current_max > global_max_E:
        global_max_E = current_max

    if n % 500 == 0:
        print(f"Step {n}/{n_steps}")

u_avg = u_sum/max(u_count, 1)

Ey_probe_out_td = np.array(Ey_probe_out_td)
Ey_input_td = np.array(Ey_input_td)
Hz_input_td = np.array(Hz_input_td)
Ey_output_td = np.array(Ey_output_td)
Hx_output_td = np.array(Hx_output_td)

t_axis = np.arange(n_steps)*dt

# 7. Plot 1: 2D Snapshot of E-Field Navigating the Bend
# =============================================================================

vmax = max(global_max_E*0.25, 1e-12)

plt.figure(figsize=(8, 7))

im = plt.imshow(snapshot_Ey.T, origin='lower', aspect='equal', cmap='bwr',
                extent=[x_um[0], x_um[-1], y_um[0], y_um[-1]],
                vmin=-vmax, vmax=vmax)

if np.min(core_mask) < np.max(core_mask):
    plt.contour(x_um, y_um, core_mask.T.astype(float), levels=[0.5],
                colors='black', linewidths=1.0)

plt.colorbar(im, label='$E_y$ amplitude')
plt.xlabel('x position (um)')
plt.ylabel('y position (um)')
plt.title(f'2D E-Field Snapshot: Pulse Navigating 90-Degree Bend\nTime Step = {snapshot_step}')
plt.tight_layout()
plt.savefig('Final-Prob6-Plot1.png', dpi=300, bbox_inches='tight')

plt.show()

# 8. Plot 2: Time-Domain Signal at Transmission Probe
# =============================================================================

plt.figure(figsize=(8, 5))
plt.plot(t_axis*1e15, Ey_probe_out_td, label='Transmission probe after bend')
plt.xlabel('Time (fs)')
plt.ylabel('$E_y$ amplitude')
plt.title('Time-Domain Signal at Transmission Probe After Bend')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('Final-Prob6-Plot2.png', dpi=300, bbox_inches='tight')

plt.show()

# 9. Plot 3: Transmission Efficiency vs Frequency
# =============================================================================
Ey_input_fd = np.fft.fft(Ey_input_td, axis=0)
Hx_input_fd = np.fft.fft(Hz_input_td, axis=0)

Ey_output_fd = np.fft.fft(Ey_output_td, axis=0)
Hz_output_fd = np.fft.fft(Hx_output_td, axis=0)

freq = np.fft.fftfreq(n_steps, d=dt)
pos = freq > 0

P_in_fd = np.sum(-0.5*np.real(Ey_input_fd*np.conj(Hx_input_fd)), axis=1)*dl
P_out_fd = np.sum(0.5*np.real(Ey_output_fd*np.conj(Hz_output_fd)), axis=1)*dl



freq_pos = freq[pos]
P_in_pos = P_in_fd[pos]
P_out_pos = P_out_fd[pos]

P_in_mag = np.abs(P_in_pos)
P_out_mag = np.abs(P_out_pos)

T_freq_diag = np.abs(P_out_pos)/np.maximum(np.abs(P_in_pos), 1e-30)

input_threshold = 0.05*np.max(P_in_mag)
valid = P_in_mag > input_threshold

T_freq = np.full_like(freq_pos, np.nan, dtype=float)
T_freq[valid] = P_out_mag[valid]/P_in_mag[valid]

band = (freq_pos >= 180e12) & (freq_pos <= 200e12) & valid

plt.figure(figsize=(8, 5))
plt.plot(freq_pos[band]/1e12, T_freq[band], 'o-', label='$P_{out}/P_{in}$')
plt.axhline(1.0, color='black', linestyle='--', linewidth=1.0, label='100% limit')
plt.xlabel('Frequency (THz)')
plt.ylabel('Transmission Efficiency')
plt.title('Transmission Efficiency Through 90-Degree SOI Bend')
plt.ylim(((0,1)))
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('Final-Prob6-Plot3.png', dpi=300, bbox_inches='tight')

plt.show()


# #------- diagnostic plot -----
# plt.figure(figsize=(8, 5))
# plt.plot(freq_pos/1e12, np.abs(P_in_pos), 'o-', label='$|P_{in}(f)|$')
# plt.plot(freq_pos/1e12, np.abs(P_out_pos), 'o-', label='$|P_{out}(f)|$')
# plt.xlim(180, 200)
# plt.xlabel('Frequency (THz)')
# plt.ylabel('Power Spectrum Magnitude')
# plt.title('Input and Output Power Spectra')
# plt.grid(True, alpha=0.3)
# plt.legend()
# plt.tight_layout()
# plt.show()


# 10. Plot 4: Time-Averaged EM Energy Density in Bend Region
# =============================================================================

bend_pad = 80

x_min = max(min(x_bend_center, x_out_center) - bend_pad, 0)
x_max = min(max(x_bend_center, x_out_center) + bend_pad, Nx)

y_min = max(min(y_center, y_bend_center) - bend_pad, 0)
y_max = min(max(y_center, y_bend_center) + bend_pad, Ny)

u_bend = u_avg[x_min:x_max, y_min:y_max]
core_bend = core_mask[x_min:x_max, y_min:y_max]

print("u_bend shape =", u_bend.shape)
print("u_bend min/max =", np.min(u_bend), np.max(u_bend))
print("core_bend cells =", np.sum(core_bend))

plt.figure(figsize=(7, 6))

im = plt.imshow(u_bend.T, origin='lower', aspect='equal',
                extent=[x_um[x_min], x_um[x_max-1],
                        y_um[y_min], y_um[y_max-1]])

if np.min(core_bend) < np.max(core_bend):
    plt.contour(x_um[x_min:x_max], y_um[y_min:y_max],
                core_bend.T.astype(float), levels=[0.5],
                colors='white', linewidths=1.0)

plt.colorbar(im, label='Time-Averaged EM Energy Density (J/m$^3$)')
plt.xlabel('x position (um)')
plt.ylabel('y position (um)')
plt.title('Time-Averaged EM Energy Density in Bend Region')
plt.tight_layout()
plt.savefig('Final-Prob6-Plot4.png', dpi=300, bbox_inches='tight')

plt.show()