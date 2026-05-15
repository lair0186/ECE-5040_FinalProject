# -*- coding: utf-8 -*-
"""
Created on Thu May 14 22:21:45 2026

@author: nsomb
"""

import numpy as np
import matplotlib.pyplot as plt
import fdtd_constants as fdtd

# 1. Physical Constants
# =============================================================================

c0 = fdtd.c0
mu0 = fdtd.mu0
eps0 = fdtd.eps0

# 2. Source and Material Parameters
# =============================================================================

f0 = 192e12
fmax = 1.2 * f0
lambda_min = c0 / fmax
lambda0 = c0 / f0
omega0 = 2.0 * np.pi * f0

n1 = 1.0    # Vacuum
n2 = 3.48   # Silicon

eps_vac = n1**2
eps_si = n2**2

#Fresnel Coefficients for Transmission and Reflection
R_ana = abs((n1 - n2) / (n1 + n2))**2
T_ana = (n2 / n1) * abs((2.0 * n1) / (n1 + n2))**2

tau = 20e-15
t0 = 6.0 * tau
source_amplitude = 1.0

# 3. Grid Parameters
# =============================================================================

dz = lambda_min / 30.0
S = 0.99
dt = S * dz / c0

Nz = 2600
Nt = 8500

src_z = 350
ref_probe_z = 650
interface_z = 1300
trans_probe_z = 1650

mur_coef = (c0 * dt - dz) / (c0 * dt + dz)

# Diagnostic Console printouts to confirm variables are properly calculated.
# print(f"lambda_min = {lambda_min * 1e6:.3f} um")
# print(f"dz = {dz * 1e9:.2f} nm")
# print(f"dt = {dt * 1e15:.3f} fs")
# print(f"Analytical R = {R_ana:.4f}")
# print(f"Analytical T = {T_ana:.4f}")
# print(f"R + T = {R_ana + T_ana:.4f}")

# 4. FDTD Simulation Function
# =============================================================================

def run_fdtd(eps_r, save_snapshot=False):
    Ez = np.zeros(Nz)
    Hy = np.zeros(Nz - 1)

    Ez_ref_probe_td = []
    Ez_trans_probe_td = []

    incident_snapshot = None
    transmitted_snapshot = None

    incident_snapshot_step = int((t0 + 0.65 * ((interface_z - src_z) * dz / c0)) / dt)

    transmitted_snapshot_step = int((t0
                                     + ((interface_z - src_z) * dz / c0)
                                     + ((trans_probe_z - interface_z) * dz * n2 / c0)
                                     + 80e-15) / dt)

    Ce = dt / (eps0 * eps_r * dz)
    Ch = dt / (mu0 * dz)

    for n in range(Nt):
        Ez_left_old = Ez[0]
        Ez_left_inner_old = Ez[1]
        Ez_right_old = Ez[-1]
        Ez_right_inner_old = Ez[-2]

        Hy[:] += Ch * (Ez[1:] - Ez[:-1])
        Ez[1:-1] += Ce[1:-1] * (Hy[1:] - Hy[:-1])

        Ez[0] = Ez_left_inner_old + mur_coef * (Ez[1] - Ez_left_old)
        Ez[-1] = Ez_right_inner_old + mur_coef * (Ez[-2] - Ez_right_old)

        t = n * dt
        pulse = source_amplitude * np.cos(omega0 * (t - t0)) * np.exp(-((t - t0) / tau)**2)
        Ez[src_z] += pulse

        Ez_ref_probe_td.append(Ez[ref_probe_z])
        Ez_trans_probe_td.append(Ez[trans_probe_z])

        if save_snapshot and n == incident_snapshot_step:
            incident_snapshot = Ez.copy()

        if save_snapshot and n == transmitted_snapshot_step:
            transmitted_snapshot = Ez.copy()

    return np.array(Ez_ref_probe_td), np.array(Ez_trans_probe_td), incident_snapshot, transmitted_snapshot

# 5. Reference and Interface Simulations
# =============================================================================

eps_ref = np.ones(Nz) * eps_vac

eps_int = np.ones(Nz) * eps_vac
eps_int[interface_z:] = eps_si

Ez_ref_probe_ref, Ez_trans_probe_ref, _, _ = run_fdtd(eps_ref)

Ez_ref_probe_int, Ez_trans_probe_int, Ez_incident_snapshot, Ez_transmitted_snapshot = run_fdtd(
    eps_int, save_snapshot=True
)

Ez_incident_td = Ez_ref_probe_ref
Ez_reflected_td = Ez_ref_probe_int - Ez_ref_probe_ref
Ez_transmitted_td = Ez_trans_probe_int

# 6. Time-Gating for Frequency-Domain Coefficients
# =============================================================================

def apply_gate(signal, half_width):
    gated = np.zeros_like(signal)
    peak_i = np.argmax(np.abs(signal))

    i1 = max(0, peak_i - half_width)
    i2 = min(len(signal), peak_i + half_width)

    gated[i1:i2] = signal[i1:i2] * np.hanning(i2 - i1)

    return gated

gate_width = 700

inc_gate = apply_gate(Ez_incident_td, gate_width)
ref_gate = apply_gate(Ez_reflected_td, gate_width)
trans_gate = apply_gate(Ez_transmitted_td, gate_width)
trans_ref_gate = apply_gate(Ez_trans_probe_ref, gate_width)

freq = np.fft.fftfreq(Nt, d=dt)

INC = np.fft.fft(inc_gate)
REF = np.fft.fft(ref_gate)
TRA = np.fft.fft(trans_gate)
TRA_REF = np.fft.fft(trans_ref_gate)

pos = freq > 0

freq_pos = freq[pos]
INC_pos = INC[pos]
REF_pos = REF[pos]
TRA_pos = TRA[pos]
TRA_REF_pos = TRA_REF[pos]

band = (freq_pos >= 180e12) & (freq_pos <= 200e12)

valid_inc = np.abs(INC_pos) > 0.02 * np.max(np.abs(INC_pos))
valid_trans = np.abs(TRA_REF_pos) > 0.02 * np.max(np.abs(TRA_REF_pos))

valid_R = band & valid_inc
valid_T = band & valid_trans

freq_R = freq_pos[valid_R]
freq_T = freq_pos[valid_T]

R_num = np.abs(REF_pos[valid_R] / INC_pos[valid_R])**2
T_num = (n2 / n1) * np.abs(TRA_pos[valid_T] / TRA_REF_pos[valid_T])**2

# 7. Plot 1: Dielectric Profile with Incident and Transmitted E-Field Snapshots
# =============================================================================

fig, ax1 = plt.subplots(figsize=(10, 8))

ax1.axvspan(0, interface_z, color='white', alpha=0.25, label='Vacuum Region')
ax1.axvspan(interface_z, Nz - 1, color='lightgrey', alpha=0.6, label='Silicon Region')

ax1.plot(eps_int, color='black', linewidth=1.5, label=r'$\epsilon_r$ Profile')
ax1.axvline(interface_z, color='black', linestyle='--', linewidth=1.2, label='Interface')

ax1.set_xlabel('Grid Index')
ax1.set_ylabel(r'Relative Permittivity $\epsilon_r$')
ax1.set_title('Dielectric Profile with Incident and Transmitted E-field Snapshots')
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(Ez_incident_snapshot, color='tab:blue', linewidth=1.1,
         label='Incident E-field Before Interface')
ax2.plot(Ez_transmitted_snapshot, color='tab:red', linewidth=1.1,
         label='Transmitted E-field Inside Silicon')
ax2.set_ylabel('Electric Field Amplitude')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
plt.savefig('Final-Prob4-Plot1.png', dpi=300, bbox_inches='tight')

plt.show()

# 8. Plot 2: Incident, Reflected, and Transmitted Time-Domain Signals
# =============================================================================

t_axis = np.arange(Nt) * dt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.plot(t_axis * 1e15, Ez_incident_td, label='Incident Signal')
ax1.set_xlabel('Time (fs)')
ax1.set_ylabel('Electric Field Amplitude')
ax1.set_title('Incident Pulse at Reflection Probe')
ax1.set_ylim(-0.6, 0.6)
ax1.grid(True, alpha=0.3)
ax1.legend()

ax2.plot(t_axis * 1e15, Ez_reflected_td, label='Reflected Signal')
ax2.plot(t_axis * 1e15, Ez_transmitted_td, label='Transmitted Signal')
ax2.set_xlabel('Time (fs)')
ax2.set_ylabel('Electric Field Amplitude')
ax2.set_title('Reflected and Transmitted Pulses After Interface Interaction')
ax2.set_ylim(-0.6, 0.6)
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig('Final-Prob4-Plot2.png', dpi=300, bbox_inches='tight')

plt.show()

# 9. Plots 3 and 4: R and T vs Frequency
# =============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.plot(freq_R / 1e12, R_num, label='FDTD Numerical R')
ax1.axhline(R_ana, color='black', linestyle='--', label=f'Fresnel R = {R_ana:.3f}')
ax1.set_xlabel('Frequency (THz)')
ax1.set_ylabel('Reflection Coefficient R')
ax1.set_title('Computed Reflection Coefficient')
ax1.set_xlim(180, 200)
ax1.set_ylim(0, 1.05)
ax1.grid(True, alpha=0.3)
ax1.legend()

ax2.plot(freq_T / 1e12, T_num, label='FDTD Numerical T')
ax2.axhline(T_ana, color='black', linestyle='--', label=f'Fresnel T = {T_ana:.3f}')
ax2.set_xlabel('Frequency (THz)')
ax2.set_ylabel('Transmission Coefficient T')
ax2.set_title('Computed Transmission Coefficient')
ax2.set_xlim(180, 200)
ax2.set_ylim(0, 1.05)
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig('Final-Prob4-Plot3&4.png', dpi=300, bbox_inches='tight')

plt.show()