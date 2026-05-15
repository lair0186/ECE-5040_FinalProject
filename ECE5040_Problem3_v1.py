# -*- coding: utf-8 -*-
"""
Problem 3: X-Band Waveguide Iris and S-Parameters
Author: nsomb
"""

import numpy as np
import matplotlib.pyplot as plt
import fdtd_constants as fdtd

# 1. Physical Constants
# =============================================================================

c0 = fdtd.c0
mu0 = fdtd.mu0
eps0 = fdtd.eps0

# 2. Waveguide and Frequency Parameters
# =============================================================================

fc = 9e9                         # Designed cutoff frequency for first mode
f_center = 10e9                  # Center of X-band
fmin_plot = 8e9
fmax_plot = 12e9

a = c0 / (2.0 * fc)              # Parallel-plate separation

# 3. Grid Parameters
# =============================================================================

fmax_grid = 12e9
lambda_min = c0 / fmax_grid

dl = lambda_min / 20.0
dt = 0.99 * dl / (c0 * np.sqrt(2.0))

Nx, Ny = 520, 70

guide_height = int(round(a / dl))
a_eff = guide_height * dl

y1 = (Ny - guide_height) // 2
y2 = y1 + guide_height

Ch = dt / (mu0 * dl)
Ce = dt / (eps0 * dl)

# 4. Source, Probe, and Time Parameters
# =============================================================================

src_x = 45

reflection_probe_x = 130
transmission_probe_x = 390

# TE1 Hz mode has a null at the centerline, so probe away from center.
probe_y = y1 + guide_height // 4

n_steps = 2600
steps_per_frame = 40

t_axis = np.arange(n_steps) * dt

# Differentiated Gaussian pulse parameters
tau = 30e-12
t0 = 6.0 * tau
source_amplitude = 1.0


# 5. Modal Source Profile
# =============================================================================

src_yvals = np.arange(y1, y2 + 1)
mode_profile = np.cos(np.pi * (src_yvals - y1) / guide_height)
mode_profile = mode_profile / np.max(np.abs(mode_profile))

# 6. Iris Geometry
# =============================================================================

iris_x = Nx // 2

# Aperture opening as a fraction of guide height.
# Smaller value = stronger blockage.
aperture_fraction = 0.45

aperture_cells = int(round(aperture_fraction * guide_height))
aperture_center = (y1 + y2) // 2

aperture_y1 = aperture_center - aperture_cells // 2
aperture_y2 = aperture_center + aperture_cells // 2

# 7. Mur ABC Parameter
# =============================================================================

mur_coef = (c0 * dt - dl) / (c0 * dt + dl)

# =============================================================================
# 8. Helper Functions
# =============================================================================

def differentiated_gaussian(t):
    """
    Broadband differentiated Gaussian source pulse.
    """
    return -((t - t0) / tau) * np.exp(-((t - t0) / tau)**2)


def build_iris_mask(include_iris):
    """
    Creates a PEC mask for the iris only.

    The top and bottom plates are enforced separately by zeroing fields outside
    the guide region.
    """
    pec_mask = np.zeros((Nx, Ny), dtype=bool)

    if include_iris:
        pec_mask[iris_x, y1:y2+1] = True
        pec_mask[iris_x, aperture_y1:aperture_y2+1] = False

    return pec_mask


def run_tez_waveguide(include_iris=False, case_name="No Iris"):
    """
    Runs the 2D TEz FDTD waveguide simulation.

    Parameters
    ----------
    include_iris : bool
        If True, inserts the PEC iris at the waveguide center.
    case_name : str
        Name used for console output.

    Returns
    -------
    result : dict
        Contains final field, stored frames, reflection/transmission probes,
        and field scaling information.
    """

    Hz = np.zeros((Nx, Ny))
    Ex = np.zeros((Nx, Ny))
    Ey = np.zeros((Nx, Ny))

    pec_mask = build_iris_mask(include_iris)

    ref_probe_td = []
    trans_probe_td = []

    Hz_frames = []
    global_max = 0.0

    print(f"\nRunning TEz FDTD Simulation: {case_name}")

    for n in range(n_steps):

        # ---------------------------------------------------------------------
        # Store old fields for Mur ABC
        # ---------------------------------------------------------------------

        Hz_left_old = Hz[0, :].copy()
        Hz_left_inner = Hz[1, :].copy()

        Hz_right_old = Hz[-1, :].copy()
        Hz_right_inner = Hz[-2, :].copy()

        # ---------------------------------------------------------------------
        # Ex Update
        # ---------------------------------------------------------------------

        Ex[:, 1:-1] += Ce * (Hz[:, 1:-1] - Hz[:, :-2])

        # ---------------------------------------------------------------------
        # Ey Update
        # ---------------------------------------------------------------------

        Ey[1:-1, :] -= Ce * (Hz[1:-1, :] - Hz[:-2, :])

        # ---------------------------------------------------------------------
        # PEC Parallel Plates
        # ---------------------------------------------------------------------

        Ex[:, :y1+1] = 0.0
        Ex[:, y2:] = 0.0

        Ey[:, :y1+1] = 0.0
        Ey[:, y2:] = 0.0

        # ---------------------------------------------------------------------
        # Hz Update
        # ---------------------------------------------------------------------

        Hz[1:-1, 1:-1] += Ch * ((Ex[1:-1, 2:] - Ex[1:-1, 1:-1])
            -(Ey[2:, 1:-1] - Ey[1:-1, 1:-1]))

        # Keep fields outside the guide zero.
        Hz[:, :y1] = 0.0
        Hz[:, y2+1:] = 0.0

        # Apply PEC iris condition.
        Ey[pec_mask] = 0.0

        # ---------------------------------------------------------------------
        # Mur ABC at Open Ends
        # ---------------------------------------------------------------------

        Hz[0, :] = Hz_left_inner + mur_coef * (Hz[1, :] - Hz_left_old)

        Hz[-1, :] = Hz_right_inner + mur_coef * (Hz[-2, :] - Hz_right_old)

        # Re-apply guide confinement after ABC.
        Hz[:, :y1] = 0.0
        Hz[:, y2+1:] = 0.0
        Hz[pec_mask] = 0.0

        # ---------------------------------------------------------------------
        # Broadband Modal Source
        # ---------------------------------------------------------------------

        t = n * dt
        source = source_amplitude * differentiated_gaussian(t)

        Hz[src_x, y1:y2+1] += source * mode_profile

        # ---------------------------------------------------------------------
        # Probe Collection using Modal Projection
        # ---------------------------------------------------------------------
        # Instead of sampling Hz at one point, project the entire transverse
        # field profile at the probe plane onto the launched mode shape.
        #
        # This gives the fundamental-mode amplitude at each probe plane.
        
        ref_line = Hz[reflection_probe_x, y1:y2+1]
        trans_line = Hz[transmission_probe_x, y1:y2+1]
        
        mode_norm = np.sum(mode_profile**2)
        
        ref_modal = np.sum(ref_line * mode_profile) / mode_norm
        trans_modal = np.sum(trans_line * mode_profile) / mode_norm
        
        ref_probe_td.append(ref_modal)
        trans_probe_td.append(trans_modal)

        # ---------------------------------------------------------------------
        # Frame Storage
        # ---------------------------------------------------------------------

        if n % steps_per_frame == 0:
            Hz_frames.append(Hz.copy())

            current_max = np.max(np.abs(Hz))
            if current_max > global_max:
                global_max = current_max

        if n % 500 == 0:
            print(f"Computed step {n}/{n_steps}")

    return {
        "Hz": Hz.copy(),
        "Hz_frames": Hz_frames,
        "ref_probe_td": np.array(ref_probe_td),
        "trans_probe_td": np.array(trans_probe_td),
        "global_max": global_max,
        "include_iris": include_iris,
        "case_name": case_name
    }


def compute_s_parameters(empty_result, test_result):
    """
    Computes S11 and S21 using empty-waveguide calibration.

    S11 denominator:
        Incident field spectrum at the reflection probe from the empty-guide run.

    S11 numerator:
        Reflected field spectrum = iris reflection probe - empty reflection probe.

    S21 denominator:
        Transmitted field spectrum at the transmission probe from the empty-guide run.

    S21 numerator:
        Transmitted field spectrum at the transmission probe from the test run.
    """
    
    # Time-domain signals from calibration and test runs
    # -------------------------------------------------------------------------

    empty_ref_td = empty_result["ref_probe_td"]          # Incident reference for S11
    empty_trans_td = empty_result["trans_probe_td"]      # Through reference for S21

    test_ref_td = test_result["ref_probe_td"]            # Total field at reflection probe
    test_trans_td = test_result["trans_probe_td"]        # Total field at transmission probe

    # Separate reflected and transmitted signals
    # -------------------------------------------------------------------------

    # Incident field at reflection probe comes from the empty-guide run.
    incident_td = empty_ref_td

    # Reflected field is the difference between iris/test and empty guide.
    reflected_td = test_ref_td - empty_ref_td

    # Transmitted field is the test transmission probe signal.
    transmitted_td = test_trans_td

    # Empty-guide transmitted field is the S21 normalization signal.
    transmitted_empty_td = empty_trans_td


    # Window signals before FFT
    # -------------------------------------------------------------------------
    
    # This reduces spectral leakage without changing the basic extraction method.
    window = np.hanning(n_steps)

    Incident_fd = np.fft.fft(incident_td * window)
    Reflected_fd = np.fft.fft(reflected_td * window)

    Transmitted_fd = np.fft.fft(transmitted_td * window)
    Transmitted_empty_fd = np.fft.fft(transmitted_empty_td * window)

    freqs = np.fft.fftfreq(n_steps, d=dt)

    # Keep only positive frequencies.
    pos = freqs > 0

    freqs_pos = freqs[pos]

    Incident_fd = Incident_fd[pos]
    Reflected_fd = Reflected_fd[pos]

    Transmitted_fd = Transmitted_fd[pos]
    Transmitted_empty_fd = Transmitted_empty_fd[pos]

    # Avoid dividing by weak reference spectrum
    # -------------------------------------------------------------------------

    incident_threshold = 5e-2 * np.max(np.abs(Incident_fd))
    transmitted_threshold = 5e-2 * np.max(np.abs(Transmitted_empty_fd))

    valid_s11 = np.abs(Incident_fd) > incident_threshold
    valid_s21 = np.abs(Transmitted_empty_fd) > transmitted_threshold

    S11 = np.full_like(Reflected_fd, np.nan, dtype=complex)
    S21 = np.full_like(Transmitted_fd, np.nan, dtype=complex)

    S11[valid_s11] = Reflected_fd[valid_s11] / Incident_fd[valid_s11]
    S21[valid_s21] = Transmitted_fd[valid_s21] / Transmitted_empty_fd[valid_s21]

    # Frequency band for plotting.
    band = (freqs_pos >= fmin_plot) & (freqs_pos <= fmax_plot)

    return {
        "freqs": freqs_pos,
        "band": band,

        # Time-domain signals for plotting
        "incident_td": incident_td,
        "reflected_td": reflected_td,
        "transmitted_td": transmitted_td,

        # Frequency-domain S-parameters
        "S11": S11,
        "S21": S21
    }


# 9. Run Simulations
# =============================================================================

empty_result = run_tez_waveguide(
    include_iris=False,
    case_name="No Iris / Empty Waveguide Calibration"
)

iris_result = run_tez_waveguide(
    include_iris=True,
    case_name="PEC Iris"
)

no_iris_sparams = compute_s_parameters(empty_result, empty_result)
iris_sparams = compute_s_parameters(empty_result, iris_result)



# 10. Plotting Functions
# =============================================================================

def plot_2d_snapshot(result, filename):
    """
    Plot 1: 2D field snapshot.
    """

    Hz = result["Hz"]
    include_iris = result["include_iris"]
    case_name = result["case_name"]

    vmax = max(np.max(np.abs(Hz)) * 0.6, 1e-12)

    plt.figure(figsize=(10, 4.5))

    plt.imshow(
        Hz.T,
        cmap="RdBu",
        origin="lower",
        vmin=-vmax,
        vmax=vmax,
        aspect="auto"
    )

    plt.colorbar(label=r"$H_z$ Field Amplitude")

    # PEC plates
    plt.hlines(y1, 0, Nx - 1, colors="black", linewidth=2)
    plt.hlines(y2, 0, Nx - 1, colors="black", linewidth=2)

    # Source and probes
    plt.vlines(src_x, y1, y2, colors="green", linestyles="--", linewidth=1.5, label="Source")
    plt.plot(reflection_probe_x, probe_y, "ko", markersize=4, label="Reflection Probe")
    plt.plot(transmission_probe_x, probe_y, "mo", markersize=4, label="Transmission Probe")

    # Iris
    if include_iris:
        plt.vlines(iris_x, y1, aperture_y1, colors="black", linewidth=4)
        plt.vlines(iris_x, aperture_y2, y2, colors="black", linewidth=4)
        plt.text(
            iris_x + 5,
            y2 + 3,
            "PEC Iris",
            fontsize=9,
            color="black"
        )

    plt.title(f"2D TEz Waveguide Field Snapshot: {case_name}")
    plt.xlabel("x (cells)")
    plt.ylabel("y (cells)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()


def plot_time_domain_signals(sparam_result, case_name, filename):
    """
    Plot 2: Reflected and transmitted time-domain signals.
    """

    reflected_td = sparam_result["reflected_td"]
    transmitted_td = sparam_result["transmitted_td"]

    plt.figure(figsize=(10, 4.5))

    plt.plot(
        t_axis * 1e9,
        reflected_td,
        label="Reflected Signal"
    )

    plt.plot(
        t_axis * 1e9,
        transmitted_td,
        label="Transmitted Signal",
        alpha=0.85
    )

    plt.title(f"Time-Domain Reflected and Transmitted Signals: {case_name}")
    plt.xlabel("Time (ns)")
    plt.ylabel(r"$H_z$ Amplitude")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()


def plot_s11(sparam_result, case_name, filename):
    """
    Plot 3: |S11| over X-band.
    """

    freqs = sparam_result["freqs"]
    band = sparam_result["band"]
    S11 = sparam_result["S11"]

    valid = band & np.isfinite(S11)

    plt.figure(figsize=(8, 4.5))

    plt.plot(
        freqs[valid] / 1e9,
        np.abs(S11[valid]),
        linewidth=2
    )

    plt.title(f"Computed |S11| Across X-Band: {case_name}")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel(r"$|S_{11}|$")
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.2)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()


def plot_s21(sparam_result, case_name, filename):
    """
    Plot 4: |S21| over X-band.
    """

    freqs = sparam_result["freqs"]
    band = sparam_result["band"]
    S21 = sparam_result["S21"]

    plt.figure(figsize=(8, 4.5))

    plt.plot(
        freqs[band] / 1e9,
        np.abs(S21[band]),
        linewidth=2
    )

    plt.title(f"Computed |S21| Across X-Band: {case_name}")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel(r"$|S_{21}|$")
    plt.grid(True, alpha=0.3)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()


# =============================================================================
# 11. Generate No-Iris Plots
# =============================================================================

# plot_2d_snapshot(
#     empty_result,
#     "Problem3_NoIris_Plot1_2D_Field.png"
# )

# plot_time_domain_signals(
#     no_iris_sparams,
#     "No Iris",
#     "Problem3_NoIris_Plot2_TimeSignals.png"
# )

# plot_s11(
#     no_iris_sparams,
#     "No Iris",
#     "Problem3_NoIris_Plot3_S11.png"
# )

# plot_s21(
#     no_iris_sparams,
#     "No Iris",
#     "Problem3_NoIris_Plot4_S21.png"
# )


# =============================================================================
# 12. Generate Iris Plots
# =============================================================================

plot_2d_snapshot(
    iris_result,
    "Final-Prob3-Plot1.png"
)

plot_time_domain_signals(
    iris_sparams,
    "PEC Iris",
    "Final-Prob3-Plot2.png"
)

plot_s11(
    iris_sparams,
    "PEC Iris",
    "Final-Prob3-Plot3.png"
)

plot_s21(
    iris_sparams,
    "PEC Iris",
    "Final-Prob3-Plot4.png"
)


# =============================================================================
# 13. Optional Diagnostic Output
# =============================================================================

print("\nSimulation Complete.")
print(f"Reflection Probe Location: x = {reflection_probe_x}, y = {probe_y}")
print(f"Transmission Probe Location: x = {transmission_probe_x}, y = {probe_y}")
print(f"Source Location: x = {src_x}")
print(f"Iris Location: x = {iris_x}")
print(f"X-band Plot Range: {fmin_plot / 1e9:.1f} to {fmax_plot / 1e9:.1f} GHz")
print("\nSaved plots:")
print("  Problem3_NoIris_Plot1_2D_Field.png")
print("  Problem3_NoIris_Plot2_TimeSignals.png")
print("  Problem3_NoIris_Plot3_S11.png")
print("  Problem3_NoIris_Plot4_S21.png")
print("  Problem3_Iris_Plot1_2D_Field.png")
print("  Problem3_Iris_Plot2_TimeSignals.png")
print("  Problem3_Iris_Plot3_S11.png")
print("  Problem3_Iris_Plot4_S21.png")




plt.figure(figsize=(8, 4.5))

freqs = iris_sparams["freqs"]
band = iris_sparams["band"]
S11 = iris_sparams["S11"]
S21 = iris_sparams["S21"]

power_sum = np.abs(S11)**2 + np.abs(S21)**2

plt.plot(freqs[band] / 1e9, power_sum[band], linewidth=2)
plt.axhline(1.0, color="black", linestyle="--", linewidth=1)

plt.title("Power Check: |S11|^2 + |S21|^2")
plt.xlabel("Frequency (GHz)")
plt.ylabel(r"$|S_{11}|^2 + |S_{21}|^2$")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()