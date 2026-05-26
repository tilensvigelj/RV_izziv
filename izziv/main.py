import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import fft, fftfreq
from scipy.stats import skew, kurtosis
from scipy.ndimage import uniform_filter1d
import argparse
import traceback
import sys
import time

'''docker run \
  -v /media/FastDataMama/data_rv_26:/data \
  -v /media/FastDataMama/tilens/izziv:/output \
  -v /media/FastDataMama/tilens/izziv:/calibration \
  hand-tracking'''




# -----------------------------------------------------------------------------
# Poti
# -----------------------------------------------------------------------------
DATA_DIR = Path("/data")
OUTPUT_DIR = Path("/output")
CALIBRATION_FILE = Path("/calibration/calibration.npz")

# -----------------------------------------------------------------------------
# MediaPipe
# -----------------------------------------------------------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
LANDMARK_NAMES = [lm.name for lm in mp.solutions.hands.HandLandmark]

CAMERA_ID = "camP_1"


# =============================================================================
# Kalibracijske funkcije
# =============================================================================

def load_calibration(cam_name: str = "mid"):
    """Naloži kalibracijske matrike za izbrano kamero."""
    if not CALIBRATION_FILE.exists():
        print("Kalibracijska datoteka ni najdena — nadaljujem brez kalibracije.")
        return None, None

    data = np.load(str(CALIBRATION_FILE))
    K = data[f"{cam_name}_K"]
    dist = data[f"{cam_name}_dist"]
    print(f"Kalibracija nalozena ({cam_name}): fx={K[0,0]:.1f}, fy={K[1,1]:.1f}")
    return K, dist


def undistort_frame(frame, K, dist):
    """Popravi distorzijo lece."""
    if K is None:
        return frame
    h, w = frame.shape[:2]
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    return cv2.undistort(frame, K, dist, None, new_K)


def px_to_mm(speed_px_s: float, K, depth_mm: float = 500.0):
    """Pretvori hitrost iz px/s v mm/s (privzeta razdalja roke 500 mm)."""
    if K is None:
        return None
    fx = K[0, 0]
    return speed_px_s * depth_mm / fx


# =============================================================================
# Ekstrakcija znacilk iz casovne serije (robustna)
# =============================================================================

def extract_timeseries_features_robust(df: pd.DataFrame, fps: float, unit: str = "mm", 
                                        t_start: float = None, t_end: float = None):
    """
    Iz casovne serije hitrosti (WRIST) izracuna dodatne znacilke z robustno detekcijo.
    Uporablja filtriranje cez 3 zaporedne vzorce za potrditev vrhov.
    
    Parametri:
    ----------
    t_start : float, optional
        Zacetek intervala v sekundah (npr. 3.0)
    t_end : float, optional
        Konec intervala v sekundah (npr. 5.0)
    
    Vrne (features_dict, plot_data_dict).
    """
    if df.empty or "WRIST_speed_mm_s" not in df.columns:
        return {}, {}
    
    speed_all = df["WRIST_speed_mm_s"].dropna().values
    time_all = df["time_s"].dropna().values[:len(speed_all)]
    
    # Filtriranje po casovnem intervalu
    if t_start is not None and t_end is not None:
        mask = (time_all >= t_start) & (time_all <= t_end)
        speed = speed_all[mask]
        time = time_all[mask]
        if len(speed) > 0:
            print(f"  Analiza intervala: {t_start}-{t_end} sekund (st. vzorcev: {len(speed)})")
    else:
        speed = speed_all
        time = time_all
        print(f"  Analiza celotnega posnetka (st. vzorcev: {len(speed)})")
    
    if len(speed) < 10:
        print(f"  Premalo podatkov za analizo (potrebnih vsaj 10 vzorcev, najdenih {len(speed)})")
        return {}, {}
    
    dt = np.mean(np.diff(time)) if len(time) > 1 else 1.0 / fps
    
    # Glajenje signala (moving average cez 3 tocke)
    speed_smoothed = uniform_filter1d(speed, size=3, mode='nearest')
    
    # Pospesek in jerk na glajenem signalu
    accel = np.gradient(speed_smoothed, dt)
    jerk = np.gradient(accel, dt)
    
    # ========================================================================
    # 1. ZNACILKE HITROSTI
    # ========================================================================
    speed_mean = np.mean(speed_smoothed)
    speed_std = np.std(speed_smoothed)
    speed_max = np.max(speed_smoothed)
    speed_min = np.min(speed_smoothed)
    speed_median = np.median(speed_smoothed)
    speed_q25 = np.percentile(speed_smoothed, 25)
    speed_q75 = np.percentile(speed_smoothed, 75)
    speed_iqr = speed_q75 - speed_q25
    speed_cv = speed_std / speed_mean if speed_mean > 0 else 0
    speed_skew = skew(speed_smoothed)
    speed_kurt = kurtosis(speed_smoothed)
    speed_rms = np.sqrt(np.mean(speed_smoothed ** 2))
    
    # ========================================================================
    # 2. ROBUSTNA DETEKCIJA VRHOV (filter 3 zaporedne tocke)
    # ========================================================================
    min_distance_samples = max(3, int(0.5 / dt))
    height_threshold = np.percentile(speed_smoothed, 50)
    
    peaks, peak_props = find_peaks(speed_smoothed, 
                                   height=height_threshold,
                                   distance=min_distance_samples,
                                   width=1,
                                   prominence=np.std(speed_smoothed) * 0.5)
    
    # Filtriranje vrhov
    filtered_peaks = []
    filtered_widths = []
    filtered_prominences = []
    
    for i, p in enumerate(peaks):
        if p < 1 or p > len(speed_smoothed) - 2:
            continue
        
        left_check = all(speed_smoothed[p] > speed_smoothed[p - j] for j in range(1, min(3, p+1)))
        right_check = all(speed_smoothed[p] > speed_smoothed[p + j] for j in range(1, min(3, len(speed_smoothed)-p)))
        
        if left_check and right_check:
            filtered_peaks.append(p)
            if 'widths' in peak_props and i < len(peak_props['widths']):
                filtered_widths.append(peak_props['widths'][i])
            if 'prominences' in peak_props and i < len(peak_props['prominences']):
                filtered_prominences.append(peak_props['prominences'][i])
    
    n_peaks = len(filtered_peaks)
    peaks_arr = np.array(filtered_peaks) if filtered_peaks else np.array([])
    
    peak_magnitudes = [speed_smoothed[p] for p in peaks_arr] if n_peaks > 0 else []
    peak_mean_magnitude = np.mean(peak_magnitudes) if n_peaks > 0 else 0
    peak_max_magnitude = np.max(peak_magnitudes) if n_peaks > 0 else 0
    peak_std_magnitude = np.std(peak_magnitudes) if n_peaks > 0 else 0
    
    if n_peaks > 0 and filtered_widths:
        widths = np.array(filtered_widths) * dt
        peak_mean_duration = np.mean(widths)
        peak_max_duration = np.max(widths)
        peak_total_duration = np.sum(widths)
    else:
        peak_mean_duration = peak_max_duration = peak_total_duration = 0
    
    high_threshold = np.percentile(speed_smoothed, 75)
    high_peaks = [p for p in peaks_arr if speed_smoothed[p] > high_threshold]
    n_high_peaks = len(high_peaks)
    high_peak_mean = np.mean([speed_smoothed[p] for p in high_peaks]) if n_high_peaks > 0 else 0
    
    time_to_first_peak = time[peaks_arr[0]] if n_peaks > 0 else np.nan
    time_to_last_peak = time[peaks_arr[-1]] if n_peaks > 0 else np.nan
    peak_duration_span = time_to_last_peak - time_to_first_peak if n_peaks > 1 else 0
    
    # ========================================================================
    # 3. ZNACILKE POSPESKA
    # ========================================================================
    accel_abs = np.abs(accel)
    accel_mean = np.mean(accel_abs)
    accel_std = np.std(accel_abs)
    accel_max = np.max(accel_abs)
    accel_median = np.median(accel_abs)
    accel_q95 = np.percentile(accel_abs, 95)
    
    accel_peaks_raw, accel_props = find_peaks(accel_abs, 
                                               height=np.percentile(accel_abs, 75),
                                               distance=int(0.3 / dt),
                                               width=1)
    
    filtered_accel_peaks = []
    filtered_accel_widths = []
    for i, p in enumerate(accel_peaks_raw):
        if p < 1 or p > len(accel_abs) - 2:
            continue
        left_check = all(accel_abs[p] > accel_abs[p - j] for j in range(1, min(3, p+1)))
        right_check = all(accel_abs[p] > accel_abs[p + j] for j in range(1, min(3, len(accel_abs)-p)))
        if left_check and right_check:
            filtered_accel_peaks.append(p)
            if 'widths' in accel_props and i < len(accel_props['widths']):
                filtered_accel_widths.append(accel_props['widths'][i])
    
    n_accel_peaks = len(filtered_accel_peaks)
    accel_peak_magnitudes = [accel_abs[p] for p in filtered_accel_peaks] if n_accel_peaks > 0 else []
    accel_peak_mean = np.mean(accel_peak_magnitudes) if n_accel_peaks > 0 else 0
    accel_peak_max = np.max(accel_peak_magnitudes) if n_accel_peaks > 0 else 0
    
    if n_accel_peaks > 0 and filtered_accel_widths:
        accel_widths = np.array(filtered_accel_widths) * dt
        accel_peak_mean_duration = np.mean(accel_widths)
        accel_peak_total_duration = np.sum(accel_widths)
    else:
        accel_peak_mean_duration = accel_peak_total_duration = 0
    
    accel_pos = accel[accel > 0]
    accel_neg = np.abs(accel[accel < 0])
    accel_pos_mean = np.mean(accel_pos) if len(accel_pos) > 0 else 0
    accel_neg_mean = np.mean(accel_neg) if len(accel_neg) > 0 else 0
    accel_balance = accel_pos_mean / (accel_neg_mean + 1e-6)
    
    # ========================================================================
    # 4. ZNACILKE JERKA
    # ========================================================================
    jerk_abs = np.abs(jerk)
    jerk_mean = np.mean(jerk_abs)
    jerk_std = np.std(jerk_abs)
    jerk_max = np.max(jerk_abs)
    jerk_median = np.median(jerk_abs)
    jerk_q95 = np.percentile(jerk_abs, 95)
    
    jerk_peaks_raw, jerk_props = find_peaks(jerk_abs, 
                                             height=np.percentile(jerk_abs, 75),
                                             distance=int(0.2 / dt),
                                             width=1)
    
    filtered_jerk_peaks = []
    filtered_jerk_widths = []
    for i, p in enumerate(jerk_peaks_raw):
        if p < 1 or p > len(jerk_abs) - 2:
            continue
        left_check = all(jerk_abs[p] > jerk_abs[p - j] for j in range(1, min(3, p+1)))
        right_check = all(jerk_abs[p] > jerk_abs[p + j] for j in range(1, min(3, len(jerk_abs)-p)))
        if left_check and right_check:
            filtered_jerk_peaks.append(p)
            if 'widths' in jerk_props and i < len(jerk_props['widths']):
                filtered_jerk_widths.append(jerk_props['widths'][i])
    
    n_jerk_peaks = len(filtered_jerk_peaks)
    jerk_peak_magnitudes = [jerk_abs[p] for p in filtered_jerk_peaks] if n_jerk_peaks > 0 else []
    jerk_peak_mean = np.mean(jerk_peak_magnitudes) if n_jerk_peaks > 0 else 0
    jerk_peak_max = np.max(jerk_peak_magnitudes) if n_jerk_peaks > 0 else 0
    
    if n_jerk_peaks > 0 and filtered_jerk_widths:
        jerk_widths = np.array(filtered_jerk_widths) * dt
        jerk_peak_mean_duration = np.mean(jerk_widths)
        jerk_peak_total_duration = np.sum(jerk_widths)
    else:
        jerk_peak_mean_duration = jerk_peak_total_duration = 0
    
    # ========================================================================
    # 5. FREKVENCNE ZNACILKE
    # ========================================================================
    n = len(speed_smoothed)
    if n > 10:
        fft_vals = fft(speed_smoothed - speed_mean)
        fft_mag = np.abs(fft_vals[:n//2])
        freqs = fftfreq(n, dt)[:n//2]
        
        if len(freqs) > 1 and np.sum(fft_mag[1:]) > 0:
            dom_freq_idx = np.argmax(fft_mag[1:]) + 1
            dominant_freq = freqs[dom_freq_idx] if dom_freq_idx < len(freqs) else 0
            spectral_energy = np.sum(fft_mag ** 2)
            mean_freq = np.sum(freqs * fft_mag) / np.sum(fft_mag) if np.sum(fft_mag) > 0 else 0
            
            cum_energy = np.cumsum(fft_mag)
            median_freq_idx = np.searchsorted(cum_energy, cum_energy[-1] / 2)
            median_freq = freqs[median_freq_idx] if median_freq_idx < len(freqs) else 0
        else:
            dominant_freq = mean_freq = median_freq = 0
            spectral_energy = 0
    else:
        dominant_freq = mean_freq = median_freq = 0
        spectral_energy = 0
    
    # ========================================================================
    # 6. CASOVNE ZNACILKE
    # ========================================================================
    speed_norm = speed_smoothed - speed_mean
    zero_crossings = np.sum(np.diff(np.sign(speed_norm)) != 0)
    mean_abs_diff = np.mean(np.abs(np.diff(speed_smoothed)))
    peak_to_avg_ratio = speed_max / speed_mean if speed_mean > 0 else 0
    time_to_max_speed = time[np.argmax(speed_smoothed)] if len(time) > 0 else np.nan
    time_from_max_to_end = time[-1] - time_to_max_speed if not np.isnan(time_to_max_speed) else np.nan
    
    try:
        area_under_curve = np.trapezoid(speed_smoothed, time)
    except AttributeError:
        area_under_curve = np.trapz(speed_smoothed, time)
    
    # ========================================================================
    # Zbrane znacilke
    # ========================================================================
    features = {
        "speed_mean_mm_s": speed_mean,
        "speed_std_mm_s": speed_std,
        "speed_max_mm_s": speed_max,
        "speed_min_mm_s": speed_min,
        "speed_median_mm_s": speed_median,
        "speed_q25_mm_s": speed_q25,
        "speed_q75_mm_s": speed_q75,
        "speed_iqr_mm_s": speed_iqr,
        "speed_cv": speed_cv,
        "speed_skew": speed_skew,
        "speed_kurtosis": speed_kurt,
        "speed_rms_mm_s": speed_rms,
        "n_peaks": n_peaks,
        "n_high_peaks": n_high_peaks,
        "peak_mean_magnitude_mm_s": peak_mean_magnitude,
        "peak_max_magnitude_mm_s": peak_max_magnitude,
        "peak_std_magnitude_mm_s": peak_std_magnitude,
        "high_peak_mean_mm_s": high_peak_mean,
        "peak_mean_duration_s": peak_mean_duration,
        "peak_max_duration_s": peak_max_duration,
        "peak_total_duration_s": peak_total_duration,
        "peak_duration_span_s": peak_duration_span,
        "time_to_first_peak_s": time_to_first_peak,
        "time_to_last_peak_s": time_to_last_peak,
        "accel_mean_mm_s2": accel_mean,
        "accel_std_mm_s2": accel_std,
        "accel_max_mm_s2": accel_max,
        "accel_median_mm_s2": accel_median,
        "accel_q95_mm_s2": accel_q95,
        "n_accel_peaks": n_accel_peaks,
        "accel_peak_mean_mm_s2": accel_peak_mean,
        "accel_peak_max_mm_s2": accel_peak_max,
        "accel_peak_mean_duration_s": accel_peak_mean_duration,
        "accel_peak_total_duration_s": accel_peak_total_duration,
        "accel_pos_mean_mm_s2": accel_pos_mean,
        "accel_neg_mean_mm_s2": accel_neg_mean,
        "accel_balance_ratio": accel_balance,
        "jerk_mean_mm_s3": jerk_mean,
        "jerk_std_mm_s3": jerk_std,
        "jerk_max_mm_s3": jerk_max,
        "jerk_median_mm_s3": jerk_median,
        "jerk_q95_mm_s3": jerk_q95,
        "n_jerk_peaks": n_jerk_peaks,
        "jerk_peak_mean_mm_s3": jerk_peak_mean,
        "jerk_peak_max_mm_s3": jerk_peak_max,
        "jerk_peak_mean_duration_s": jerk_peak_mean_duration,
        "jerk_peak_total_duration_s": jerk_peak_total_duration,
        "dominant_freq_hz": dominant_freq,
        "median_freq_hz": median_freq,
        "mean_freq_hz": mean_freq,
        "spectral_energy": spectral_energy,
        "zero_crossings": zero_crossings,
        "mean_abs_diff_mm_s": mean_abs_diff,
        "peak_to_avg_ratio": peak_to_avg_ratio,
        "time_to_max_speed_s": time_to_max_speed,
        "time_from_max_to_end_s": time_from_max_to_end,
        "area_under_curve_mm": area_under_curve,
    }
    
    plot_data = {
        'time': time,
        'speed_smoothed': speed_smoothed,
        'peaks': peaks_arr,
        'peak_widths': filtered_widths,
        'accel_abs': accel_abs,
        'accel_peaks': np.array(filtered_accel_peaks) if filtered_accel_peaks else np.array([]),
        'jerk_abs': jerk_abs,
        'jerk_peaks': np.array(filtered_jerk_peaks) if filtered_jerk_peaks else np.array([])
    }
    
    return features, plot_data


# =============================================================================
# Plotly graf z vsemi sklepi (enote v metrih)
# =============================================================================

def save_speed_plot(df: pd.DataFrame, output_dir: Path, video_name: str, features: dict = None, plot_data: dict = None):
    """
    Izrise interaktivni Plotly graf s hitrostmi, pospeski in akumulirano potjo za vse sklepe.
    Opcijsko oznaci vrhove in njihovo trajanje (ce so podani features in plot_data).
    Enote so v metrih (m, m/s, m/s²).
    """
    # Pretvori enote iz mm v m
    use_mm = any("_speed_mm_s" in col for col in df.columns)
    speed_suffix = "_speed_mm_s" if use_mm else "_speed_px_s"
    accel_suffix = "_accel_mm_s2" if use_mm else "_accel_px_s2"
    path_suffix = "_path_mm" if use_mm else "_path_px"
    
    # Enote v metrih
    speed_unit = "m/s" if use_mm else "px/s"
    accel_unit = "m/s²" if use_mm else "px/s²"
    path_unit = "m" if use_mm else "px"
    
    # Pretvorba faktorja (mm -> m)
    conv_factor = 0.001 if use_mm else 1.0
    
    joints = [col.replace(speed_suffix, "") for col in df.columns if col.endswith(speed_suffix)]
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[
            f"Hitrost [{speed_unit}]",
            f"Pospesek [{accel_unit}]",
            f"Akumulirana pot [{path_unit}]",
        ],
        shared_xaxes=True,
        vertical_spacing=0.08,
    )
    
    # Dodaj podatke za vsak sklep
    for joint in joints:
        visible = True if joint == "WRIST" else "legendonly"
        
        # Hitrost (pretvorjena v metre)
        if f"{joint}{speed_suffix}" in df.columns:
            speed_data = df[f"{joint}{speed_suffix}"].values * conv_factor
            fig.add_trace(go.Scatter(
                x=df["time_s"], y=speed_data,
                name=joint, mode="lines", line=dict(width=1.5),
                visible=visible, legendgroup=joint,
            ), row=1, col=1)
        
        # Pospesek (pretvorjen v metre)
        if f"{joint}{accel_suffix}" in df.columns:
            accel_data = df[f"{joint}{accel_suffix}"].values * conv_factor
            fig.add_trace(go.Scatter(
                x=df["time_s"], y=accel_data,
                name=joint, mode="lines", line=dict(width=1.5),
                visible=visible, legendgroup=joint, showlegend=False,
            ), row=2, col=1)
        
        # Pot (pretvorjena v metre)
        if f"{joint}{path_suffix}" in df.columns:
            path_data = df[f"{joint}{path_suffix}"].values * conv_factor
            fig.add_trace(go.Scatter(
                x=df["time_s"], y=path_data,
                name=joint, mode="lines", line=dict(width=1.5),
                visible=visible, legendgroup=joint, showlegend=False,
            ), row=3, col=1)
    
    # Označi vrhove in trajanje (samo za WRIST, ce so podani)
    if features is not None and plot_data is not None:
        if 'peaks' in plot_data and len(plot_data['peaks']) > 0:
            peak_times = plot_data['time'][plot_data['peaks']]
            peak_values = plot_data['speed_smoothed'][plot_data['peaks']] * conv_factor
            
            fig.add_trace(go.Scatter(
                x=peak_times, y=peak_values,
                mode='markers', name=f'Vrhovi (n={len(peak_times)})',
                marker=dict(color='red', size=10, symbol='triangle-down',
                           line=dict(color='darkred', width=1)),
                legendgroup='peaks',
            ), row=1, col=1)
            
            # Označi trajanje vrhov
            if 'peak_widths' in plot_data and len(plot_data['peak_widths']) == len(plot_data['peaks']):
                dt_est = plot_data['time'][1] - plot_data['time'][0] if len(plot_data['time']) > 1 else 0.033
                for i, (p_idx, width_samples) in enumerate(zip(plot_data['peaks'], plot_data['peak_widths'])):
                    width_seconds = width_samples * dt_est
                    peak_height = plot_data['speed_smoothed'][p_idx] * conv_factor
                    
                    fig.add_annotation(
                        x=plot_data['time'][p_idx], y=peak_height,
                        text=f'{width_seconds:.2f}s',
                        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1,
                        arrowcolor='red', ax=20, ay=-30,
                        font=dict(size=9, color='red'),
                        row=1, col=1
                    )
        
        # Označi sunke (vrhove pospeska)
        if 'accel_peaks' in plot_data and len(plot_data['accel_peaks']) > 0:
            accel_peak_times = plot_data['time'][plot_data['accel_peaks']]
            accel_peak_values = plot_data['accel_abs'][plot_data['accel_peaks']] * conv_factor
            
            fig.add_trace(go.Scatter(
                x=accel_peak_times, y=accel_peak_values,
                mode='markers', name=f'Sunki (n={len(accel_peak_times)})',
                marker=dict(color='darkgreen', size=8, symbol='triangle-up',
                           line=dict(color='black', width=0.5)),
                legendgroup='peaks',
            ), row=2, col=1)
        
        # Označi vrhove jerka
        if 'jerk_peaks' in plot_data and len(plot_data['jerk_peaks']) > 0:
            jerk_peak_times = plot_data['time'][plot_data['jerk_peaks']]
            jerk_peak_values = plot_data['jerk_abs'][plot_data['jerk_peaks']] * conv_factor
            
            fig.add_trace(go.Scatter(
                x=jerk_peak_times, y=jerk_peak_values,
                mode='markers', name=f'Jerk vrhovi (n={len(jerk_peak_times)})',
                marker=dict(color='purple', size=8, symbol='diamond',
                           line=dict(color='black', width=0.5)),
                legendgroup='peaks',
            ), row=2, col=1)
    
    # Izracunaj in dodaj statistiko v zgornji desni kot
    if features is not None:
        stats_text = f"<b>Statistika gibanja (zapestje)</b><br>"
        stats_text += f"Povprecna hitrost: {features.get('speed_mean_mm_s', 0) * conv_factor:.2f} m/s<br>"
        stats_text += f"Maksimalna hitrost: {features.get('speed_max_mm_s', 0) * conv_factor:.2f} m/s<br>"
        stats_text += f"Stevilo vrhov: {features.get('n_peaks', 0)}<br>"
        stats_text += f"Povp. trajanje vrha: {features.get('peak_mean_duration_s', 0):.3f} s<br>"
        stats_text += f"Stevilo sunkov: {features.get('n_accel_peaks', 0)}<br>"
        stats_text += f"Dominantna frekvenca: {features.get('dominant_freq_hz', 0):.2f} Hz"
        
        fig.add_annotation(
            x=0.98, y=0.98, xref="paper", yref="paper",
            text=stats_text, showarrow=False, font=dict(size=10, color="white"),
            align="left", bgcolor="rgba(0,0,0,0.7)", bordercolor="black", borderwidth=1,
            xanchor="right", yanchor="top"
        )
    
    # Posodobi izgled
    fig.update_layout(
        title=f"Kinematika sklepov — {video_name}",
        xaxis3_title="Cas [s]",
        legend=dict(
            title="Sklepi",
            itemclick="toggle",
            itemdoubleclick="toggleothers",
            orientation="v",
            yanchor="top", y=0.99, xanchor="left", x=1.02
        ),
        hovermode="x unified",
        template="plotly_white",
        height=1000,
        margin=dict(r=200)
    )
    
    fig.update_yaxes(title_text=f"Hitrost [{speed_unit}]", row=1, col=1)
    fig.update_yaxes(title_text=f"Pospesek [{accel_unit}]", row=2, col=1)
    fig.update_yaxes(title_text=f"Pot [{path_unit}]", row=3, col=1)
    fig.update_xaxes(title_text="Cas [s]", row=3, col=1)
    
    plot_path = output_dir / (video_name + "_kinematics.html")
    fig.write_html(str(plot_path))
    print(f"  → Interaktivni graf: {plot_path}")


# =============================================================================
# Batch funkcije za zbiranje znacilk in histogramov
# =============================================================================

def collect_batch_features(df: pd.DataFrame, patient_id: str, output_dir: Path,
                           t_start: float = None, t_end: float = None) -> tuple:
    """
    Zbere vse znacilke za enega pacienta v batch nacinu.
    Vrne (features_dict, frames_list) kjer frames_list vsebuje podatke za vsak frame.
    """
    if df.empty or "WRIST_speed_mm_s" not in df.columns:
        return {}, []
    
    fps = 30.0
    features, _ = extract_timeseries_features_robust(df, fps, unit="mm", 
                                                      t_start=t_start, t_end=t_end)
    
    # Zberi podatke za vsak frame
    frames_data = []
    joints = ['WRIST', 'THUMB_TIP', 'INDEX_FINGER_TIP'] 
    
    for idx, row in df.iterrows():
        frame_record = {
            'patient': patient_id,
            'frame': row.get('frame', idx),
            'time_s': row.get('time_s', np.nan),
        }
        for joint in joints:
            speed_col = f"{joint}_speed_mm_s"
            accel_col = f"{joint}_accel_mm_s2"
            path_col = f"{joint}_path_mm"
            if speed_col in row:
                frame_record[f"{joint.lower()}_speed_mm_s"] = row[speed_col]
            if accel_col in row:
                frame_record[f"{joint.lower()}_accel_mm_s2"] = row[accel_col]
            if path_col in row:
                frame_record[f"{joint.lower()}_path_mm"] = row[path_col]
        frames_data.append(frame_record)
    
    # Dodaj metrike v features
    if features:
        if "time_s" in df.columns:
            if t_start is not None and t_end is not None:
                mask = (df["time_s"] >= t_start) & (df["time_s"] <= t_end)
                features["analysis_interval_s"] = f"{t_start}-{t_end}"
                features["frames_in_interval"] = mask.sum()
            features["video_duration_s"] = round(df["time_s"].max(), 2)
        
        features["frames_with_hand"] = len(df)
        features["patient"] = patient_id
        
        if "speed_mean_mm_s" not in features and speed_col:
            features["speed_mean_mm_s"] = df[speed_col].mean()
        if "speed_max_mm_s" not in features and speed_col:
            features["speed_max_mm_s"] = df[speed_col].max()
    
    return features, frames_data


def save_batch_features_summary(all_features: list, all_frames_data: list, output_dir: Path):
    """Shrani zbrane znacilke v CSV in naredi histogram uspesnosti."""
    
    # Shrani znacilke
    if all_features:
        df_features = pd.DataFrame(all_features)
        features_path = output_dir / "batch_features_all.csv"
        df_features.to_csv(features_path, index=False)
        print(f"\n→ Znacilke shranjene: {features_path}")
    
    # Shrani vse frame podatke v eno CSV datoteko
    if all_frames_data:
        df_all_frames = pd.DataFrame(all_frames_data)
        frames_output_path = output_dir / "all_patients_frames_data.csv"
        df_all_frames.to_csv(frames_output_path, index=False)
        print(f"\n→ Skupna CSV datoteka z vsemi frami: {frames_output_path}")
        print(f"  Stevilo pacientov: {df_all_frames['patient'].nunique()}")
        print(f"  Skupno stevilo framov: {len(df_all_frames)}")
        
        # Shrani tudi po pacientih loceno
        for patient in df_all_frames['patient'].unique():
            patient_frames = df_all_frames[df_all_frames['patient'] == patient]
            patient_frames_path = output_dir / patient / "frames_data.csv"
            patient_frames.to_csv(patient_frames_path, index=False)
        print(f"  → Podatki po pacientih shranjeni v podmapah")
    
    # Histogrami
    if all_features:
        df_features = pd.DataFrame(all_features)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        if "processing_time_s" in df_features.columns:
            axes[0, 0].hist(df_features["processing_time_s"].dropna(), bins=20, color='steelblue', edgecolor='black')
            axes[0, 0].set_xlabel('Cas obdelave (s)')
            axes[0, 0].set_ylabel('Stevilo pacientov')
            axes[0, 0].set_title('Porazdelitev casa obdelave')
            axes[0, 0].grid(True, alpha=0.3)
        
        if "frames_with_hand" in df_features.columns:
            axes[0, 1].hist(df_features["frames_with_hand"].dropna(), bins=20, color='coral', edgecolor='black')
            axes[0, 1].set_xlabel('Stevilo okvirjev z roko')
            axes[0, 1].set_ylabel('Stevilo pacientov')
            axes[0, 1].set_title('Porazdelitev zaznanih okvirjev')
            axes[0, 1].grid(True, alpha=0.3)
        
        if "speed_mean_mm_s" in df_features.columns:
            axes[1, 0].hist(df_features["speed_mean_mm_s"].dropna(), bins=20, color='seagreen', edgecolor='black')
            axes[1, 0].set_xlabel('Povprecna hitrost (mm/s)')
            axes[1, 0].set_ylabel('Stevilo pacientov')
            axes[1, 0].set_title('Porazdelitev povprecne hitrosti')
            axes[1, 0].grid(True, alpha=0.3)
        
        if "n_peaks" in df_features.columns:
            axes[1, 1].hist(df_features["n_peaks"].dropna(), bins=20, color='purple', edgecolor='black')
            axes[1, 1].set_xlabel('Stevilo vrhov')
            axes[1, 1].set_ylabel('Stevilo pacientov')
            axes[1, 1].set_title('Porazdelitev stevila vrhov')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Histogrami uspesnosti procesiranja in kljucnih parametrov', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'batch_processing_histograms.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"→ Histogrami: {output_dir / 'batch_processing_histograms.png'}")
        
        # Statistika
        print("\n" + "=" * 60)
        print("STATISTIKA BATCH PROCESIRANJA")
        print("=" * 60)
        print(f"Stevilo uspesno obdelanih pacientov: {len(df_features)}")
        
        if "processing_time_s" in df_features.columns:
            print(f"Povprecen cas obdelave: {df_features['processing_time_s'].mean():.2f} s")
            print(f"Skupen cas obdelave: {df_features['processing_time_s'].sum():.2f} s")
        
        if "frames_with_hand" in df_features.columns:
            print(f"Povprecno stevilo okvirjev z roko: {df_features['frames_with_hand'].mean():.0f}")
        
        if "n_peaks" in df_features.columns:
            print(f"Povprecno stevilo vrhov: {df_features['n_peaks'].mean():.1f}")
        
        if "peak_mean_duration_s" in df_features.columns:
            print(f"Povprecno trajanje vrha: {df_features['peak_mean_duration_s'].mean():.3f} s")


# =============================================================================
# Obdelava enega videa
# =============================================================================

def process_video(
    video_path: Path,
    output_dir: Path,
    K=None,
    dist_coeffs=None,
    save_overlay: bool = True,
) -> pd.DataFrame:
    """Obdela en video in vrne DataFrame s kinematicnimi parametri."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Video ni mogoce odpreti: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if save_overlay:
        out_video_path = output_dir / (video_path.stem + "_overlay.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

    records = []
    prev_landmarks = None
    prev_speed: dict = {}
    frame_idx = 0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = undistort_frame(frame, K, dist_coeffs)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                lm_list = result.multi_hand_landmarks[0].landmark

                if save_overlay:
                    mp_drawing.draw_landmarks(
                        frame,
                        result.multi_hand_landmarks[0],
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

                curr = {
                    name: (lm.x * width, lm.y * height)
                    for name, lm in zip(LANDMARK_NAMES, lm_list)
                }

                row = {"frame": frame_idx, "time_s": frame_idx / fps}

                for name, (x, y) in curr.items():
                    row[f"{name}_x"] = x
                    row[f"{name}_y"] = y

                    if prev_landmarks and name in prev_landmarks:
                        prev_x, prev_y = prev_landmarks[name]
                        pixel_dist = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
                        speed_px = pixel_dist * fps
                        row[f"{name}_speed_px_s"] = speed_px

                        if K is not None:
                            speed_mm = px_to_mm(speed_px, K)
                            row[f"{name}_speed_mm_s"] = speed_mm
                            if name in prev_speed:
                                row[f"{name}_accel_mm_s2"] = (speed_mm - prev_speed[name]) * fps
                            else:
                                row[f"{name}_accel_mm_s2"] = np.nan
                            prev_speed[name] = speed_mm
                        else:
                            if name in prev_speed:
                                row[f"{name}_accel_px_s2"] = (speed_px - prev_speed[name]) * fps
                            else:
                                row[f"{name}_accel_px_s2"] = np.nan
                            prev_speed[name] = speed_px
                    else:
                        row[f"{name}_speed_px_s"] = np.nan
                        if K is not None:
                            row[f"{name}_speed_mm_s"] = np.nan
                            row[f"{name}_accel_mm_s2"] = np.nan
                        else:
                            row[f"{name}_accel_px_s2"] = np.nan
                        prev_speed.pop(name, None)

                if save_overlay:
                    wrist_speed_px = row.get("WRIST_speed_px_s", 0) or 0
                    cv2.putText(frame, f"WRIST: {wrist_speed_px:.1f} px/s",
                                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    if K is not None:
                        wrist_speed_mm = px_to_mm(wrist_speed_px, K)
                        cv2.putText(frame, f"WRIST: {wrist_speed_mm:.1f} mm/s",
                                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

                records.append(row)
                prev_landmarks = curr
            else:
                prev_landmarks = None

            if writer:
                writer.write(frame)
            frame_idx += 1

    cap.release()
    if writer:
        writer.release()
        print(f"  → Overlay video: {out_video_path}")

    df = pd.DataFrame(records)

    # Akumulirana pot
    use_mm = K is not None
    suffix = "_speed_mm_s" if use_mm else "_speed_px_s"
    unit = "mm" if use_mm else "px"
    for name in LANDMARK_NAMES:
        speed_col = f"{name}{suffix}"
        if speed_col in df.columns:
            df[f"{name}_path_{unit}"] = df[speed_col].fillna(0).cumsum() / fps

    return df


# =============================================================================
# Poisci video camP_0 za pacienta
# =============================================================================

def find_camP0_video(patient_dir: Path) -> Path | None:
    """Vrne pot do videa s kamero camP_0, ali None ce ga ni."""
    candidates = list(patient_dir.glob(f"*{CAMERA_ID}*.mp4"))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0]


# =============================================================================
# Paketni nacin
# =============================================================================

def run_batch(max_patients: int, K, dist, output_dir: Path,
              t_start: float = None, t_end: float = None):
    """Paketna obdelava: zbira znacilke v CSV in histogrami, ter vse framove v en CSV."""
    data_root = DATA_DIR / "Data"
    if not data_root.exists():
        data_root = DATA_DIR

    patient_dirs = sorted(
        [d for d in data_root.iterdir() if d.is_dir() and d.name.startswith("patient_")]
    )[:max_patients]

    if not patient_dirs:
        print(f"Ni najdenih map pacientov v {data_root}")
        sys.exit(1)

    print(f"Paketna obdelava: {len(patient_dirs)} pacientov ...")
    if t_start is not None and t_end is not None:
        print(f"Interval analize: {t_start} - {t_end} sekund")
    else:
        print(f"Analiza celotnega posnetka")
    print("-" * 60)

    summary_rows = []
    all_features = []
    all_frames_data = []

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        video_path = find_camP0_video(patient_dir)

        row = {"patient": patient_id, "status": "ok", "error": ""}

        if video_path is None:
            print(f"  [{patient_id}] Ni videa — preskoceno.")
            row.update({"status": "skip_no_video"})
            summary_rows.append(row)
            continue

        row["video"] = video_path.name
        print(f"  [{patient_id}] Obdelava: {video_path.name}")

        out_dir = output_dir / patient_id
        out_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()

        try:
            df = process_video(
                video_path, out_dir,
                K=K, dist_coeffs=dist,
                save_overlay=False,
            )
            elapsed = time.perf_counter() - t0
            row["processing_time_s"] = round(elapsed, 2)

            if df.empty:
                print(f"  [{patient_id}] Ni zaznane roke.")
                row["status"] = "no_hand_detected"
                summary_rows.append(row)
                continue

            # Shrani CSV za tega pacienta (podrobni podatki)
            csv_path = out_dir / "results.csv"
            df.to_csv(csv_path, index=False)

            # Zberi znacilke in frame podatke z intervalom
            features, frames_data = collect_batch_features(df, patient_id, out_dir, 
                                                            t_start=t_start, t_end=t_end)
            if features:
                features["processing_time_s"] = elapsed
                features["status"] = "ok"
                all_features.append(features)
            
            if frames_data:
                all_frames_data.extend(frames_data)

            # Osnovne metrike za summary
            use_mm = K is not None
            speed_col = "WRIST_speed_mm_s" if use_mm else "WRIST_speed_px_s"
            path_col = "WRIST_path_mm" if use_mm else "WRIST_path_px"

            if speed_col in df.columns:
                row["wrist_speed_mean"] = round(df[speed_col].mean(), 3)
                row["wrist_speed_max"] = round(df[speed_col].max(), 3)
            if path_col in df.columns:
                row["wrist_total_path"] = round(df[path_col].iloc[-1], 3)

            row["duration_s"] = round(df["time_s"].max(), 2)
            row["frames_with_hand"] = len(df)

            if features:
                row["n_peaks"] = features.get("n_peaks", 0)
                row["peak_mean_duration_s"] = features.get("peak_mean_duration_s", 0)
                row["n_accel_peaks"] = features.get("n_accel_peaks", 0)

            print(f"  [{patient_id}] OK - hitrost: {row['wrist_speed_mean']:.1f} mm/s, vrhov: {row.get('n_peaks', 0)}, framov: {len(df)}")

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            row["status"] = "error"
            row["error"] = str(exc)
            row["processing_time_s"] = round(elapsed, 2)
            print(f"  [{patient_id}] Napaka: {exc}")
            traceback.print_exc()

        summary_rows.append(row)

    # Shrani povzetni CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "batch_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n→ Povzetni CSV: {summary_path}")

    # Shrani vse znacilke in frame podatke
    save_batch_features_summary(all_features, all_frames_data, output_dir)

    # Statusi
    status_counts = summary_df["status"].value_counts()
    print("\n" + "=" * 60)
    print("STATUS PROCESIRANJA:")
    for status, count in status_counts.items():
        print(f"  {status}: {count}")
    print("=" * 60)


# =============================================================================
# Enojni nacin
# =============================================================================

def run_single(video_path: Path, K, dist, output_dir: Path,
               t_start: float = None, t_end: float = None):
    """Obdelava enega videa z izrisom Plotly grafa in oznacenimi vrhovi."""
    out_dir = output_dir / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Obdelava: {video_path.name} ...")
    if t_start is not None and t_end is not None:
        print(f"Interval analize: {t_start} - {t_end} sekund")
    
    df = process_video(video_path, out_dir, K=K, dist_coeffs=dist, save_overlay=True)

    if df.empty:
        print("Ni zaznane roke.")
        return

    # Izracunaj znacilke in plot podatke za oznacevanje vrhov
    features = None
    plot_data = None
    
    if "WRIST_speed_mm_s" in df.columns and K is not None:
        fps = 30.0
        features, plot_data = extract_timeseries_features_robust(df, fps, unit="mm",
                                                                   t_start=t_start, t_end=t_end)
        
        if features:
            features_df = pd.DataFrame([features])
            features_df.to_csv(out_dir / "timeseries_features.csv", index=False)
            
            conv_factor = 0.001
            print(f"\n  ZNACILKE ZA PACIENTA {video_path.stem}:")
            print(f"    Stevilo vrhov hitrosti: {features.get('n_peaks', 0)}")
            print(f"    Povprecno trajanje vrha: {features.get('peak_mean_duration_s', 0):.3f} s")
            print(f"    Stevilo sunkov: {features.get('n_accel_peaks', 0)}")
            print(f"    Dominantna frekvenca: {features.get('dominant_freq_hz', 0):.3f} Hz")
            print(f"    Povprecna hitrost: {features.get('speed_mean_mm_s', 0) * conv_factor:.3f} m/s")
            print(f"    Maksimalna hitrost: {features.get('speed_max_mm_s', 0) * conv_factor:.3f} m/s")
    
    # Izrisi Plotly graf z vsemi sklepi in oznacenimi vrhovi
    save_speed_plot(df, out_dir, video_path.stem, features, plot_data)

    csv_path = out_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  → CSV: {csv_path}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Sledenje sklepom roke iz videa (9HPC).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Primeri:
  # Enojni video:
  python test.py --video /data/Data/patient_001/patient_001camP_0_20241121_10_21_17.mp4

  # Enojni video s casovnim intervalom (3-5 sekund):
  python test.py --video /data/Data/patient_001/patient_001camP_0_20241121_10_21_17.mp4 --t-start 3.0 --t-end 5.0

  # Paketni nacin (prvih 100 pacientov):
  python test.py --batch

  # Paketni nacin s casovnim intervalom:
  python test.py --batch --t-start 3.0 --t-end 5.0

  # Paketni nacin z omejenim stevilom:
  python test.py --batch --max-patients 20

  # Brez kalibracije:
  python test.py --video /data/... --no-calib
        """,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--video", type=Path, help="Pot do MP4 videa za enojno analizo.")
    mode.add_argument("--batch", action="store_true", help="Paketna analiza.")

    parser.add_argument("--max-patients", type=int, default=336, metavar="N",
                        help="Maks. stevilo pacientov v paketnem nacinu (privzeto: 100).")
    parser.add_argument("--cam", default="mid", help="Ime kamere v kalibracijskem .npz (privzeto: mid).")
    parser.add_argument("--no-calib", action="store_true", help="Preskoci kalibracijo.")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR,
                        help=f"Izhodni direktorij (privzeto: {OUTPUT_DIR}).")
    parser.add_argument("--t-start", type=float, default=5.0,
                        help="Zacetek intervala za analizo v sekundah (npr. 3.0)")
    parser.add_argument("--t-end", type=float, default=10.0,
                        help="Konec intervala za analizo v sekundah (npr. 5.0)")

    return parser.parse_args()


def main():
    args = parse_args()
    
    # Preveri interval
    if (args.t_start is not None) != (args.t_end is not None):
        print("Napaka: --t-start in --t-end morata biti podana skupaj ali pa nobeden.")
        sys.exit(1)
    
    K, dist = (None, None) if args.no_calib else load_calibration(args.cam)
    args.output.mkdir(parents=True, exist_ok=True)

    if args.batch:
        run_batch(args.max_patients, K, dist, args.output,
                  t_start=args.t_start, t_end=args.t_end)
    else:
        if not args.video.exists():
            print(f"Video ne obstaja: {args.video}")
            sys.exit(1)
        run_single(args.video, K, dist, args.output,
                   t_start=args.t_start, t_end=args.t_end)


if __name__ == "__main__":
    main()