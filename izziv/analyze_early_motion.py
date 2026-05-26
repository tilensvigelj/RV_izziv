#!/usr/bin/env python3
# analyze_early_motion.py

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr, ttest_ind
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq
from scipy.stats import skew, kurtosis, entropy
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("/output")
ANALYSIS_START = 5.0
ANALYSIS_END = 10.0

# Seznam sklepov, ki jih želimo analizirati (mala imena, kot so v CSV)
JOINTS = [ 'wrist', 'thumb_tip', 'index_finger_tip' ]

def load_frames_data(output_dir: Path):
    frames_path = output_dir / "all_patients_frames_data.csv"
    if not frames_path.exists():
        print(f"Napaka: {frames_path} ne obstaja!")
        print("Najprej poženi: python test.py --batch")
        return None
    df = pd.read_csv(frames_path)
    print(f"Naloženih {len(df)} framov, {df['patient'].nunique()} pacientov")
    print(f"Razpoložljivi stolpci: {df.columns.tolist()}")
    return df

def compute_joint_features(speed_series, time_series, dt, joint_name):
    if len(speed_series) < 5:
        return {}
    
    speed = speed_series.values
    time = time_series.values[:len(speed)]
    
    speed_mean = np.mean(speed)
    speed_std = np.std(speed)
    speed_max = np.max(speed)
    speed_min = np.min(speed)
    speed_median = np.median(speed)
    speed_q25 = np.percentile(speed, 25)
    speed_q75 = np.percentile(speed, 75)
    speed_iqr = speed_q75 - speed_q25
    speed_cv = speed_std / speed_mean if speed_mean > 0 else 0
    speed_skew = skew(speed)
    speed_kurt = kurtosis(speed)
    speed_rms = np.sqrt(np.mean(speed ** 2))
    
    accel = np.gradient(speed, dt)
    jerk = np.gradient(accel, dt)
    accel_abs = np.abs(accel)
    jerk_abs = np.abs(jerk)
    
    accel_mean = np.mean(accel_abs)
    accel_max = np.max(accel_abs)
    accel_q95 = np.percentile(accel_abs, 95)
    jerk_mean = np.mean(jerk_abs)
    jerk_max = np.max(jerk_abs)
    jerk_q95 = np.percentile(jerk_abs, 95)
    
    min_dist = max(2, int(0.3 / dt))
    peaks, peak_props = find_peaks(speed, distance=min_dist, height=np.percentile(speed, 50), width=1)
    n_peaks = len(peaks)
    if n_peaks > 0:
        peak_mags = speed[peaks]
        peak_mean_mag = np.mean(peak_mags)
        peak_max_mag = np.max(peak_mags)
        if 'widths' in peak_props:
            widths = peak_props['widths'] * dt
            peak_mean_dur = np.mean(widths)
            peak_total_dur = np.sum(widths)
        else:
            peak_mean_dur = peak_total_dur = 0
    else:
        peak_mean_mag = peak_max_mag = 0
        peak_mean_dur = peak_total_dur = 0
    
    high_thresh = np.percentile(speed, 75)
    n_high_peaks = sum(1 for p in peaks if speed[p] > high_thresh)
    
    accel_peaks, _ = find_peaks(accel_abs, distance=max(2, int(0.2/dt)), height=np.percentile(accel_abs, 75))
    n_accel_peaks = len(accel_peaks)
    
    jerk_peaks, _ = find_peaks(jerk_abs, distance=max(2, int(0.15/dt)), height=np.percentile(jerk_abs, 75))
    n_jerk_peaks = len(jerk_peaks)
    
    total_path = np.sum(np.abs(np.diff(speed))) * dt if len(speed) > 1 else 0
    area_curve = np.trapezoid(speed, time) if len(speed) > 1 else 0
    kinetic_energy = np.mean(speed ** 2)
    
    n_fft = len(speed)
    if n_fft > 10:
        fft_vals = fft(speed - speed_mean)
        fft_mag = np.abs(fft_vals[:n_fft//2])
        freqs = fftfreq(n_fft, dt)[:n_fft//2]
        if len(freqs) > 1 and np.sum(fft_mag[1:]) > 0:
            dom_idx = np.argmax(fft_mag[1:]) + 1
            dominant_freq = freqs[dom_idx] if dom_idx < len(freqs) else 0
            spectral_energy = np.sum(fft_mag ** 2)
            mean_freq = np.sum(freqs * fft_mag) / (np.sum(fft_mag) + 1e-6)
            fft_norm = fft_mag / (np.sum(fft_mag) + 1e-6)
            spectral_entropy = entropy(fft_norm + 1e-10)
        else:
            dominant_freq = mean_freq = 0
            spectral_energy = 0
            spectral_entropy = 0
    else:
        dominant_freq = mean_freq = 0
        spectral_energy = 0
        spectral_entropy = 0
    
    zero_crossings = np.sum(np.diff(np.sign(speed - speed_mean)) != 0)
    mean_abs_diff = np.mean(np.abs(np.diff(speed)))
    peak_to_avg = speed_max / speed_mean if speed_mean > 0 else 0
    
    def sample_entropy(ts, m=2, r=0.2):
        if len(ts) < 10:
            return np.nan
        r = r * np.std(ts)
        n = len(ts)
        def _maxdist(xi, xj):
            return max([abs(ua-va) for ua,va in zip(xi,xj)])
        def _phi(mval):
            x = [[ts[j] for j in range(i, i+mval-1)] for i in range(n-mval+1)]
            B = 0
            for i in range(n-mval):
                cnt = 0
                for j in range(n-mval):
                    if i != j and _maxdist(x[i], x[j]) <= r:
                        cnt += 1
                B += cnt / (n-mval-1)
            return B / (n-mval)
        try:
            Bm = _phi(m)
            Bmp1 = _phi(m+1)
            if Bm > 0 and Bmp1 > 0:
                return -np.log(Bmp1 / Bm)
            return np.nan
        except:
            return np.nan
    samp_ent = sample_entropy(speed)
    
    prefix = f"{joint_name}_"
    return {
        f"{prefix}speed_mean_mm_s": speed_mean,
        f"{prefix}speed_std_mm_s": speed_std,
        f"{prefix}speed_max_mm_s": speed_max,
        f"{prefix}speed_min_mm_s": speed_min,
        f"{prefix}speed_median_mm_s": speed_median,
        f"{prefix}speed_q25_mm_s": speed_q25,
        f"{prefix}speed_q75_mm_s": speed_q75,
        f"{prefix}speed_iqr_mm_s": speed_iqr,
        f"{prefix}speed_cv": speed_cv,
        f"{prefix}speed_skew": speed_skew,
        f"{prefix}speed_kurtosis": speed_kurt,
        f"{prefix}speed_rms_mm_s": speed_rms,
        f"{prefix}n_peaks": n_peaks,
        f"{prefix}n_high_peaks": n_high_peaks,
        f"{prefix}peak_mean_magnitude_mm_s": peak_mean_mag,
        f"{prefix}peak_max_magnitude_mm_s": peak_max_mag,
        f"{prefix}peak_mean_duration_s": peak_mean_dur,
        f"{prefix}peak_total_duration_s": peak_total_dur,
        f"{prefix}accel_mean_mm_s2": accel_mean,
        f"{prefix}accel_max_mm_s2": accel_max,
        f"{prefix}accel_q95_mm_s2": accel_q95,
        f"{prefix}n_accel_peaks": n_accel_peaks,
        f"{prefix}jerk_mean_mm_s3": jerk_mean,
        f"{prefix}jerk_max_mm_s3": jerk_max,
        f"{prefix}jerk_q95_mm_s3": jerk_q95,
        f"{prefix}n_jerk_peaks": n_jerk_peaks,
        f"{prefix}total_path_mm": total_path,
        f"{prefix}area_under_curve_mm_s": area_curve,
        f"{prefix}kinetic_energy_mm2_s2": kinetic_energy,
        f"{prefix}dominant_freq_hz": dominant_freq,
        f"{prefix}mean_freq_hz": mean_freq,
        f"{prefix}spectral_energy": spectral_energy,
        f"{prefix}spectral_entropy": spectral_entropy,
        f"{prefix}zero_crossings": zero_crossings,
        f"{prefix}mean_abs_diff_mm_s": mean_abs_diff,
        f"{prefix}peak_to_avg_ratio": peak_to_avg,
        f"{prefix}sample_entropy": samp_ent,
    }

def extract_interval_features_for_all_joints(df_frames: pd.DataFrame, start_time: float, end_time: float):
    patients = df_frames['patient'].unique()
    all_features = []
    
    for patient in patients:
        patient_data = df_frames[df_frames['patient'] == patient].sort_values('time_s')
        total_duration = patient_data['time_s'].max() - patient_data['time_s'].min()
        interval = patient_data[(patient_data['time_s'] >= start_time) & (patient_data['time_s'] <= end_time)].copy()
        if len(interval) < 5:
            continue
        
        dt = np.mean(np.diff(interval['time_s'].values)) if len(interval) > 1 else 0.033
        features = {'patient': patient, 'total_duration_s': total_duration}
        
        for joint in JOINTS:
            speed_col = f"{joint}_speed_mm_s"
            if speed_col not in interval.columns:
                continue
            speed_series = interval[speed_col].dropna()
            if len(speed_series) < 5:
                continue
            time_series = interval['time_s'].loc[speed_series.index]
            joint_feat = compute_joint_features(speed_series, time_series, dt, joint)
            features.update(joint_feat)
        
        if len(features) > 2:
            all_features.append(features)
    
    df_features = pd.DataFrame(all_features)
    print(f"Izračunane značilke za {len(df_features)} pacientov")
    return df_features

def main():
    print("="*80)
    print("ANALIZA KORELACIJ - INTERVAL 5-10 SEKUND ZA VEČ SKLEPOV")
    print("="*80)
    print(f"Interval: {ANALYSIS_START} - {ANALYSIS_END} s")
    print(f"Iščemo sklepe: {', '.join(JOINTS)}")
    
    df_frames = load_frames_data(OUTPUT_DIR)
    if df_frames is None:
        return
    
    df_feat = extract_interval_features_for_all_joints(df_frames, ANALYSIS_START, ANALYSIS_END)
    if df_feat is None or df_feat.empty:
        print("Ni podatkov za analizo!")
        return
    
    target = 'total_duration_s'
    feature_cols = [c for c in df_feat.columns if c not in ['patient', target]]
    
    correlations = []
    for col in feature_cols:
        valid = df_feat[[col, target]].dropna()
        if len(valid) > 5:
            rho, p = spearmanr(valid[target], valid[col])
            r_pearson, p_pearson = pearsonr(valid[target], valid[col])
            correlations.append({
                'feature': col,
                'spearman_rho': rho,
                'spearman_p': p,
                'pearson_r': r_pearson,
                'pearson_p': p_pearson,
                'n': len(valid)
            })
    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('spearman_rho', key=abs, ascending=False)
    corr_df.to_csv(OUTPUT_DIR / "correlation_all_joints.csv", index=False)
    
    print("\nNAJBOLJŠIH 10 KORELACIJ:")
    print(corr_df.head(10).to_string(index=False))
    
    for joint in JOINTS:
        joint_corr = corr_df[corr_df['feature'].str.startswith(joint+'_')]
        if not joint_corr.empty:
            best = joint_corr.iloc[0]
            print(f"\nNajboljša korelacija za {joint}: {best['feature']}  ρ={best['spearman_rho']:.3f} (p={best['spearman_p']:.4f})")
    
    # Heatmap, scatter ploti, bar plot (enako kot prej)
    top_features = corr_df.head(15)['feature'].tolist()
    if top_features:
        heatmap_cols = [target] + top_features
        corr_matrix = df_feat[heatmap_cols].corr(method='spearman')
        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                    square=True, linewidths=0.5, annot_kws={'size': 8})
        plt.title('Korelacijska matrika (Spearman)')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'correlation_heatmap.png', dpi=150)
        plt.close()
        print(f"Heatmap shranjena: {OUTPUT_DIR / 'correlation_heatmap.png'}")
        
        top9 = top_features[:9]
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()
        for i, feat in enumerate(top9):
            ax = axes[i]
            valid = df_feat[[feat, target]].dropna()
            ax.scatter(valid[target], valid[feat], alpha=0.6, s=50, c='steelblue', edgecolors='black')
            z = np.polyfit(valid[target], valid[feat], 1)
            x_line = np.linspace(valid[target].min(), valid[target].max(), 100)
            ax.plot(x_line, np.polyval(z, x_line), 'r--', linewidth=2)
            rho = corr_df[corr_df['feature'] == feat]['spearman_rho'].values[0]
            p = corr_df[corr_df['feature'] == feat]['spearman_p'].values[0]
            ax.text(0.95, 0.95, f'ρ = {rho:.3f}\np = {p:.4f}', transform=ax.transAxes,
                    verticalalignment='top', horizontalalignment='right', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax.set_xlabel('Celoten čas (s)')
            ax.set_ylabel(feat)
            ax.set_title(feat[:50], fontsize=9)
            ax.grid(True, alpha=0.3)
        for i in range(len(top9), len(axes)):
            axes[i].set_visible(False)
        plt.suptitle('Scatter ploti za najbolj korelirane značilke')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'scatter_top9.png', dpi=150)
        plt.close()
        print(f"Scatter ploti shranjeni: {OUTPUT_DIR / 'scatter_top9.png'}")
        
        plt.figure(figsize=(12, max(8, len(corr_df)*0.2)))
        colors = ['red' if x<0 else 'green' for x in corr_df['spearman_rho']]
        plt.barh(corr_df['feature'], corr_df['spearman_rho'], color=colors, alpha=0.7, edgecolor='black')
        plt.axvline(0, color='black', linewidth=1)
        plt.axvline(0.3, color='gray', linestyle='--', alpha=0.5)
        plt.axvline(-0.3, color='gray', linestyle='--', alpha=0.5)
        plt.xlabel('Spearmanov korelacijski koeficient (ρ)')
        plt.title('Korelacije vseh značilk s celotnim časom')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'correlation_barplot.png', dpi=150)
        plt.close()
        print(f"Bar plot shranjen: {OUTPUT_DIR / 'correlation_barplot.png'}")
    
    print("\nAnaliza končana.")

if __name__ == "__main__":
    main()