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
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("/output")

# Interval analize (v sekundah)
ANALYSIS_START = 5.0
ANALYSIS_END = 10.0


def load_frames_data(output_dir: Path):
    """Naloži CSV z vsemi frami vseh pacientov."""
    frames_path = output_dir / "all_patients_frames_data.csv"
    
    if not frames_path.exists():
        print(f"❌ Napaka: {frames_path} ne obstaja!")
        print("   Najprej poženi: python test.py --batch")
        return None
    
    df = pd.read_csv(frames_path)
    print(f"✅ Naloženih {len(df)} framov, {df['patient'].nunique()} pacientov")
    return df


def extract_interval_features(df_frames: pd.DataFrame, 
                              start_time: float = 5.0, 
                              end_time: float = 10.0) -> pd.DataFrame:
    """
    Iz vsakega pacienta izlušči interval [start_time, end_time] in izračuna značilke.
    Vrne DataFrame z značilkami za vsakega pacienta.
    """
    if df_frames is None or df_frames.empty:
        return None
    
    patients = df_frames['patient'].unique()
    features_list = []
    
    for patient in patients:
        # Podatki za tega pacienta
        patient_data = df_frames[df_frames['patient'] == patient].copy()
        patient_data = patient_data.sort_values('time_s')
        
        # Celotno trajanje videa (iz celotnega posnetka)
        total_duration = patient_data['time_s'].max() - patient_data['time_s'].min()
        
        # Izreži interval [start_time, end_time]
        interval_data = patient_data[(patient_data['time_s'] >= start_time) & 
                                      (patient_data['time_s'] <= end_time)].copy()
        
        if len(interval_data) < 5:
            continue
        
        # Hitrost v intervalu
        speed = interval_data['wrist_speed_mm_s'].dropna().values
        time_interval = interval_data['time_s'].dropna().values[:len(speed)]
        
        if len(speed) < 5:
            continue
        
        dt = np.mean(np.diff(time_interval)) if len(time_interval) > 1 else 0.033
        
        # ====================================================================
        # 1. OSNOVNE STATISTIKE HITROSTI V INTERVALU
        # ====================================================================
        speed_mean = np.mean(speed)
        speed_std = np.std(speed)
        speed_max = np.max(speed)
        speed_min = np.min(speed)
        speed_median = np.median(speed)
        speed_q25 = np.percentile(speed, 25)
        speed_q75 = np.percentile(speed, 75)
        speed_iqr = speed_q75 - speed_q25
        speed_cv = speed_std / speed_mean if speed_mean > 0 else 0
        speed_rms = np.sqrt(np.mean(speed ** 2))
        
        from scipy.stats import skew, kurtosis
        speed_skew = skew(speed)
        speed_kurt = kurtosis(speed)
        
        # ====================================================================
        # 2. AKUMULIRANA POT IN INTEGRALI
        # ====================================================================
        # Skupna prepotovana pot v intervalu
        total_path = np.sum(np.abs(np.diff(speed))) * dt if len(speed) > 1 else 0
        
        # Površina pod krivuljo hitrosti (integral hitrosti) - POPRAVLJENO
        try:
            area_under_curve = np.trapezoid(speed, time_interval) if len(speed) > 1 else 0
        except AttributeError:
            area_under_curve = np.trapz(speed, time_interval) if len(speed) > 1 else 0
        
        # Povprečna hitrost iz integrala (pot / čas)
        interval_duration = end_time - start_time
        speed_from_path = total_path / interval_duration if interval_duration > 0 else 0
        
        # Energija gibanja (povprečje kvadrata hitrosti)
        kinetic_energy = np.mean(speed ** 2)
        
        # ====================================================================
        # 3. VRHOVI HITROSTI V INTERVALU
        # ====================================================================
        min_distance = max(2, int(0.3 / dt))
        peaks, peak_props = find_peaks(speed, 
                                        distance=min_distance,
                                        height=np.percentile(speed, 50))
        
        n_peaks = len(peaks)
        
        if n_peaks > 0:
            peak_magnitudes = speed[peaks]
            peak_mean_magnitude = np.mean(peak_magnitudes)
            peak_max_magnitude = np.max(peak_magnitudes)
            peak_std_magnitude = np.std(peak_magnitudes)
            
            if 'widths' in peak_props and len(peak_props['widths']) > 0:
                widths = peak_props['widths'] * dt
                peak_mean_duration = np.mean(widths)
                peak_total_duration = np.sum(widths)
            else:
                peak_mean_duration = 0
                peak_total_duration = 0
        else:
            peak_mean_magnitude = 0
            peak_max_magnitude = 0
            peak_std_magnitude = 0
            peak_mean_duration = 0
            peak_total_duration = 0
        
        # Visoki vrhovi (nad 75. percentil)
        high_threshold = np.percentile(speed, 75)
        n_high_peaks = sum(1 for p in peaks if speed[p] > high_threshold)
        
        # ====================================================================
        # 4. POSPEŠEK V INTERVALU
        # ====================================================================
        accel = np.gradient(speed, dt)
        accel_abs = np.abs(accel)
        accel_mean = np.mean(accel_abs)
        accel_max = np.max(accel_abs)
        accel_std = np.std(accel_abs)
        accel_q95 = np.percentile(accel_abs, 95)
        
        accel_peaks, _ = find_peaks(accel_abs, 
                                     distance=max(2, int(0.2 / dt)),
                                     height=np.percentile(accel_abs, 75))
        n_accel_peaks = len(accel_peaks)
        
        # ====================================================================
        # 5. JERK V INTERVALU
        # ====================================================================
        jerk = np.gradient(accel, dt)
        jerk_abs = np.abs(jerk)
        jerk_mean = np.mean(jerk_abs)
        jerk_max = np.max(jerk_abs)
        jerk_std = np.std(jerk_abs)
        jerk_q95 = np.percentile(jerk_abs, 95)
        
        jerk_peaks, _ = find_peaks(jerk_abs, 
                                    distance=max(2, int(0.15 / dt)),
                                    height=np.percentile(jerk_abs, 75))
        n_jerk_peaks = len(jerk_peaks)
        
        # ====================================================================
        # 6. FREKVENČNE ZNAČILKE
        # ====================================================================
        from scipy.fft import fft, fftfreq
        from scipy.stats import entropy
        
        n = len(speed)
        if n > 10:
            fft_vals = fft(speed - speed_mean)
            fft_mag = np.abs(fft_vals[:n//2])
            freqs = fftfreq(n, dt)[:n//2]
            
            if len(freqs) > 1 and np.sum(fft_mag[1:]) > 0:
                # Dominantna frekvenca
                dom_freq_idx = np.argmax(fft_mag[1:]) + 1
                dominant_freq = freqs[dom_freq_idx] if dom_freq_idx < len(freqs) else 0
                
                # Spektralna energija
                spectral_energy = np.sum(fft_mag ** 2)
                
                # Povprečna frekvenca
                mean_freq = np.sum(freqs * fft_mag) / np.sum(fft_mag) if np.sum(fft_mag) > 0 else 0
                
                # Spektralna entropija
                fft_norm = fft_mag / (np.sum(fft_mag) + 1e-6)
                spectral_entropy = entropy(fft_norm + 1e-10)
                
                # Frekvenca z največjo energijo
                max_freq_idx = np.argmax(fft_mag)
                peak_freq = freqs[max_freq_idx] if max_freq_idx < len(freqs) else 0
            else:
                dominant_freq = mean_freq = peak_freq = 0
                spectral_energy = 0
                spectral_entropy = 0
        else:
            dominant_freq = mean_freq = peak_freq = 0
            spectral_energy = 0
            spectral_entropy = 0
        
        # ====================================================================
        # 7. VARIABILNOST IN KOMPLEKSNOST
        # ====================================================================
        mean_abs_diff = np.mean(np.abs(np.diff(speed)))
        peak_to_avg_ratio = speed_max / speed_mean if speed_mean > 0 else 0
        
        # Prehodi čez povprečje
        speed_norm = speed - speed_mean
        zero_crossings = np.sum(np.diff(np.sign(speed_norm)) != 0)
        
        # Čas do prvega vrha
        time_to_first_peak = time_interval[peaks[0]] - start_time if n_peaks > 0 else np.nan
        
        # ====================================================================
        # 8. SAMPLE ENTROPY (kompleksnost)
        # ====================================================================
        def sample_entropy(ts, m=2, r=0.2):
            """Izračuna sample entropy (kompleksnost časovne serije)."""
            if len(ts) < 10:
                return np.nan
            
            r = r * np.std(ts)
            n = len(ts)
            
            def _maxdist(xi, xj):
                return max([abs(ua - va) for ua, va in zip(xi, xj)])
            
            def _phi(m_val):
                x = [[ts[j] for j in range(i, i + m_val - 1)] for i in range(n - m_val + 1)]
                B = 0
                for i in range(n - m_val):
                    cnt = 0
                    for j in range(n - m_val):
                        if i != j and _maxdist(x[i], x[j]) <= r:
                            cnt += 1
                    B += cnt / (n - m_val - 1)
                return B / (n - m_val)
            
            try:
                Bm = _phi(m)
                Bmp1 = _phi(m + 1)
                if Bm > 0 and Bmp1 > 0:
                    return -np.log(Bmp1 / Bm)
                return np.nan
            except:
                return np.nan
        
        samp_entropy = sample_entropy(speed)
        
        # ====================================================================
        # Zbrane značilke
        # ====================================================================
        features_list.append({
            'patient': patient,
            'total_duration_s': total_duration,
            
            # Osnovne hitrostne značilke
            'interval_speed_mean': speed_mean,
            'interval_speed_std': speed_std,
            'interval_speed_max': speed_max,
            'interval_speed_min': speed_min,
            'interval_speed_median': speed_median,
            'interval_speed_iqr': speed_iqr,
            'interval_speed_cv': speed_cv,
            'interval_speed_skew': speed_skew,
            'interval_speed_kurtosis': speed_kurt,
            'interval_speed_rms': speed_rms,
            
            # Akumulirana pot in integrali
            'interval_total_path_mm': total_path,
            'interval_area_under_curve': area_under_curve,
            'interval_speed_from_path': speed_from_path,
            'interval_kinetic_energy': kinetic_energy,
            
            # Vrhovi
            'interval_n_peaks': n_peaks,
            'interval_n_high_peaks': n_high_peaks,
            'interval_peak_mean_magnitude': peak_mean_magnitude,
            'interval_peak_max_magnitude': peak_max_magnitude,
            'interval_peak_std_magnitude': peak_std_magnitude,
            'interval_peak_mean_duration': peak_mean_duration,
            'interval_peak_total_duration': peak_total_duration,
            'interval_time_to_first_peak': time_to_first_peak,
            
            # Pospešek
            'interval_accel_mean': accel_mean,
            'interval_accel_max': accel_max,
            'interval_accel_q95': accel_q95,
            'interval_n_accel_peaks': n_accel_peaks,
            
            # Jerk
            'interval_jerk_mean': jerk_mean,
            'interval_jerk_max': jerk_max,
            'interval_jerk_q95': jerk_q95,
            'interval_n_jerk_peaks': n_jerk_peaks,
            
            # Frekvenčne značilke
            'interval_dominant_freq_hz': dominant_freq,
            'interval_mean_freq_hz': mean_freq,
            'interval_peak_freq_hz': peak_freq,
            'interval_spectral_energy': spectral_energy,
            'interval_spectral_entropy': spectral_entropy,
            
            # Variabilnost
            'interval_mean_abs_diff': mean_abs_diff,
            'interval_peak_to_avg_ratio': peak_to_avg_ratio,
            'interval_zero_crossings': zero_crossings,
            
            # Kompleksnost
            'interval_sample_entropy': samp_entropy,
        })
    
    df_features = pd.DataFrame(features_list)
    print(f"✅ Izračunanih {len(df_features.columns)-2} značilk za {len(df_features)} pacientov")
    
    return df_features


def calculate_correlations(df_features: pd.DataFrame) -> pd.DataFrame:
    """Izračuna korelacije med vsemi značilkami in total_duration_s."""
    
    target = 'total_duration_s'
    
    # Izberi vse stolpce razen patient in target
    feature_cols = [c for c in df_features.columns if c not in ['patient', target]]
    
    correlations = []
    for col in feature_cols:
        valid = df_features[[col, target]].dropna()
        if len(valid) > 5:
            r_spearman, p_spearman = spearmanr(valid[target], valid[col])
            r_pearson, p_pearson = pearsonr(valid[target], valid[col])
            correlations.append({
                'feature': col,
                'spearman_rho': r_spearman,
                'spearman_p': p_spearman,
                'pearson_r': r_pearson,
                'pearson_p': p_pearson,
                'n': len(valid)
            })
    
    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('spearman_rho', key=abs, ascending=False)
    
    return corr_df


def save_correlation_heatmap(df_features: pd.DataFrame, corr_df: pd.DataFrame, output_dir: Path):
    """Izriše heatmap korelacijske matrike za top parametre."""
    
    top_features = corr_df.head(15)['feature'].tolist()
    all_features = ['total_duration_s'] + top_features
    
    # Izračunaj korelacijsko matriko
    corr_matrix = df_features[all_features].corr(method='spearman')
    
    # Ustvari heatmap
    plt.figure(figsize=(14, 12))
    
    # Maskiraj zgornji trikotnik
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Heatmap s clusteriranjem
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.3f', 
                cmap='RdBu_r', center=0, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.8},
                annot_kws={'size': 9})
    
    plt.title('Korelacijska matrika (Spearman) - značilke v intervalu 5-10s vs. celoten čas',
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Heatmap: {output_dir / 'correlation_heatmap.png'}")


def save_scatter_plots(df_features: pd.DataFrame, corr_df: pd.DataFrame, output_dir: Path):
    """Izriše scatter plot za top 9 značilk proti total_duration_s."""
    
    top_features = corr_df.head(9)['feature'].tolist()
    
    rows, cols = 3, 3
    fig, axes = plt.subplots(rows, cols, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, feat in enumerate(top_features):
        ax = axes[i]
        valid = df_features[[feat, 'total_duration_s']].dropna()
        
        # Scatter plot
        ax.scatter(valid['total_duration_s'], valid[feat], 
                   alpha=0.6, s=60, c='steelblue', edgecolors='black', linewidth=0.5)
        
        # Regresijska črta
        z = np.polyfit(valid['total_duration_s'], valid[feat], 1)
        x_line = np.linspace(valid['total_duration_s'].min(), valid['total_duration_s'].max(), 100)
        ax.plot(x_line, np.polyval(z, x_line), 'r--', linewidth=2, label='Linearna regresija')
        
        # Korelacijski koeficient
        r = corr_df[corr_df['feature'] == feat]['spearman_rho'].values[0]
        p = corr_df[corr_df['feature'] == feat]['spearman_p'].values[0]
        
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        ax.text(0.95, 0.95, f'ρ = {r:.3f} {sig}\np = {p:.4f}', 
                transform=ax.transAxes, verticalalignment='top',
                horizontalalignment='right', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Celoten čas posnetka (s)', fontsize=10)
        ax.set_ylabel(feat, fontsize=9)
        ax.set_title(feat, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    # Skrij neuporabljene podgrafe
    for i in range(len(top_features), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Korelacija med značilkami v intervalu 5-10s in celotnim časom posnetka',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'scatter_plots_top9.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Scatter ploti: {output_dir / 'scatter_plots_top9.png'}")


def save_correlation_barplot(corr_df: pd.DataFrame, output_dir: Path):
    """Izriše bar plot korelacijskih koeficientov za vse značilke."""
    
    # Uredi po absolutni vrednosti
    corr_sorted = corr_df.sort_values('spearman_rho', key=abs, ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, max(8, len(corr_sorted) * 0.3)))
    
    colors = ['red' if x < 0 else 'green' for x in corr_sorted['spearman_rho']]
    
    bars = ax.barh(corr_sorted['feature'], corr_sorted['spearman_rho'], 
                   color=colors, alpha=0.7, edgecolor='black')
    
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.axvline(x=0.3, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(x=-0.3, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0.5, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    ax.axvline(x=-0.5, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    
    ax.set_xlabel('Spearmanov korelacijski koeficient (ρ)', fontsize=12)
    ax.set_ylabel('Značilka (interval 5-10s)', fontsize=12)
    ax.set_title('Korelacija značilk v intervalu 5-10s s celotnim časom posnetka',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Dodaj p-vrednosti
    for i, (idx, row) in enumerate(corr_sorted.iterrows()):
        p_val = row['spearman_p']
        if p_val < 0.001:
            sig = '***'
        elif p_val < 0.01:
            sig = '**'
        elif p_val < 0.05:
            sig = '*'
        else:
            sig = ''
        
        ax.text(row['spearman_rho'] + (0.02 if row['spearman_rho'] >= 0 else -0.15), 
                i, sig, va='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_barplot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Bar plot: {output_dir / 'correlation_barplot.png'}")


def save_histograms(df_features: pd.DataFrame, corr_df: pd.DataFrame, output_dir: Path):
    """Shrani histograme top značilk."""
    
    top_features = corr_df.head(12)['feature'].tolist()
    
    n_features = len(top_features)
    cols = 3
    rows = (n_features + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
    axes = axes.flatten()
    
    for i, feat in enumerate(top_features):
        ax = axes[i]
        data = df_features[feat].dropna()
        
        ax.hist(data, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Povprečje: {data.mean():.3f}')
        ax.axvline(data.median(), color='green', linestyle='--', linewidth=2, 
                   label=f'Mediana: {data.median():.3f}')
        
        rho = corr_df[corr_df['feature'] == feat]['spearman_rho'].values[0]
        ax.set_xlabel(feat)
        ax.set_ylabel('Število pacientov')
        ax.set_title(f'{feat}\n(ρ = {rho:.3f})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    for i in range(len(top_features), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Porazdelitve značilk v intervalu 5-10s', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'histograms_top_features.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Histogrami: {output_dir / 'histograms_top_features.png'}")


def save_boxplot_best_feature(df_features: pd.DataFrame, corr_df: pd.DataFrame, output_dir: Path):
    """Izriše box plot za najbolj korelirano značilko."""
    
    if corr_df.empty:
        return
    
    best_feat = corr_df.iloc[0]['feature']
    best_rho = corr_df.iloc[0]['spearman_rho']
    
    # Razdeli v skupine glede na mediano
    median_val = df_features[best_feat].median()
    high_group = df_features[df_features[best_feat] > median_val]['total_duration_s']
    low_group = df_features[df_features[best_feat] <= median_val]['total_duration_s']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bp = ax.boxplot([high_group, low_group], 
                    labels=[f'Visok {best_feat}\n(n={len(high_group)})', 
                           f'Nizek {best_feat}\n(n={len(low_group)})'],
                    patch_artist=True)
    
    # Barvanje škatel
    bp['boxes'][0].set_facecolor('lightcoral')
    bp['boxes'][1].set_facecolor('lightblue')
    
    # Barvanje median - POPRAVLJENO (medians je seznam)
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2)
    
    # Barvanje mean (povprečje) - če obstaja
    if 'means' in bp and bp['means']:
        for mean in bp['means']:
            mean.set_marker('D')
            mean.set_markeredgecolor('black')
            mean.set_markerfacecolor('yellow')
            mean.set_markersize(8)
    
    # T-test
    t_stat, p_val = ttest_ind(high_group, low_group)
    
    ax.text(0.95, 0.95, f'T-test: t = {t_stat:.3f}\np = {p_val:.4f}',
            transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_ylabel('Celoten čas posnetka (s)', fontsize=12)
    ax.set_title(f'Primerjava časa glede na {best_feat}\n(ρ = {best_rho:.3f})', 
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'best_feature_boxplot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Box plot: {output_dir / 'best_feature_boxplot.png'}")


def main():
    print("=" * 80)
    print("ANALIZA KORELACIJ - INTERVAL 5-10 SEKUND vs. CELOTEN ČAS POSNETKA")
    print("=" * 80)
    print(f"\n📊 Interval analize: {ANALYSIS_START} - {ANALYSIS_END} sekund")
    
    # Naloži frame podatke
    df_frames = load_frames_data(OUTPUT_DIR)
    if df_frames is None:
        return
    
    # Izračunaj značilke v intervalu
    df_features = extract_interval_features(df_frames, ANALYSIS_START, ANALYSIS_END)
    
    if df_features is None or df_features.empty:
        print("❌ Ni veljavnih podatkov za analizo!")
        return
    
    print(f"\n📈 Število pacientov v analizi: {len(df_features)}")
    
    # Izračunaj korelacije
    corr_df = calculate_correlations(df_features)
    
    # Shrani korelacije v CSV
    corr_df.to_csv(OUTPUT_DIR / "correlation_analysis_results.csv", index=False)
    print(f"\n✅ Korelacije shranjene: {OUTPUT_DIR / 'correlation_analysis_results.csv'}")
    
    # Izpis najboljših korelacij
    print("\n" + "=" * 80)
    print("NAJBOLJŠE KORELACIJE S CELOTNIM ČASOM POSNETKA:")
    print("=" * 80)
    print(f"{'Značilka':<40s} {'Spearman ρ':>12s} {'p-value':>12s}")
    print("-" * 80)
    
    for _, row in corr_df.head(15).iterrows():
        sig = "***" if row['spearman_p'] < 0.001 else "**" if row['spearman_p'] < 0.01 else "*" if row['spearman_p'] < 0.05 else ""
        print(f"{row['feature']:<40s} {row['spearman_rho']:12.3f} {sig} {row['spearman_p']:12.4f}")
    
    # Statistični povzetek
    n_sig = len(corr_df[corr_df['spearman_p'] < 0.05])
    n_strong = len(corr_df[abs(corr_df['spearman_rho']) > 0.5])
    n_moderate = len(corr_df[(abs(corr_df['spearman_rho']) > 0.3) & (abs(corr_df['spearman_rho']) <= 0.5)])
    
    print("\n" + "=" * 80)
    print("STATISTIČNI POVZETEK:")
    print("=" * 80)
    print(f"  Število značilk s p < 0.05: {n_sig}")
    print(f"  Število značilk z |ρ| > 0.5 (močna korelacija): {n_strong}")
    print(f"  Število značilk z 0.3 < |ρ| ≤ 0.5 (srednja korelacija): {n_moderate}")
    
    # Ustvari vse grafe
    print("\n" + "=" * 80)
    print("🎨 USTVARJAM GRAFE...")
    print("=" * 80)
    
    # 1. Heatmap korelacijske matrike
    save_correlation_heatmap(df_features, corr_df, OUTPUT_DIR)
    
    # 2. Scatter ploti top 9 značilk
    save_scatter_plots(df_features, corr_df, OUTPUT_DIR)
    
    # 3. Bar plot vseh korelacij
    save_correlation_barplot(corr_df, OUTPUT_DIR)
    
    # 4. Histogrami top značilk
    save_histograms(df_features, corr_df, OUTPUT_DIR)
    
    # 5. Box plot za najboljšo značilko
    save_boxplot_best_feature(df_features, corr_df, OUTPUT_DIR)
    
    # Interpretacija
    print("\n" + "=" * 80)
    print("📋 INTERPRETACIJA REZULTATOV")
    print("=" * 80)
    
    if not corr_df.empty:
        best_row = corr_df.iloc[0]
        if best_row['spearman_rho'] < 0:
            print(f"\n  ✅ Negativna korelacija: višji '{best_row['feature']}' v intervalu 5-10s")
            print(f"     → krajši celoten čas posnetka")
            print(f"     (ρ = {best_row['spearman_rho']:.3f}, p = {best_row['spearman_p']:.4f})")
        else:
            print(f"\n  ⚠️ Pozitivna korelacija: višji '{best_row['feature']}' v intervalu 5-10s")
            print(f"     → daljši celoten čas posnetka")
            print(f"     (ρ = {best_row['spearman_rho']:.3f}, p = {best_row['spearman_p']:.4f})")
    
    print("\n" + "=" * 80)
    print("✨ ANALIZA KONČANA!")
    print("=" * 80)
    print(f"\n📁 Vsi rezultati shranjeni v: {OUTPUT_DIR}")
    print("   - correlation_analysis_results.csv  (vse korelacije)")
    print("   - correlation_heatmap.png           (heatmap korelacijske matrike)")
    print("   - scatter_plots_top9.png            (scatter ploti)")
    print("   - correlation_barplot.png           (bar plot korelacij)")
    print("   - histograms_top_features.png       (histogrami)")
    print("   - best_feature_boxplot.png          (box plot)")


if __name__ == "__main__":
    main()