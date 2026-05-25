import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt
import argparse
import traceback
import sys


'''docker run \
  -v /media/FastDataMama/data_rv_26:/data \
  -v /media/FastDataMama/tilens/izziv:/izziv \
  -v /media/FastDataMama/tilens/izziv:/output \
  -v /media/FastDataMama/tilens/izziv:/calibration \
  hand-tracking python /izziv/test.py \
    --cam mid \
    --video /data/Data/patient_001/patient_001camP_0_20241121_10_21_17.mp4'''
    
'''docker run \
  -v /media/FastDataMama/data_rv_26:/data \
  -v /media/FastDataMama/tilens/izziv:/izziv \
  -v /media/FastDataMama/tilens/izziv:/output \
  -v /media/FastDataMama/tilens/izziv:/calibration \
  hand-tracking python /izziv/test.py --batch

'''

# ─────────────────────────────────────────────
# Poti
# ─────────────────────────────────────────────
DATA_DIR = Path("/data")
OUTPUT_DIR = Path("/output")
CALIBRATION_FILE = Path("/calibration/calibration.npz")

# ─────────────────────────────────────────────
# MediaPipe
# ─────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
LANDMARK_NAMES = [lm.name for lm in mp.solutions.hands.HandLandmark]

CAMERA_ID = "camP_0"   # edina podprta kamera


# ══════════════════════════════════════════════════════════════════════════════
# Kalibracijske funkcije
# ══════════════════════════════════════════════════════════════════════════════

def load_calibration(cam_name: str = "mid"):
    """Naloži kalibracijske matrike za izbrano kamero."""
    if not CALIBRATION_FILE.exists():
        print("Kalibracijska datoteka ni najdena — nadaljujem brez kalibracije.")
        return None, None

    data = np.load(str(CALIBRATION_FILE))
    K = data[f"{cam_name}_K"]
    dist = data[f"{cam_name}_dist"]
    print(f"Kalibracija naložena ({cam_name}): fx={K[0,0]:.1f}, fy={K[1,1]:.1f}")
    return K, dist


def undistort_frame(frame, K, dist):
    """Popravi distorzijo leče."""
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


# ══════════════════════════════════════════════════════════════════════════════
# Obdelava enega videa
# ══════════════════════════════════════════════════════════════════════════════

def process_video(
    video_path: Path,
    output_dir: Path,
    K=None,
    dist_coeffs=None,
    save_overlay: bool = True,
) -> pd.DataFrame:
    """
    Obdela en video in vrne DataFrame s kinematičnimi parametri.

    Parametri
    ---------
    save_overlay : bool
        Če True, shrani video z narisanimi skeletonom.
        V paketnem načinu (batch) je False za hitrejšo obdelavo.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Video ni mogoče odpreti: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
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


# ══════════════════════════════════════════════════════════════════════════════
# Graf za en video
# ══════════════════════════════════════════════════════════════════════════════

def save_speed_plot(df: pd.DataFrame, output_dir: Path, video_name: str):
    from plotly.subplots import make_subplots

    use_mm = any("_speed_mm_s" in col for col in df.columns)
    speed_suffix = "_speed_mm_s" if use_mm else "_speed_px_s"
    accel_suffix = "_accel_mm_s2" if use_mm else "_accel_px_s2"
    path_suffix = "_path_mm" if use_mm else "_path_px"
    speed_unit = "mm/s" if use_mm else "px/s"
    accel_unit = "mm/s²" if use_mm else "px/s²"
    path_unit = "mm" if use_mm else "px"

    joints = [col.replace(speed_suffix, "") for col in df.columns if col.endswith(speed_suffix)]

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[
            f"Hitrost [{speed_unit}]",
            f"Pospešek [{accel_unit}]",
            f"Akumulirana pot [{path_unit}]",
        ],
        shared_xaxes=True,
        vertical_spacing=0.08,
    )

    for joint in joints:
        visible = True if joint == "WRIST" else "legendonly"

        if f"{joint}{speed_suffix}" in df.columns:
            fig.add_trace(go.Scatter(
                x=df["time_s"], y=df[f"{joint}{speed_suffix}"],
                name=joint, mode="lines", line=dict(width=1.5),
                visible=visible, legendgroup=joint,
            ), row=1, col=1)

        if f"{joint}{accel_suffix}" in df.columns:
            fig.add_trace(go.Scatter(
                x=df["time_s"], y=df[f"{joint}{accel_suffix}"],
                name=joint, mode="lines", line=dict(width=1.5),
                visible=visible, legendgroup=joint, showlegend=False,
            ), row=2, col=1)

        if f"{joint}{path_suffix}" in df.columns:
            fig.add_trace(go.Scatter(
                x=df["time_s"], y=df[f"{joint}{path_suffix}"],
                name=joint, mode="lines", line=dict(width=1.5),
                visible=visible, legendgroup=joint, showlegend=False,
            ), row=3, col=1)

    fig.update_layout(
        title=f"Kinematika sklepov — {video_name}",
        xaxis3_title="Čas [s]",
        legend=dict(title="Sklepi", itemclick="toggle", itemdoubleclick="toggleothers"),
        hovermode="x unified",
        template="plotly_white",
        height=900,
    )

    plot_path = output_dir / (video_name + "_kinematics.html")
    fig.write_html(str(plot_path))
    print(f"  → Interaktivni graf: {plot_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Poišči video camP_0 za pacienta
# ══════════════════════════════════════════════════════════════════════════════

def find_camP0_video(patient_dir: Path) -> Path | None:
    """
    Vrne pot do videa s kamero camP_0, ali None če ga ni.
    Podprta sta oba vzorca:
      patient_001camP_0_20241121_10_21_17.mp4
      patient_001_camP_0_20241121_10_21_17.mp4
    """
    candidates = list(patient_dir.glob(f"*{CAMERA_ID}*.mp4"))
    if not candidates:
        return None
    # Vzemi prvega (po imenu) če jih je več
    candidates.sort()
    return candidates[0]


# ══════════════════════════════════════════════════════════════════════════════
# Paketni histogrami
# ══════════════════════════════════════════════════════════════════════════════

def save_batch_histograms(summary_df: pd.DataFrame, output_dir: Path):
    """
    Iz povzetnega CSV nariše histograme vseh numeričnih kinematičnih parametrov
    in časa izvajanja, ter jih shrani kot en HTML.
    """
    from plotly.subplots import make_subplots

    numeric_cols = [
        c for c in summary_df.columns
        if c not in ("patient", "video", "status", "error")
        and pd.api.types.is_numeric_dtype(summary_df[c])
    ]

    if not numeric_cols:
        print("Ni numeričnih stolpcev za histogram.")
        return

    n = len(numeric_cols)
    cols_per_row = 3
    rows = (n + cols_per_row - 1) // cols_per_row

    fig = make_subplots(
        rows=rows, cols=cols_per_row,
        subplot_titles=numeric_cols,
        vertical_spacing=0.06,
        horizontal_spacing=0.06,
    )

    for i, col in enumerate(numeric_cols):
        r = i // cols_per_row + 1
        c = i % cols_per_row + 1
        data = summary_df[col].dropna()
        fig.add_trace(go.Histogram(x=data, name=col, showlegend=False), row=r, col=c)

    fig.update_layout(
        title="Primerjava kinematičnih parametrov — vsi pacienti",
        template="plotly_white",
        height=300 * rows,
    )

    out_path = output_dir / "batch_histograms.html"
    fig.write_html(str(out_path))
    print(f"  → Histogrami: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Paketni način
# ══════════════════════════════════════════════════════════════════════════════

def run_batch(max_patients: int, K, dist, output_dir: Path):
    """
    Obdela do max_patients pacientov (mapa patient_XXX v DATA_DIR/Data).
    Za vsakega vzame samo video camP_0.
    Rezultate zbere v summary CSV in nariše histograme.
    """
    data_root = DATA_DIR / "Data"
    if not data_root.exists():
        data_root = DATA_DIR   # fallback če mapa Data ne obstaja

    patient_dirs = sorted(
        [d for d in data_root.iterdir() if d.is_dir() and d.name.startswith("patient_")]
    )[:max_patients]

    if not patient_dirs:
        print(f"Ni najdenih map pacientov v {data_root}")
        sys.exit(1)

    print(f"Paketna obdelava: {len(patient_dirs)} pacientov ...")

    summary_rows = []

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        video_path = find_camP0_video(patient_dir)

        row: dict = {"patient": patient_id, "status": "ok", "error": ""}

        # ── Failsafe: ni videa ────────────────────────────────────────────────
        if video_path is None:
            print(f"  [{patient_id}] ✗ Ni videa {CAMERA_ID} — preskočeno.")
            row.update({"video": "", "status": "skip_no_video"})
            summary_rows.append(row)
            continue

        row["video"] = video_path.name
        print(f"  [{patient_id}] Obdelava: {video_path.name}")

        out_dir = output_dir / patient_id
        out_dir.mkdir(parents=True, exist_ok=True)

        import time
        t0 = time.perf_counter()

        try:
            df = process_video(
                video_path, out_dir,
                K=K, dist_coeffs=dist,
                save_overlay=False,      # brez overlaya v paketnem načinu
            )
            elapsed = time.perf_counter() - t0
            row["processing_time_s"] = round(elapsed, 2)

            if df.empty:
                print(f"  [{patient_id}] ⚠ Ni zaznane roke v videu.")
                row["status"] = "no_hand_detected"
                summary_rows.append(row)
                continue

            # Shrani CSV za tega pacienta
            csv_path = out_dir / "results.csv"
            df.to_csv(csv_path, index=False)
            print(f"  [{patient_id}] → CSV: {csv_path}")

            # ── Izračun povzetnih statistik ───────────────────────────────────
            use_mm = K is not None
            speed_col = "WRIST_speed_mm_s" if use_mm else "WRIST_speed_px_s"
            accel_col = "WRIST_accel_mm_s2" if use_mm else "WRIST_accel_px_s2"
            path_col = "WRIST_path_mm" if use_mm else "WRIST_path_px"

            if speed_col in df.columns:
                row["wrist_speed_mean"] = round(df[speed_col].mean(), 3)
                row["wrist_speed_max"] = round(df[speed_col].max(), 3)
            if accel_col in df.columns:
                row["wrist_accel_mean"] = round(df[accel_col].abs().mean(), 3)
                row["wrist_accel_max"] = round(df[accel_col].abs().max(), 3)
            if path_col in df.columns:
                row["wrist_total_path"] = round(df[path_col].iloc[-1], 3)

            row["duration_s"] = round(df["time_s"].max(), 2)
            row["frames_with_hand"] = len(df)

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            row["status"] = "error"
            row["error"] = str(exc)
            row["processing_time_s"] = round(elapsed, 2)
            print(f"  [{patient_id}] ✗ Napaka: {exc}")
            traceback.print_exc()

        summary_rows.append(row)

    # ── Shrani povzetni CSV ───────────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "batch_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n→ Povzetni CSV: {summary_path}")

    # ── Histogrami ────────────────────────────────────────────────────────────
    ok_df = summary_df[summary_df["status"] == "ok"]
    if not ok_df.empty:
        save_batch_histograms(ok_df, output_dir)
    else:
        print("Ni uspešnih obdelav — histogrami preskočeni.")


# ══════════════════════════════════════════════════════════════════════════════
# Enojni način
# ══════════════════════════════════════════════════════════════════════════════

def run_single(video_path: Path, K, dist, output_dir: Path):
    out_dir = output_dir / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Obdelava: {video_path.name} ...")
    df = process_video(video_path, out_dir, K=K, dist_coeffs=dist, save_overlay=True)

    if df.empty:
        print("Ni zaznane roke.")
        return

    save_speed_plot(df, out_dir, video_path.stem)

    csv_path = out_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  → CSV: {csv_path}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Sledenje sklepom roke iz videa (9HPC).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Primeri:
  # Enojni video:
  python main.py --video /data/Data/patient_001/patient_001camP_0_20241121_10_21_17.mp4

  # Paketni način (prvih 100 pacientov):
  python main.py --batch

  # Paketni način z omejenim številom:
  python main.py --batch --max-patients 20

  # Brez kalibracije:
  python main.py --video /data/... --no-calib
        """,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--video", type=Path,
        help="Pot do MP4 videa za enojno analizo.",
    )
    mode.add_argument(
        "--batch", action="store_true",
        help="Paketna analiza: en camP_0 video za vsakega pacienta.",
    )

    parser.add_argument(
        "--max-patients", type=int, default=100, metavar="N",
        help="Maks. število pacientov v paketnem načinu (privzeto: 100).",
    )
    parser.add_argument(
        "--cam", default="mid",
        help="Ime kamere v kalibracijskem .npz (privzeto: mid).",
    )
    parser.add_argument(
        "--no-calib", action="store_true",
        help="Preskoči kalibracijo (hitrost v px namesto mm).",
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_DIR,
        help=f"Izhodni direktorij (privzeto: {OUTPUT_DIR}).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Kalibracija
    K, dist = (None, None) if args.no_calib else load_calibration(args.cam)

    args.output.mkdir(parents=True, exist_ok=True)

    if args.batch:
        run_batch(args.max_patients, K, dist, args.output)
    else:
        # Validacija videa
        if not args.video.exists():
            print(f"Video ne obstaja: {args.video}")
            sys.exit(1)
        run_single(args.video, K, dist, args.output)


if __name__ == "__main__":
    main()