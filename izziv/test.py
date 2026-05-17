import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

'''docker run \
  -v /media/FastDataMama/data_rv_26:/data \
  -v /media/FastDataMama/tilens/izziv:/output \
  -v /media/FastDataMama/tilens/izziv:/calibration \
  hand-tracking python /izziv/izziv/test.py'''

DATA_DIR = Path("/data")
OUTPUT_DIR = Path("/output")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
LANDMARK_NAMES = [lm.name for lm in mp.solutions.hands.HandLandmark]


def process_video(video_path: Path, output_dir: Path, K=None, dist_coeffs=None) -> pd.DataFrame:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_video_path = output_dir / (video_path.stem + "_overlay.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

    records = []
    prev_landmarks = None
    frame_idx = 0
    prev_speed = {} 

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
                            row[f"{name}_speed_mm_s"] = px_to_mm(speed_px, K)
                        else:
                            row[f"{name}_speed_px_s"] = np.nan
                            if K is not None:
                                row[f"{name}_speed_mm_s"] = np.nan

                wrist_speed_px = row.get("WRIST_speed_px_s", 0) or 0

                cv2.putText(frame, f"WRIST: {wrist_speed_px:.1f} px/s",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                if K is not None:
                    wrist_speed_mm = px_to_mm(wrist_speed_px, K)
                    row["WRIST_speed_mm_s"] = wrist_speed_mm
                    cv2.putText(frame, f"WRIST: {wrist_speed_mm:.1f} mm/s",
                                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
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

                            # Pospešek [mm/s²]
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
        
                records.append(row)
                prev_landmarks = curr
            else:
                prev_landmarks = None

            writer.write(frame)
            frame_idx += 1

    cap.release()
    writer.release()
    print(f"  → Overlay video: {out_video_path}")

    # Akumulirana pot 
    df = pd.DataFrame(records)

    use_mm = K is not None
    suffix = "_speed_mm_s" if use_mm else "_speed_px_s"
    unit = "mm" if use_mm else "px"

    for name in LANDMARK_NAMES:
        speed_col = f"{name}{suffix}"
        if speed_col in df.columns:
            df[f"{name}_path_{unit}"] = df[speed_col].fillna(0).cumsum() / fps

    return df 


import plotly.graph_objects as go

def save_speed_plot(df: pd.DataFrame, output_dir: Path, video_name: str):
    use_mm = any("_speed_mm_s" in col for col in df.columns)
    speed_suffix = "_speed_mm_s" if use_mm else "_speed_px_s"
    accel_suffix = "_accel_mm_s2" if use_mm else "_accel_px_s2"
    path_suffix = "_path_mm" if use_mm else "_path_px"
    speed_unit = "mm/s"if use_mm else "px/s"
    accel_unit = "mm/s²" if use_mm else "px/s²"
    path_unit = "mm" if use_mm else "px"

    joints = [col.replace(speed_suffix, "") for col in df.columns if col.endswith(speed_suffix)]

    from plotly.subplots import make_subplots
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

        # Hitrost
        if f"{joint}{speed_suffix}" in df.columns:
            fig.add_trace(go.Scatter(
                x=df["time_s"], y=df[f"{joint}{speed_suffix}"],
                name=joint, mode="lines", line=dict(width=1.5),
                visible=visible, legendgroup=joint,
            ), row=1, col=1)

        # Pospešek
        if f"{joint}{accel_suffix}" in df.columns:
            fig.add_trace(go.Scatter(
                x=df["time_s"], y=df[f"{joint}{accel_suffix}"],
                name=joint, mode="lines", line=dict(width=1.5),
                visible=visible, legendgroup=joint, showlegend=False,
            ), row=2, col=1)

        # Pot
        if f"{joint}{path_suffix}" in df.columns:
            fig.add_trace(go.Scatter(
                x=df["time_s"], y=df[f"{joint}{path_suffix}"],
                name=joint, mode="lines", line=dict(width=1.5),
                visible=visible, legendgroup=joint, showlegend=False,
            ), row=3, col=1)

    fig.update_layout(
        title=f"Kinematika sklepov — {video_name}",
        xaxis3_title="Čas [s]",
        legend=dict(
            title="Sklepi",
            itemclick="toggle",
            itemdoubleclick="toggleothers",
        ),
        hovermode="x unified",
        template="plotly_white",
        height=900,
    )

    plot_path = output_dir / (video_name + "_kinematics.html")
    fig.write_html(str(plot_path))
    print(f"  → Interaktivni graf: {plot_path}")


CALIBRATION_FILE = Path("/calibration/calibration.npz")


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
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    return cv2.undistort(frame, K, dist, None, new_K)


def px_to_mm(speed_px_s: float, K, depth_mm: float = 500.0):
    """
    Pretvori hitrost iz px/s v mm/s.
    depth_mm: ocenjena razdalja roke od kamere v mm (default 500mm = 50cm).
    """
    if K is None:
        return None
    fx = K[0, 0]
    return speed_px_s * depth_mm / fx

def main():
    K, dist = load_calibration(cam_name="mid")
    video_path = Path("/data/Data/patient_001/patient_001camP_0_20241121_10_21_17.mp4")

    out_dir = OUTPUT_DIR / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Obdelava: {video_path.name} ...")
    df = process_video(video_path, out_dir, K=K, dist_coeffs=dist)

    if df.empty:
        print("Ni zaznane roke.")
        return

    save_speed_plot(df, out_dir, video_path.stem)

    csv_path = out_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  → CSV: {csv_path}")


if __name__ == "__main__":
    main()