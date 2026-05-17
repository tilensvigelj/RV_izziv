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
                        row[f"{name}_speed_px_s"] = pixel_dist * fps
                    else:
                        row[f"{name}_speed_px_s"] = np.nan

                wrist_speed_px = row.get("WRIST_speed_px_s", 0) or 0

                cv2.putText(frame, f"WRIST: {wrist_speed_px:.1f} px/s",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                if K is not None:
                    wrist_speed_mm = px_to_mm(wrist_speed_px, K)
                    row["WRIST_speed_mm_s"] = wrist_speed_mm
                    cv2.putText(frame, f"WRIST: {wrist_speed_mm:.1f} mm/s",
                                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

                records.append(row)
                prev_landmarks = curr
            else:
                prev_landmarks = None

            writer.write(frame)
            frame_idx += 1

    cap.release()
    writer.release()
    print(f"  → Overlay video: {out_video_path}")
    return pd.DataFrame(records)


def save_speed_plot(df: pd.DataFrame, output_dir: Path, video_name: str):
    # Izberi zanimive sklepe
    joints = ["WRIST", "INDEX_FINGER_TIP", "THUMB_TIP", "MIDDLE_FINGER_TIP"]

    fig, ax = plt.subplots(figsize=(14, 5))

    col = "WRIST_speed_mm_s" if "WRIST_speed_mm_s" in df.columns else "WRIST_speed_px_s"
    unit = "mm/s" if "WRIST_speed_mm_s" in df.columns else "px/s"

    ax.plot(df["time_s"], df[col], linewidth=1.2, color="steelblue")

    ax.set_xlabel("Čas [s]")
    ax.set_ylabel(f"Hitrost [{unit}]")
    ax.set_title(f"Hitrost zapestja — {video_name}")
    ax.grid(True, alpha=0.3)

    plot_path = output_dir / (video_name + "_speed.png")
    plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Graf: {plot_path}")


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