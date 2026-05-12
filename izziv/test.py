import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

'''docker run \
  -v /media/FastDataMama/data_rv_26:/data \
  -v /media/FastDataMama/tilens/izziv:/output \
  hand-tracking python /izziv/izziv/test.py'''

DATA_DIR = Path("/data")
OUTPUT_DIR = Path("/output")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
LANDMARK_NAMES = [lm.name for lm in mp.solutions.hands.HandLandmark]


def process_video(video_path: Path, output_dir: Path) -> pd.DataFrame:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Video writer za overlay
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

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                lm_list = result.multi_hand_landmarks[0].landmark

                # Nariši skeletonk na frame
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
                        px, py = prev_landmarks[name]
                        dist = np.sqrt((x - px) ** 2 + (y - py) ** 2)
                        row[f"{name}_speed_px_s"] = dist * fps
                    else:
                        row[f"{name}_speed_px_s"] = np.nan

                # Hitrost zapestja na video
                wrist_speed = row.get("WRIST_speed_px_s", 0) or 0
                cv2.putText(
                    frame,
                    f"WRIST speed: {wrist_speed:.1f} px/s",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )

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

    for joint in joints:
        col = f"{joint}_speed_px_s"
        if col in df.columns:
            ax.plot(df["time_s"], df[col], label=joint, linewidth=1.2)

    ax.set_xlabel("Čas [s]")
    ax.set_ylabel("Hitrost [px/s]")
    ax.set_title(f"Hitrost sklepov — {video_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plot_path = output_dir / (video_name + "_speed.png")
    plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Graf: {plot_path}")


def main():
    video_path = Path("/data/Data/patient_001/patient_001camP_0_20241121_10_21_17.mp4")

    out_dir = OUTPUT_DIR / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Obdelava: {video_path.name} ...")
    df = process_video(video_path, out_dir)

    if df.empty:
        print("Ni zaznane roke.")
        return

    save_speed_plot(df, out_dir, video_path.stem)

    csv_path = out_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  → CSV: {csv_path}")


if __name__ == "__main__":
    main()