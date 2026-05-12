import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("/data")  # mountan z -v

# MediaPipe setup
mp_hands = mp.solutions.hands
LANDMARK_NAMES = [lm.name for lm in mp.solutions.hands.HandLandmark]


def process_video(video_path: Path) -> pd.DataFrame:
    """Iz enega videa izračuna hitrosti za vsak sklepni točki roke."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

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
                # Vzamemo samo prvo roko
                lm_list = result.multi_hand_landmarks[0].landmark

                # Koordinate v pikslih
                curr = {
                    name: (lm.x * width, lm.y * height)
                    for name, lm in zip(LANDMARK_NAMES, lm_list)
                }

                row = {"frame": frame_idx, "time_s": frame_idx / fps}

                for name, (x, y) in curr.items():
                    row[f"{name}_x"] = x
                    row[f"{name}_y"] = y

                    # Hitrost [px/s] = razdalja med frames / čas med frames
                    if prev_landmarks and name in prev_landmarks:
                        px, py = prev_landmarks[name]
                        dist = np.sqrt((x - px) ** 2 + (y - py) ** 2)
                        row[f"{name}_speed_px_s"] = dist * fps
                    else:
                        row[f"{name}_speed_px_s"] = np.nan

                records.append(row)
                prev_landmarks = curr
            else:
                prev_landmarks = None  # Roka ni vidna

            frame_idx += 1

    cap.release()
    return pd.DataFrame(records)


def main():
    patient_dirs = sorted(DATA_DIR.glob("patient_*"))

    if not patient_dirs:
        print(f"Ni najdenih map v {DATA_DIR}")
        return

    for patient_dir in patient_dirs:
        videos = sorted(patient_dir.glob("*.mp4"))
        print(f"\n=== {patient_dir.name} ({len(videos)} videov) ===")

        for video_path in videos:
            print(f"  Obdelava: {video_path.name} ...", end=" ", flush=True)

            df = process_video(video_path)

            if df.empty:
                print("ni zaznane roke, preskočeno.")
                continue

            # Povprečna hitrost zapestja (WRIST) kot summary
            wrist_speed = df["WRIST_speed_px_s"].dropna()
            print(f"{len(df)} frameov | "
                  f"WRIST avg: {wrist_speed.mean():.1f} px/s | "
                  f"max: {wrist_speed.max():.1f} px/s")

            # Shrani CSV zraven videa
            out_path = video_path.with_suffix(".csv")
            df.to_csv(out_path, index=False)
            print(f"    → {out_path}")


if __name__ == "__main__":
    main()