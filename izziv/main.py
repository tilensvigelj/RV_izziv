import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from pathlib import Path

#docker build -t hand-tracking .
#docker run -v /media/FastDataMama/data_rv_26:/data hand-tracking

DATA_DIR = Path("/data/Data")  # mountan z -v

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
    # Določi kateri video želiš obdelati
    video_path = Path("/data/Data/patient_001/patient_001camP_0_20241121_10_21_17.mp4")

    print(f"Obdelava: {video_path.name} ...")
    df = process_video(video_path)

    if df.empty:
        print("Ni zaznane roke.")
        return

    print(df[["frame", "time_s", "WRIST_speed_px_s"]].head(20))

    # Shrani v /output znotraj containerja
    out_path = Path("/output") / video_path.stem
    out_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path / "results.csv", index=False)
    print(f"\nShranjen: {out_path}/results.csv")


if __name__ == "__main__":
    main()