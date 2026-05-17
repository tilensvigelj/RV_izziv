import cv2
import numpy as np
from pathlib import Path
import json

'''
docker run   -v /media/FastDataMama/zigab/calibration_photos:/calibration_photos  
-v /media/FastDataMama/tilens/izziv:/output 
hand-tracking python /izziv/izziv/calibrate.py

'''

# Kalibracija šahovnice
CHESSBOARD = (9, 6)       # notranja vogala (polja - 1)
SQUARE_SIZE = 20.0        # mm

CAM_DIRS = {
    "mid":   Path("/calibration_photos/cam_mid_resized2"),
}

OUTPUT_FILE = Path("/output/calibration.npz")


def get_3d_points():
    """Pripravi 3D točke šahovnice v koordinatnem sistemu šahovnice (z=0)."""
    objp = np.zeros((CHESSBOARD[0] * CHESSBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE
    return objp


def calibrate_camera(image_dir: Path, name: str):
    """Kalibrira eno kamero iz mape s kalibracijskimi slikami."""
    images = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
    print(f"Iščem slike v: {image_dir}")
    print(f"Najdenih slik: {len(images)}")
    for img in images:
        print(f"  {img}")

    objp = get_3d_points()
    obj_points = []   # 3D točke v prostoru
    img_points = []   # 2D točke na sliki
    img_size = None
    found_count = 0

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for img_path in images:
        img = cv2.imread(str(img_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_size = gray.shape[::-1]  # (width, height)

        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD, None)

        if ret:
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(objp)
            img_points.append(corners_refined)
            found_count += 1
            print(f"  ✓ {img_path.name}")
        else:
            print(f"  ✗ {img_path.name} — šahovnica ni najdena")

    print(f"  Najdenih: {found_count}/{len(images)}")

    if found_count < 5:
        print(f"  OPOZORILO: premalo slik za zanesljivo kalibracijo!")
        return None

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_size, None, None
    )

    # Reprojection error — nižje je boljše (< 1.0 je ok)
    mean_error = 0
    for i in range(len(obj_points)):
        projected, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, dist)
        mean_error += cv2.norm(img_points[i], projected, cv2.NORM_L2) / len(projected)
    mean_error /= len(obj_points)

    print(f"  Reprojection error: {mean_error:.4f} px")
    print(f"  Fokalna razdalja: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}")
    print(f"  Optično središče: cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")
    print(f"\n  Kalibrijska matrika (K):")
    print(f"  [[{K[0,0]:.8f}, 0.0, {K[0,2]:.8f}]")
    print(f"   [0.0, {K[1,1]:.8f}, {K[1,2]:.8f}]")
    print(f"   [0.0, 0.0, 1.0]]")
    print(f"\n  Distorzijski koeficienti:")
    print(f"  {dist.flatten().tolist()}")
    
    return {
        "K": K,
        "dist": dist,
        "img_size": img_size,
        "reprojection_error": mean_error,
        "obj_points": obj_points,
        "img_points": img_points,
    }




def main():
    results = {}

    # Kalibriraj vsako kamero posebej
    for name, directory in CAM_DIRS.items():
        cam_data = calibrate_camera(directory, name)
        if cam_data:
            results[name] = cam_data

    # Shrani rezultate
    save_data = {}
    for name, data in results.items():
        save_data[f"{name}_K"] = data["K"]
        save_data[f"{name}_dist"] = data["dist"]
        save_data[f"{name}_img_size"] = np.array(data["img_size"])
        save_data[f"{name}_reprojection_error"] = np.array(data["reprojection_error"])


    np.savez(str(OUTPUT_FILE), **save_data)
    print(f"\nKalibracija shranjena: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()