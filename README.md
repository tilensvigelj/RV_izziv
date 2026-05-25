Grajenje projekta: 
docker build -t hand-tracking .

1.) Analiza enega videa:
    docker run \
    -v /media/FastDataMama/data_rv_26:/data \
    -v /media/FastDataMama/tilens/izziv:/output \
    -v /media/FastDataMama/tilens/izziv:/calibration \
    hand-tracking python main.py \
        --video /data/Data/patient_001/patient_001camP_0_20241121_10_21_17.mp4 \
        --cam mid

output:
   - video overlay
   - kinematics html graf
   - results.csv

2.) Analiza več videov (default 100):
    docker run \
  -v /media/FastDataMama/data_rv_26:/data \
  -v /media/FastDataMama/tilens/izziv:/output \
  -v /media/FastDataMama/tilens/izziv:/calibration \
  hand-tracking python main.py --batch

  poljubno število pacientov
    ... python main.py --batch --max-patients 20

output:
    batch_summary.csv
    batch_histograms.html

3.) kalibracija:
    docker run \
  -v /media/FastDataMama/data_rv_26:/data \
  -v /media/FastDataMama/tilens/izziv:/output \
  -v /media/FastDataMama/tilens/izziv:/calibration \
  hand-tracking python calibrate.py

output:
    calibration.npz




