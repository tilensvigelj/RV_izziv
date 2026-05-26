```text
================================================================================
HAND TRACKING FOR 9-HOLE PEG TEST (9HPT)
================================================================================

Avtomatska analiza kinematike roke iz video posnetkov standardiziranega testa
fine motorike.

----------------------------------------------------------------------------
PREGLED
----------------------------------------------------------------------------

Sistem uporablja racunalniski vid za sledenje 21 tockam na roki (MediaPipe)
in izracun kinematicnih parametrov:
- hitrost, pospesek in sunek (jerk) za vsak sklep
- akumulirana pot in stevilo vrhov hitrosti
- frekvencna analiza gibanja (FFT)
- detekcija sunkov in sunkovitosti gibanja

Deluje v dveh nacinih:
- Single mode - podrobna analiza enega videa z interaktivnim grafom
- Batch mode - populacijska analiza vec pacientov

----------------------------------------------------------------------------
ZAHTEVE
----------------------------------------------------------------------------

- Docker (namestitev: https://docker.com)
- Podatkovna mapa s posnetki v strukturi:

    data_rv_26/
    └── Data/
        ├── patient_001/
        │   └── patient_001camP_0_*.mp4
        ├── patient_002/
        │   └── patient_002camP_0_*.mp4
        └── ...

----------------------------------------------------------------------------
GRADNJA DOCKER SLIKE
----------------------------------------------------------------------------

Iz korenske mape projekta (kjer je Dockerfile):

    docker build -t hand-tracking .

----------------------------------------------------------------------------
KALIBRACIJA KAMERE (enkratno)
----------------------------------------------------------------------------

Pred prvo uporabo je potrebno kalibrirati kamero.

    docker run --rm \
      -v /pot/do/data_rv_26:/data \
      -v /pot/do/izziv:/output \
      -v /pot/do/izziv:/calibration \
      hand-tracking python calibrate.py

Izhod: calibration.npz (vsebuje matriko K in koeficiente distorzije)

Opomba: Zamenjajte /pot/do/... z dejanskimi potmi na vasem sistemu.

----------------------------------------------------------------------------
ANALIZA ENEGA VIDEA
----------------------------------------------------------------------------

    docker run --rm \
      -v /pot/do/data_rv_26:/data \
      -v /pot/do/izziv:/output \
      -v /pot/do/izziv:/calibration \
      hand-tracking python izziv/main.py \
        --video /data/Data/patient_001/patient_001camP_0_20241121_10_21_17.mp4 \
        --cam mid

Parametri:
  --video       Pot do MP4 videa (znotraj containerja)
  --cam         Ime kamere v kalibracijski datoteki (privzeto: mid)
  --no-calib    Preskoci kalibracijo (hitrost v px/s)
  --t-start     Zacetek intervala za analizo v sekundah
  --t-end       Konec intervala za analizo v sekundah

Primer s casovnim intervalom (3-5 sekund):

    docker run --rm \
      -v /pot/do/data_rv_26:/data \
      -v /pot/do/izziv:/output \
      -v /pot/do/izziv:/calibration \
      hand-tracking python izziv/main.py \
        --video /data/Data/patient_001/patient_001camP_0_20241121_10_21_17.mp4 \
        --cam mid --t-start 3.0 --t-end 5.0

Izhodi:

    /output/
    └── ime_videa/
        ├── results.csv                 # podatki po okvirjih
        ├── ime_videa_overlay.mp4       # video s skeletonom
        ├── ime_videa_kinematics.html   # interaktivni Plotly graf
        └── timeseries_features.csv     # znacilke za ta video

----------------------------------------------------------------------------
PAKETNA ANALIZA (vec pacientov)
----------------------------------------------------------------------------

Osnovni zagon (do 100 pacientov):

    docker run --rm \
      -v /pot/do/data_rv_26:/data \
      -v /pot/do/izziv:/output \
      -v /pot/do/izziv:/calibration \
      hand-tracking python izziv/main.py --batch

Omejitev stevila pacientov (npr. 20):

    docker run --rm \
      -v /pot/do/data_rv_26:/data \
      -v /pot/do/izziv:/output \
      -v /pot/do/izziv:/calibration \
      hand-tracking python izziv/main.py --batch --max-patients 20

Analiza samo dolocenega casovnega intervala za vse paciente:

    docker run --rm \
      -v /pot/do/data_rv_26:/data \
      -v /pot/do/izziv:/output \
      -v /pot/do/izziv:/calibration \
      hand-tracking python izziv/main.py --batch --t-start 5.0 --t-end 10.0

Brez kalibracije (hitrost v px/s):

    docker run --rm \
      -v /pot/do/data_rv_26:/data \
      -v /pot/do/izziv:/output \
      hand-tracking python izziv/main.py --batch --no-calib

Izhodi:

    /output/
    ├── batch_summary.csv                 # povzetek vseh pacientov
    ├── batch_features_all.csv            # agregirane znacilke
    ├── all_patients_frames_data.csv      # vsi okvirji vseh pacientov
    ├── batch_processing_histograms.png   # histogrami uspesnosti
    └── patient_XXX/
        └── results.csv                   # podrobni podatki po pacientih

----------------------------------------------------------------------------
KORELACIJSKA ANALIZA (po batch obdelavi)
----------------------------------------------------------------------------

Ko imate opravljeno batch obdelavo, lahko pozenete korelacijsko analizo:

    docker run --rm \
      -v /pot/do/izziv:/output \
      hand-tracking python izziv/analyze_early_motion.py

Izhodi:

    /output/
    ├── correlation_all_joints.csv   # Spearmanove in Pearsonove korelacije
    ├── correlation_heatmap.png      # toplotna karta korelacijske matrike
    ├── scatter_top9.png             # razsevni grafi za top 9 znacilk
    ├── correlation_barplot.png      # barvni graf vseh koeficientov
    └── best_feature_boxplot.png     # skatlasti graf za najboljso znacilko

----------------------------------------------------------------------------
STRUKTURA PODATKOVNIH MAP NA GOSTITELJU
----------------------------------------------------------------------------

    /pot/do/data_rv_26/              # gostitelj
    └── Data/
        ├── patient_001/
        │   └── patient_001camP_0_20241121_10_21_17.mp4
        ├── patient_002/
        │   └── patient_002camP_0_20241122_09_15_33.mp4
        └── ...

    /pot/do/izziv/                   # gostitelj (projektna mapa)
    ├── Dockerfile
    ├── calibrate.py
    ├── main.py
    ├── analyze_early_motion.py
    ├── calibration.npz              # ustvari kalibracija
    └── output/                      # vsi rezultati (ustvari program)

----------------------------------------------------------------------------
POGOSTI PROBLEMI IN RESITVE
----------------------------------------------------------------------------

Problem: "Permission denied" pri mountanju volumnov
Resitev: Dodajte :Z k volumnom (za SELinux):
    -v /pot/do/data_rv_26:/data:Z

Problem: Video ni zaznan / kamera ne deluje
Resitev: Preverite poti in imena datotek. Video mora vsebovati "camP_0"
         v imenu datoteke.

Problem: Roka ni zaznana v nobenem okvirju
Resitev: Preverite osvetlitev in polozaj roke. MediaPipe najbolje deluje
         pri dobri osvetlitvi in ko je roka obrnjena z dlanjo proti kameri.

Problem: Calibration.npz ne obstaja
Resitev: Najprej pozenite kalibracijo (korak 1).

Problem: Batch analiza se prekine ob izklopu SSH
Resitev: Uporabite screen ali tmux:

    screen -S batch_job
    docker run ... --batch
    Ctrl+A, D (odklop)
    screen -r batch_job (ponovni priklop)

    Ali pa uporabite nohup:

    nohup docker run ... --batch > batch.log 2>&1 &

----------------------------------------------------------------------------
POMEMBNE OPMBE
----------------------------------------------------------------------------

- Zamenjava poti: V vseh zgornjih ukazih zamenjajte /pot/do/... z dejanskimi
  potmi na vasem sistemu.

- Dovoljenja: Ce uporabljate SELinux, dodajte :Z k volumenom.

- Brez kalibracije: Hitrosti bodo izrazene v px/s namesto mm/s. To je
  uporabno za primerjavo znotraj istega videa, ne pa za primerjavo med
  razlicnimi posnetki.

- Casovni interval: Za korelacijsko analizo je priporocljiv interval
  5-10 sekund (izloci zacetni prijem in morebitne artefakte).

----------------------------------------------------------------------------
DATOTEKE IN NJIHOV POMEN
----------------------------------------------------------------------------

| Datoteka | Vsebina |
|----------|---------|
| calibration.npz | Kalibracijska matrika K in koeficienti distorzije |
| results.csv | Podatki za vsak okvir: polozaji, hitrosti, pospeski, pot |
| *_overlay.mp4 | Video z narisanim skeletom roke |
| *_kinematics.html | Interaktivni Plotly graf (hitrost, pospesek, pot) |
| batch_summary.csv | Povzetek vseh pacientov (cas, hitrost, pot, status) |
| batch_features_all.csv | Agregirane znacilke za vsakega pacienta |
| all_patients_frames_data.csv | Vsi okvirji vseh pacientov v eni datoteki |
| correlation_*.csv/png | Rezultati korelacijske analize |

