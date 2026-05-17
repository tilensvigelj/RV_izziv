FROM python:3.11-slim

WORKDIR /izziv

RUN pip install numpy
RUN pip install opencv-python
RUN pip install matplotlib
RUN pip install mediapipe==0.10.9
RUN pip install pandas
RUN pip install protobuf==3.20.3
RUN pip install plotly

RUN apt-get update --fix-missing && apt-get install -y \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY izziv/ ./izziv/

CMD ["python", "-m", "izziv.main"]