FROM python:trixie

WORKDIR /izziv

RUN pip install numpy
RUN pip install opencv-python
RUN pip install matplotlib
RUN pip install mediapipe 
RUN pip install pandas

COPY izziv/ ./izziv/

CMD ["python", "-m", "izziv.main"]