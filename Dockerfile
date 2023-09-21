# app/Dockerfile

FROM python:3.7-slim

EXPOSE ${PORT}

WORKDIR /app

RUN apt-get update && apt-get install -y \
  build-essential \
  ffmpeg \
  libsm6 \
  libxext6 \
  curl \
  software-properties-common \
  git \
  && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/wiktorlazarski/head-segmentation.git .

RUN pip3 install -e .

ENTRYPOINT exec python -m streamlit run apps/web_checking.py --server.port=$PORT --server.address=0.0.0.0
