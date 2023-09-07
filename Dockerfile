# app/Dockerfile

FROM python:3.9-slim

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

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "apps/web_checking.py", "--server.port=8501", "--server.address=0.0.0.0"]