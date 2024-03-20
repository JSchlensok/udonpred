FROM python:3.12
RUN git clone https://github.com/JSchlensok/pp2-2023 ./app
WORKDIR ./app

RUN pip install -r requirements.txt
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

RUN python -m src.pipeline.caid.download_prostt5_weights

ENTRYPOINT ["python", "-m", "src.pipeline.caid.run", "--prostt5-cache", "./prostt5_cache"]