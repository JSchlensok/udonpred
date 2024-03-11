FROM python:3.12
RUN git clone https://github.com/JSchlensok/pp2-2023 ./app
WORKDIR ./app

RUN pip install -r requirements.txt
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

ENTRYPOINT ["python", "-m", "src.pipeline.caid.run"]