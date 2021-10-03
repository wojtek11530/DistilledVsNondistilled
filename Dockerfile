FROM python:3.8-slim-buster

WORKDIR /app
COPY ./src ./src
COPY ./models/.gitkeep ./models/.gitkeep
COPY ./data/.gitkeep ./data/.gitkeep
COPY ./requirements.txt .
COPY ./setup.cfg .
RUN pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 \
-f https://download.pytorch.org/whl/torch_stable.html
RUN pip install -r requirements.txt
