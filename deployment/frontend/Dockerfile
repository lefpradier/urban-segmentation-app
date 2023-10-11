# backend/Dockerfile

FROM python:3.10.6

COPY deployment/frontend .

WORKDIR .

RUN pip install -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 libgl1-mesa-glx -y

EXPOSE 5000

CMD ["python", "app.py"]
