# backend/Dockerfile

FROM python:3.10.6



WORKDIR .

COPY deployment/backend/ .

RUN pip install -r requirements.txt

EXPOSE 8080

CMD ["python", "backend/main.py"]
