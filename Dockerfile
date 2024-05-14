FROM python:3.12

WORKDIR /work
COPY . /work

RUN apt update
RUN pip install --upgrade pip
RUN pip install -r requirements.txt