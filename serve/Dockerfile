FROM python:latest

ENV ENVIRONMENT production

RUN apt-get update && apt-get install -y software-properties-common \
    nginx

RUN python --version
RUN pip3 install --upgrade pip

RUN mkdir /app

COPY . /app
COPY nginx.conf /usr/share/nginx/nginx.conf

WORKDIR  /app

RUN pip3 install Cython numpy
RUN pip3 install --no-cache-dir -r requirements.txt

ENTRYPOINT [ "python", "serve.py"] 