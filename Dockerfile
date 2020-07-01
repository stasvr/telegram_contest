FROM python:3.6

COPY . /opt/app/

WORKDIR /opt/app/

RUN apt-get update

RUN pip install -r requirements.txt
RUN pip install gunicorn

WORKDIR /opt/app/src/

CMD gunicorn -w 2 main:app -b "0.0.0.0:8321" 
