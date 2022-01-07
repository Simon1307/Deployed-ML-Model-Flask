FROM ubuntu:latest

RUN apt-get update \
  && apt-get upgrade -y \
  && apt-get install -y python3-pip python3-dev \
  && pip3 install --upgrade pip

COPY requirements.txt /requirements.txt

RUN pip install --requirement requirements.txt

ADD src /src
ADD static /static
ADD templates /templates
ADD app.py app.py

EXPOSE 5000
CMD ["python3",  "app.py"]
