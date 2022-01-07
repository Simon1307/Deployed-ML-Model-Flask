FROM python:3.8

COPY requirements.txt /requirements.txt

RUN pip install --requirement requirements.txt

ADD src /src
ADD static /static
ADD templates /templates
ADD app.py app.py

EXPOSE 5000
CMD ["python3",  "app.py"]
