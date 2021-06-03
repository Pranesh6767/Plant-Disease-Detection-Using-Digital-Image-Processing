FROM python:3.6-alpine as base

FROM base as builder

RUN mkdir /install
WORKDIR /install

# Install app dependencies
COPY ./requirements.txt /requirements.txt

RUN pip install --install-option="--prefix=/install" -r /requirements.txt

FROM base
COPY --from=builder /install /usr/local
COPY . /app

WORKDIR /app

EXPOSE 5000

CMD [ "python", "server.py" ]