# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install any needed packages specified in requirements.txt
RUN apt-get update && apt-get install -y \
    net-tools \
    iputils-ping \
    iproute2 \
    curl

ENV POETRY_HOME="/opt/poetry"
ENV PATH="$POETRY_HOME/bin:$PATH"
RUN curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /home
RUN mkdir sfc-emulator
COPY ./packages/ /home/sfc-emulator/packages/

COPY ./apps/sfc_classifier/ /home/sfc-emulator/apps/sfc_classifier/
COPY config.yaml /home/sfc-emulator/apps/sfc_classifier/

WORKDIR /home/sfc-emulator/apps/sfc_classifier/
RUN poetry config virtualenvs.in-project true && \
    poetry install --no-root --only main

CMD poetry run python sfc_classifier.py
