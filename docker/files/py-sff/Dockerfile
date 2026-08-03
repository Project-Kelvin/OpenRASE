FROM python:3.10-slim

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
COPY ./apps/sff-rx/ ./sfc-emulator/apps/sff/
COPY config.yaml /home/sfc-emulator/apps/sff/

WORKDIR /home/sfc-emulator/apps/sff/
RUN poetry config virtualenvs.in-project true && \
    poetry install --no-root --only main

CMD poetry run python sff.py
