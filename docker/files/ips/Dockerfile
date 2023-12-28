FROM ubuntu:20.04

RUN apt-get update && apt-get install software-properties-common -y
RUN add-apt-repository ppa:oisf/suricata-stable -y
RUN add-apt-repository ppa:deadsnakes/ppa -y

RUN apt-get update && apt-get dist-upgrade -y

RUN apt-get install -y \
    suricata \
    jq \
    python3.10 \
    python3-pip \net-tools \
    iputils-ping \
    iproute2 \
    curl \
    bash \
    iptables

ENV POETRY_HOME="/opt/poetry"
ENV PATH="$POETRY_HOME/bin:$PATH"
RUN curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /home
RUN mkdir sfc-emulator
COPY ./shared/ /home/sfc-emulator/shared/
COPY poetry.lock pyproject.toml /home/sfc-emulator/

COPY ./apps/vnf_proxy/ /home/sfc-emulator/apps/vnf_proxy/
COPY config.yaml /home/sfc-emulator/apps/vnf_proxy/

WORKDIR /home/sfc-emulator/apps/vnf_proxy/
RUN poetry config virtualenvs.in-project true && \
    poetry install --no-root --only main

WORKDIR /home
COPY ./docker/files/ips/shared/startup.sh /home/startup.sh
RUN chmod +x startup.sh

CMD ./startup.sh
