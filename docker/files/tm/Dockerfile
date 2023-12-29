FROM ntop/ntopng:stable

RUN apt-get update && apt-get install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa -y

RUN apt-get update && apt-get install -y \
    bash \
    tcpdump \
    iperf \
    iproute2 \
    curl \
    python3.10 \
    python3-pip

ENV POETRY_HOME="/opt/poetry"
ENV PATH="$POETRY_HOME/bin:$PATH"
RUN curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /home
RUN mkdir sfc-emulator
COPY ./packages/ /home/sfc-emulator/packages/

COPY ./apps/vnf_proxy/ /home/sfc-emulator/apps/vnf_proxy/
COPY config.yaml /home/sfc-emulator/apps/vnf_proxy/

WORKDIR /home/sfc-emulator/apps/vnf_proxy/
RUN poetry config virtualenvs.in-project true && \
    poetry install --no-root --only main

WORKDIR /home
COPY ./docker/files/tm/shared/startup.sh /home/startup.sh
RUN chmod +x startup.sh

ENTRYPOINT [ "./startup.sh" ]
