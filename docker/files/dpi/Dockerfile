
FROM dreadl0ck/netcap:alpine-v0.6.9

RUN apk add --update --no-cache \
    bash \
    tcpdump \
    iperf \
    busybox-extras \
    iproute2 \
    iputils \
    curl \
    python3

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
COPY ./docker/files/dpi/shared/startup.sh /home/startup.sh
RUN chmod +x startup.sh

RUN mkdir /home/ncap
WORKDIR /home/ncap

ENTRYPOINT ["/home/startup.sh"]
