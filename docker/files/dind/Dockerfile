FROM docker:24.0-dind

RUN apk add --update --no-cache \
    bash \
    tcpdump \
    iperf \
    busybox-extras \
    iproute2 \
    iputils \
    curl

COPY ./docker/files/dind/shared/daemon.json /etc/docker/daemon.json

WORKDIR /home/
