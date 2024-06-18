
FROM dreadl0ck/netcap:alpine-v0.6.9

RUN apk add --update --no-cache \
    bash \
    tcpdump \
    iperf \
    busybox-extras \
    iproute2 \
    iputils \
    curl

RUN curl -fsSL https://unofficial-builds.nodejs.org/download/release/v22.2.0/node-v22.2.0-linux-x64-musl.tar.gz | tar -xz -C /usr/local --strip-components=1
RUN corepack enable pnpm

WORKDIR /home
RUN mkdir OpenRASE
COPY ./package.json /home/OpenRASE/package.json
COPY ./pnpm-lock.yaml /home/OpenRASE/pnpm-lock.yaml
COPY ./pnpm-workspace.yaml /home/OpenRASE/pnpm-workspace.yaml
COPY ./packages/typescript /home/OpenRASE/packages/typescript
COPY ./apps/vnf_proxy/ /home/OpenRASE/apps/vnf_proxy/
COPY config.yaml /home/OpenRASE/apps/vnf_proxy/config.yaml
WORKDIR /home/OpenRASE
RUN pnpm install

WORKDIR /home/OpenRASE/apps/vnf_proxy/
RUN pnpm install

WORKDIR /home
COPY ./docker/files/dpi/shared/startup.sh /home/startup.sh
RUN chmod +x startup.sh

RUN mkdir /home/ncap
WORKDIR /home/ncap

ENTRYPOINT ["/home/startup.sh"]
