FROM ntop/ntopng:stable

RUN apt-get update && apt-get install -y \
    bash \
    tcpdump \
    iperf \
    iproute2 \
    curl

RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
RUN apt-get install -y nodejs

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
COPY ./docker/files/tm/shared/startup.sh /home/startup.sh
RUN chmod +x startup.sh

ENTRYPOINT [ "./startup.sh" ]
