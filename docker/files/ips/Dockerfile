FROM ubuntu:20.04

RUN apt-get update && apt-get install software-properties-common -y
RUN add-apt-repository ppa:oisf/suricata-stable -y

RUN apt-get update && apt-get dist-upgrade -y

RUN apt-get install -y \
    suricata \
    jq \
    net-tools \
    iputils-ping \
    iproute2 \
    curl \
    bash \
    iptables

RUN curl -fsSL https://deb.nodesource.com/setup_23.x | bash -
RUN apt-get install -y nodejs

RUN corepack enable pnpm
RUN npm install -g pm2

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
WORKDIR /home/OpenRASE/packages/typescript/shared
RUN pnpm install
RUN pnpm build
WORKDIR /home/OpenRASE/apps/vnf_proxy/
RUN pnpm install

WORKDIR /home
COPY ./docker/files/ips/shared/startup.sh /home/startup.sh
RUN chmod +x startup.sh

CMD ./startup.sh
