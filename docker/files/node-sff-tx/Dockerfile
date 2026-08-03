FROM node:23-alpine3.20

RUN apk add --update --no-cache \
    bash \
    tcpdump \
    iperf \
    busybox-extras \
    iproute2 \
    iputils \
    curl

RUN corepack enable pnpm
RUN npm install -g pm2
WORKDIR /home
RUN mkdir OpenRASE
COPY ./package.json /home/OpenRASE/package.json
COPY ./pnpm-lock.yaml /home/OpenRASE/pnpm-lock.yaml
COPY ./pnpm-workspace.yaml /home/OpenRASE/pnpm-workspace.yaml
COPY ./packages/typescript /home/OpenRASE/packages/typescript
COPY ./apps/sff-tx/ /home/OpenRASE/apps/sff-tx/
COPY config.yaml /home/OpenRASE/apps/sff-tx/config.yaml
WORKDIR /home/OpenRASE
RUN pnpm install
WORKDIR /home/OpenRASE/packages/typescript/shared
RUN pnpm install
RUN pnpm build
WORKDIR /home/OpenRASE/apps/sff-tx/
RUN pnpm install

CMD pnpm run start
