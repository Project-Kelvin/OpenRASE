FROM node:22-alpine3.19

RUN apk add --update --no-cache \
    bash \
    tcpdump \
    iperf \
    busybox-extras \
    iproute2 \
    iputils \
    curl

RUN corepack enable pnpm
WORKDIR /home
RUN mkdir OpenRASE
COPY ./package.json /home/OpenRASE/package.json
COPY ./pnpm-lock.yaml /home/OpenRASE/pnpm-lock.yaml
COPY ./pnpm-workspace.yaml /home/OpenRASE/pnpm-workspace.yaml
COPY ./packages/typescript /home/OpenRASE/packages/typescript
COPY ./apps/sfc_classifier/ /home/OpenRASE/apps/sfc_classifier/
COPY config.yaml /home/OpenRASE/apps/sfc_classifier/config.yaml
WORKDIR /home/OpenRASE
RUN pnpm install

WORKDIR /home/OpenRASE/apps/sfc_classifier/
RUN pnpm install

CMD pnpm run start
