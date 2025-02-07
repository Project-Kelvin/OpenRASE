FROM caddy:2.9-builder-alpine AS builder

RUN xcaddy build --with github.com/corazawaf/coraza-caddy/v2

FROM caddy:2.9-alpine

RUN apk add --update --no-cache \
    bash \
    tcpdump \
    iperf \
    busybox-extras \
    iproute2 \
    iputils

COPY --from=builder /usr/bin/caddy /usr/bin/caddy
