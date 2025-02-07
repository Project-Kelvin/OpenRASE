
FROM golang:1.23-alpine3.21 as builder

WORKDIR $GOPATH/src/go.k6.io/k6

RUN apk --no-cache add git
RUN go install go.k6.io/xk6/cmd/xk6@latest
RUN xk6 build --with github.com/grafana/xk6-output-influxdb --output /tmp/k6

FROM alpine:3.17
RUN apk add --no-cache ca-certificates

COPY --from=builder /tmp/k6 /usr/bin/k6

WORKDIR /home/k6

CMD ["tail", "-f", "/dev/null"]
