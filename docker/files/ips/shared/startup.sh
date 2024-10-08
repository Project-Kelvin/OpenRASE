#!/bin/bash

suricata -c /etc/suricata/suricata.yaml -q 0 -D

iptables -I INPUT -p tcp --sport 80  -j NFQUEUE
iptables -I OUTPUT -p tcp --dport 80 -j NFQUEUE

cd /home/OpenRASE/apps/vnf_proxy
pnpm run start &

# Wait for any process to exit
wait -n

# Exit with the exit code of the first process that exits non-zero
exit $?
