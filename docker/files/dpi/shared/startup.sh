#!/bin/bash

net capture -iface eth0 -dpi -debug &

cd /home/sfc-emulator/apps/vnf_proxy
poetry run python vnf_proxy.py &

# Wait for any process to exit
wait -n

# Exit with the exit code of the first process that exits non-zero
exit $?
