#!/bin/bash

net capture -iface eth0 -dpi -debug &

cd /home/OpenRASE/apps/vnf_proxy
pnpm run start &

# Wait for any process to exit
wait -n

# Exit with the exit code of the first process that exits non-zero
exit $?
