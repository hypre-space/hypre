#!/bin/sh

prob=beam
p=4
echo ${prob}.dat
poe distExtBeamDriver.x ${prob}.dat \
            -euilib us -adapter_use dedicated -cpu_use unique \
            -procs $p -rmpool 0 -euidevice css0 -labelio yes

