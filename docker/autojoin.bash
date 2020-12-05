#!/usr/bin/env bash
#
# Typical usage: ./join.bash vrx
#

IMG="vrx"

xhost +
containerid=$(docker ps -aqf "ancestor=${IMG}")
docker exec --privileged -e DISPLAY=${DISPLAY} -e LINES=`tput lines` -it ${containerid} bash
xhost -
