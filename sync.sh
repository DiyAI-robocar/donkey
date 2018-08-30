#!/bin/bash

if [ -z "$1" ]; then
  set donkeypi
fi

rsync -avR --no-implied-dirs cars/* pi@$1:
rsync -avR --no-implied-dirs donkeycar pi@$1:env/lib/python3.5/site-packages/
