#!/bin/bash
srcpath="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
python3 $srcpath/main.py $@ 2>/dev/null
