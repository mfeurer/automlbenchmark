#!/usr/bin/env bash
HERE=$(dirname "$0")
#AMLB_DIR="$1"
#VERSION=${2:-"v.0.6.0"}

# creating local venv
. $HERE/../shared/setup.sh $HERE

TARGET_DIR="$HERE/lib/Auto-PyTorch"
if [[ ! -e "$TARGET_DIR" ]]; then
    git clone https://github.com/automl/Auto-PyTorch $TARGET_DIR
    cd $TARGET_DIR
    git checkout develop
    cd $HERE
fi
PIP install --no-cache-dir -r $HERE/requirements.txt
PIP install --no-cache-dir -r $TARGET_DIR/requirements.txt
PIP install --no-cache-dir -e $TARGET_DIR
