#!/usr/bin/env bash
HERE=$(dirname "$0")
#AMLB_DIR="$1"
#VERSION=${2:-"v.0.6.0"}

# creating local venv
. $HERE/../shared/setup.sh $HERE

if [[ -x "$(command -v apt-get)" ]]; then
    SUDO apt-get install -y build-essential swig
fi
# by passing the module directory to `setup.sh`, it tells it to automatically create a virtual env under the current module.
# this virtual env is then used to run the exec.py only, and can be configured here using `PIP` and `PY` commands.
#curl "https://raw.githubusercontent.com/automl/auto-sklearn/${VERSION}/requirements.txt" | sed '/^$/d' | while read -r i; do PIP install "$i"; done
#PIP install --no-cache-dir -r "https://raw.githubusercontent.com/automl/auto-sklearn/${VERSION}/requirements.txt"
TARGET_DIR="$HERE/lib/autosklearn"
if [[ ! -e "$TARGET_DIR" ]]; then
    git clone https://github.com/franchuterivera/auto-sklearn.git $TARGET_DIR
    cd "$TARGET_DIR"
    git checkout precision
    git log --name-status HEAD^..HEAD
    cd "$HERE"
fi
PIP install --no-cache-dir -r $TARGET_DIR/requirements.txt
PIP install --no-cache-dir -e $TARGET_DIR
