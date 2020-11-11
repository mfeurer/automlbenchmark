#!/usr/bin/env bash
HERE=$(dirname "$0")
AMLB_DIR="$1"
VERSION="0.10.0"
REPO=${3:-"https://github.com/automl/auto-sklearn.git"}
PKG=${4:-"auto-sklearn"}
if [[ "$VERSION" == "latest" ]]; then
    VERSION="master"
fi

# creating local venv
. $HERE/../shared/setup.sh $HERE

TARGET_DIR="$HERE/lib/Auto-PyTorch"

git clone https://github.com/automl/Auto-PyTorch $TARGET_DIR
cd $TARGET_DIR
git checkout test_singlerun
cd $HERE


echo $HERE/requirements.txt
PIP install --no-cache-dir -r $HERE/requirements.txt
echo $TARGET_DIR/requirements.txt
PIP install --no-cache-dir -r $TARGET_DIR/requirements.txt
PIP uninstall autopytorch
PIP install --no-cache-dir -e $TARGET_DIR
