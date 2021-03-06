#!/usr/bin/env bash
HERE=$(dirname "$0")
VERSION=${1:-"latest"}
REPO=${2:-"https://github.com/hyperopt/hyperopt-sklearn.git"}
PKG=${3:-"hyperopt-sklearn"}
if [[ "$VERSION" == "latest" ]]; then
    VERSION="master"
fi

. ${HERE}/../shared/setup.sh ${HERE}

if [[ "$VERSION" =~ ^[0-9] ]]; then
    PIP install --no-cache-dir ${PKG}==${VERSION}
else
#    PIP install --no-cache-dir -e git+${REPO}@${VERSION}#egg=${PKG}
    LIB=$(echo ${PKG} | sed "s/\[.*\]//")
    TARGET_DIR="${HERE}/lib/${LIB}"
    rm -Rf ${TARGET_DIR}
    git clone --depth 1 --single-branch --branch ${VERSION} --recurse-submodules ${REPO} ${TARGET_DIR}
    PIP install -e ${HERE}/lib/${PKG}
fi
