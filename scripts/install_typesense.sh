#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

set -ex

# Install programs into the active python virtualenv, under
# $VIRTUAL_ENV/.thinkingbox/
# and link executables to $VIRTUAL_ENV/bin/

# Check if we are in a Python virtual environment
if [[ -n "$VIRTUAL_ENV" && -d "$VIRTUAL_ENV" ]]; then
    VENV_DIR="$(readlink -f "$VIRTUAL_ENV")"
    echo "Python virtualenv: $VENV_DIR"
else
    echo "Error: Not inside a Python virtual environment." >&2
    exit 1
fi

INSTALL_DIR="$VENV_DIR/.thinkingbox"

# Create temporary directory and ensure it's deleted on exit
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

download_check_extract() {
    # download_check_extract $NAME $DOWNLOAD_URL $EXPECTED_CHECKSUM $EXTRACT_DEST
    local NAME="$1"
    local DOWNLOAD_URL="$2"
    local EXPECTED_CHECKSUM="$3"
    local EXTRACT_DEST="$4"

    # Download and install typesense in the virtual environment
    DOWNLOAD_DEST="$TMP_DIR/${NAME}.tar.gz"
    curl -fL "$DOWNLOAD_URL" -o "$DOWNLOAD_DEST"
    DOWNLOAD_CHECKSUM=$(sha256sum "$DOWNLOAD_DEST" | cut -d' ' -f1)
    if [[ "$EXPECTED_CHECKSUM" != "$DOWNLOAD_CHECKSUM" ]]; then
        echo "Error: $NAME checksum does not match!" >&2
        exit 1
    fi
    mkdir -p "$EXTRACT_DEST"
    tar -xf "$DOWNLOAD_DEST" -C "$EXTRACT_DEST"
    chmod -R +rw "$EXTRACT_DEST"
    echo "$NAME installed at: $EXTRACT_DEST"
}

link_exe() {
    # link_exe $SOURCE $NAME
    # link an executable to the bin directory of the active python virtualenv
    local SOURCE="$1"
    local NAME="$2"

    chmod +x "$SOURCE"
    ln -sf "$(readlink -f "$SOURCE")" "$VENV_DIR/bin/$NAME"
}

OS=$(uname -s)
ARCH=$(uname -m)

if [[ "$OS" = "Darwin" ]]; then
    if [[ "$ARCH" = "arm64" ]]; then
        download_check_extract \
            typesense \
            "https://dl.typesense.org/releases/30.1/typesense-server-30.1-darwin-arm64.tar.gz" \
            dcbf09f30ecea7c3ff3245043611f6ec2eea87a6851b4462572825bb87e3be33 \
            "$INSTALL_DIR/typesense"
    elif [[ "$ARCH" = "x86_64" ]]; then
        download_check_extract \
            typesense \
            "https://dl.typesense.org/releases/30.1/typesense-server-30.1-darwin-amd64.tar.gz" \
            34fe199ed1462a7dc54418f360ac97a5265806a786298fb5863a2ea95f70d47c \
            "$INSTALL_DIR/typesense"
    else
        echo "Unsupported architecture on macOS: $ARCH" >&2
        exit 1
    fi
elif [[ "$OS" = "Linux" ]]; then
    if [[ "$ARCH" = "x86_64" ]]; then
        download_check_extract \
            typesense \
            "https://dl.typesense.org/releases/30.1/typesense-server-30.1-linux-amd64.tar.gz" \
            4a1c9ce33efa70b26e24a043bd48792856212845ee0061ea6b9f8a2b7ad76c89 \
            "$INSTALL_DIR/typesense"
    elif [[ "$ARCH" = "aarch64" ]]; then
        download_check_extract \
            typesense \
            "https://dl.typesense.org/releases/30.1/typesense-server-30.1-linux-arm64.tar.gz" \
            39212289689b763ae219f322cb87d977b751662f5856b68543bccdaa448200ae \
            "$INSTALL_DIR/typesense"
    else
        echo "Unsupported architecture on Linux: $ARCH" >&2
        exit 1
    fi
else
    echo "Unsupported platform: $OS" >&2
    exit 1
fi

link_exe "$INSTALL_DIR/typesense/typesense-server" typesense-server
