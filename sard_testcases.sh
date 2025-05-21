#!/bin/bash
PUBLIC_URL="https://disk.yandex.ru/d/JBuGu9n-DZTCFA"
API_URL="https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key=${PUBLIC_URL}"
DOWNLOAD_URL=$(curl -s "$API_URL" | grep -oP '"href"\s*:\s*"\K[^"]+')
ARCHIVE_PATH="$(pwd)/sard_testcases.tar.gz"
EXTRACT_DIR="$ML_TRIAGE_HOME/dataset"

if [ -z "$DOWNLOAD_URL" ]; then
    echo "Can't obtain download link from Yandex API"
    exit 1
fi

echo "Downloading archive to:"
wget --quiet --show-progress "$DOWNLOAD_URL" -O "$ARCHIVE_PATH"

echo "Extracting archive to: $EXTRACT_DIR"
tar -xzf $ARCHIVE_PATH -C $EXTRACT_DIR
