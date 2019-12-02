#! /usr/bin/bash
# Copyright (c) 2019 seanchang
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

OUTPUT_DIR="embeddings"
FASTTEXT_EMBEDDING="wiki.zh"
DOMAIN="https://dl.fbaipublicfiles.com/fasttext/vectors-wiki"
DOWNLOAD_URL="${DOMAIN}/${FASTTEXT_EMBEDDING}.zip"

if [ ! -e "${OUTPUT_DIR}/${FASTTEXT_EMBEDDING}.zip" ]
then
    wget --no-check-certificate  "${DOWNLOAD_URL}" -P "${OUTPUT_DIR}/"
fi
if [ ! -e "${OUTPUT_DIR}/${FASTTEXT_EMBEDDING}.bin" ]
then
    unzip "${OUTPUT_DIR}/${FASTTEXT_EMBEDDING}.zip" -d ${OUTPUT_DIR}
fi