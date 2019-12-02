#! /usr/bin/bash
# Copyright (c) 2019 seanchang
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

GLOVE_NAME="glove.840B.300d"
OUTPUT_DIR="embeddings"
REMOTE_PATH="http://nlp.stanford.edu/data/${GLOVE_NAME}.zip"

if [ ! -e ${OUTPUT_DIR} ]
then
    mkdir ${OUTPUT_DIR}
fi
if [ ! -e "${OUTPUT_DIR}/${GLOVE_NAME}.zip" ]
then
    wget --no-check-certificate  ${REMOTE_PATH} -P "${OUTPUT_DIR}/"
fi
if [ ! -e "${GLOVE_NAME}.txt" ]
then
    unzip "${OUTPUT_DIR}/${GLOVE_NAME}.zip" -d ${OUTPUT_DIR}
fi