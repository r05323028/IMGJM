#! /bin/bash
# Copyright (c) 2019 seanchang
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

pipenv install --skip-lock
pipenv run python -m IMGJM.utils.transform_glove_to_gensim_format