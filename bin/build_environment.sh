#! /bin/bash

pipenv install --skip-lock
pipenv run python -m IMGJM.utils.transform_glove_to_gensim_format