#!/bin/bash

wget -O data/mls_polish.tar.gz https://dl.fbaipublicfiles.com/mls/mls_polish.tar.gz
tar xzvf data/mls_polish.tar.gz -C data  && rm -r data/mls_polish.tar.gz

python preprocess.py --model-size tiny --input-dir data/mls_polish/train/audio --output-dir data/train_mels --transcripts-path data/mls_polish/train/transcripts.txt --metadata-output-path data/train_metadata.csv
python preprocess.py --model-size tiny --input-dir data/mls_polish/dev/audio --output-dir data/valid_mels --transcripts-path data/mls_polish/dev/transcripts.txt --metadata-output-path data/valid_metadata.csv
python preprocess.py --model-size tiny --input-dir data/mls_polish/test/audio --output-dir data/test_mels --transcripts-path data/mls_polish/test/transcripts.txt --metadata-output-path data/test_metadata.csv