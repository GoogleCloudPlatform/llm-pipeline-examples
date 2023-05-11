#!/bin/bash -ev


# Copyright 2022 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if [[ -z $1 ]]; then
    echo "Usage:"
    echo "Option 1: ./preprocess.sh <GCS destination>"
    echo "          This will download wikipedia dataset, preprocess it and uplaod it to <GCS destination>"
    echo "Option 2: ./preprocess.sh <GCS source> <GCS destination>"
    echo "          This will preprocess data from <GCS source> and uplaod it to <GCS destination>"
fi

if [[ -z $2 ]]; then
    wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
    wikiextractor --json enwiki-latest-pages-articles.xml.bz2
    for file in ./text/*/*; do cat $file >> wikidata.txt; done
    data_file=wikidata.txt
    destination=$1
else
    gsutil cp $1 .
    data_file=$(basename $1)
    destination=$2
fi

wget https://huggingface.co/gpt2/raw/main/vocab.json
wget https://huggingface.co/gpt2/raw/main/merges.txt
python3 ../Megatron-DeepSpeed/tools/preprocess_data.py \
    --input $data_file \
    --output-prefix wiki_data \
    --vocab vocab.json \
    --dataset-impl mmap \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file merges.txt \
    --workers 35 \
    --append-eod 

gsutil cp vocab.json $destination/gpt2-vocab.json
gsutil cp merges.txt $destination/gpt2-merges.txt
gsutil cp wiki_data_text_document.bin $destination
gsutil cp wiki_data_text_document.idx $destination
