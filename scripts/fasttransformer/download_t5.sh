# Arg1 - Model path on HuggingFace or GCS
cd FasterTransformer/build

if [[ "$1" == /gcs* ]] || [[ "$1" == gs://* ]] ;
then
    export GCS_PATH=${MODEL_OUTPUT/\/gcs\//gs:\/\/}
    gsutil cp -r $GCS_PATH .
else
    # Download model
    apt-get update
    apt-get install git-lfs
    git lfs install
    git lfs clone https://huggingface.co/$1
fi
cd ../..