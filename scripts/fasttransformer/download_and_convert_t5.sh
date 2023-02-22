# Arg1 - NumGpus
# Arg2 - Model Path on HuggingFace
./download_t5.sh $2

IFS="/"
MODEL_NAME=$2
for x in $2
do
    MODEL_NAME = x
done

./convert_existing_t5.sh $1 $MODEL_NAME