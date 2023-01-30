# Arg1 - Num GPUs
# Arg2 - Num Computes
./faster_transformer_install.sh $2
./convert_t5-11b.sh
/opt/tritonserver/bin/tritonserver --model-repository=/workspace/all_models/t5/