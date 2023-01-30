# Arg0 - Num GPUs
# Arg1 - Num Computes
./faster_transformer_install.sh $1
./convert_t5-11b.sh
/opt/tritonserver/bin/tritonserver --model-repository=/workspace/all_models/t5/