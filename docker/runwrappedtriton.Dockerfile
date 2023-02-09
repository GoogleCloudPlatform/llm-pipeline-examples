FROM chris113113/triton_with_ft:22.07

RUN pip3 install absl-py flask gcsfs transformers tritonclient[http]

ADD src/predicttriton.py .
ADD src/tritonlocal.py .

ENTRYPOINT ["/bin/python3", "predicttriton.py"]

# python3 predicttriton.py --hf_model_path=google/t5-v1_1-base --model_path=gs://pirillo-sct-bucket/all_models/t5-v1_1-base