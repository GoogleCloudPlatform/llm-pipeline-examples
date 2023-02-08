FROM chris113113/triton_with_ft:22.07

RUN pip3 install absl-py flask gcsfs transformers tritonclient[http]

ADD src/predicttriton.py .
ADD src/tritonlocal.py .

ENTRYPOINT ["/bin/python3", "predicttriton.py"]