FROM FROM nvcr.io/nvidia/pytorch:23.05-py3
WORKDIR /usr/local/
ADD FSDP .
RUN pip install -r requirements.txt
RUN chmod +x download_dataset.sh
RUN ./download_dataset.sh