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
"""Simple Flask prediction for a model using FasterTransformer_Triton.

Simple Flask prediction for a model using FasterTransformer_Triton..
"""
import json
import struct

import numpy as np
import torch
from transformers import AutoTokenizer
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype

from utils import timer

class TritonProcessorBase:
  """Base Processor class for any FasterTransformer Triton Backend model."""

  def __init__(self, model_path, host, port):
    self.client = httpclient.InferenceServerClient(
        f"{host}:{port}", verbose=True
    )

    # Initialize tokenizers from HuggingFace to do pre and post processings
    # (convert text into tokens and backward) at the client side
    self.tokenizer = AutoTokenizer.from_pretrained(model_path)

  def _get_payload(
      self,
      text,
      outputs=None,
      request_id="",
      sequence_id=0,
      sequence_start=False,
      sequence_end=False,
      priority=0,
      timeout=None,
      headers=None,
      response_compression_algorithm=None,
  ):
    """Generates a raw http payload from the provided text."""
    inputs = self._preprocess(text)
    request_body, json_size = self.client._get_inference_request(
        inputs=inputs,
        request_id=request_id,
        outputs=outputs,
        sequence_id=sequence_id,
        sequence_start=sequence_start,
        sequence_end=sequence_end,
        priority=priority,
        timeout=timeout,
    )

    if response_compression_algorithm == "gzip":
      if headers is None:
        headers = {}
      headers["Accept-Encoding"] = "gzip"
    elif response_compression_algorithm == "deflate":
      if headers is None:
        headers = {}
      headers["Accept-Encoding"] = "deflate"

    if json_size is not None:
      if headers is None:
        headers = {}
      headers["Inference-Header-Content-Length"] = str(json_size)

    return request_body, headers

  def _get_inference_request(
      self,
      inputs,
      request_id,
      outputs,
      sequence_id,
      sequence_start,
      sequence_end,
      priority,
      timeout,
  ):
    """Generates a TensorRT payload from a provided numpy array."""
    infer_request = {}
    parameters = {}
    if request_id:
      infer_request["id"] = request_id
    if sequence_id != 0:
      parameters["sequence_id"] = sequence_id
      parameters["sequence_start"] = sequence_start
      parameters["sequence_end"] = sequence_end
    if priority != 0:
      parameters["priority"] = priority
    if timeout is not None:
      parameters["timeout"] = timeout

    infer_request["inputs"] = [
        this_input._get_tensor() for this_input in inputs
    ]
    if outputs:
      infer_request["outputs"] = [
          this_output._get_tensor() for this_output in outputs
      ]
    else:
      # no outputs specified so set 'binary_data_output' True in the
      # request so that all outputs are returned in binary format
      parameters["binary_data_output"] = True

    if parameters:
      infer_request["parameters"] = parameters

    request_body = json.dumps(infer_request)
    json_size = len(request_body)
    binary_data = None
    for input_tensor in inputs:
      raw_data = input_tensor._get_binary_data()
      if raw_data is not None:
        if binary_data is not None:
          binary_data += raw_data
        else:
          binary_data = raw_data

    if binary_data is not None:
      request_body = struct.pack(
          "{}s{}s".format(len(request_body), len(binary_data)),
          request_body.encode(),
          binary_data,
      )
      return request_body, json_size

    return request_body, None


class T5TritonProcessor(TritonProcessorBase):
  """Processor for the T5 model family."""

  def __init__(self, hf_model_path="t5-base", host="localhost", port=8000):
    super().__init__(hf_model_path, host, port)

  @timer
  def infer(self, task=None, text=None):
    """Run inferencing on a series of inputs."""
    if task is not None:
      text = f"{task}: {text}"
    inputs = self._preprocess(text)
    result = self.client.infer("fastertransformer", inputs)
    return self._postprocess(result)

  @timer
  def _preprocess(self, string_input):
    """Implement the function that takes text, converts it into the tokens using HFtokenizer and prepares tensorts for sending to Triton."""
    input_token = self.tokenizer(
        string_input, return_tensors="pt", padding=True, truncation=True
    )
    input_ids = input_token.input_ids.numpy().astype(np.uint32)

    mem_seq_len = (
        torch.sum(input_token.attention_mask, dim=1).numpy().astype(np.uint32)
    )
    mem_seq_len = mem_seq_len.reshape([mem_seq_len.shape[0], 1])
    max_output_len = np.array([[128]], dtype=np.uint32)
    runtime_top_k = (1.0 * np.ones([input_ids.shape[0], 1])).astype(np.uint32)

    inputs = [
        httpclient.InferInput(
            "input_ids", input_ids.shape, np_to_triton_dtype(input_ids.dtype)
        ),
        httpclient.InferInput(
            "sequence_length",
            mem_seq_len.shape,
            np_to_triton_dtype(mem_seq_len.dtype),
        ),
        httpclient.InferInput(
            "max_output_len",
            max_output_len.shape,
            np_to_triton_dtype(mem_seq_len.dtype),
        ),
        httpclient.InferInput(
            "runtime_top_k",
            runtime_top_k.shape,
            np_to_triton_dtype(runtime_top_k.dtype),
        ),
    ]
    inputs[0].set_data_from_numpy(input_ids, False)
    inputs[1].set_data_from_numpy(mem_seq_len, False)
    inputs[2].set_data_from_numpy(max_output_len, False)
    inputs[3].set_data_from_numpy(runtime_top_k, False)

    return inputs

  # Implement function that takes tokens from Triton's response and converts
  # them into text
  @timer
  def _postprocess(self, result):
    ft_decoding_outputs = result.as_numpy("output_ids")
    ft_decoding_seq_lens = result.as_numpy("sequence_length")
    tokens = self.tokenizer.decode(
        ft_decoding_outputs[0][0][: ft_decoding_seq_lens[0][0]],
        skip_special_tokens=True,
    )
    return tokens
