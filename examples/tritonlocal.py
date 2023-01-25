import torch
import numpy as np
from transformers import (
        T5Tokenizer,
        T5TokenizerFast
    ) 
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype
import json
import struct

class T5TritonProcessor:
    def __init__(self, local_triton=False):
        # Initialize client
        self.client = httpclient.InferenceServerClient(
            "localhost:8000",verbose=True
        )

        # Initialize tokenizers from HuggingFace to do pre and post processings 
        # (convert text into tokens and backward) at the client side
        self.tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-base")

    # Implement the function that takes text converts it into the tokens using 
    # HFtokenizer and prepares tensorts for sending to Triton
    def preprocess(self, t5_task_input):
        input_token = self.tokenizer(t5_task_input, return_tensors="pt", padding=True,truncation=True)
        input_ids = input_token.input_ids.numpy().astype(np.uint32)

        mem_seq_len = (
            torch.sum(input_token.attention_mask, dim=1).numpy().astype(np.uint32)
        )
        mem_seq_len = mem_seq_len.reshape([mem_seq_len.shape[0], 1])
        max_output_len = np.array([[128]], dtype=np.uint32)
        runtime_top_k = (1.0 * np.ones([input_ids.shape[0], 1])).astype(np.uint32)
        
        inputs = [
            httpclient.InferInput(
                "input_ids", 
                input_ids.shape, 
                np_to_triton_dtype(input_ids.dtype)
            ),
            httpclient.InferInput(
                "sequence_length",
                mem_seq_len.shape,
                np_to_triton_dtype(mem_seq_len.dtype),
            ),
            httpclient.InferInput(
                "max_output_len", 
                max_output_len.shape , 
                np_to_triton_dtype(mem_seq_len.dtype)
            ),
            httpclient.InferInput(
                "runtime_top_k", 
                runtime_top_k.shape, 
                np_to_triton_dtype(runtime_top_k.dtype)
            )
        ]
        inputs[0].set_data_from_numpy(input_ids, False)
        inputs[1].set_data_from_numpy(mem_seq_len, False)
        inputs[2].set_data_from_numpy(max_output_len, False)
        inputs[3].set_data_from_numpy(runtime_top_k, False)

        return inputs
        

    # Implement function that takes tokens from Triton's response and converts 
    # them into text
    def postprocess(self, result):
        ft_decoding_outputs = result.as_numpy("output_ids")
        ft_decoding_seq_lens = result.as_numpy("sequence_length")
        # print(type(ft_decoding_outputs), type(ft_decoding_seq_lens))
        # print(ft_decoding_outputs, ft_decoding_seq_lens)
        tokens = self.tokenizer.decode(
            ft_decoding_outputs[0][0][: ft_decoding_seq_lens[0][0]],
            skip_special_tokens=True,
        )
        print(tokens)
        return tokens

    def get_payload(self,
                    text,
                    outputs=None,
                    request_id="",
                    sequence_id=0,
                    sequence_start=False,
                    sequence_end=False,
                    priority=0,
                    timeout=None,
                    headers=None,
                    query_params=None,
                    request_compression_algorithm=None,
                    response_compression_algorithm=None):
        inputs = self.preprocess(text)
        request_body, json_size = self.client._get_inference_request(
            inputs=inputs,
            request_id=request_id,
            outputs=outputs,
            sequence_id=sequence_id,
            sequence_start=sequence_start,
            sequence_end=sequence_end,
            priority=priority,
            timeout=timeout)

        if response_compression_algorithm == "gzip":
            if headers is None:
                headers = {}
            headers["Accept-Encoding"] = "gzip"
        elif response_compression_algorithm == 'deflate':
            if headers is None:
                headers = {}
            headers["Accept-Encoding"] = "deflate"

        if json_size is not None:
            if headers is None:
                headers = {}
            headers["Inference-Header-Content-Length"] = json_size

        return request_body, headers

    def test_local(self):
        # Run translation task with T5
        text = "Sandwiched between a second-hand bookstore and record shop in Cape Town's charmingly grungy suburb of Observatory is a blackboard reading 'Tapi Tapi -- Handcrafted, authentic African ice cream.' The parlor has become one of Cape Town's most talked about food establishments since opening in October 2020. And in its tiny kitchen, Jeff is creating ice cream flavors like no one else. Handwritten in black marker on the shiny kitchen counter are today's options: Salty kapenta dried fish (blitzed), toffee and scotch bonnet chile Sun-dried blackjack greens and caramel, Malted millet ,Hibiscus, cloves and anise. Using only flavors indigenous to the African continent, Guzha's ice cream has become the tool through which he is reframing the narrative around African food. 'This (is) ice cream for my identity, for other people's sake,' Jeff tells CNN. 'I think the (global) food story doesn't have much space for Africa ... unless we're looking at the generic idea of African food,' he adds. 'I'm not trying to appeal to the global universe -- I'm trying to help Black identities enjoy their culture on a more regular basis.'"
        inputs = self.preprocess(text)
        result = self.client.infer("fastertransformer", inputs)
        self.postprocess(result)

    
    def _get_inference_request(self, inputs, request_id, outputs, sequence_id,
                            sequence_start, sequence_end, priority, timeout):
        infer_request = {}
        parameters = {}
        if request_id != "":
            infer_request['id'] = request_id
        if sequence_id != 0 and sequence_id != "":
            parameters['sequence_id'] = sequence_id
            parameters['sequence_start'] = sequence_start
            parameters['sequence_end'] = sequence_end
        if priority != 0:
            parameters['priority'] = priority
        if timeout is not None:
            parameters['timeout'] = timeout

        infer_request['inputs'] = [
            this_input._get_tensor() for this_input in inputs
        ]
        if outputs:
            infer_request['outputs'] = [
                this_output._get_tensor() for this_output in outputs
            ]
        else:
            # no outputs specified so set 'binary_data_output' True in the
            # request so that all outputs are returned in binary format
            parameters['binary_data_output'] = True

        if parameters:
            infer_request['parameters'] = parameters

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
                '{}s{}s'.format(len(request_body), len(binary_data)),
                request_body.encode(), binary_data)
            return request_body, json_size

        return request_body, None
