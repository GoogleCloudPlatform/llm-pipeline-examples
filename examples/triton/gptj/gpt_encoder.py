import json
import os

batch_size=1
beam_width=1
num_runs=1
request_output_len = 128

json_template = """
{
    "request": [
        {
            "name": "input_ids",
            "dtype": "uint32",
            "data": []
        },
        {
            "name": "input_lengths",
            "dtype": "uint32",
            "data": []
        },
        {
            "name": "request_output_len",
            "dtype": "uint32",
            "data": []
        },
        {
            "name": "beam_width",
            "dtype": "uint32",
            "data": []
        }
    ]
}
"""

with open('cnn_dailymail_calibration_tokenized.json') as f:
    tokenized_inputs = json.load(f)

payload = json.loads(json_template)

for i in range(batch_size):
    ids = tokenized_inputs[i]["input_ids"]
    if len(ids) > 1919:
        ids = ids[:1919]
    else:
        missing = 1919-len(ids)
        ids = ids.extend([50256] * missing)
    payload["request"][0]["data"].append(ids)
    payload["request"][1]["data"].append([len(ids)])
    payload["request"][2]["data"].append([request_output_len])
    payload["request"][3]["data"].append([beam_width])

file_name = f"requests/cnn_bs{batch_size}_bw{beam_width}_len{request_output_len}.json"
with open(file_name, "w") as output_file:
    json.dump(payload, output_file, indent=None, ensure_ascii=False)