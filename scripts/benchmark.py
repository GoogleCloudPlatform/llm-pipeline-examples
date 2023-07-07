from absl import app as absl_app
from absl import flags
from absl import logging
from absl.flags import argparse_flags
import argparse
from typing import List, Tuple
import requests
import numpy as np
from pprint import pprint

FLAGS = flags.FLAGS
flags.DEFINE_string(
   "hf_dataset_id",
   "cnn_dailymail",
   ("ID of the dataset on huggingface to use for benchmarking. Gated datasets are not supported.")
)
flags.DEFINE_string(
   "dataset_split_id",
   "test",
   ("Name of the split within the dataset to pull from.")
)
flags.DEFINE_string(
   "dataset_row_text_key",
   "article",
   ("Json key in the dataset for the payload to send.")
)
flags.DEFINE_string(
   "dataset_config_id",
   "1.0.0"
   ("Config within list of splits to use.")
)
flags.DEFINE_integer(
   "num_rows",
   100,
   ("Number of rows to take from the top of the dataset"),
   lower_bound=1,
   upper_bound=100
)
flags.DEFINE_integer(
   "total_requests",
   100,
   ("Total number of requests to send. (total_requests = c * num_rows)"),
   lower_bound=1
)
flags.DEFINE_integer(
   "batch_size",
   1,
   ("Number of rows of inputs to send in a single request"),
   lower_bound=1
)
flags.DEFINE_string(
   "target_url",
   "localhost:5000/infer"
   ("Url to infer against.")
)

def main(argv):
    rows = get_rows(FLAGS.hf_dataset_id, FLAGS.dataset_split_id, FLAGS.dataset_config_id, FLAGS.num_rows)
    results = benchmark(FLAGS.target_url, rows, FLAGS.total_requests, FLAGS.dataset_row_text_key, FLAGS.batch_size)
    pprint(results)

def query(url):
    response = requests.get(url)
    return response.json()

def get_rows(dataset_id, split_id, config_id, num_rows):
    base_url="https://datasets-server.huggingface.co"
    print("Downloading dataset " + dataset_id)
    validity_url = f"{base_url}/is-valid?dataset={dataset_id}"
    data = query(validity_url)
    print(data)
    if (not data["viewer"]):
       print(f"Provided dataset {dataset_id} is invalid")
       raise
    print("Validated provided dataset.")
    print("Retrieving splits for dataset")
    splits_url = f"{base_url}/splits?dataset={dataset_id}"
    data = query(splits_url)
    #print(data)
    split = [x for x in data['splits'] if x['split'] == split_id and x['config'] == config_id]
    if not len(split) == 1:
        print(f"Invalid split ({split_id}) and config ({config_id}) specified.")
        print(f"Available splits: {[('split: ' + x['split'] + ', config: ' + x['config']) for x in data['splits']][:]})")
        raise

    print(f"Downloading top {num_rows} rows from split: {split_id}, config: {config_id}")
    rows_url = f"{base_url}/rows?dataset={dataset_id}&split={split_id}&config={config_id}&length={num_rows}"
    return query(rows_url)

def benchmark(target_url, rows_obj, total_requests, text_key, batch_size):
    requests_sent = 0
    result_metrics = {
       'num_success': 0,
       'num_failed': 0,
       'latencies': [],
       'errors': [],
       'avg_latency': 0,
       'latency_percentiles': [],
       'unit': 'milliseconds',
       'batch_size': batch_size
    }
    rows = rows_obj['rows']

    while (requests_sent < total_requests):
        if requests_sent % 20 == 0:
            print(f"Sending request number {requests_sent}...")
        payload = {'instances': []}
        for _ in range(batch_size):
            text = rows[requests_sent % len(rows)]['row'][text_key]
            payload['instances'].append(text)
            requests_sent += 1
        response = requests.post(target_url, json=payload)
        result_metrics['latencies'].append(response.elapsed.microseconds / 1000)
        if response.ok:
            result_metrics['num_success'] += 1
        else:
            result_metrics['num_failed'] += 1
            result_metrics['errors'].append(response.text)

    result_metrics['avg_latency'] = np.average(result_metrics['latencies'])
    requested_percentiles = [50, 90, 99, 99.9, 99.99]
    latency_percentiles = np.percentile(result_metrics['latencies'], requested_percentiles)
    result_metrics['latency_percentiles'] = [{'percentile': x, 'value': y} for (x,y) in zip(requested_percentiles, latency_percentiles)]

    return result_metrics


def parse_flags(argv: List[str]) -> Tuple[argparse.Namespace, List[str]]:
    """Parses command line arguments entry_point.

    Args:
    argv: Unparsed arguments.

    Returns:
    Tuple of an argparse result and remaining args.
    """
    parser = argparse_flags.ArgumentParser(allow_abbrev=False)
    return parser.parse_known_args(argv)

if __name__ == "__main__":
    absl_app.run(main, flags_parser=parse_flags)