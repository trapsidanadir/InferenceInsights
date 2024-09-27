import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import docker
import pandas as pd
import requests


def get_image_size(image_name):
    client = docker.from_env()
    image = client.images.get(image_name)
    return image.attrs["Size"] / (1024 * 1024)  # Convert to MB


def call_predict_api(url, file_path):
    start_time = time.time()
    with open(file_path, "rb") as file:
        files = {"file": (file_path.name, file, "image/jpeg")}
        response = requests.post(f"{url}/predict", files=files)
    end_time = time.time()
    result = response.json()
    result["total_request_time"] = end_time - start_time
    result["file_name"] = file_path.name
    return result


def benchmark_api(url, image_files, num_calls):
    results = []

    # If there are fewer images than num_calls, we'll cycle through the images
    image_files = (image_files * (num_calls // len(image_files) + 1))[:num_calls]

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(call_predict_api, url, file_path)
            for file_path in image_files
        ]
        for future in futures:
            results.append(future.result())
    return results


def parser():
    parser = argparse.ArgumentParser(description="Process a number of calls.")
    parser.add_argument(
        "--num_calls",
        "-n",
        type=int,
        default=30,
        help="Number of calls to make",
        required=False,
    )

    return parser.parse_args()


def main():
    args = parser()
    # Define your API endpoints and image names
    apis = [
        {
            "name": "onnx",
            "url": "http://onnxapi:8000",
            "image": "inferenceinsights-onnx-api",
        },
        {
            "name": "pytorch",
            "url": "http://pytorchapi:8000",
            "image": "inferenceinsights-pytorch-api",
        },
        {
            "name": "tensorflow",
            "url": "http://tensorflowapi:8000",
            "image": "inferenceinsights-tensorflow-api",
        },
    ]

    data_folder = "/app/data"
    results_folder = "/app/results"

    # Ensure results folder exists
    os.makedirs(results_folder, exist_ok=True)

    all_results = []

    # list of images
    image_files = list(Path(data_folder).glob("*"))

    for api in apis:
        print(f"Benchmarking {api['name']}...")
        image_size = get_image_size(api["image"])
        results = benchmark_api(api["url"], image_files, num_calls=args.num_calls)

        for result in results:
            result["api_name"] = api["name"]
            result["image_size_mb"] = image_size

        all_results.extend(results)

    # Convert results to DataFrame
    df = pd.DataFrame(all_results)

    # Save results to CSV and JSON in the results folder
    csv_path = os.path.join(results_folder, "benchmark_results.csv")
    json_path = os.path.join(results_folder, "benchmark_results.json")

    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records")

    print(f"Benchmark completed. Results saved to {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
