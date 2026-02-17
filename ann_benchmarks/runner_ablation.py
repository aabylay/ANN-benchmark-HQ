"""
Runner for ablation study with segment size tracking.
Modified from runner.py to pass segment_size through the chain.
"""
import argparse
import json
import logging
import os
import threading
import time
from typing import Dict, Optional, Tuple, List, Union

import colors
import docker
import numpy
import psutil

import re

from ann_benchmarks.algorithms.base.module import BaseANN

from .definitions import Definition, instantiate_algorithm
from .datasets import DATASETS, get_dataset, get_train_dataset, get_filters, get_workload_dataset
from .distance import dataset_transform, metrics
from .results_ablation import store_results  # Use ablation version


def run_individual_query(algo: BaseANN, X_train: numpy.array, X_test: numpy.array, distance: str, count: int, 
                         run_count: int, batch: bool, filter: str, X_attr: numpy.array = None, faiss_algo: bool = False) -> Tuple[dict, list]:
    """Run a search query using the provided algorithm and report the results."""
    prepared_queries = (batch and hasattr(algo, "prepare_batch_query")) or (
        (not batch) and hasattr(algo, "prepare_query")
    )

    best_search_time = float("inf")
    for i in range(run_count):
        print("Run %d/%d..." % (i + 1, run_count))
        n_items_processed = [0]

        def single_query(v: numpy.array, filter: str) -> Tuple[float, List[Tuple[int, float]]]:
            if prepared_queries:
                algo.prepare_query(v, count)
                start = time.time()
                algo.run_prepared_query()
                total = time.time() - start
                candidates = algo.get_prepared_query_results()
            else:
                start = time.time()
                if faiss_algo and filter != ["No_filter"]:
                    candidates = algo.query(v, count, filter, X_attr)
                else:
                    candidates = algo.query(v, count, filter)
                total = time.time() - start

            candidates = numpy.delete(candidates, numpy.where(candidates == -1))                
            assert len(candidates) == len(set(candidates)), "Implementation returned duplicated candidates"

            candidates = [
                (int(idx), float(metrics[distance].distance(v, X_train[idx]))) for idx in candidates
            ]
            n_items_processed[0] += 1
            if n_items_processed[0] % 1000 == 0:
                print("Processed %d/%d queries..." % (n_items_processed[0], len(X_test)))
            if len(candidates) > count:
                print(
                    "warning: algorithm %s returned %d results, but count is only %d)" % (algo, len(candidates), count)
                )
            return (total, candidates)

        def batch_query(X: numpy.array) -> List[Tuple[float, List[Tuple[int, float]]]]:
            if prepared_queries:
                algo.prepare_batch_query(X, count)
                start = time.time()
                algo.run_batch_query()
                total = time.time() - start
            else:
                start = time.time()
                algo.batch_query(X, count)
                total = time.time() - start
            results = algo.get_batch_results()
            if hasattr(algo, "get_batch_latencies"):
                batch_latencies = algo.get_batch_latencies()
            else:
                batch_latencies = [total / float(len(X))] * len(X)

            for res in results:
                assert len(res) == len(set(res)), "Implementation returned duplicated candidates"

            candidates = [
                [(int(idx), float(metrics[distance].distance(v, X_train[idx]))) for idx in single_results]
                for v, single_results in zip(X, results)
            ]
            return [(latency, v) for latency, v in zip(batch_latencies, candidates)]

        if batch:
            results = batch_query(X_test)
        else:
            results = [single_query(x, filter) for x in X_test]

        total_time = sum(time for time, _ in results)
        total_candidates = sum(len(candidates) for _, candidates in results)
        search_time = total_time / len(X_test)
        avg_candidates = total_candidates / len(X_test)
        best_search_time = min(best_search_time, search_time)

    verbose = hasattr(algo, "query_verbose")
    attrs = {
        "batch_mode": batch,
        "best_search_time": best_search_time,
        "candidates": avg_candidates,
        "expect_extra": verbose,
        "name": str(algo),
        "run_count": run_count,
        "distance": distance,
        "count": int(count),
    }
    additional = algo.get_additional()
    for k in additional:
        attrs[k] = additional[k]
    return (attrs, results)


def load_train_dataset(dataset_type: str, dataset_size: str) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, int]:
    """Loads the training dataset."""
    D, dimension = get_train_dataset(dataset_type, dataset_size)

    if dataset_type == "movies":
        train_vecs = numpy.array(D["train_storyline_vec"])
        train_ids = numpy.array(D["train_title_id"])
        train_attr1 = numpy.array(D["train_avgRating"])
        train_attr2 = numpy.array(D["train_is_adult"])
        train_attr3 = numpy.array(D["train_genre"])
        train_attr4 = numpy.array(D["train_num_votes"])
        train_attr5 = numpy.array(D["train_start_year"])
    elif dataset_type == "reviews":
        train_vecs = numpy.array(D["train_review_vec"])
        train_ids = numpy.array(D["train_review_ids"])
        train_attr1 = numpy.array(D["train_title_ids"])
        train_attr2 = numpy.array(D["train_user_ids"])
        train_attr3 = numpy.array(D["train_up_votes"])
        train_attr4 = numpy.array(D["train_down_votes"])    
        train_attr5 = numpy.array(D["train_total_votes"])

    train_attrs = numpy.array([train_attr1, train_attr2, train_attr3, train_attr4, train_attr5])
    print(f"Got a train set with ***{dataset_type}*** of size: ({train_vecs.shape[0]} * {dimension})")
    return train_ids, train_vecs, train_attrs, dimension


def load_workload_dataset(dataset_type: str, filter_id: str, dataset_size: str) -> Tuple[numpy.ndarray, str]:
    """Loads the workload dataset based on the dataset type and filter."""
    Q = get_workload_dataset(dataset_type, filter_id, dataset_size)
    X_test = numpy.array(Q["test"])
    distance = "angular"
    print(f"Got a test set of size ({X_test.shape[0]} * {X_test.shape[1]})")
    return X_test, distance


def load_filters(dataset_type: str, dataset_size: str) -> Tuple[List[int], List[str]]:
    """Loads the filters for a specific dataset type."""
    F = get_filters(dataset_type, dataset_size)
    filters = numpy.array(F["filters"])
    filters = [f.decode("utf-8") if isinstance(f, bytes) else f for f in filters]
    filter_ids = [int(fid) for fid in range(len(filters))]
    print(f"Got {len(filters)} filters for dataset type {dataset_type}")
    return filter_ids, filters


def compile_filter(filter: str) -> List[str]:
    """Compile a filter string into a list of selection parts."""
    filter = filter.replace(" ", "")
    if filter == "No_filter":
        return ["No_filter"]
    parts = filter.split("AND")
    selection = []
    for part in parts:
        match = re.match(r"(\w+)\s*(>=|<=|>|<|=)\s*(\d+)", part)
        if match:
            attribute, operator, value = match.groups()
            selection.append((attribute, operator, int(value)))
        else:
            raise ValueError(f"Invalid filter part: {part}")
    return selection


def build_index(algo: BaseANN, X_train_ids: numpy.ndarray, X_train: numpy.ndarray, 
                X_attrs: numpy.ndarray, dataset_type: str, algorithm: str) -> Tuple[float, float]:
    """Builds the index for the algorithm."""
    if dataset_type == "movies":
        X_attr = X_attrs[0]
    elif dataset_type == "reviews":
        X_attr = X_attrs[4]

    t0 = time.time()
    algo.fit(X_train_ids, X_train, X_attr, dataset_type)
    algo.fit2()
    build_time = time.time() - t0
    index_size = algo.get_memory_usage() if hasattr(algo, "get_memory_usage") else 0

    print(f"Built index in {build_time:.4f}s, size {index_size}")
    return build_time, index_size


def run(definition: Definition, dataset_name: str, dataset_size: str, run_count: int, batch: bool, 
        segment_size: int = 1024) -> None:
    """Run the algorithm benchmarking with segment size tracking for ablation study.

    Args:
        definition (Definition): The algorithm definition.
        dataset_name (str): The name of the dataset.
        dataset_size (str): Size of the dataset.
        run_count (int): The number of runs.
        batch (bool): If true, runs in batch mode.
        segment_size (int): Milvus segment size in MB for ablation study.
    """
    print(f"\n[ABLATION] Running with segment_size={segment_size}MB\n")
    
    algo = instantiate_algorithm(definition)
    assert not definition.query_argument_groups or hasattr(
        algo, "set_query_arguments"
    ), f"query argument groups specified but algorithm doesn't implement set_query_arguments"

    for dataset_type in ["movies", "reviews"]:
        print(f"Running on dataset type: {dataset_type}")
        X_train_ids, X_train, X_attrs, dimension = load_train_dataset(dataset_type, dataset_size)
        filter_ids, filters = load_filters(dataset_type, dataset_size)

        try:
            if hasattr(algo, "supports_prepared_queries"):
                algo.supports_prepared_queries()
                
            build_time, index_size = build_index(algo, X_train_ids, X_train, X_attrs, dataset_type, definition.algorithm)

            for att_idx in [0]:
                if att_idx: 
                    algo.fit_idx(dataset_type)
                for fid, ff in zip(filter_ids, filters):
                    kk_values = [10, 40, 100, 1]
                    X_test, distance = load_workload_dataset(dataset_type, fid, dataset_size)
                    ff = compile_filter(ff)
                    print(f"Running with filter: {ff}")
                    # kk = 1

                    for kk in kk_values:
                        query_argument_groups = definition.query_argument_groups.copy() or [[]]
                        print(definition.algorithm)
                        if definition.algorithm in ["pgvector_ivf", "faiss-ivf", "milvus-ivfflat"]:
                            if dataset_type == "reviews":
                                extra_query_arguments = [[query_argument_groups[-1][-1] * 2]]
                            else: 
                                extra_query_arguments = []
                        else:
                            extra_query_arguments = [[k] for k in kk_values[:-1] if k >= kk]
                        query_argument_groups.extend(extra_query_arguments)
                        print("Running on query argument groups:", query_argument_groups)
                        
                        for pos, query_arguments in enumerate(query_argument_groups, 1): 
                            print(f"Running query argument group {pos} of {len(query_argument_groups)}...")
                            if query_arguments:
                                algo.set_query_arguments(*query_arguments)
                                
                                if definition.algorithm in ["faiss-ivf", "hnsw(faiss)"] and ff != ["No_filter"]:
                                    if dataset_type == "movies":
                                        X_attr = X_attrs[0]
                                    elif dataset_type == "reviews":
                                        X_attr = X_attrs[4]
                                    X_attr = X_attr.astype(numpy.float32)
                                    descriptor, results = run_individual_query(algo, X_train, X_test, distance, kk, run_count, batch, ff, X_attr, True)
                                else:
                                    descriptor, results = run_individual_query(algo, X_train, X_test, distance, kk, run_count, batch, ff)

                                descriptor.update({
                                    "build_time": build_time,
                                    "index_size": index_size,
                                    "algo": definition.algorithm,
                                    "dataset": dataset_name,
                                    "segment_size": segment_size  # Add segment_size to descriptor
                                })

                                # Use ablation store_results with segment_size
                                store_results(dataset_name, kk, definition, query_arguments, descriptor, results, 
                                             batch, fid, dataset_size, dataset_type, att_idx, segment_size)
                    
        finally:
            if dataset_type == "reviews" and definition.algorithm == "milvus-ivfflat":
                algo.done(final_call=True)
            else:
                algo.done()


def run_from_cmdline():
    """Calls the function `run` using arguments from the command line for ablation study."""
    parser = argparse.ArgumentParser(description="Ablation study runner")
    parser.add_argument("--dataset", choices=DATASETS.keys(), help="Dataset to benchmark on.", required=True)
    parser.add_argument("--algorithm", help="Name of algorithm for saving the results.", required=True)
    parser.add_argument("--module", help='Python module containing algorithm.', required=True)
    parser.add_argument("--constructor", help='Constructor to load from module.', required=True)
    parser.add_argument("--dataset_size", choices=["large", "medium", "small"], help="Choose dataset size", required=True)
    parser.add_argument("--runs", help="Number of times to run the algorithm.", required=True, type=int)
    parser.add_argument("--batch", help='Run in batch mode.', action="store_true")
    parser.add_argument("--segment_size", help="Milvus segment size in MB for ablation study.", type=int, default=1024)
    parser.add_argument("build", help='JSON of arguments to pass to the constructor.')
    parser.add_argument("queries", help="JSON of arguments to pass to the queries.", nargs="*", default=[])
    args = parser.parse_args()

    algo_args = json.loads(args.build)
    print(f"[ABLATION] Algo args: {algo_args}")
    print(f"[ABLATION] Segment size: {args.segment_size}MB")
    query_args = [json.loads(q) for q in args.queries]

    definition = Definition(
        algorithm=args.algorithm,
        docker_tag=None,
        module=args.module,
        constructor=args.constructor,
        arguments=algo_args,
        query_argument_groups=query_args,
        disabled=False,
    )
    run(definition, args.dataset, args.dataset_size, args.runs, args.batch, args.segment_size)


def run_docker(
    definition: Definition,
    dataset: str,
    dataset_size: str,
    runs: int,
    timeout: int,
    batch: bool,
    cpu_limit: str,
    mem_limit: Optional[int] = None,
    segment_size: int = 1024
) -> None:
    """Runs `run_from_cmdline` within a Docker container with segment_size for ablation study."""
    cmd = [
        "--dataset", dataset,
        "--algorithm", definition.algorithm,
        "--module", definition.module,
        "--constructor", definition.constructor,
        "--runs", str(runs),
        "--dataset_size", dataset_size,
        "--segment_size", str(segment_size),  # Pass segment_size to container
    ]
    if batch:
        cmd += ["--batch"]
    cmd.append(json.dumps(definition.arguments))
    cmd += [json.dumps(qag) for qag in definition.query_argument_groups]

    client = docker.from_env()
    if mem_limit is None:
        mem_limit = psutil.virtual_memory().available
    
    print(f"[ABLATION] Running with segment_size={segment_size}MB, CPU limit: {cpu_limit}, memory limit: {mem_limit}")
    # Pass host project dir so container can fix volume paths for snap Docker
    host_project_dir = os.path.abspath(".")
    container = client.containers.run(
        definition.docker_tag,
        cmd,
        volumes={
            os.path.abspath("/var/run/docker.sock"): {"bind": "/var/run/docker.sock", "mode": "rw"},
            os.path.abspath("ann_benchmarks"): {"bind": "/home/app/ann_benchmarks", "mode": "ro"},
            os.path.abspath("data"): {"bind": "/home/app/data", "mode": "ro"},
            os.path.abspath("results"): {"bind": "/home/app/results", "mode": "rw"},
        },
        environment={
            "HOST_PROJECT_DIR": host_project_dir,
        },
        network_mode="host",
        cpuset_cpus=cpu_limit,
        mem_limit=mem_limit,
        detach=True,
    )
    logger = logging.getLogger(f"annb.ablation.{container.short_id}")

    logger.info(
        f"[ABLATION] Created container {container.short_id}: segment_size={segment_size}MB, CPU limit {cpu_limit}, mem limit {mem_limit}, timeout {timeout}, command {cmd}"
    )

    def stream_logs():
        for line in container.logs(stream=True):
            logger.info(colors.color(line.decode().rstrip(), fg="blue"))

    t = threading.Thread(target=stream_logs, daemon=True)
    t.start()

    try:
        return_value = container.wait(timeout=timeout)
        _handle_container_return_value(return_value, container, logger)
    except Exception as e:
        logger.error(f"Container.wait for container {container.short_id} failed with exception")
        logger.error(str(e))
    finally:
        logger.info("Removing container")
        container.remove(force=True)


def _handle_container_return_value(
    return_value: Union[Dict[str, Union[int, str]], int],
    container,
    logger: logging.Logger
) -> None:
    """Handles the return value of a Docker container."""
    base_msg = f"Child process for container {container.short_id} "
    msg = base_msg + "returned exit code {}"

    if isinstance(return_value, dict):
        error_msg = return_value.get("Error", "")
        exit_code = return_value["StatusCode"]
        msg = msg.format(f"{exit_code} with message {error_msg}")
    else:
        exit_code = return_value
        msg = msg.format(exit_code)

    if exit_code not in [0, None]:
        for line in container.logs(stream=True):
            logger.error(colors.color(line.decode(), fg="red"))
        logger.error(msg)
    else:
        logger.info(msg)
