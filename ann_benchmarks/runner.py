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

from .results import store_results
# Uncomment this to import results for ablation study
# from .results_ablation import store_results


def run_individual_query(algo: BaseANN, X_train: numpy.array, X_test: numpy.array, distance: str, count: int, 
                         run_count: int, batch: bool, filter: str, X_attr: numpy.array = None, faiss_algo: bool = False) -> Tuple[dict, list]:
    """Run a search query using the provided algorithm and report the results.

    Args:
        algo (BaseANN): An instantiated ANN algorithm.
        X_train (numpy.array): The training data.
        X_test (numpy.array): The testing data.
        distance (str): The type of distance metric to use.
        count (int): The number of nearest neighbors to return.
        run_count (int): The number of times to run the query.
        batch (bool): Flag to indicate whether to run in batch mode or not.

    Returns:
        tuple: A tuple with the attributes of the algorithm run and the results.
    """
    prepared_queries = (batch and hasattr(algo, "prepare_batch_query")) or (
        (not batch) and hasattr(algo, "prepare_query")
    )

    best_search_time = float("inf")
    for i in range(run_count):
        print("Run %d/%d..." % (i + 1, run_count))
        # a bit dumb but can't be a scalar since of Python's scoping rules
        n_items_processed = [0]

        def single_query(v: numpy.array, filter: str) -> Tuple[float, List[Tuple[int, float]]]:
            """Executes a single query on an instantiated, ANN algorithm.

            Args:
                v (numpy.array): Vector to query.

            Returns:
                List[Tuple[float, List[Tuple[int, float]]]]: Tuple containing
                    1. Total time taken for each query 
                    2. Result pairs consisting of (point index, distance to candidate data )
            """
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
                    #if filter != ["No_filter"]:
                    #    print("--------------------------------")
                    #    print("Algo:", faiss_algo, "Filter:", filter)
                    candidates = algo.query(v, count, filter)
                total = time.time() - start
                # print("Candidates:", candidates)
                # raise Exception("Debugging")
            
            # Early return for query plan
            # return (total, candidates) 

            # make sure all returned indices are unique and valid (drop -1 for not found cases)
            candidates = numpy.delete(candidates, numpy.where(candidates == -1))                
            assert len(candidates) == len(set(candidates)), "Implementation returned duplicated candidates"

            candidates = [
                (int(idx), float(metrics[distance].distance(v, X_train[idx]))) for idx in candidates  # noqa
            ]
            n_items_processed[0] += 1
            if n_items_processed[0] % 1000 == 0:
                print("Processed %d/%d queries..." % (n_items_processed[0], len(X_test)))
            if len(candidates) > count:
                print(
                    "warning: algorithm %s returned %d results, but count"
                    " is only %d)" % (algo, len(candidates), count)
                )
            return (total, candidates)

        def batch_query(X: numpy.array) -> List[Tuple[float, List[Tuple[int, float]]]]:
            """Executes a batch of queries on an instantiated, ANN algorithm.

            Args:
                X (numpy.array): Array containing multiple vectors to query.

            Returns:
                List[Tuple[float, List[Tuple[int, float]]]]: List of tuples, each containing
                    1. Total time taken for each query 
                    2. Result pairs consisting of (point index, distance to candidate data )
            """
            # TODO: consider using a dataclass to represent return value.
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

            # make sure all returned indices are unique
            for res in results:
                assert len(res) == len(set(res)), "Implementation returned duplicated candidates"

            candidates = [
                [(int(idx), float(metrics[distance].distance(v, X_train[idx]))) for idx in single_results]  # noqa
                for v, single_results in zip(X, results)
            ]
            return [(latency, v) for latency, v in zip(batch_latencies, candidates)]

        if batch:
            results = batch_query(X_test)
        else:
            """
            print("checking shapes...")
            print(X_test.shape)
            print(X_test[:1].shape)
            print(X_test[0].shape)
            print(X_test[:, 0].shape)
            print([x.shape for x in X_test[:1]])
            print([x.shape for x in X_test[:2]])
            
            for x in X_test:
                print(x.shape)
                break 
            
            raise Exception("Debugging")
            """

            results = [single_query(x, filter) for x in X_test]
            
            # NOTE: X_test[idx] is for query plan testing only, should be X_test for actual experiments
            # Uncomment this for query plan testing only (replaces vector numbers with "vector" text)
            """
            random_index = numpy.random.randint(0, len(X_test) - 1)
            results = [single_query(x, filter) for x in [X_test[random_index]]]
            
            for i, result in enumerate(results):
                # replace pattern f'[{float numbers array where at least 10 numbers are present}]' with text '[vector]'
                # Pattern matches arrays like [-0.012313394,-0.046120428,...] including scientific notation
                vector_pattern = r'\[-?\d+\.?\d*(?:[eE][+-]?\d+)?(?:,-?\d+\.?\d*(?:[eE][+-]?\d+)?)+\]'
                processed_candidates = [re.sub(vector_pattern, '[vector]', candidate) for candidate in result[1]]
                results[i] = (result[0], processed_candidates)
            """

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


def load_and_transform_dataset(dataset_name: str, k, filter) -> Tuple[
        Union[numpy.ndarray, List[numpy.ndarray]],
        Union[numpy.ndarray, List[numpy.ndarray]],
        str]:
    """Loads and transforms the dataset.

    Args:
        dataset_name (str): The name of the dataset.

    Returns:
        Tuple: Transformed datasets.
    """
    D, dimension = get_dataset(dataset_name, k, filter)
    X_train = numpy.array(D["train"])
    X_test = numpy.array(D["test"])
    distance = D.attrs["distance"]

    print(f"Got a train set of size ({X_train.shape[0]} * {dimension})")
    print(f"Got {len(X_test)} queries")

    train, test = dataset_transform(D)
    train_ids = numpy.array(D["train_ids"])
    train_attr = numpy.array(D["train_ratings"])
    return train_ids, train, train_attr, test, distance
    
# --- NEW CODE: START ---------------------------------------------------------------------------------

# Function to load the training dataset based on the dataset type.
def load_train_dataset(dataset_type: str, dataset_size: str) -> Tuple[
        numpy.ndarray, 
        numpy.ndarray, 
        numpy.ndarray]:
    """Loads the training dataset.
    
    Args:
        dataset_type (str): The type of the dataset.

    Returns:
        Tuple: The training vectors, IDs, attributes, and dimension.
    """

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


# Function to load a workload dataset based on the dataset type and filter.
def load_workload_dataset(dataset_type: str, filter_id: str, dataset_size: str) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Loads the workload dataset based on the dataset type and filter.

    Args:
        dataset_type (str): The type of the dataset.
        filter (str): The filter to apply.

    Returns:
        Tuple: The test data and distance metric.
    """
    Q = get_workload_dataset(dataset_type, filter_id, dataset_size)
    X_test = numpy.array(Q["test"])
    distance = "angular"
    print(f"Got a test set of size ({X_test.shape[0]} * {X_test.shape[1]})")
    return X_test, distance


# Function to load filters for a specific dataset type.
def load_filters(dataset_type: str, dataset_size: str) -> Tuple[List[int], List[str]]:
    """Loads the filters for a specific dataset type.

    Args:
        dataset_type (str): The type of the dataset.

    Returns:
        Tuple: A tuple containing filter IDs and filter strings.
    """
    F = get_filters(dataset_type, dataset_size)
    # filter_ids = numpy.array(F["filter_ids"])
    filters = numpy.array(F["filters"])
    filters = [f.decode("utf-8") if isinstance(f, bytes) else f for f in filters]
    filter_ids = [int(fid) for fid in range(len(filters))]
    print(f"Got {len(filters)} filters for dataset type {dataset_type}")
    return filter_ids, filters


# Function to compile a filter string into a list of selection parts.
def compile_filter(filter: str) -> List[str]:
    # remove all whitespace from the filter string
    print(f"Compiling filter: {filter}")
    if filter == ["No_filter"]: return filter
    elif filter == "No_filter": return [filter]
    
    filter = filter.replace(" ", "")
    # each filter part then should be split in three parts: attribute, operator, value
    # operators include '=', '!=', '<', '>', '<=', '>='
    if '<=' in filter:
        attr, value = filter.split('<=')
        compiled_filter = [attr.strip(), '<=', value.strip()]
    elif '>=' in filter:
        attr, value = filter.split('>=')
        compiled_filter = [attr.strip(), '>=', value.strip()]
    elif '!=' in filter:
        attr, value = filter.split('!=')
        compiled_filter = [attr.strip(), '!=', value.strip()]    
    elif "=" in filter:
        attr, value = filter.split("=")
        compiled_filter = [attr.strip(), '=', value.strip()]
    elif '<' in filter:
        attr, value = filter.split('<')
        compiled_filter = [attr.strip(), '<', value.strip()]
    elif '>' in filter:
        attr, value = filter.split('>')
        compiled_filter = [attr.strip(), '>', value.strip()]
    else:
        raise ValueError(f"Unknown filter format: {filter}")
    return compiled_filter

# --- NEW CODE: END ---------------------------------------------------------------------------------

def build_index(algo: BaseANN, X_train_ids: numpy.ndarray, X_train: numpy.ndarray, X_attrs: numpy.ndarray, dataset_type: str, algo_name="") -> Tuple:
    """Builds the ANN index for a given ANN algorithm on the training data.

    Args:
        algo (Any): The algorithm instance.
        X_train (Any): The training data.

    Returns:
        Tuple: The build time and index size.
    """
    t0 = time.time()
    memory_usage_before = algo.get_memory_usage()
    print("Memory usage before building index: ", memory_usage_before)
    if dataset_type == "movies":
        X_attr = X_attrs[0]
    elif dataset_type == "reviews":
        X_attr = X_attrs[4]
        print("--- ATTR FOR REVIEW ---:", X_attr[0], type(X_attr[0]), type(X_attr))
    algo.fit(X_train_ids, X_train, X_attr, dataset_type)

    if algo_name == "milvus-ivfflat" or algo_name == "milvus-hnsw":
        time.sleep(3)
        memory_usage_before = algo.get_memory_usage()
        print("Milvus memory usage before building index:", memory_usage_before)
        algo.fit2()
        print("Loaded Milvus collection successfully!!!")

    build_time = time.time() - t0

    ##### IMPORTANT! #####
    # Change index size
    
    # uncomment this for QUERY running:
    memory_usage_after = algo.get_memory_usage()
    print("Memory usage after building index: ", memory_usage_after)
    index_size = memory_usage_after - memory_usage_before 
    
    # uncomment this for GROUND TRUTH:
    # index_size = memory_usage_before

    # END: Change index size
    print("Built index in", build_time)
    print("Index size: ", index_size, flush=True)

    return build_time, index_size


# --- CHANGE: The main run function to support filters -------------------------------

# run function to check query plans in PG-Vector
'''
def run(definition: Definition, dataset_name: str, dataset_size: str, run_count: int, batch: bool) -> None:
    """Run the algorithm benchmarking.

    Args:
        definition (Definition): The algorithm definition.
        dataset_name (str): The name of the dataset.
        max_k (int): The maximum number of nearest neighbors to return.
        run_count (int): The number of runs.
        batch (bool): If true, runs in batch mode.
    """
    # Map dataset sizes to their corresponding names for paths
    algo = instantiate_algorithm(definition)
    assert not definition.query_argument_groups or hasattr(
        algo, "set_query_arguments"
    ), f"""\
error: query argument groups have been specified for {definition.module}.{definition.constructor}({definition.arguments}), but the \
algorithm instantiated from it does not implement the set_query_arguments \
function"""

    for dataset_type in ["movies", "reviews"]: # hardcoded for now, can be changed later ["movies", "reviews"]
        print(f"Running on dataset type: {dataset_type}")
        X_train_ids, X_train, X_attrs, dimension = load_train_dataset(dataset_type, dataset_size)
        filter_ids, filters = load_filters(dataset_type, dataset_size)

        try:
            if hasattr(algo, "supports_prepared_queries"):
                algo.supports_prepared_queries()

            build_time, index_size = build_index(algo, X_train_ids, X_train, X_attrs, dataset_type)

            for att_idx in [0, 1]:
                if att_idx: algo.fit_idx(dataset_type)
                for fid, ff in zip(filter_ids, filters):
                    kk_values = [10]
                    X_test, distance = load_workload_dataset(dataset_type, fid, dataset_size)
                    ff = compile_filter(ff)
                    print(f"Running with filter: {ff}")
                    # if ff == ['No_filter']:
                    #    print("Skipping filter for No_filter")
                    #    continue
                    for kk in kk_values:
                        query_argument_groups = definition.query_argument_groups.copy() or [[]]  # Ensure at least one iteration
                        extra_query_arguments = [[k] for k in kk_values[:-1] if k >= kk] # except last k value
                        # append query_argument_groups with extra_query_arguments
                        query_argument_groups.extend(extra_query_arguments)
                        print("Running on query argument groups:", query_argument_groups)
                        
                        for pos, query_arguments in enumerate(query_argument_groups, 1): 
                        
                            print(f"Running query argument group {pos} of {len(query_argument_groups)}...")
                            if query_arguments:
                                algo.set_query_arguments(*query_arguments)
                                
                                for q_id in range(3):
                                    descriptor, results = run_individual_query(algo, X_train, X_test, distance, kk, run_count, batch, ff)
                                    # print("Results:", results)
                                    # raise Exception("Debugging")

                                    descriptor.update({
                                        "build_time": build_time,
                                        "index_size": index_size,
                                        "algo": definition.algorithm,
                                        "dataset": dataset_name
                                    })

                                    print(f"======= Query plan for the following search parameters:")
                                    print(f"======= Dataset type: {dataset_type}, Dataset size: {dataset_size}, M / EF Construction: {definition.arguments[1]}")
                                    print(f"======= Filter: {ff}, K: {kk}, EF Search: {query_arguments[0]}, K: {kk}, Att idx: {att_idx}")
                                    print(f"======= Results\n: {results}")
                                    # store_results(dataset_name, kk, definition, query_arguments, descriptor, results, batch, fid, dataset_size, dataset_type, att_idx)
                    

        finally:
            algo.done()
'''

# run function changed to support filters
def run(definition: Definition, dataset_name: str, dataset_size: str, run_count: int, batch: bool, segment_size: Optional[int] = None) -> None:
    """Run the algorithm benchmarking.

    Args:
        definition (Definition): The algorithm definition.
        dataset_name (str): The name of the dataset.
        max_k (int): The maximum number of nearest neighbors to return.
        run_count (int): The number of runs.
        batch (bool): If true, runs in batch mode.
    """
    # Map dataset sizes to their corresponding names for paths
    algo = instantiate_algorithm(definition)
    assert not definition.query_argument_groups or hasattr(
        algo, "set_query_arguments"
    ), f"""\
error: query argument groups have been specified for {definition.module}.{definition.constructor}({definition.arguments}), but the \
algorithm instantiated from it does not implement the set_query_arguments \
function"""

    for dataset_type in ["movies", "reviews"]: # hardcoded for now, can be changed later ["movies", "reviews"]
        print(f"Running on dataset type: {dataset_type}")
        X_train_ids, X_train, X_attrs, dimension = load_train_dataset(dataset_type, dataset_size)
        filter_ids, filters = load_filters(dataset_type, dataset_size)

        try:
            if hasattr(algo, "supports_prepared_queries"):
                algo.supports_prepared_queries()
                
            build_time, index_size = build_index(algo, X_train_ids, X_train, X_attrs, dataset_type, definition.algorithm)
            # need to skip the rest for index building stats only

            for att_idx in [0]: # [0 - for no attr idx, | 1 - for attr idx]
                if att_idx: algo.fit_idx(dataset_type)
                for fid, ff in zip(filter_ids, filters):
                    kk_values = [10, 40, 100, 1]
                    X_test, distance = load_workload_dataset(dataset_type, fid, dataset_size)
                    ff = compile_filter(ff)
                    print(f"Running with filter: {ff}")
                    # kk = 1
                    # if ff == ['No_filter']:
                    #    print("Skipping filter for No_filter")
                    #    continue
                    for kk in kk_values:
                        query_argument_groups = definition.query_argument_groups.copy() or [[]]  # Ensure at least one iteration
                        print(definition.algorithm)
                        if definition.algorithm in ["pgvector_ivf", "faiss-ivf", "milvus-ivfflat"]:
                            if dataset_type == "reviews":
                                extra_query_arguments = [[query_argument_groups[-1][-1] * 2]] # probes
                            else: extra_query_arguments = []
                        else:
                            extra_query_arguments = [[k] for k in kk_values[:-1] if k >= kk] # except last k value
                        # print(extra_query_arguments)
                        # append query_argument_groups with extra_query_arguments
                        query_argument_groups.extend(extra_query_arguments)
                        print("Running on query argument groups:", query_argument_groups)
                        
                        for pos, query_arguments in enumerate(query_argument_groups, 1): 
                        
                            print(f"Running query argument group {pos} of {len(query_argument_groups)}...")
                            if query_arguments:
                                algo.set_query_arguments(*query_arguments)
                                
                                if definition.algorithm in ["faiss-ivf", "hnsw(faiss)"] and ff != ["No_filter"]:
                                    # Select the correct attribute array that matches what was used in build_index
                                    if dataset_type == "movies":
                                        X_attr = X_attrs[0]
                                    elif dataset_type == "reviews":
                                        X_attr = X_attrs[4]
                                    X_attr = X_attr.astype(numpy.float32)
                                    descriptor, results = run_individual_query(algo, X_train, X_test, distance, kk, run_count, batch, ff, X_attr, True)
                                else:
                                    #if ff != ["No_filter"]:
                                    #    print("\n", "-"*50, "\nRunning individual query...")
                                    #    print("Filter:", ff, "| Algo:", definition.algorithm, "Not in:", ["faiss-ivf", "hnsw(faiss)"])
                                        
                                    descriptor, results = run_individual_query(algo, X_train, X_test, distance, kk, run_count, batch, ff)
                                
                                # print("Results:", results)
                                # raise Exception("Debugging")

                                descriptor.update({
                                    "build_time": build_time,
                                    "index_size": index_size,
                                    "algo": definition.algorithm,
                                    "dataset": dataset_name
                                })

                                store_results(dataset_name, kk, definition, query_arguments, descriptor, results, batch, fid, dataset_size, dataset_type, att_idx, segment_size)
                                
                    
        finally: # making sure that milvus finished and didn't crash
            if dataset_type == "reviews" and definition.algorithm == "milvus-ivfflat":
                algo.done(final_call=True)
            else:
                algo.done()

# --------------------------------------------------------------

def run_from_cmdline():
    """Calls the function `run` using arguments from the command line. See `ArgumentParser` for 
    arguments, all run it with `--help`.
    """
    parser = argparse.ArgumentParser(
        """

            NOTICE: You probably want to run.py rather than this script.

"""
    )
    parser.add_argument("--dataset", choices=DATASETS.keys(), help="Dataset to benchmark on.", required=True)
    parser.add_argument("--algorithm", help="Name of algorithm for saving the results.", required=True)
    parser.add_argument(
        "--module", help='Python module containing algorithm. E.g. "ann_benchmarks.algorithms.annoy"', required=True
    )
    parser.add_argument("--constructor", help='Constructer to load from modulel. E.g. "Annoy"', required=True)
    #parser.add_argument(
    #    "--max_k", help="K: Max number of nearest neighbours for the algorithm to return.", required=True, type=int
    #)
    parser.add_argument("--dataset_size", choices=["large", "medium", "small"], help="Choose dataset size from: large, medium, small", required=True)
    parser.add_argument(
        "--runs",
        help="Number of times to run the algorihm. Will use the fastest run-time over the bunch.",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--batch",
        help='If flag included, algorithms will be run in batch mode, rather than "individual query" mode.',
        action="store_true",
    )
    parser.add_argument("--segment_size", type=int, default=None,
                        help="Milvus segment size in MB for ablation study (extracted from docker_tag when running from host)")
    # New argument to parser
    parser.add_argument(
        "--filter",
        help='If flag included, algorithms will look for dataset with given filter in the name.',
        type=str,
        default=None,
    )
    
    parser.add_argument("build", help='JSON of arguments to pass to the constructor. E.g. ["angular", 100]')
    parser.add_argument("queries", help="JSON of arguments to pass to the queries. E.g. [100]", nargs="*", default=[])
    args = parser.parse_args()

    algo_args = json.loads(args.build)
    print(algo_args)
    query_args = [json.loads(q) for q in args.queries]

    definition = Definition(
        algorithm=args.algorithm,
        docker_tag=None,  # not needed
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
    mem_limit: Optional[int] = None
) -> None:
    """Runs `run_from_cmdline` within a Docker container with specified parameters and logs the output.

    See `run_from_cmdline` for details on the args.
    """
    # Extract segment size from docker_tag (e.g., "ann-benchmarks-milvus-seg16384" -> 16384)
    segment_size = 1024  # default
    if definition.docker_tag and "-seg" in definition.docker_tag:
        try:
            segment_size = int(definition.docker_tag.split("-seg")[-1])
        except ValueError:
            pass

    cmd = [
        "--dataset",
        dataset,
        "--algorithm",
        definition.algorithm,
        "--module",
        definition.module,
        "--constructor",
        definition.constructor,
        "--runs",
        str(runs),
        "--dataset_size",
        dataset_size,
        "--segment_size",
        str(segment_size),
    ]
    if batch:
        cmd += ["--batch"]
    cmd.append(json.dumps(definition.arguments))
    cmd += [json.dumps(qag) for qag in definition.query_argument_groups]

    client = docker.from_env()
    if mem_limit is None:
        mem_limit = psutil.virtual_memory().available
    print("Running with CPU limit:", cpu_limit, "and memory limit:", mem_limit)
    # Pass host project dir so the container can rewrite docker-compose.yml
    # volume paths to be accessible by snap Docker (which can't see /tmp)
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
    logger = logging.getLogger(f"annb.{container.short_id}")

    # NEW: check installed packages in the container
    # pip_list_result = container.exec_run("pip list")
    # logger.info(f"Installed packages in container {container.short_id}:\n{pip_list_result.output.decode()}")
    
    logger.info(
        "Created container %s: CPU limit %s, mem limit %s, timeout %d, command %s"
        % (container.short_id, cpu_limit, mem_limit, timeout, cmd)
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
        logger.error("Container.wait for container %s failed with exception", container.short_id)
        logger.error(str(e))
    finally:
        logger.info("Removing container")
        container.remove(force=True)


def _handle_container_return_value(
    return_value: Union[Dict[str, Union[int, str]], int],
    container: docker.models.containers.Container,
    logger: logging.Logger
) -> None:
    """Handles the return value of a Docker container and outputs error and stdout messages (with colour).

    Args:
        return_value (Union[Dict[str, Union[int, str]], int]): The return value of the container.
        container (docker.models.containers.Container): The Docker container.
        logger (logging.Logger): The logger instance.
    """

    base_msg = f"Child process for container {container.short_id} "
    msg = base_msg + "returned exit code {}"

    if isinstance(return_value, dict):  # The return value from container.wait changes from int to dict in docker 3.0.0
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
