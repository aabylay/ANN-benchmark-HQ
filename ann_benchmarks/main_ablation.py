"""
Main entry point for ablation study with segment size tracking.
Modified from main.py to support segment_size parameter.
"""
import argparse
from dataclasses import replace
import logging
import logging.config
import multiprocessing
import multiprocessing.pool
import os
import random
import shutil
import sys
from typing import List

import docker
import psutil

from .definitions import (Definition, InstantiationStatus, algorithm_status,
                          get_definitions, list_algorithms)
from .constants import INDEX_DIR
from .datasets import DATASETS, get_train_dataset
from .results_ablation import build_result_filepath
from .runner_ablation import run, run_docker


logging.config.fileConfig("logging.conf")
logger = logging.getLogger("annb.ablation")


def positive_int(input_str: str) -> int:
    """Validates if the input string can be converted to a positive integer."""
    try:
        i = int(input_str)
        if i < 1:
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError(f"{input_str} is not a positive integer")
    return i


def run_worker(cpu: int, mem_limit: int, args: argparse.Namespace, queue: multiprocessing.Queue) -> None:
    """Executes the algorithm based on the provided parameters for ablation study."""
    print(f"[ABLATION] Worker started with CPU: {cpu}, Memory limit: {mem_limit}, Segment size: {args.segment_size}MB", flush=True)
    print("Queue size at start:", queue.qsize(), flush=True)
    
    while not queue.empty():
        definition = queue.get()
        cpu_limit = str(cpu) if not args.batch else f"0-{multiprocessing.cpu_count() - 1}"
        print(f"[ABLATION] Running in Docker with CPU limit: {cpu_limit}, memory limit: {mem_limit}, segment_size: {args.segment_size}MB")
        run_docker(definition, args.dataset, args.dataset_size, args.runs, args.timeout, args.batch, 
                   cpu_limit, mem_limit, args.segment_size)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Ablation study runner with segment size support")
    parser.add_argument(
        "--dataset",
        metavar="NAME",
        help="the dataset to load training points from",
        default="glove-100-angular",
        choices=DATASETS.keys(),
    )
    parser.add_argument(
        "-k", "--count", default=1, type=positive_int, help="the number of near neighbours to search for"
    )
    parser.add_argument(
        "--dataset_size", default="small", help="choose the size of the dataset to use from options: (small, medium, large)"
    )
    parser.add_argument(
        "--segment_size", type=int, default=1024, 
        help="Milvus segment size in MB for ablation study (512, 1024, 2048, 4096, 8192, 16384)"
    )
    parser.add_argument(
        "--definitions", metavar="FOLDER", help="base directory of algorithms", default="ann_benchmarks/algorithms"
    )
    parser.add_argument("--algorithm", metavar="NAME", help="run only the named algorithm", default=None)
    parser.add_argument(
        "--docker-tag", metavar="NAME", help="run only algorithms in a particular docker image", default=None
    )
    parser.add_argument(
        "--list-algorithms", help="print the names of all known algorithms and exit", action="store_true"
    )
    parser.add_argument("--force", help="re-run algorithms even if their results already exist", action="store_true")
    parser.add_argument(
        "--runs",
        metavar="COUNT",
        type=positive_int,
        help="run each algorithm instance %(metavar)s times and use only the best result",
        default=1,
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Timeout (in seconds) for each individual algorithm run, or -1 if no timeout should be set",
        default=200 * 3600,
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="If set, run everything locally rather than using Docker",
    )
    parser.add_argument("--batch", action="store_true", help="If set, algorithms get all queries at once")
    parser.add_argument(
        "--max-n-algorithms", type=int, help="Max number of algorithms to run", default=-1
    )
    parser.add_argument("--run-disabled", help="run algorithms that are disabled in algos.yml", action="store_true")
    parser.add_argument("--parallelism", type=positive_int, help="Number of Docker containers in parallel", default=1)

    args = parser.parse_args()
    if args.timeout == -1:
        args.timeout = None
    return args


def filter_already_run_definitions(
    definitions: List[Definition], 
    dataset: str, 
    count: int, 
    batch: bool, 
    force: bool,
    segment_size: int
) -> List[Definition]:
    """Filters out the algorithm definitions based on whether they have already been run."""
    filtered_definitions = []

    for definition in definitions:
        not_yet_run = [
            query_args 
            for query_args in (definition.query_argument_groups or [[]])
            if force or not os.path.exists(build_result_filepath(dataset, count, definition, query_args, batch, segment_size=segment_size))
        ]

        if not_yet_run:
            definition = replace(definition, query_argument_groups=not_yet_run) if definition.query_argument_groups else definition
            filtered_definitions.append(definition)
            
    return filtered_definitions


def filter_by_available_docker_images(definitions: List[Definition]) -> List[Definition]:
    """Filters out the algorithm definitions that do not have an associated, available Docker images."""
    docker_client = docker.from_env()
    docker_tags = {tag.split(":")[0] for image in docker_client.images.list() for tag in image.tags}

    missing_docker_images = set(d.docker_tag for d in definitions).difference(docker_tags)
    if missing_docker_images:
        logger.info(f"not all docker images available, only: {docker_tags}")
        logger.info(f"missing docker images: {missing_docker_images}")
        definitions = [d for d in definitions if d.docker_tag in docker_tags]
    
    return definitions


def check_module_import_and_constructor(df: Definition) -> bool:
    """Verifies if the algorithm module can be imported and its constructor exists."""
    status = algorithm_status(df)
    if status == InstantiationStatus.NO_CONSTRUCTOR:
        raise Exception(
            f"{df.module}.{df.constructor}({df.arguments}): error: the module '{df.module}' does not expose the named constructor"
        )
    if status == InstantiationStatus.NO_MODULE:
        logging.warning(
            f"{df.module}.{df.constructor}({df.arguments}): the module '{df.module}' could not be loaded; skipping"
        )
        return False
    
    return True


def create_workers_and_execute(definitions: List[Definition], args: argparse.Namespace):
    """Manages the creation, execution, and termination of worker processes for ablation study."""
    
    cpu_count = multiprocessing.cpu_count()
    if args.parallelism > cpu_count - 1:
        raise Exception(f"Parallelism larger than {cpu_count - 1}! (CPU count minus one)")

    if args.batch and args.parallelism > 1:
        raise Exception(
            f"Batch mode uses all available CPU resources, --parallelism should be set to 1. (Was: {args.parallelism})"
        )

    print(f"[ABLATION] Tasks for workers (segment_size={args.segment_size}MB):", definitions, flush=True)
    task_queue = multiprocessing.Queue()
    for definition in definitions:
        task_queue.put(definition)

    memory_margin = 500e6
    mem_limit = int((psutil.virtual_memory().available - memory_margin) / args.parallelism)
    
    cpu_usage = 48

    try:
        workers = [multiprocessing.Process(target=run_worker, args=(cpu_usage, mem_limit, args, task_queue)) 
                   for i in range(args.parallelism)]
        [worker.start() for worker in workers]
        [worker.join() for worker in workers]
    finally:
        logger.info("Terminating %d workers" % len(workers))
        [worker.terminate() for worker in workers]


def filter_disabled_algorithms(definitions: List[Definition]) -> List[Definition]:
    """Excludes disabled algorithms from the given list of definitions."""
    disabled_algorithms = [d for d in definitions if d.disabled]
    if disabled_algorithms:
        logger.info(f"Not running disabled algorithms {disabled_algorithms}")
    return [d for d in definitions if not d.disabled]


def limit_algorithms(definitions: List[Definition], limit: int) -> List[Definition]:
    """Limits the number of algorithm definitions based on the given limit."""
    return definitions if limit < 0 else definitions[:limit]


def main():
    args = parse_arguments()
    
    print(f"\n{'='*60}")
    print(f"[ABLATION STUDY] Segment size: {args.segment_size}MB")
    print(f"{'='*60}\n")
    
    if args.list_algorithms:
        list_algorithms(args.definitions)
        sys.exit(0)

    count = args.count if args.count else [args.count]    
    
    if os.path.exists(INDEX_DIR):
        shutil.rmtree(INDEX_DIR)
        
    dataset, dimension = get_train_dataset("movies", args.dataset_size)
    
    definitions: List[Definition] = get_definitions(
        dimension=dimension,
        point_type=dataset.attrs.get("point_type", "float"),
        distance_metric="angular",
        count=args.count,
        base_dir=args.definitions,
    )
    random.shuffle(definitions)
    
    if args.algorithm:
        logger.info(f"running only {args.algorithm}")
        definitions = [d for d in definitions if d.algorithm == args.algorithm]
    
    if not args.local:
        definitions = filter_by_available_docker_images(definitions)
    else:
        definitions = list(filter(check_module_import_and_constructor, definitions))
    
    definitions = filter_disabled_algorithms(definitions) if not args.run_disabled else definitions
    definitions = limit_algorithms(definitions, args.max_n_algorithms)

    logger.info(f"[ABLATION] Order: {definitions}")
    create_workers_and_execute(definitions, args)


if __name__ == "__main__":
    main()
