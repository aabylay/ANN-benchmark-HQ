import json
import os
import re
import traceback
from typing import Any, Optional, Set, Tuple, Iterator
import h5py

from ann_benchmarks.definitions import Definition


def build_result_filepath(dataset_name: Optional[str] = None, 
                          count: Optional[int] = None, 
                          definition: Optional[Definition] = None, 
                          query_arguments: Optional[Any] = None, 
                          batch_mode: bool = False,
                          filter_id: int = 0,
                          dataset_size: str = "small",
                          data_table: str = "movies",
                          att_idx: int = 0,
                          segment_size: Optional[int] = None) -> str:
    """
    Constructs the filepath for storing the results.

    Args:
        dataset_name (str, optional): The name of the dataset.
        count (int, optional): The count of records.
        definition (Definition, optional): The definition of the algorithm.
        query_arguments (Any, optional): Additional arguments for the query.
        batch_mode (bool, optional): If True, the batch mode is activated.
        segment_size (int, optional): Milvus segment size in MB; when set, uses ablation path results/ablation_seg{N}/.

    Returns:
        str: The constructed filepath.
    """
    if segment_size is not None:
        d = ["results", f"ablation_seg{segment_size}", f"MoRe_UPD_{dataset_size}_attidx_{att_idx}"]
    else:
        d = ["results", f"MoRe_UPD_{dataset_size}_attidx_{att_idx}"]
    if filter_id: # adding filter to result file name 
        d.append("fid" + str(filter_id))
    else: d.append("fid0")
    if count:
        d.append(str(count))
    if definition:
        d.append(definition.algorithm + ("-batch" if batch_mode else ""))
        data = definition.arguments + query_arguments
        d.append(data_table + "_" + re.sub(r"\W+", "_", json.dumps(data, sort_keys=True)).strip("_") + ".hdf5")
    return os.path.join(*d)


def store_results(dataset_name: str, count: int, definition: Definition, query_arguments: Any, attrs, results, batch, filter_id, dataset_size, data_table, att_idx=0, segment_size: Optional[int] = None):
    """
    Stores results for an algorithm (and hyperparameters) running against a dataset in a HDF5 file.

    Args:
        dataset_name (str): The name of the dataset.
        count (int): The count of records.
        definition (Definition): The definition of the algorithm.
        query_arguments (Any): Additional arguments for the query.
        attrs (dict): Attributes to be stored in the file.
        results (list): Results to be stored.
        batch (bool): If True, the batch mode is activated.
        segment_size (int, optional): Milvus segment size in MB; when set, uses ablation path.
    """
    filename = build_result_filepath(dataset_name, count, definition, query_arguments, batch, filter_id, dataset_size, data_table, att_idx, segment_size)
    directory, _ = os.path.split(filename)
    print(f"===========\nSaving results with filename: {filename}\n===========")
    if not os.path.isdir(directory):
        os.makedirs(directory)
        
    # delete existing file
    if os.path.isfile(filename):
        os.remove(filename)

    with h5py.File(filename, "w") as f:
        if segment_size is not None:
            attrs = {**attrs, "segment_size": segment_size}
        for k, v in attrs.items():
            f.attrs[k] = v
        times = f.create_dataset("times", (len(results),), "f")
        neighbors = f.create_dataset("neighbors", (len(results), count), "i")
        distances = f.create_dataset("distances", (len(results), count), "f")
        
        for i, (time, ds) in enumerate(results): 
            times[i] = time
            try:
                neighbors[i] = [n for n, d in ds] + [-1] * (count - len(ds))
                distances[i] = [d for n, d in ds] + [float("inf")] * (count - len(ds))
            except Exception as e:
                print(f"Error storing neighbors for result {i}: {ds}", flush=True)
                raise e
            


def load_all_results(dataset: Optional[str] = None, 
                 count: Optional[int] = None, 
                 batch_mode: bool = False) -> Iterator[Tuple[dict, h5py.File]]:
    """
    Loads all the results from the HDF5 files in the specified path.

    Args:
        dataset (str, optional): The name of the dataset.
        count (int, optional): The count of records.
        batch_mode (bool, optional): If True, the batch mode is activated.

    Yields:
        tuple: A tuple containing properties as a dictionary and an h5py file object.
    """
    temp_filepath = build_result_filepath(dataset, count)
    print(temp_filepath)
    for root, whatisit, files in os.walk(temp_filepath):
        print("==========================", "\n", root, "\n\n", whatisit, "\n\n", files, "\n", "==========================")
        for filename in files:
            if os.path.splitext(filename)[-1] != ".hdf5":
                continue
            print(root, filename)
            try:
                with h5py.File(os.path.join(root, filename), "r+") as f:
                    properties = dict(f.attrs)
                    if batch_mode != properties["batch_mode"]:
                        continue
                    yield properties, f
            except Exception:
                print(f"Was unable to read {filename}")
                traceback.print_exc()


def get_unique_algorithms() -> Set[str]:
    """
    Retrieves unique algorithm names from the results.

    Returns:
        set: A set of unique algorithm names.
    """
    algorithms = set()
    for batch_mode in [False, True]:
        for properties, _ in load_all_results(batch_mode=batch_mode):
            algorithms.add(properties["algo"])
    return algorithms