import yaml

def make_yaml_ablation(segment_size, algo="milvus-hnsw", m=None, ef_c=None, ef_s_list=None, 
                       ivf_algo=False, dataset_size=None):
    """
    Generate Milvus config.yml for ablation study with specific segment size.
    
    Args:
        segment_size: Segment size in MB (512, 1024, 2048, 4096, 8192, 16384)
        algo: Algorithm name
        m: HNSW M parameter
        ef_c: HNSW efConstruction parameter
        ef_s_list: List of efSearch values
        ivf_algo: Whether this is an IVF algorithm
        dataset_size: Dataset size (small, medium, large)
    """
    
    # Docker tag includes segment size
    docker_tag = f"ann-benchmarks-milvus-seg{segment_size}"
    
    if ivf_algo:
        clusters = [1000]
        probes = [50]
    else:
        clusters = None
        probes = None
    
    output_path = "ann_benchmarks/algorithms/milvus/config.yml"
    
    data = {
        "float": {
            "any": [
                {
                    "base_args": ["@metric", "@dimension"],
                    "constructor": "MilvusFLAT",
                    "disabled": False,
                    "docker_tag": docker_tag,
                    "module": "ann_benchmarks.algorithms.milvus",
                    "name": "milvus-flat",
                    "run_groups": {
                        "FLAT": {
                            "args": {"placeholder": [0]}
                        }
                    }
                },
                {
                    "base_args": ["@metric", "@dimension"],
                    "constructor": "MilvusIVFFLAT",
                    "disabled": False,
                    "docker_tag": docker_tag,
                    "module": "ann_benchmarks.algorithms.milvus",
                    "name": "milvus-ivfflat",
                    "run_groups": {
                        "IVFFLAT": {
                            "args": {"nlist": clusters if clusters else [1000]},
                            "query_args": [probes if probes else [50]]
                        }
                    }
                },
                {
                    "base_args": ["@metric", "@dimension"],
                    "constructor": "MilvusIVFSQ8",
                    "disabled": False,
                    "docker_tag": docker_tag,
                    "module": "ann_benchmarks.algorithms.milvus",
                    "name": "milvus-ivfsq8",
                    "run_groups": {
                        "IVFSQ8": {
                            "args": {"nlist": [128, 256, 512, 1024, 2048, 4096]},
                            "query_args": [[1, 10, 20, 50, 100]]
                        }
                    }
                },
                {
                    "base_args": ["@metric", "@dimension"],
                    "constructor": "MilvusIVFPQ",
                    "disabled": False,
                    "docker_tag": docker_tag,
                    "module": "ann_benchmarks.algorithms.milvus",
                    "name": "milvus-ivfpq",
                    "run_groups": {
                        "IVFPQ": {
                            "args": {
                                "nlist": [128, 256, 512, 1024, 2048, 4096],
                                "m": [2, 4]
                            },
                            "query_args": [[1, 10, 20, 50, 100]]
                        }
                    }
                },
                {
                    "base_args": ["@metric", "@dimension"],
                    "constructor": "MilvusHNSW",
                    "disabled": False,
                    "docker_tag": docker_tag,
                    "module": "ann_benchmarks.algorithms.milvus",
                    "name": "milvus-hnsw",
                    "run_groups": {
                        "HNSW": {
                            "args": {
                                "M": [m] if m else [10],
                                "efConstruction": [ef_c] if ef_c else [50]
                            },
                            "query_args": [ef_s_list if ef_s_list else [200]]
                        }
                    }
                },
                {
                    "base_args": ["@metric", "@dimension"],
                    "constructor": "MilvusSCANN",
                    "disabled": False,
                    "docker_tag": docker_tag,
                    "module": "ann_benchmarks.algorithms.milvus",
                    "name": "milvus-scann",
                    "run_groups": {
                        "SCANN": {
                            "args": {"nlist": [64, 128, 256, 512, 1024, 2048, 4096, 8192]},
                            "query_args": [[1, 10, 20, 30, 50]]
                        }
                    }
                }
            ]
        }
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)
    
    print(f"Generated config.yml with docker_tag: {docker_tag}")
    return docker_tag


if __name__ == "__main__":
    # Example usage
    import argparse
    parser = argparse.ArgumentParser(description="Generate Milvus config for ablation study")
    parser.add_argument("--segment_size", type=int, required=True, 
                        help="Segment size in MB (512, 1024, 2048, 4096, 8192, 16384)")
    parser.add_argument("--algo", type=str, default="milvus-hnsw", help="Algorithm name")
    parser.add_argument("--m", type=int, default=15, help="HNSW M parameter")
    parser.add_argument("--ef_c", type=int, default=75, help="HNSW efConstruction parameter")
    parser.add_argument("--dataset_size", type=str, default="small", help="Dataset size")
    
    args = parser.parse_args()
    
    ef_s_list = [200]
    ivf_algo = args.algo in ["milvus-ivfflat"]
    
    make_yaml_ablation(
        segment_size=args.segment_size,
        algo=args.algo,
        m=args.m,
        ef_c=args.ef_c,
        ef_s_list=ef_s_list,
        ivf_algo=ivf_algo,
        dataset_size=args.dataset_size
    )
