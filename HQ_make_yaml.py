import yaml

def make_yaml(m, ef_c, ef_s, output_path="ann_benchmarks/algorithms/milvus/config.yml"):
    data = {
        "float": {
            "any": [
                {
                    "base_args": ["@metric", "@dimension"],
                    "constructor": "MilvusFLAT",
                    "disabled": False,
                    "docker_tag": "ann-benchmarks-milvus",
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
                    "docker_tag": "ann-benchmarks-milvus",
                    "module": "ann_benchmarks.algorithms.milvus",
                    "name": "milvus-ivfflat",
                    "run_groups": {
                        "IVFFLAT": {
                            "args": {"nlist": [320, 800, 2000, 5000]},
                            "query_args": [[5, 10, 20, 50, 100]]
                        }
                    }
                },
                {
                    "base_args": ["@metric", "@dimension"],
                    "constructor": "MilvusIVFSQ8",
                    "disabled": False,
                    "docker_tag": "ann-benchmarks-milvus",
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
                    "docker_tag": "ann-benchmarks-milvus",
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
                    "docker_tag": "ann-benchmarks-milvus",
                    "module": "ann_benchmarks.algorithms.milvus",
                    "name": "milvus-hnsw",
                    "run_groups": {
                        "HNSW": {
                            "args": {
                                "M": [m],
                                "efConstruction": [ef_c]
                            },
                            "query_args": [[ef_s]]
                        }
                    }
                },
                {
                    "base_args": ["@metric", "@dimension"],
                    "constructor": "MilvusSCANN",
                    "disabled": False,
                    "docker_tag": "ann-benchmarks-milvus",
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
