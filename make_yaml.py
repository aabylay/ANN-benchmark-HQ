import yaml

def make_yaml(algo, m=None, ef_c=None, ef_s_list=None, ivf_algo=False, dataset_size=None):
    if ivf_algo:
        if dataset_size == 'small':
            clusters = [100, 500]
            probes = [1, 5, 10, 20, 50]
        elif dataset_size == 'medium':
            clusters = [300, 1200]
            probes = [1, 5, 10, 50, 150]
        elif dataset_size == 'large':
            clusters = [750,1600]
            probes = [1, 5, 10, 50, 300]
    else:
        clusters = None
        probes = None
    
    if algo == "milvus-hnsw" or algo == "milvus-ivfflat":
        output_path=f"ann_benchmarks/algorithms/milvus/config.yml"
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
                                "args": {"nlist": clusters},
                                "query_args": [probes]
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
                                "query_args": [ef_s_list]
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

    elif algo == "pgvector":
        output_path=f"ann_benchmarks/algorithms/pgvector/config.yml"
        data = {
            "float": {
                "any": [
                    {
                        "base_args": ["@metric"],
                        "constructor": "PGVector",
                        "disabled": False,
                        "docker_tag": "ann-benchmarks-pgvector",
                        "module": "ann_benchmarks.algorithms.pgvector",
                        "name": "pgvector",
                        "run_groups": {
                            "HNSW": {
                                "args": {
                                    "M": [m],
                                    "efConstruction": [ef_c]
                                },
                                "query_args": [ef_s_list]
                            }
                        }
                    }
                ]
            }
        }

    elif algo == "pgvector_ivf":
        output_path=f"ann_benchmarks/algorithms/pgvector_ivf/config.yml"
        data = {
            "float": {
                "any": [
                    {
                        "base_args": ["@metric"],
                        "constructor": "PGVector",
                        "disabled": False,
                        "docker_tag": "ann-benchmarks-pgvector",
                        "module": "ann_benchmarks.algorithms.pgvector_ivf",
                        "name": "pgvector_ivf",
                        "run_groups": {
                            "IVFFlat": {
                                "args": {"clusters": clusters},
                                "query_args": [probes]
                            }
                        }
                    }
                ]
            }
        }
    
    elif algo == "hnsw(faiss)":
        output_path = f"ann_benchmarks/algorithms/faiss_hnsw/config.yml"
        data = {
            "float": {
                "any": [
                    {
                        "base_args": ["@metric"],
                        "constructor": "FaissHNSW",
                        "disabled": False,
                        "docker_tag": "custom-hnsw-faiss",
                        "module": "ann_benchmarks.algorithms.faiss_hnsw",
                        "name": "hnsw(faiss)",
                        "run_groups": {
                            f"M-{m}": {
                                "arg_groups": [{"M": m, "efConstruction": ef_c}],
                                "args": {},
                                "query_args": [ef_s_list]
                            }
                        }
                    }
                ]
            }
        }

    elif algo == "faiss-ivf":
        output_path = f"ann_benchmarks/algorithms/faiss/config.yml"
        data = {
            "float": {
                "any": [
                    {
                        "base_args": ["@metric"],
                        "constructor": "FaissIVF",
                        "disabled": False,
                        "docker_tag": "custom-hnsw-faiss",
                        "module": "ann_benchmarks.algorithms.faiss",
                        "name": "faiss-ivf",
                        "run_groups": {
                            f"base": {
                                "args": {"clusters": clusters},
                                "query_args": [probes]
                            }
                        }
                    }
                ]
            }
        }
    
    with open(output_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)
