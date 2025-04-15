import os
from make_yaml import make_yaml

filters = [6, 6.5, 7, 7.5, 8, 8.3, 9, 9.5]
k_vals  = [1, 2, 5, 10, 20, 50, 100]

for filter in filters:
    for m in [4, 16, 64]:
        for k in k_vals:
            ef_c = m*4
            ef_s = max(40, k*2)
            make_yaml(m, ef_c, ef_s)
            print(f"Running for k: {k}")
            os.system(f"python run.py --algorithm milvus-hnsw --dataset glove-100-angular --count {k} --filter {filter}")
            print(f"Finished for k: {k}")
