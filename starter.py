import os
from HQ_make_yaml import make_yaml

filters = [None, 6, 6.5, 7, 7.5, 8, 8.3, 9, 9.5] # [6.5, 7.5, 8.5, 9.5] # 
k_vals  = [1, 2, 5, 10, 20]

for filter in filters:
    for m in [4, 8, 16]: # [4, 16, 64]:
        for k in k_vals:
            for ef_s in [20, 40, 60, 80, 100, 200, 300, 400, 500]:
                ef_c = m*4
                # ef_s = max(40, k*2)
                make_yaml(m, ef_c, ef_s)
                print(f"Running for k: {k}")
                if (filter is None) or (filter == 0) or (filter == "0"):
                    os.system(f"python run.py --algorithm milvus-hnsw --dataset glove-100-angular --count {k}")
                else:
                    os.system(f"python run.py --algorithm milvus-hnsw --dataset glove-100-angular --count {k} --filter {filter}")
                print(f"Finished for k: {k}")
