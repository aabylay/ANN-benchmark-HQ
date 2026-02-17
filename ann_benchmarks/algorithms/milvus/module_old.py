from time import sleep
from pymilvus import DataType, connections, utility, Collection, CollectionSchema, FieldSchema, DataType
import os

from ..base.module import BaseANN


def metric_mapping(_metric: str):
    _metric_type = {"angular": "COSINE", "euclidean": "L2"}.get(_metric, None)
    if _metric_type is None:
        raise Exception(f"[Milvus] Not support metric type: {_metric}!!!")
    return _metric_type


class Milvus(BaseANN):
    def __init__(self, metric, dim, index_param):
        self._metric = metric
        self._dim = dim
        self._metric_type = metric_mapping(self._metric)
        self.start_milvus()
        self.connects = connections
        max_trys = 10
        for try_num in range(max_trys):
            try:
                self.connects.connect("default", host='localhost', port='19530')
                break
            except Exception as e:
                if try_num == max_trys - 1:
                    raise Exception(f"[Milvus] connect to milvus failed: {e}!!!")
                print(f"[Milvus] try to connect to milvus again...")
                sleep(1)
        print(f"[Milvus] Milvus version: {utility.get_server_version()}")
        self.collection_name = "test_milvus"
        for attempt in range(3):
            if utility.has_collection(self.collection_name):
                print(f"[Milvus] collection {self.collection_name} already exists, drop it...")
                try:
                    utility.drop_collection(self.collection_name)
                    print(f"Collection dropped.", flush=True)

                except Exception as e:
                    if "InvalidateCollectionMetaCache" in str(e) and attempt < 2:
                        print(f"[Milvus] drop collection failed due to InvalidateCollectionMetaCache, try again...")
                        sleep(5)
                    else:
                        raise (f"[Milvus] drop collection failed: {e}!!!")
                
            else:
                break

    def start_milvus(self):
        try:
            os.system("docker compose down")
            os.system("docker compose up -d")
            print("[Milvus] docker compose up successfully!!!")
        except Exception as e:
            print(f"[Milvus] docker compose up failed: {e}!!!")

    def stop_milvus(self):
        try:
            os.system("docker compose down")
            print("[Milvus] docker compose down successfully!!!")
        except Exception as e:
            print(f"[Milvus] docker compose down failed: {e}!!!")

    def create_collection(self, dataset_type):
        
        filed_id = FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True
        )
        filed_vec = FieldSchema(
            name="vector",
            dtype=DataType.FLOAT_VECTOR,
            dim=self._dim
        )
        
        if dataset_type == "movies":
            title_id = FieldSchema(
                name="title_id",
                dtype=DataType.VARCHAR,
                max_length=16
            )            
            avg_rating = FieldSchema(
                name="avg_rating",
                dtype=DataType.FLOAT
            )
            """
            is_adult = FieldSchema(
                name="is_adult",
                dtype=DataType.BOOL
            )
            genre = FieldSchema(
                name="genre",
                dtype=DataType.VARCHAR,
                max_length=64
            )
            num_votes = FieldSchema(
                name="numvotes",
                dtype=DataType.INT64
            )
            start_year = FieldSchema(
                name="start_year",
                dtype=DataType.INT64
            )"""
            
        elif dataset_type == "reviews":
            review_id = FieldSchema(
                name="review_id",
                dtype=DataType.VARCHAR,
                max_length=16
            )
            """
            title_id = FieldSchema(
                name="title_id",
                dtype=DataType.VARCHAR,
                max_length=16
            )            
            user_id = FieldSchema(
                name="user_id",
                dtype=DataType.VARCHAR,
                max_length=16
            )
            upvotes = FieldSchema(
                name="upvotes",
                dtype=DataType.INT64
            )
            downvotes = FieldSchema(
                name="downvotes",
                dtype=DataType.INT64
            )"""
            totalvotes = FieldSchema(
                name="totalvotes",
                dtype=DataType.INT64
            )
        
        if dataset_type == "movies":
            schema = CollectionSchema(
                fields=[filed_id, filed_vec, title_id, avg_rating],
                description="Test milvus search on MOVIES dataset",
            )
        elif dataset_type == "reviews":
            schema = CollectionSchema(
                fields=[filed_id, filed_vec, review_id, totalvotes],
                description="Test milvus search on REVIEWS dataset",
            )
        self.collection = Collection(
            self.collection_name,
            schema,
            consistence_level="STRONG"
        )
        print(f"[Milvus] Create collection {self.collection.describe()} successfully!!!")

    def insert(self, X_ids, X, X_attr):
        # insert data
        print(f"[Milvus] Insert {len(X)} data into collection {self.collection_name}...")
        batch_size = 1000
        for i in range(0, len(X), batch_size):
            batch_ids = X_ids[i: min(i + batch_size, len(X))]
            # if i == 0: print("BATCH IDS:", batch_ids)  # Print first 3 IDs for debugging
            batch_ids = [tid.decode('utf-8') for tid in batch_ids]
            batch_data = X[i: min(i + batch_size, len(X))]
            batch_attr = X_attr[i: min(i + batch_size, len(X))]
            entities = [
                [i for i in range(i, min(i + batch_size, len(X)))],
                batch_data.tolist(),
                batch_ids,
                batch_attr.tolist(),
            ]
            
            self.collection.insert(entities)
        self.collection.flush()
        print(f"[Milvus] {self.collection.num_entities} data has been inserted into collection {self.collection_name}!!!")

    def get_index_param(self):
        raise NotImplementedError()

    def create_index(self):
        # create index
        print(f"[Milvus] Create index for collection {self.collection_name} with params \n {self.get_index_param()}...")
        self.collection.create_index(
            field_name = "vector",
            index_params = self.get_index_param(),
            index_name = "vector_index"
        )
        utility.wait_for_index_building_complete(
            collection_name = self.collection_name,
            index_name = "vector_index"
        )
        index = self.collection.index(index_name = "vector_index")
        index_progress =  utility.index_building_progress(
            collection_name = self.collection_name,
            index_name = "vector_index"
        )
        print(f"[Milvus] Create index {index.to_dict()} {index_progress} for collection {self.collection_name} successfully!!!")

    def load_collection(self):
        # load collection
        print(f"[Milvus] Load collection {self.collection_name}...")
        self.collection.load()
        utility.wait_for_loading_complete(self.collection_name)
        print(f"[Milvus] Load collection {self.collection_name} successfully!!!")
    
    def create_attr_index(self, dataset_type):
        # create attribute index
        if dataset_type == "movies":
            field_name = "avg_rating"
            index_name = "avg_rating_index"
        elif dataset_type == "reviews":
            field_name = "totalvotes"
            index_name = "totalvotes_index"

        print(f"[Milvus] Create attribute index for collection {self.collection_name} on attribute {field_name}...")
        self.collection.create_index(
            field_name = field_name,
            index_name = index_name
        )
        utility.wait_for_index_building_complete(
            collection_name = self.collection_name,
            index_name = index_name
        )
        index = self.collection.index(index_name = index_name)
        index_progress =  utility.index_building_progress(
            collection_name = self.collection_name,
            index_name = index_name
        )
        print(f"[Milvus] Create attribute index {index.to_dict()} {index_progress} for collection {self.collection_name} successfully!!!")

    def fit(self, X_ids, X, X_attr, dataset_type):
        self.create_collection(dataset_type)
        if dataset_type == "movies":
            self.insert(X_ids, X, X_attr)
        elif dataset_type == "reviews":
            self.insert(X_ids, X, X_attr)
        
        self.create_index()
        self.load_collection()
    
    def load_col(self):
        self.load_collection()
        
    def fit_idx(self, dataset_type):
        self.create_attr_index(dataset_type) # COMMENT OUT IF NOT NEEDED

    def query(self, v, n, filter = ["No_filter"]):
        
        if filter == ["No_filter"]:
            # print("Pure vector search")
            results = self.collection.search(
                data = [v],
                anns_field = "vector",
                param = self.search_params,
                limit = n,
                output_fields=["id"]
            )
        else:
            try: filter_expr = f"{filter[0][0]} {filter[0][1]} {filter[0][2]}"
            except Exception as e: raise Exception(f"WARNING! Unrecognized filter format: {filter}")

            results = self.collection.search(
                data = [v],
                anns_field = "vector",
                param = self.search_params,
                filter_expr = filter_expr, # added for queries with filters
                limit = n,
                output_fields=["id"]
            )
                        
        ids = [r.entity.get("id") for r in results[0]]
        return ids #

    def done(self, final_call=False):
        print(f"[Milvus] Releasing and dropping collection {self.collection_name}...", flush=True)
        self.collection.release()
        print(f"[Milvus] Collection released.", flush=True)
        utility.drop_collection(self.collection_name)
        print(f"[Milvus] Collection dropped.", flush=True)
        if final_call:
            self.stop_milvus()


class MilvusFLAT(Milvus):
    def __init__(self, metric, dim, index_param):
        super().__init__(metric, dim, index_param)
        self.name = f"MilvusFLAT metric:{self._metric}"

    def get_index_param(self):
        return {
            "index_type": "FLAT",
            "metric_type": self._metric_type
        }

    def query(self, v, n):
        self.search_params = {
            "metric_type": self._metric_type,
        }
        results = self.collection.search(
            data = [v],
            anns_field = "vector",
            param = self.search_params,
            limit = n,
            output_fields=["id"]
        )
        ids = [r.entity.get("id") for r in results[0]]
        return ids


class MilvusIVFFLAT(Milvus):
    def __init__(self, metric, dim, index_param):
        super().__init__(metric, dim, index_param)
        self._index_nlist = index_param.get("nlist", None)

    def get_index_param(self):
        return {
            "index_type": "IVF_FLAT",
            "params": {
                "nlist": self._index_nlist
            },
            "metric_type": self._metric_type
        }

    def set_query_arguments(self, nprobe):
        self.search_params = {
            "metric_type": self._metric_type,
            "params": {"nprobe": nprobe}
        }
        self.name = f"MilvusIVFFLAT metric:{self._metric}, index_nlist:{self._index_nlist}, search_nprobe:{nprobe}"


class MilvusIVFSQ8(Milvus):
    def __init__(self, metric, dim, index_param):
        super().__init__(metric, dim, index_param)
        self._index_nlist = index_param.get("nlist", None)

    def get_index_param(self):
        return {
            "index_type": "IVF_SQ8",
            "params": {
                "nlist": self._index_nlist
            },
            "metric_type": self._metric_type
        }

    def set_query_arguments(self, nprobe):
        self.search_params = {
            "metric_type": self._metric_type,
            "params": {"nprobe": nprobe}
        }
        self.name = f"MilvusIVFSQ8 metric:{self._metric}, index_nlist:{self._index_nlist}, search_nprobe:{nprobe}"


class MilvusIVFPQ(Milvus):
    def __init__(self, metric, dim, index_param):
        super().__init__(metric, dim, index_param)
        self._index_nlist = index_param.get("nlist", None)
        self._index_m = index_param.get("m", None)
        self._index_nbits = index_param.get("nbits", None)

    def get_index_param(self):
        assert self._dim % self._index_m == 0, "dimension must be able to be divided by m"
        return {
            "index_type": "IVF_PQ",
            "params": {
                "nlist": self._index_nlist,
                "m": self._index_m,
                "nbits": self._index_nbits if self._index_nbits else 8 
            },
            "metric_type": self._metric_type
        }
    
    def set_query_arguments(self, nprobe):
        self.search_params = {
            "metric_type": self._metric_type,
            "params": {"nprobe": nprobe}
        }
        self.name = f"MilvusIVFPQ metric:{self._metric}, index_nlist:{self._index_nlist}, search_nprobe:{nprobe}"


class MilvusHNSW(Milvus):
    def __init__(self, metric, dim, index_param):
        super().__init__(metric, dim, index_param)
        self._index_m = index_param.get("M", None)
        self._index_ef = index_param.get("efConstruction", None)

    def get_index_param(self):
        return {
            "index_type": "HNSW",
            "params": {
                "M": self._index_m,
                "efConstruction": self._index_ef
            },
            "metric_type": self._metric_type
        }

    def set_query_arguments(self, ef):
        self.search_params = {
            "metric_type": self._metric_type,
            "params": {"ef": ef}
        }
        self.name = f"MilvusHNSW metric:{self._metric}, index_M:{self._index_m}, index_ef:{self._index_ef}, search_ef={ef}"


class MilvusSCANN(Milvus):
    def __init__(self, metric, dim, index_param):
        super().__init__(metric, dim, index_param)
        self._index_nlist = index_param.get("nlist", None)

    def get_index_param(self):
        return {
            "index_type": "SCANN",
            "params": {
                "nlist": self._index_nlist
            },
            "metric_type": self._metric_type
        }

    def set_query_arguments(self, nprobe):
        self.search_params = {
            "metric_type": self._metric_type,
            "params": {"nprobe": nprobe}
        }
        self.name = f"MilvusSCANN metric:{self._metric}, index_nlist:{self._index_nlist}, search_nprobe:{nprobe}"
