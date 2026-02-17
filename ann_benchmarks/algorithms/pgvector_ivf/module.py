"""
This module supports connecting to a PostgreSQL instance and performing vector
indexing and search using the pgvector extension. The default behavior uses
the "ann" value of PostgreSQL user name, password, and database name, as well
as the default host and port values of the psycopg driver.

If PostgreSQL is managed externally, e.g. in a cloud DBaaS environment, the
environment variable overrides listed below are available for setting PostgreSQL
connection parameters:

ANN_BENCHMARKS_PG_USER
ANN_BENCHMARKS_PG_PASSWORD
ANN_BENCHMARKS_PG_DBNAME
ANN_BENCHMARKS_PG_HOST
ANN_BENCHMARKS_PG_PORT

This module starts the PostgreSQL service automatically using the "service"
command. The environment variable ANN_BENCHMARKS_PG_START_SERVICE could be set
to "false" (or e.g. "0" or "no") in order to disable this behavior.

This module will also attempt to create the pgvector extension inside the
target database, if it has not been already created.
"""

import subprocess
import sys
import os

import pgvector.psycopg
import psycopg

from typing import Dict, Any, Optional

from ..base.module import BaseANN
from ...util import get_bool_env_var


def get_pg_param_env_var_name(pg_param_name: str) -> str:
    return f'ANN_BENCHMARKS_PG_{pg_param_name.upper()}'


def get_pg_conn_param(
        pg_param_name: str,
        default_value: Optional[str] = None) -> Optional[str]:
    env_var_name = get_pg_param_env_var_name(pg_param_name)
    env_var_value = os.getenv(env_var_name, default_value)
    if env_var_value is None or len(env_var_value.strip()) == 0:
        return default_value
    return env_var_value


class PGVector(BaseANN):
    def __init__(self, metric, method_param):
        metric = "angular" # forced to be angular
        self._metric = metric
        self._clusters = method_param['clusters']
        self._cur = None
        self._query = None

    def set_query(self, metric, filter):
        if metric == "angular":
            if filter == ["No_filter"] or filter == "No_filter":
                
                self._query = """
                    SELECT id
                    FROM items
                    ORDER BY embedding <=> %s
                    LIMIT %s
                """
                
            else:
                attr, literal, value = filter[0], filter[1], filter[2]
                
                self._query = f"""
                    SELECT id 
                    FROM items 
                    WHERE filter_attr {literal} {value}::FLOAT
                    ORDER BY embedding <=> %s
                    LIMIT %s
                """
            # self._query = """SELECT COUNT(*) FROM items WHERE averagerating >= %s"""
            # self._query = """SELECT COUNT(*) FROM items WHERE averagerating >= %s AND embedding <=> %s LIMIT %s"""
            # self._query = """SELECT COUNT(*) FROM items"""
        elif metric == "euclidean":
            self._query = "SELECT id FROM items ORDER BY embedding <-> %s LIMIT %s"
        else:
            raise RuntimeError(f"unknown metric {metric}")
        return self._query

    def ensure_pgvector_extension_created(self, conn: psycopg.Connection) -> None:
        """
        Ensure that `CREATE EXTENSION vector` has been executed.
        """
        with conn.cursor() as cur:
            # We have to use a separate cursor for this operation.
            # If we reuse the same cursor for later operations, we might get
            # the following error:
            # KeyError: "couldn't find the type 'vector' in the types registry"
            cur.execute(
                "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')")
            pgvector_exists = cur.fetchone()[0]
            if pgvector_exists:
                print("vector extension already exists")
            else:
                print("vector extension does not exist, creating")
                cur.execute("CREATE EXTENSION vector")

    def fit(self, X_tid, X, X_attr, dataset_type):
        psycopg_connect_kwargs: Dict[str, Any] = dict(
            autocommit=True,
        )
        for arg_name in ['user', 'password', 'dbname']:
            # The default value is "ann" for all of these parameters.
            psycopg_connect_kwargs[arg_name] = get_pg_conn_param(
                arg_name, 'ann')

        # If host/port are not specified, leave the default choice to the
        # psycopg driver.
        pg_host: Optional[str] = get_pg_conn_param('host')
        if pg_host is not None:
            psycopg_connect_kwargs['host'] = pg_host

        pg_port_str: Optional[str] = get_pg_conn_param('port')
        if pg_port_str is not None:
            psycopg_connect_kwargs['port'] = int(pg_port_str)

        should_start_service = get_bool_env_var(
            get_pg_param_env_var_name('start_service'),
            default_value=True)
        if should_start_service:
            subprocess.run(
                "service postgresql start",
                shell=True,
                check=True,
                stdout=sys.stdout,
                stderr=sys.stderr)
        else:
            print(
                "Assuming that PostgreSQL service is managed externally. "
                "Not attempting to start the service.")

        conn = psycopg.connect(**psycopg_connect_kwargs)
        self.ensure_pgvector_extension_created(conn)
        pgvector.psycopg.register_vector(conn)        
        cur = conn.cursor()        
        
        #cur.execute("SET max_parallel_workers_per_gather = 32")
        #cur.execute("SET max_parallel_maintenance_workers = 32")
        #cur.execute("SET max_parallel_workers = 48")
        #cur.execute("SET maintenance_work_mem = '8GB'")
        #cur.execute("SET work_mem = '1GB'")

        cur.execute("DROP TABLE IF EXISTS items")
        cur.execute("CREATE TABLE items (id int, embedding vector(%d), filter_attr FLOAT)" % X.shape[1])
        cur.execute("ALTER TABLE items ALTER COLUMN embedding SET STORAGE PLAIN")
        # SHOW shared_buffers;
        print("copying data...")
        try:
            with cur.copy("COPY items (id, embedding, filter_attr) FROM STDIN WITH (FORMAT BINARY)") as copy:
                copy.set_types(["int4", "vector", "float8"])
                for i, embedding in enumerate(X):
                    copy.write_row((i, embedding.tolist(), float(X_attr[i])))
        except Exception as e:
            print(f"ID: {i}, === SHAPE: {X_attr.shape}, === VALUE: {float(X_attr[i])}, === TYPE: {type(X_attr)}, === VALUE TYPE: {type(float(X_attr[i]))}")
            print(f"Error during COPY: {e}")
            raise

        # Create vector index
        print("creating index...")
        if self._metric == "angular":
            cur.execute(
                "CREATE INDEX ON items USING ivfflat (embedding vector_cosine_ops) WITH (lists = %d)" % (self._clusters)
            )
        elif self._metric == "euclidean":
            cur.execute("CREATE INDEX ON items USING ivfflat (embedding vector_l2_ops) WITH (lists = %d)" % (self._clusters))
        else:
            raise RuntimeError(f"unknown metric {self._metric}")
        print("done!")
        
        self._cur = cur
        
    # NEW FUNCTION: create index on filter attribute
    def fit_idx(self, dataset_type):
        print("creating attribute index...")
        self._cur.execute("CREATE INDEX ON items (filter_attr)")
        print(f"[PGVector] Created index on filter_attr")
        
    def set_query_arguments(self, probes):
        self._probes = probes
        self._cur.execute("SET ivfflat.probes = %d" % probes)
        # pass

    def query(self, v, n, filter):
        
        # Check if the filter is None, 0, "0", or "none" (case insensitive) and set params accordingly
        if filter == ["No_filter"] or filter == "No_filter":
            params = (v, n)
        elif len(filter) == 3:
            params = (v, n)
        else:
            raise ValueError(f"Invalid filter value: {filter}. Expected 'No_filter' or a tuple of three values.")

        self._query = self.set_query(self._metric, filter)
        # print(f"Query to be executed:\n{self._query}")
        try:
            self._cur.execute(self._query, params, binary=True, prepare=True)
        except Exception as e:
            # Debug prints
            # print(f"DEBUG: Query: {self._query}")
            print(f"DEBUG: Query: {self._query}")
            print(f"DEBUG: Filter value: {filter}")
            print(f"DEBUG: Error during query execution: {str(e)}")
            raise
        
        # self._cur.execute(self._query, binary=True, prepare=True)
        result = [id for id, in self._cur.fetchall()]
        # print(f"Result IDs: {result}")
        return result

    def get_memory_usage(self):
        if self._cur is None:
            return 0
        self._cur.execute("SELECT pg_relation_size('items_embedding_idx')")
        return self._cur.fetchone()[0] / 1024

    def __str__(self):
        try: self._ef_search
        except AttributeError: self._ef_search = 0            
        return f"PGVector(m={self._clusters}, ef_search={self._probes})"
