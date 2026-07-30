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
from ..pgvector_common import (
    copy_items_rows,
    create_attr_indexes,
    create_items_table,
    filtered_order_query,
    prepare_attrs,
)
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
        self._lists = None  # actual lists used (auto = round(sqrt(|D|)) when configured <= 0)
        self._n = 0

    def set_query(self, metric, filter):
        self._query = filtered_order_query(metric, filter, self._dataset_type)
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

        self._n = int(X.shape[0])
        self._dataset_type = dataset_type
        attrs = prepare_attrs(X_attr, dataset_type)
        self._attrs = attrs
        # Fixed construction param: number of IVF lists ~ sqrt(|D|), computed
        # per table. A configured value of 0 (or negative) requests this.
        lists = int(self._clusters) if self._clusters is not None else 0
        if lists <= 0:
            lists = max(1, int(round(self._n ** 0.5)))
        self._lists = lists
        print(f"pgvector_ivf: building IVFFlat with lists={lists} for n={self._n} ({dataset_type})")

        cols = create_items_table(cur, X.shape[1], dataset_type, attrs)
        print("copying data...")
        try:
            copy_items_rows(cur, X, attrs, cols)
        except Exception as e:
            print(f"Error during COPY: {e}")
            raise

        # Create vector index
        print("creating index...")
        if self._metric == "angular":
            cur.execute(
                "CREATE INDEX ON items USING ivfflat (embedding vector_cosine_ops) WITH (lists = %d)" % (self._lists)
            )
        elif self._metric == "euclidean":
            cur.execute("CREATE INDEX ON items USING ivfflat (embedding vector_l2_ops) WITH (lists = %d)" % (self._lists))
        else:
            raise RuntimeError(f"unknown metric {self._metric}")
        print("done!")
        
        self._cur = cur
        
    # NEW FUNCTION: create index on filter attribute
    def fit_idx(self, dataset_type):
        print("creating attribute index...")
        create_attr_indexes(self._cur, dataset_type)
        
    def set_query_arguments(self, probes):
        self._probes = probes
        self._cur.execute("SET ivfflat.probes = %d" % probes)
        # Post-filtering with iterative scan turned ON: pgvector keeps probing
        # additional lists (up to ivfflat.max_probes) until enough rows pass the
        # WHERE filter. relaxed_order lets the scan grow efficiently.
        self._cur.execute("SET ivfflat.iterative_scan = relaxed_order")
        # Permit probing up to all lists so low-selectivity filters can still
        # reach high recall; `probes` remains the swept starting point.
        self._cur.execute("SET ivfflat.max_probes = %d" % max(100, probes))

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
        try: self._probes
        except AttributeError: self._probes = 0
        return f"PGVectorIVF(lists={self._lists}, probes={self._probes})"
