"""
PG-Vector brute-force (exact) plan.

This module loads the vectors into a PostgreSQL table WITHOUT building any
vector index (and without an index on the filter attribute). Every query is
therefore an exact sequential scan: the filter predicate is applied and the
remaining rows are ordered by the cosine distance operator and limited to k.
This yields exact filtered top-k results (recall == 1.0) and serves as the
PG-Vector brute-force baseline for the FANNS experiments.

Connection parameters follow the same conventions as the pgvector module
(see ANN_BENCHMARKS_PG_* environment variables).
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
    def __init__(self, metric):
        metric = "angular"  # forced to be angular
        self._metric = metric
        self._cur = None
        self._query = None

    def set_query(self, metric, filter):
        self._query = filtered_order_query(metric, filter, self._dataset_type)
        return self._query

    def ensure_pgvector_extension_created(self, conn: psycopg.Connection) -> None:
        with conn.cursor() as cur:
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
            psycopg_connect_kwargs[arg_name] = get_pg_conn_param(arg_name, 'ann')

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

        self._dataset_type = dataset_type
        attrs = prepare_attrs(X_attr, dataset_type)
        self._attrs = attrs
        cols = create_items_table(cur, X.shape[1], dataset_type, attrs)
        print("copying data...")
        try:
            copy_items_rows(cur, X, attrs, cols)
        except Exception as e:
            print(f"Error during COPY: {e}")
            raise

        # Brute force: deliberately do NOT create any index (neither a vector
        # index nor an index on filter attributes). Queries run as exact seq scans.
        print("brute-force plan: no index created, analyzing table...")
        cur.execute("ANALYZE items")
        print("done!")

        self._cur = cur

    def set_query_arguments(self, placeholder=0):
        # No search parameters to sweep for the exact brute-force plan.
        self._placeholder = placeholder

    def query(self, v, n, filter):
        if filter == ["No_filter"] or filter == "No_filter":
            params = (v, n)
        elif len(filter) == 3:
            params = (v, n)
        else:
            raise ValueError(f"Invalid filter value: {filter}. Expected 'No_filter' or a tuple of three values.")

        self._query = self.set_query(self._metric, filter)
        try:
            self._cur.execute(self._query, params, binary=True, prepare=True)
        except Exception as e:
            print(f"DEBUG: Query: {self._query}")
            print(f"DEBUG: Filter value: {filter}")
            print(f"DEBUG: Error during query execution: {str(e)}")
            raise

        result = [id for id, in self._cur.fetchall()]
        return result

    def get_memory_usage(self):
        if self._cur is None:
            return 0
        try:
            self._cur.execute("SELECT pg_relation_size('items')")
            return self._cur.fetchone()[0] / 1024
        except Exception:
            return 0

    def __str__(self):
        return "PGVectorBruteForce(metric=%s)" % self._metric
