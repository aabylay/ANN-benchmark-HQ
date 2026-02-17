"""
Entry point for Milvus Docker containers. Supports --segment_size for ablation study.
Uses runner.run_from_cmdline (segment_size merged into main runner).
"""
from ann_benchmarks.runner import run_from_cmdline

run_from_cmdline()
