# CaseLaw_7M Upstream Sources and Predicate Mapping

## Upstream Repository and Commit

- Repository: `https://github.com/harsha-simhadri/big-ann-benchmarks`
- Commit: `89a3abaafa63dda46b94b308bdf039e699841b3b` (2026-07-13)
- Dataset class: `CaselawDataset(7414023)` registered as `caselaw-7M` in `benchmark/datasets.py`
- Documentation: `dataset_preparation/caselaw_dataset.md`
- Source corpus: Collaborative Open Legal Data (COLD) Cases (8,362,175 source cases; CC0 1.0). Embeddings and auxiliary files are licensed under CDLA-2.0.
- Embedding model: OpenAI `text-embedding-3-small` (single-vector `.fbin` header dimension `d = 1536`, metric `euclidean`).

## Official Single-Vector Full-Base URLs (`caselaw-7M`)

Base URL: `https://comp21storage.z5.web.core.windows.net/caselaw/single_vector/`

| Filename | URL | Size (bytes) | Rows / Format |
| --- | --- | ---: | --- |
| `caselaw_base_embeddings.bin` | `https://comp21storage.z5.web.core.windows.net/caselaw/single_vector/caselaw_base_embeddings.bin` | `45,551,757,320` | `N = 7,414,023`, `d = 1,536`, `.fbin` (`int32 n, d` + `float32[n, d]`) |
| `caselaw_query_embeddings.bin` | `https://comp21storage.z5.web.core.windows.net/caselaw/single_vector/caselaw_query_embeddings.bin` | `551,804,936` | `nq = 89,812`, `d = 1,536`, `.fbin` (`int32 n, d` + `float32[n, d]`) |
| `caselaw_gt.bin` | `https://comp21storage.z5.web.core.windows.net/caselaw/single_vector/caselaw_gt.bin` | `71,849,608` | Unfiltered top-100 ground truth on `caselaw-7M` (`uint32 nq=89812, k=100`, `int32 ids[nq, 100]`, `float32 dists[nq, 100]`) |
| `caselaw_gt_filtered.bin` | `https://comp21storage.z5.web.core.windows.net/caselaw/single_vector/caselaw_gt_filtered.bin` | `36,162,440` | Filtered top-100 range ground truth on `caselaw-7M` (`int32 nq=89812, total_res=8950796`, `int32 nres[nq]`, `int32 ids[total_res]`) |
| `caselaw_base_metadata.jsonl` | `https://comp21storage.z5.web.core.windows.net/caselaw/single_vector/caselaw_base_metadata.jsonl` | `1,301,991,216` | `7,414,023` JSON lines (`doc_id`, `case_id`, `date_filed`, `court_jurisdiction`, `court_type`, `court_full_name`) |
| `caselaw_query_metadata.jsonl` | `https://comp21storage.z5.web.core.windows.net/caselaw/single_vector/caselaw_query_metadata.jsonl` | `15,595,530` | `89,812` JSON lines |
| `caselaw_query_filters.jsonl` | `https://comp21storage.z5.web.core.windows.net/caselaw/single_vector/caselaw_query_filters.jsonl` | `11,849,200` | `89,812` JSON lines (`query_id`, `filter`) |

## Filter JSON Schema and Canonical Predicate Mapping

Sample records from `single_vector/caselaw_query_filters.jsonl`:

```json
{"query_id": 0, "filter": {"$and": [{"date_filed": {"$gte": 734700}}, {"date_filed": {"$lte": 736161}}]}}
{"query_id": 1, "filter": {"$and": [{"court_full_name": {"$eq": "Court of Appeals for the Fifth Circuit"}}, {"date_filed": {"$gte": 729296}}, {"date_filed": {"$lte": 730757}}]}}
{"query_id": 3, "filter": {"court_full_name": {"$eq": "Supreme Court of Kansas"}}}
{"query_id": 6, "filter": {"$and": [{"court_type": {"$eq": "SA"}}, {"court_jurisdiction": {"$eq": "Pennsylvania, PA"}}]}}
```

Mapping rules (`ann_benchmarks.corpora.canonical_filter_from_json`):

1. Top-level `filter` is either a single field comparison object `{"<ident>": {"<op>": <val>}}` or `{"$and": [ ... ]}` containing 2 or 3 leaf comparisons (a filing-date range is encoded as two leaf comparisons `$gte` and `$lte` on `date_filed`, optionally conjoined with one court equality predicate).
2. JSON operators map as:
   - `"$eq"` $\to$ `"="`
   - `"$ne"` $\to$ `"!="`
   - `"$gte"` $\to$ `">="`
   - `"$lte"` $\to$ `"<="`
   - `"$gt"` $\to$ `">"`
   - `"$lt"` $\to$ `"<"`
3. `date_filed` values are formatted as base-10 integers (`date_filed >= 734700 AND date_filed <= 736161`).
4. `court_jurisdiction`, `court_type`, and `court_full_name` values are formatted as double-quoted UTF-8 strings (`court_full_name = "Supreme Court of Kansas"`).
5. Multiple clauses are joined with `" AND "` (four characters: space-A-N-D-space) in their published order.
