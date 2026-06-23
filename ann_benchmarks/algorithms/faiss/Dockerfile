FROM ann-benchmarks

# Standard (upstream) FAISS, modern enough to expose the filtered-search API the
# FAISS modules rely on: SearchParametersHNSW / SearchParametersIVF /
# IDSelectorBitmap. This replaces the old conda faiss-cpu 1.7.1 image (and the
# out-of-tree "custom-hnsw-faiss" image), which lacked those symbols.
RUN pip3 install --no-cache-dir faiss-cpu==1.12.0

RUN python3 -c "import faiss; print('faiss', faiss.__version__)"
RUN python3 -c "import faiss; missing=[a for a in ['IndexFlatL2','IndexHNSWFlat','SearchParametersHNSW','SearchParametersIVF','IDSelectorBitmap'] if not hasattr(faiss, a)]; assert not missing, 'missing faiss API: %s' % missing; print('faiss filtered-search API OK')"
