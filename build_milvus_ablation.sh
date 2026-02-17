#!/bin/bash
# Build Milvus Docker images with different segment sizes for ablation study

set -e

DOCKERFILE_PATH="ann_benchmarks/algorithms/milvus"

# Segment sizes to test (in MB)
SEGMENT_SIZES=(512 1024 2048 4096 8192 16384)

echo "=========================================="
echo "Building Milvus images for ablation study"
echo "=========================================="

# Copy ablation entry point to milvus directory for Docker build context
echo "Copying run_algorithm_ablation.py to build context..."
cp run_algorithm_ablation.py ${DOCKERFILE_PATH}/

for seg_size in "${SEGMENT_SIZES[@]}"; do
    disk_seg_size=$((seg_size * 2))
    image_tag="ann-benchmarks-milvus-seg${seg_size}"
    
    echo ""
    echo "Building image: ${image_tag}"
    echo "  - maxSize: ${seg_size} MB"
    echo "  - diskSegmentMaxSize: ${disk_seg_size} MB"
    echo ""
    
    docker build \
        --build-arg SEGMENT_MAX_SIZE=${seg_size} \
        --build-arg DISK_SEGMENT_MAX_SIZE=${disk_seg_size} \
        -t ${image_tag} \
        ${DOCKERFILE_PATH}/
    
    echo "Successfully built: ${image_tag}"
done

# Clean up copied file
echo "Cleaning up..."
rm -f ${DOCKERFILE_PATH}/run_algorithm_ablation.py

echo ""
echo "=========================================="
echo "All images built successfully!"
echo "=========================================="
echo ""
echo "Available images:"
for seg_size in "${SEGMENT_SIZES[@]}"; do
    echo "  - ann-benchmarks-milvus-seg${seg_size}"
done
