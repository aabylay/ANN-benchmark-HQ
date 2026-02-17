FROM ann-benchmarks

# Build arguments for segment size configuration (defaults to 1024 MB)
ARG SEGMENT_MAX_SIZE=1024
ARG DISK_SEGMENT_MAX_SIZE=2048

# Add Docker's official GPG key:
RUN apt-get update
RUN apt-get install ca-certificates curl
RUN install -m 0755 -d /etc/apt/keyrings
RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
RUN chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
RUN echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
    tee /etc/apt/sources.list.d/docker.list > /dev/null

RUN apt-get update 
RUN apt-get install -y wget docker-ce-cli docker-compose-plugin

RUN pip3 install pymilvus==2.6.3
RUN python3 -c "from pymilvus import Collection"

# Install milvus standalone by docker-compose.yml
RUN wget https://github.com/milvus-io/milvus/releases/download/v2.6.6/milvus-standalone-docker-compose.yml -O docker-compose.yml
RUN sed -i 's|${DOCKER_VOLUME_DIRECTORY:-.}|/tmp|g' docker-compose.yml

# Create custom Milvus configuration with segment size settings
# Using build args to parameterize segment sizes
RUN echo "# Custom Milvus configuration for ANN benchmarks (segment size: ${SEGMENT_MAX_SIZE} MB)" > /tmp/user.yaml && \
    echo "dataCoord:" >> /tmp/user.yaml && \
    echo "  segment:" >> /tmp/user.yaml && \
    echo "    maxSize: ${SEGMENT_MAX_SIZE}" >> /tmp/user.yaml && \
    echo "    diskSegmentMaxSize: ${DISK_SEGMENT_MAX_SIZE}" >> /tmp/user.yaml && \
    echo "    sealProportion: 0.12" >> /tmp/user.yaml

# Add user.yaml volume mount to docker-compose.yml for standalone service
# Find the first volume entry under standalone and add our config mount after it
RUN sed -i '/- \/tmp\/volumes\/milvus:/a\      - /tmp/user.yaml:/milvus/configs/user.yaml' docker-compose.yml

# Copy ablation-specific entry point that supports --segment_size argument
COPY run_algorithm_ablation.py ./

# Override entrypoint to use ablation runner
ENTRYPOINT ["python3", "-u", "run_algorithm_ablation.py"]
