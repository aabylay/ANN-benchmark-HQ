FROM ann-benchmarks

RUN apt update && apt install -y wget software-properties-common

# Add Ubuntu toolchain PPA for GCC 12 (provides GLIBCXX_3.4.32)
RUN add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
    apt update && \
    apt install -y g++-12 libstdc++6

RUN wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
RUN bash Anaconda3-2023.09-0-Linux-x86_64.sh -b -p /root/anaconda3

ENV PATH /root/anaconda3/bin:$PATH

RUN python3 -m pip install ansicolors==1.1.8 docker==5.0.2
RUN python3 -c 'import sys; print("Path", sys.path)'

# Update conda and install dependencies
RUN conda install -c pytorch -c conda-forge h5py

# Upgrade numpy with pip, since numpy _core module is missing in conda version"
RUN pip install numpy==1.26.1


RUN python -c "import numpy; print('NumPy version:', numpy.__version__)"
RUN python -c "import numpy._core; print('NumPy _core module is present')"

COPY faiss-1.12.0-py3-none-any.whl /root/
RUN ls /root/
RUN pip install /root/faiss-1.12.0-py3-none-any.whl

# Override Anaconda's libstdc++.so.6 with system's (to fix GLIBCXX_3.4.30)
RUN ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /root/anaconda3/lib/libstdc++.so.6

# https://developpaper.com/a-pit-of-mkl-library-in-linux-anaconda/
ENV LD_PRELOAD /root/anaconda3/lib/libmkl_core.so:/root/anaconda3/lib/libmkl_sequential.so 

RUN python3 -c 'import faiss; print(faiss.__version__)'
RUN python3 -c 'import faiss; print(faiss.IndexHNSWFlat)'
