FROM dustynv/pytorch:2.1-r36.2.0
RUN apt-get update
RUN apt-get install  libasound2-dev alsa-base alsa-utils libsndfile1 vim libportaudio2 ccache ffmpeg -y
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
WORKDIR /root
ENV PATH="/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
RUN mkdir -p /usr/local/app
WORKDIR /usr/local/app
RUN pip install --ignore-installed TTS
RUN wget https://nvidia.box.com/shared/static/0h6tk4msrl9xz3evft9t0mpwwwkw7a32.whl -O torch-2.1.0-cp310-cp310-linux_aarch64.whl
RUN pip install torch-2.1.0-cp310-cp310-linux_aarch64.whl --force-reinstall
ENV TORCHAUDIO_VERSION 'v2.1.0'
ENV TORCH_CUDA_ARCH_LIST "5.3;6.2;7.2;8.7"
RUN git clone --branch ${TORCHAUDIO_VERSION} --recursive --depth=1 https://github.com/pytorch/audio torchaudio && \
    cd torchaudio && \
    git checkout ${TORCHAUDIO_VERSION} && \
    sed -i 's#  URL https://zlib.net/zlib-1.2.11.tar.gz#  URL https://github.com/madler/zlib/archive/refs/tags/v1.2.12.tar.gz#g' third_party/zlib/CMakeLists.txt || echo "failed to patch torchaudio/third_party/zlib/CMakeLists.txt" && \
    sed -i 's#  URL_HASH SHA256=c3e5e9fdd5004dcb542feda5ee4f0ff0744628baf8ed2dd5d66f8ca1197cb1a1#  URL_HASH SHA256=d8688496ea40fb61787500e863cc63c9afcbc524468cedeb478068924eb54932#g' third_party/zlib/CMakeLists.txt || echo "failed to patch torchaudio/third_party/zlib/CMakeLists.txt" && \
    BUILD_SOX=1 python3 setup.py bdist_wheel && \
    cp dist/torchaudio*.whl /opt && \
    pip3 install --no-cache-dir --verbose /opt/torchaudio*.whl && \
    cd ../ && \
    rm -rf torchaudio
ENV LLAMA_CUBLAS=1
ENV CUDA_ARCHITECTURES=87
RUN apt-get install libspeexdsp-dev portaudio19-dev -y
RUN pip install https://github.com/dscripka/openWakeWord/releases/download/v0.1.1/speexdsp_ns-0.1.2-cp310-cp310-linux_aarch64.whl
ENV LLAMA_CPP_PYTHON_REPO=abetlen/llama-cpp-python
ENV LLAMA_CPP_PYTHON_BRANCH=main

ADD https://api.github.com/repos/${LLAMA_CPP_PYTHON_REPO}/git/refs/heads/${LLAMA_CPP_PYTHON_BRANCH} /tmp/llama_cpp_python_version.json
RUN git clone --branch=${LLAMA_CPP_PYTHON_BRANCH} --recursive https://github.com/${LLAMA_CPP_PYTHON_REPO}
RUN cd llama-cpp-python && git reset a05b4da80a67f6079e54cac4cd769cf877022639 --hard
# RUN { cd /usr/local/app/llama-cpp-python/vendor/llama.cpp; patch -p1 < /usr/local/app/llama_cpp_path.patch; }
RUN ln -s llama-cpp-python/vendor/llama.cpp llama.cpp
ENV CUDA_HOME="/usr/local/cuda"
ENV LD_LIBRARY_PATH="/usr/local/cuda/compat:/usr/local/cuda/lib64:"
# build C++ libraries
RUN cd llama-cpp-python/vendor/llama.cpp && \
    git reset 0e18b2e7d0b5c0a509ea40098def234b8d4a938a --hard && \
    mkdir build && \
    cd build && \
    cmake .. -DLLAMA_CUBLAS=on -DLLAMA_CUDA_F16=1 -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} && \
    cmake --build . --config Release --parallel 4
    
RUN ln -s build/bin llama.cpp/bin

# # apply patches
# #RUN cd llama-cpp-python/vendor/llama.cpp && \
# #    git apply /opt/llama.cpp/patches.diff && \
# #    git diff
    
# # build Python bindings
RUN cd llama-cpp-python && \
    CMAKE_ARGS="-DLLAMA_CUBLAS=on -DLLAMA_CUDA_F16=1 -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}" FORCE_CMAKE=1 \
    pip3 wheel -w dist --verbose . 
 
# # install the wheel
RUN cp llama-cpp-python/dist/llama_cpp_python*.whl /opt && \
    pip3 install --no-cache-dir --force-reinstall --verbose /opt/llama_cpp_python*.whl

COPY . .

RUN pip install -r requirements.txt
CMD ["/usr/bin/python3", "/usr/local/app/start-voice-assistant.py"]

