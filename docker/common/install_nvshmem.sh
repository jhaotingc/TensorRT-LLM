git clone https://github.com/NVIDIA/gdrcopy.git
cd gdrcopy
git checkout v2.4.4

apt update
apt install -y nvidia-dkms-535
apt install -y build-essential devscripts debhelper fakeroot pkg-config dkms
apt install -y check libsubunit0 libsubunit-dev

cd packages
CUDA=/usr/local/cuda ./build-deb-packages.sh
dpkg -i gdrdrv-dkms_*.deb
dpkg -i libgdrapi_*.deb
dpkg -i gdrcopy-tests_*.deb
dpkg -i gdrcopy_*.deb

GDRCOPY_HOME=/usr/src/gdrdrv-2.4.4/

ARCH=$(uname -m)
# IBGDA dependency
ln -s /usr/lib/${ARCH}-linux-gnu/libmlx5.so.1 /usr/lib/${ARCH}-linux-gnu/libmlx5.so
apt-get install -y libfabric-dev

cd ../..
wget https://developer.download.nvidia.com/compute/redist/nvshmem/3.2.5/source/nvshmem_src_3.2.5-1.txz
tar -xf nvshmem_src_3.2.5-1.txz \
    && mv nvshmem_src nvshmem
cd nvshmem
git apply ../nvshmem.patch
cd ..
mv nvshmem /usr/local/nvshmem
cd /usr/local/nvshmem
# WORKDIR /sgl-workspace/nvshmem
export CUDA_HOME=/usr/local/cuda

NVSHMEM_SHMEM_SUPPORT=0 \
NVSHMEM_UCX_SUPPORT=0 \
NVSHMEM_USE_NCCL=0 \
NVSHMEM_MPI_SUPPORT=0 \
NVSHMEM_IBGDA_SUPPORT=1 \
NVSHMEM_PMIX_SUPPORT=0 \
NVSHMEM_TIMEOUT_DEVICE_POLLING=0 \
NVSHMEM_USE_GDRCOPY=1 \
cmake -S . -B build/ -DCMAKE_INSTALL_PREFIX=/usr/local/nvshmem/install -DCMAKE_CUDA_ARCHITECTURES=90 \
&& cd build \
&& make install -j