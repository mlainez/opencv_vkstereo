#!/usr/bin/env bash
# Cross-builds a static vkstereo_bench for the OKdo Nano C100 running the
# Nerves system, using that system's Buildroot toolchain. The result needs
# only the board's Vulkan loader and libc.
#
#   NERVES_SYSTEM_ARTIFACT=.../nerves_system_okdo_c100-portable-0.1.0 \
#   OPENCV_SOURCE=/path/to/opencv-4.10.0 scripts/build-jetson.sh [build-dir]
set -euo pipefail
artifact=${NERVES_SYSTEM_ARTIFACT:?Nerves system artifact directory required}
source_dir=${OPENCV_SOURCE:?OpenCV source directory required}
out=${1:-$PWD/build-jetson}
toolchain=$artifact/host/share/buildroot/toolchainfile.cmake
project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$out"

# Minimal static OpenCV: core, imgproc and calib3d (which pulls features2d and flann).
cmake -S "$source_dir" -B "$out/opencv" -G Ninja -DCMAKE_TOOLCHAIN_FILE="$toolchain" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$out/opencv-install" -DBUILD_SHARED_LIBS=OFF \
  -DBUILD_LIST=core,imgproc,calib3d -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF \
  -DBUILD_opencv_apps=OFF -DBUILD_opencv_python3=OFF -DBUILD_JAVA=OFF -DBUILD_DOCS=OFF \
  -DWITH_OPENCL=OFF -DWITH_CUDA=OFF -DWITH_EIGEN=OFF -DWITH_IPP=OFF -DWITH_ADE=OFF -DWITH_OPENEXR=OFF \
  -DWITH_JPEG=OFF -DWITH_PNG=OFF -DWITH_TIFF=OFF -DWITH_WEBP=OFF -DWITH_OPENJPEG=OFF -DWITH_JASPER=OFF \
  -DWITH_FFMPEG=OFF -DWITH_GTK=OFF -DWITH_V4L=OFF -DWITH_GSTREAMER=OFF -DWITH_PROTOBUF=OFF -DWITH_ITT=OFF \
  -DWITH_LAPACK=OFF -DWITH_OPENMP=OFF -DWITH_TBB=OFF -DBUILD_ZLIB=ON -DOPENCV_GENERATE_PKGCONFIG=OFF \
  -DCPU_BASELINE=NEON
ninja -C "$out/opencv" install

cmake -S "$project" -B "$out/vkstereo" -G Ninja -DCMAKE_TOOLCHAIN_FILE="$toolchain" \
  -DOpenCV_DIR="$out/opencv-install/lib/cmake/opencv4" -DVKSTEREO_WITH_IMGCODECS=OFF \
  -DVKSTEREO_BUILD_TESTS=OFF -DCMAKE_EXE_LINKER_FLAGS="-static-libstdc++ -static-libgcc"
ninja -C "$out/vkstereo"
echo "built $out/vkstereo/modules/vkstereo/vkstereo_bench"
