# opencv_vkstereo

Semi-global stereo matching on Vulkan compute, packaged as an OpenCV module:
`cv::vkstereo::StereoSGM`, a `cv::StereoMatcher` with the semantics of
Fixstars libSGM (as `cv::cuda::StereoSGM` has in opencv_contrib), running on
any Vulkan 1.1 device with subgroup shuffles instead of CUDA.

The kernels are the Vulkan port of libSGM's CUDA kernels developed for the
OKdo Nano C100 (Jetson Nano) under Nerves, where they run faster than NVIDIA's
own driver executes the same graph (49.6 FPS at 640x480, 17.3 FPS at 1280x720
with 64 disparities). This module makes that matcher usable from plain C++ and
OpenCV, outside Elixir.

## What it computes

9x7 Census features (48 bits), raw matching costs stored as bytes, three
aggregation paths (left to right, right to left, top to bottom) with penalties
P1 and P2, winner selection with a uniqueness ratio and float subpixel
refinement, OpenCV's projected left/right consistency check, and a 3x3 median.
Inputs are 8-bit grayscale images of equal size; the output is a CV_32F
disparity map in pixels, with -1 where the match was rejected. Supported: 64 or
128 disparities, minimum disparity 0, widths up to 4096 pixels, P2 + 48 < 128.

## Build

```sh
sudo apt install libopencv-dev libvulkan-dev glslang-tools libgtest-dev cmake ninja-build
cmake -S . -B build -G Ninja
ninja -C build
./build/modules/vkstereo/test_vkstereo
```

Shaders are compiled at build time into the library; image size and penalties
are specialization constants, so no shader compiler is needed at run time. The
device is the first discrete GPU; `VKSTEREO_DEVICE=<index>` selects another
physical device, `VKSTEREO_VALIDATION=1` enables the Khronos validation layer. The library is
built static by default; pass `-DBUILD_SHARED_LIBS=ON` for a shared one.

## Use

```cpp
#include <opencv2/vkstereo.hpp>

auto matcher = cv::vkstereo::StereoSGM::create(64, /*P1*/ 8, /*P2*/ 64, /*uniqueness*/ 0.95f);
cv::Mat disparity;  // CV_32F, -1 where rejected
matcher->compute(leftGray, rightGray, disparity);
```

The matcher keeps its Vulkan buffers, pipelines and command buffer for the
current image size and parameters; `compute` only uploads the two images,
submits one command buffer and reads the result back.

## Benchmark

`vkstereo_bench` runs `cv::StereoSGBM` (MODE_SGBM_3WAY, block 3, P1 72, P2 288)
and this matcher on the same pair:

```sh
./build/modules/vkstereo/vkstereo_bench left.png right.png --frames 200 [--gt truth.bin]
./build/modules/vkstereo/vkstereo_bench left.bin right.bin --width 640 --height 480
```

Results, 2026-10-23, 64 disparities, synchronous single-frame calls with
8-bit input and float output on the CPU side of each call:

| Image | Host: cv::StereoSGBM 3WAY, 16 threads | Host: vkstereo, RX 7800 XT (RADV) | Jetson Nano: cv::StereoSGBM 3WAY, 4 threads | Jetson Nano: vkstereo, GM20B (NVK) |
|---|---:|---:|---:|---:|
| 320x136 Sintel | 892 FPS | 1094 FPS | | |
| 640x480 | 148 FPS | 1094 FPS | 15.0 FPS | 42.8 FPS |
| 1280x720 | 48 FPS | (not re-measured) | 3.4 FPS | 14.8 FPS |

On the Sintel frame both matchers reject or misplace by more than one pixel
about 14.5% of the ground-truth pixels in the region where every candidate is
in frame (14.33% Vulkan, 14.57% CPU). The two matchers use different costs
(Census against OpenCV's Birchfield-Tomasi block cost) and penalties, so the
rates are comparable only as a sanity check.

On the Jetson the same shaders reach 49.6 / 17.3 FPS when the graph runs with
resident inputs and no readback; the difference is the per-call CPU work of a
Mat interface on four Cortex-A57 cores: the 8-bit to 32-bit conversion, cache
maintenance on the shared memory and the copy of the result. Host-visible
buffers use cached memory with explicit flush and invalidate; the coherent
type is uncached on both tested devices and made readback several times
slower.

## Jetson build

`scripts/build-jetson.sh` cross-compiles a static `vkstereo_bench` for the
OKdo Nano C100's Nerves system with that system's Buildroot toolchain, after
building a minimal static OpenCV (core, imgproc, calib3d) for it. The binary
needs only the board's Vulkan loader and libc, so nothing in the firmware
changes; raw 32-bit inputs are passed with `--width` and `--height`.

## Tests

`test_vkstereo` compares the output with frozen reference maps produced by the
same graph on the Jetson Nano and validated there against NVIDIA's driver
(`vga.*.bin`, `720.*.bin` under `$VKSTEREO_TEST_DATA`, default
`~/.local/share/c100-gpu/baselines/l4t-sgm-20261022/input`). Differences must
stay below 1e-4 with identical rejections; the tests are skipped when the data
is absent.

## Layout and provenance

`modules/vkstereo` follows the opencv_contrib module layout (`include/opencv2`,
`src`, `test`, `samples`) so it can be built standalone, as here, or dropped
into an opencv_contrib checkout later. The shaders in `modules/vkstereo/shaders`
are the ones validated in the Nerves/NxIREE pipeline (`nx_iree/bench/stereo/census`,
`VIDEO_FUSION=cost`), with size and penalties turned into specialization
constants; `cost_dp.glsl` and `cost_select.glsl` adapt libSGM (Apache-2.0,
Fixstars), `consistency_projected.comp` adapts OpenCV's SGBM3Way projection
(BSD-3-Clause). See `LICENSE`.
