// Compares cv::StereoSGBM on the CPU with cv::vkstereo::StereoSGM on the GPU.
// SPDX-License-Identifier: Apache-2.0
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "opencv2/vkstereo.hpp"

namespace {

const char* kUsage =
    "vkstereo_bench <left> <right> [--gt file.bin] [--frames N] [--width W --height H] [--disparities 64|128]\n"
    "  Images are read with cv::imread as grayscale, or as raw 32-bit pixels when --width/--height are given.\n"
    "  --gt names a raw float32 ground-truth disparity map (Sintel style) for the bad-pixel rates.\n";

cv::Mat readImage(const std::string& path, int width, int height) {
    if (width > 0) {
        std::ifstream in(path, std::ios::binary);
        std::vector<int32_t> pixels(static_cast<size_t>(width) * height);
        in.read(reinterpret_cast<char*>(pixels.data()), pixels.size() * sizeof(int32_t));
        cv::Mat image(height, width, CV_8UC1);
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x) image.at<uchar>(y, x) = static_cast<uchar>(pixels[y * width + x]);
        return image;
    }
    cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (image.empty()) throw std::runtime_error("cannot read " + path);
    return image;
}

template <typename F>
double framesPerSecond(F&& run, int frames) {
    for (int i = 0; i < 3; ++i) run();
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < frames; ++i) run();
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return frames / seconds;
}

// Fraction of ground-truth pixels (x >= disparities, where every candidate is
// in frame) that are rejected or more than one pixel off.
double badOrRejected(const cv::Mat& disparity, const cv::Mat& truth, int disparities) {
    int counted = 0, bad = 0;
    for (int y = 0; y < truth.rows; ++y)
        for (int x = disparities; x < truth.cols; ++x) {
            float t = truth.at<float>(y, x);
            if (!(t >= 0)) continue;
            ++counted;
            float d = disparity.at<float>(y, x);
            if (d < 0 || std::abs(d - t) > 1.0f) ++bad;
        }
    return counted ? 100.0 * bad / counted : 0.0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << kUsage;
        return 2;
    }
    std::string gtPath;
    int frames = 100, width = 0, height = 0, disparities = 64;
    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--gt") gtPath = next();
        else if (a == "--frames") frames = std::stoi(next());
        else if (a == "--width") width = std::stoi(next());
        else if (a == "--height") height = std::stoi(next());
        else if (a == "--disparities") disparities = std::stoi(next());
        else {
            std::cerr << kUsage;
            return 2;
        }
    }
    cv::Mat left = readImage(argv[1], width, height), right = readImage(argv[2], width, height);
    std::cout << "image " << left.cols << "x" << left.rows << ", " << disparities << " disparities, " << frames
              << " timed frames after 3 warmups\n";

    // OpenCV's three-way SGBM with its usual penalties for a 3x3 block.
    auto cpu = cv::StereoSGBM::create(0, disparities, 3, 8 * 9, 32 * 9, 1, 31, 10, 0, 0, cv::StereoSGBM::MODE_SGBM_3WAY);
    cv::Mat cpuDisparity16, cpuDisparity;
    double cpuFps = framesPerSecond([&] { cpu->compute(left, right, cpuDisparity16); }, frames);
    cpuDisparity16.convertTo(cpuDisparity, CV_32F, 1.0 / 16.0);
    cpuDisparity.setTo(-1.0f, cpuDisparity16 < 0);

    auto gpu = cv::vkstereo::StereoSGM::create(disparities, 8, 64, 0.95f);
    cv::Mat gpuDisparity;
    double gpuFps = framesPerSecond([&] { gpu->compute(left, right, gpuDisparity); }, frames);

    std::cout << "cv::StereoSGBM MODE_SGBM_3WAY (" << cv::getNumThreads() << " threads): " << cpuFps << " FPS\n";
    std::cout << "cv::vkstereo::StereoSGM on " << gpu->getDeviceName() << ": " << gpuFps << " FPS\n";
    if (!gtPath.empty()) {
        std::ifstream in(gtPath, std::ios::binary);
        cv::Mat truth(left.size(), CV_32F);
        in.read(reinterpret_cast<char*>(truth.data), truth.total() * sizeof(float));
        std::cout << "bad-or-rejected (>1 px, x >= " << disparities << "): CPU " << badOrRejected(cpuDisparity, truth, disparities)
                  << " %, Vulkan " << badOrRejected(gpuDisparity, truth, disparities) << " %\n";
    }
    return 0;
}
