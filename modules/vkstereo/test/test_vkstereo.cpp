// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>

#include <cstdlib>
#include <fstream>
#include <vector>

#include "opencv2/vkstereo.hpp"

namespace {

// Frozen references: 32-bit grayscale inputs and the float disparity maps of
// the same graph, as validated on the Jetson Nano against NVIDIA's driver.
std::string dataDir() {
    if (const char* d = std::getenv("VKSTEREO_TEST_DATA")) return d;
    return std::string(std::getenv("HOME")) + "/.local/share/c100-gpu/baselines/l4t-sgm-20261022/input";
}

template <typename T>
std::vector<T> readRaw(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::vector<T> values(bytes.size() / sizeof(T));
    std::memcpy(values.data(), bytes.data(), values.size() * sizeof(T));
    return values;
}

cv::Mat grayFrom32(const std::vector<int32_t>& pixels, cv::Size size) {
    cv::Mat image(size, CV_8UC1);
    for (int y = 0; y < size.height; ++y)
        for (int x = 0; x < size.width; ++x) image.at<uchar>(y, x) = static_cast<uchar>(pixels[y * size.width + x]);
    return image;
}

void checkAgainstReference(const std::string& prefix, cv::Size size) {
    auto left = readRaw<int32_t>(dataDir() + "/" + prefix + ".left.bin");
    auto right = readRaw<int32_t>(dataDir() + "/" + prefix + ".right.bin");
    auto expected = readRaw<float>(dataDir() + "/" + prefix + ".expected.bin");
    if (left.empty() || right.empty() || expected.empty()) GTEST_SKIP() << "reference data missing in " << dataDir();
    ASSERT_EQ(left.size(), static_cast<size_t>(size.area()));
    ASSERT_EQ(expected.size(), static_cast<size_t>(size.area()));
    auto matcher = cv::vkstereo::StereoSGM::create(64, 8, 64, 0.95f);
    cv::Mat disparity;
    matcher->compute(grayFrom32(left, size), grayFrom32(right, size), disparity);
    ASSERT_EQ(disparity.type(), CV_32F);
    ASSERT_EQ(disparity.size(), size);
    double worst = 0;
    int rejectionMismatches = 0;
    for (int i = 0; i < size.area(); ++i) {
        float actual = disparity.ptr<float>()[i], reference = expected[i];
        if ((actual < 0) != (reference < 0)) ++rejectionMismatches;
        worst = std::max(worst, static_cast<double>(std::abs(actual - reference)));
    }
    EXPECT_EQ(rejectionMismatches, 0);
    EXPECT_LT(worst, 1e-4) << "max abs difference " << worst;
}

}  // namespace

TEST(VkStereo, MatchesVgaReference) { checkAgainstReference("vga", {640, 480}); }
TEST(VkStereo, Matches720pReference) { checkAgainstReference("720", {1280, 720}); }

TEST(VkStereo, ReusesResourcesAcrossFrames) {
    auto left = readRaw<int32_t>(dataDir() + "/vga.left.bin");
    auto right = readRaw<int32_t>(dataDir() + "/vga.right.bin");
    if (left.empty() || right.empty()) GTEST_SKIP();
    auto matcher = cv::vkstereo::StereoSGM::create();
    cv::Mat l = grayFrom32(left, {640, 480}), r = grayFrom32(right, {640, 480}), first, second;
    matcher->compute(l, r, first);
    matcher->compute(l, r, second);
    EXPECT_EQ(cv::countNonZero(first != second), 0);
}

TEST(VkStereo, SupportsOneHundredTwentyEightDisparities) {
    cv::Mat left(100, 200, CV_8UC1), right(100, 200, CV_8UC1);
    cv::randu(left, 0, 255);
    cv::randu(right, 0, 255);
    auto matcher = cv::vkstereo::StereoSGM::create(128);
    cv::Mat disparity;
    matcher->compute(left, right, disparity);
    ASSERT_EQ(disparity.size(), left.size());
    double lo, hi;
    cv::minMaxLoc(disparity, &lo, &hi);
    EXPECT_GE(lo, -1.0);
    EXPECT_LT(hi, 128.0);
}

TEST(VkStereo, RejectsUnsupportedParameters) {
    EXPECT_THROW(cv::vkstereo::StereoSGM::create(96), cv::Exception);
    EXPECT_THROW(cv::vkstereo::StereoSGM::create(64, 8, 120), cv::Exception);
    auto matcher = cv::vkstereo::StereoSGM::create();
    EXPECT_THROW(matcher->setMinDisparity(5), cv::Exception);
    EXPECT_FALSE(matcher->getDeviceName().empty());
}
