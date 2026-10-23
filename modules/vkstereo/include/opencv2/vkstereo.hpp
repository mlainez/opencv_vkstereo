// Vulkan semi-global matching for OpenCV.
// SPDX-License-Identifier: Apache-2.0
#ifndef OPENCV_VKSTEREO_HPP
#define OPENCV_VKSTEREO_HPP

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

/** @defgroup vkstereo Semi-global matching on Vulkan compute

Stereo matching with the semantics of Fixstars libSGM, as in
cv::cuda::StereoSGM, running on any Vulkan 1.1 device with subgroup shuffles:
9x7 Census features, three aggregation paths (left to right, right to left
and top to bottom), penalties P1/P2, a uniqueness ratio, float subpixel
refinement, a projected left/right consistency check and a 3x3 median.
*/
namespace cv {
namespace vkstereo {

/** @brief Semi-global matcher executing on a Vulkan device.

The matcher keeps its Vulkan resources and reuses them for every frame of the
same size. Input images are 8-bit grayscale; the output is a CV_32F disparity
map in pixels, with -1 where the match was rejected. Widths up to 4096 pixels
and 64 or 128 disparities are supported; the minimum disparity is 0.

The device is the first discrete GPU unless `VKSTEREO_DEVICE` names another
physical device index; `VKSTEREO_VALIDATION=1` enables the validation layer.
*/
class CV_EXPORTS_W StereoSGM : public cv::StereoMatcher {
public:
    /** @param numDisparities 64 or 128
        @param P1 penalty for a disparity change of one
        @param P2 penalty for larger changes; P2 + 48 must stay below 128
        @param uniqueness ratio in (0, 1]: the second-best cost outside the
        winner's neighbourhood must exceed best / uniqueness, as in libSGM */
    CV_WRAP static Ptr<StereoSGM> create(int numDisparities = 64, int P1 = 8, int P2 = 64,
                                         float uniqueness = 0.95f);

    CV_WRAP virtual int getP1() const = 0;
    CV_WRAP virtual void setP1(int P1) = 0;
    CV_WRAP virtual int getP2() const = 0;
    CV_WRAP virtual void setP2(int P2) = 0;
    CV_WRAP virtual float getUniqueness() const = 0;
    CV_WRAP virtual void setUniqueness(float uniqueness) = 0;

    /** @brief Name of the Vulkan physical device in use. */
    CV_WRAP virtual String getDeviceName() const = 0;
};

}  // namespace vkstereo
}  // namespace cv

#endif
