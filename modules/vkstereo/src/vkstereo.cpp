// Vulkan semi-global matching for OpenCV.
// SPDX-License-Identifier: Apache-2.0
#include "opencv2/vkstereo.hpp"

#include <vulkan/vulkan_raii.hpp>

#include <array>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "census_pair_d128.h"
#include "census_pair_d64.h"
#include "consistency_d128.h"
#include "consistency_d64.h"
#include "cost_first_d128.h"
#include "cost_first_d64.h"
#include "cost_reverse_d128.h"
#include "cost_reverse_d64.h"
#include "cost_vertical_d128.h"
#include "cost_vertical_d64.h"
#include "median_d128.h"
#include "median_d64.h"

namespace cv {
namespace vkstereo {
namespace {

constexpr int kMaxWidth = 4096;
constexpr int kCensusCostBound = 48;  // 9x7 classic Census: 48 compared bits

struct ShaderCode {
    const uint32_t* words;
    size_t bytes;
};

// The six dispatches of one frame, in order. Bindings follow each shader's
// declaration order: inputs first, then outputs.
enum Buffer { LEFT, RIGHT, FEATURES, COST, FORWARD, REVERSE, DISPARITY, WINNERS, CONSISTENT, OUTPUT, BUFFER_COUNT };

struct StageSpec {
    ShaderCode code;
    std::vector<Buffer> bindings;
};

#define VKSTEREO_CODE(name) ShaderCode{name, sizeof(name)}

std::array<StageSpec, 6> stagesFor(int disparities) {
    const bool d64 = disparities == 64;
    return {{
        {d64 ? VKSTEREO_CODE(census_pair_d64) : VKSTEREO_CODE(census_pair_d128), {LEFT, RIGHT, FEATURES}},
        {d64 ? VKSTEREO_CODE(cost_first_d64) : VKSTEREO_CODE(cost_first_d128), {FEATURES, FORWARD, COST}},
        {d64 ? VKSTEREO_CODE(cost_reverse_d64) : VKSTEREO_CODE(cost_reverse_d128), {COST, REVERSE}},
        {d64 ? VKSTEREO_CODE(cost_vertical_d64) : VKSTEREO_CODE(cost_vertical_d128),
         {COST, FORWARD, REVERSE, DISPARITY, WINNERS}},
        {d64 ? VKSTEREO_CODE(consistency_d64) : VKSTEREO_CODE(consistency_d128), {DISPARITY, WINNERS, CONSISTENT}},
        {d64 ? VKSTEREO_CODE(median_d64) : VKSTEREO_CODE(median_d128), {CONSISTENT, OUTPUT}},
    }};
}

struct Specialization {
    int32_t width, height, P1, P2;
    float uniqueness;  // best * uniqueness < second, the shader's convention
};

struct DeviceBuffer {
    vk::raii::Buffer buffer = nullptr;
    vk::raii::DeviceMemory memory = nullptr;
    vk::DeviceSize size = 0;
    void* mapped = nullptr;
    bool coherent = true;
};

struct Plan {
    Size size;
    int disparities = 0, P1 = 0, P2 = 0;
    float uniqueness = 0;
    std::array<DeviceBuffer, BUFFER_COUNT> buffers;
    std::vector<vk::raii::DescriptorSetLayout> layouts;
    std::vector<vk::raii::PipelineLayout> pipelineLayouts;
    std::vector<vk::raii::ShaderModule> modules;
    std::vector<vk::raii::Pipeline> pipelines;
    vk::raii::DescriptorPool descriptorPool = nullptr;
    std::vector<vk::raii::DescriptorSet> sets;
    vk::raii::CommandBuffer commands = nullptr;
    vk::raii::Fence fence = nullptr;
};

class StereoSGMImpl : public StereoSGM {
public:
    StereoSGMImpl(int numDisparities, int P1, int P2, float uniqueness)
        : disparities_(numDisparities), P1_(P1), P2_(P2), uniqueness_(uniqueness) {
        validateParameters();
        initDevice();
    }

    void compute(InputArray left, InputArray right, OutputArray disparity) CV_OVERRIDE {
        Mat l = left.getMat(), r = right.getMat();
        CV_Assert(l.type() == CV_8UC1 && r.type() == CV_8UC1);
        CV_Assert(l.size() == r.size());
        CV_Assert(l.cols > 8 && l.rows > 6 && l.cols <= kMaxWidth);
        if (!plan_ || plan_->size != l.size() || plan_->disparities != disparities_ || plan_->P1 != P1_ ||
            plan_->P2 != P2_ || plan_->uniqueness != uniqueness_)
            plan_ = buildPlan(l.size());
        upload(l, plan_->buffers[LEFT]);
        upload(r, plan_->buffers[RIGHT]);
        vk::SubmitInfo submit({}, {}, *plan_->commands);
        queue_.submit(submit, *plan_->fence);
        if (device_.waitForFences(*plan_->fence, true, UINT64_MAX) != vk::Result::eSuccess)
            CV_Error(Error::StsInternal, "Vulkan fence wait failed");
        device_.resetFences(*plan_->fence);
        disparity.create(l.size(), CV_32F);
        Mat out = disparity.getMat();
        DeviceBuffer& result = plan_->buffers[OUTPUT];
        if (!result.coherent) device_.invalidateMappedMemoryRanges(vk::MappedMemoryRange(*result.memory, 0, VK_WHOLE_SIZE));
        Mat(out.size(), CV_32F, result.mapped).copyTo(out);
    }

    int getMinDisparity() const CV_OVERRIDE { return 0; }
    void setMinDisparity(int minDisparity) CV_OVERRIDE { CV_Assert(minDisparity == 0); }
    int getNumDisparities() const CV_OVERRIDE { return disparities_; }
    void setNumDisparities(int numDisparities) CV_OVERRIDE {
        disparities_ = numDisparities;
        validateParameters();
    }
    // The 9x7 Census window is fixed; OpenCV's block size has no equivalent.
    int getBlockSize() const CV_OVERRIDE { return 0; }
    void setBlockSize(int blockSize) CV_OVERRIDE { CV_Assert(blockSize == 0); }
    int getSpeckleWindowSize() const CV_OVERRIDE { return 0; }
    void setSpeckleWindowSize(int speckleWindowSize) CV_OVERRIDE { CV_Assert(speckleWindowSize == 0); }
    int getSpeckleRange() const CV_OVERRIDE { return 0; }
    void setSpeckleRange(int speckleRange) CV_OVERRIDE { CV_Assert(speckleRange == 0); }
    int getDisp12MaxDiff() const CV_OVERRIDE { return 1; }
    void setDisp12MaxDiff(int disp12MaxDiff) CV_OVERRIDE { CV_Assert(disp12MaxDiff == 1); }
    int getP1() const CV_OVERRIDE { return P1_; }
    void setP1(int P1) CV_OVERRIDE {
        P1_ = P1;
        validateParameters();
    }
    int getP2() const CV_OVERRIDE { return P2_; }
    void setP2(int P2) CV_OVERRIDE {
        P2_ = P2;
        validateParameters();
    }
    float getUniqueness() const CV_OVERRIDE { return uniqueness_; }
    void setUniqueness(float uniqueness) CV_OVERRIDE {
        uniqueness_ = uniqueness;
        validateParameters();
    }
    String getDeviceName() const CV_OVERRIDE { return deviceName_; }

private:
    void validateParameters() const {
        CV_Assert(disparities_ == 64 || disparities_ == 128);
        CV_Assert(P1_ >= 0 && P1_ < P2_);
        // Both horizontal path bytes are summed in packed form, so each stays below 128.
        CV_Assert(2 * (P2_ + kCensusCostBound) <= 255);
        CV_Assert(uniqueness_ > 0.0f && uniqueness_ <= 1.0f);
    }

    void initDevice() {
        vk::ApplicationInfo app("opencv_vkstereo", 1, nullptr, 0, VK_API_VERSION_1_1);
        std::vector<const char*> layers;
        if (const char* v = std::getenv("VKSTEREO_VALIDATION"); v && *v == '1')
            layers.push_back("VK_LAYER_KHRONOS_validation");
        instance_ = vk::raii::Instance(context_, vk::InstanceCreateInfo({}, &app, layers));
        vk::raii::PhysicalDevices devices(instance_);
        CV_Assert(!devices.empty());
        size_t index = devices.size();
        if (const char* v = std::getenv("VKSTEREO_DEVICE")) {
            index = static_cast<size_t>(std::atoi(v));
            CV_Assert(index < devices.size());
        } else {
            for (size_t i = 0; i < devices.size(); ++i)
                if (devices[i].getProperties().deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
                    index = i;
                    break;
                }
            if (index == devices.size()) index = 0;
        }
        physical_ = std::move(devices[index]);
        auto props = physical_.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceSubgroupProperties>();
        const auto& subgroup = props.get<vk::PhysicalDeviceSubgroupProperties>();
        CV_Assert(props.get<vk::PhysicalDeviceProperties2>().properties.apiVersion >= VK_API_VERSION_1_1);
        CV_Assert(subgroup.subgroupSize >= 32);
        CV_Assert(subgroup.supportedOperations & vk::SubgroupFeatureFlagBits::eShuffle);
        CV_Assert(subgroup.supportedStages & vk::ShaderStageFlagBits::eCompute);
        deviceName_ = props.get<vk::PhysicalDeviceProperties2>().properties.deviceName.data();
        auto families = physical_.getQueueFamilyProperties();
        queueFamily_ = UINT32_MAX;
        for (uint32_t i = 0; i < families.size(); ++i)
            if (families[i].queueFlags & vk::QueueFlagBits::eCompute) {
                queueFamily_ = i;
                if (!(families[i].queueFlags & vk::QueueFlagBits::eGraphics)) break;
            }
        CV_Assert(queueFamily_ != UINT32_MAX);
        float priority = 1.0f;
        vk::DeviceQueueCreateInfo queueInfo({}, queueFamily_, 1, &priority);
        device_ = vk::raii::Device(physical_, vk::DeviceCreateInfo({}, queueInfo));
        queue_ = device_.getQueue(queueFamily_, 0);
        pool_ = vk::raii::CommandPool(device_, {vk::CommandPoolCreateFlagBits::eResetCommandBuffer, queueFamily_});
        memoryProperties_ = physical_.getMemoryProperties();
    }

    uint32_t memoryType(uint32_t typeBits, vk::MemoryPropertyFlags required, uint32_t fallback = UINT32_MAX) const {
        for (uint32_t i = 0; i < memoryProperties_.memoryTypeCount; ++i)
            if ((typeBits & (1u << i)) && (memoryProperties_.memoryTypes[i].propertyFlags & required) == required)
                return i;
        if (fallback != UINT32_MAX) return fallback;
        CV_Error(Error::StsInternal, "no suitable Vulkan memory type");
    }

    // Host-visible buffers prefer cached memory with explicit flushes: on
    // unified-memory devices the coherent type is uncached, and reading the
    // result through it is slow.
    DeviceBuffer allocate(vk::DeviceSize size, bool hostVisible) {
        DeviceBuffer b;
        b.size = size;
        b.buffer = vk::raii::Buffer(device_, vk::BufferCreateInfo({}, size, vk::BufferUsageFlagBits::eStorageBuffer));
        auto requirements = b.buffer.getMemoryRequirements();
        uint32_t type;
        if (hostVisible) {
            auto cached = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCached;
            auto coherent = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
            type = memoryType(requirements.memoryTypeBits, cached, memoryType(requirements.memoryTypeBits, coherent));
            b.coherent = bool(memoryProperties_.memoryTypes[type].propertyFlags & vk::MemoryPropertyFlagBits::eHostCoherent);
        } else {
            type = memoryType(requirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
        }
        b.memory = vk::raii::DeviceMemory(device_, vk::MemoryAllocateInfo(requirements.size, type));
        b.buffer.bindMemory(*b.memory, 0);
        if (hostVisible) b.mapped = b.memory.mapMemory(0, size);
        return b;
    }

    void upload(const Mat& image, DeviceBuffer& buffer) const {
        // The Census shader reads 32-bit pixels.
        Mat dst(image.size(), CV_32S, buffer.mapped);
        image.convertTo(dst, CV_32S);
        if (!buffer.coherent) device_.flushMappedMemoryRanges(vk::MappedMemoryRange(*buffer.memory, 0, VK_WHOLE_SIZE));
    }

    std::unique_ptr<Plan> buildPlan(Size size) {
        auto plan = std::make_unique<Plan>();
        plan->size = size;
        plan->disparities = disparities_;
        plan->P1 = P1_;
        plan->P2 = P2_;
        plan->uniqueness = uniqueness_;
        const int width = size.width, height = size.height;
        const int tile = 1024 / disparities_;  // scan lines per workgroup, rows per volume tile
        const int tiledRows = (height + tile - 1) / tile * tile;
        const vk::DeviceSize pixels = static_cast<vk::DeviceSize>(width) * height;
        const vk::DeviceSize tiledPixels = static_cast<vk::DeviceSize>(width) * tiledRows;
        plan->buffers[LEFT] = allocate(pixels * 4, true);
        plan->buffers[RIGHT] = allocate(pixels * 4, true);
        plan->buffers[FEATURES] = allocate(tiledPixels * 16, false);
        for (Buffer b : {COST, FORWARD, REVERSE}) plan->buffers[b] = allocate(tiledPixels * disparities_, false);
        for (Buffer b : {DISPARITY, WINNERS, CONSISTENT}) plan->buffers[b] = allocate(pixels * 4, false);
        plan->buffers[OUTPUT] = allocate(pixels * 4, true);

        auto stages = stagesFor(disparities_);
        Specialization spec{width, height, P1_, P2_, 1.0f / uniqueness_};
        std::array<vk::SpecializationMapEntry, 5> entries = {{
            {0, offsetof(Specialization, width), 4},
            {1, offsetof(Specialization, height), 4},
            {2, offsetof(Specialization, P1), 4},
            {3, offsetof(Specialization, P2), 4},
            {4, offsetof(Specialization, uniqueness), 4},
        }};
        vk::SpecializationInfo specialization(entries.size(), entries.data(), sizeof(spec), &spec);
        std::vector<vk::DescriptorPoolSize> poolSizes = {{vk::DescriptorType::eStorageBuffer, 32}};
        plan->descriptorPool = vk::raii::DescriptorPool(
            device_, vk::DescriptorPoolCreateInfo(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 8, poolSizes));
        for (const auto& stage : stages) {
            std::vector<vk::DescriptorSetLayoutBinding> bindings;
            for (uint32_t i = 0; i < stage.bindings.size(); ++i)
                bindings.emplace_back(i, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute);
            plan->layouts.emplace_back(device_, vk::DescriptorSetLayoutCreateInfo({}, bindings));
            plan->pipelineLayouts.emplace_back(device_, vk::PipelineLayoutCreateInfo({}, *plan->layouts.back()));
            plan->modules.emplace_back(device_, vk::ShaderModuleCreateInfo({}, stage.code.bytes, stage.code.words));
            vk::PipelineShaderStageCreateInfo shaderStage({}, vk::ShaderStageFlagBits::eCompute, *plan->modules.back(),
                                                         "main", &specialization);
            plan->pipelines.emplace_back(device_, nullptr,
                                         vk::ComputePipelineCreateInfo({}, shaderStage, *plan->pipelineLayouts.back()));
            vk::raii::DescriptorSets sets(device_, vk::DescriptorSetAllocateInfo(*plan->descriptorPool, *plan->layouts.back()));
            plan->sets.push_back(std::move(sets.front()));
            std::vector<vk::DescriptorBufferInfo> infos;
            for (Buffer b : stage.bindings) infos.emplace_back(*plan->buffers[b].buffer, 0, plan->buffers[b].size);
            std::vector<vk::WriteDescriptorSet> writes;
            for (uint32_t i = 0; i < infos.size(); ++i)
                writes.emplace_back(*plan->sets.back(), i, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &infos[i]);
            device_.updateDescriptorSets(writes, {});
        }

        const uint32_t lines = tile;
        std::array<std::array<uint32_t, 3>, 6> groups = {{
            {uint32_t((width + 128 / tile - 1) / (128 / tile)), uint32_t((height + 4 * tile - 1) / (4 * tile)), 1},
            {uint32_t((height + lines - 1) / lines), 1, 1},
            {uint32_t((height + lines - 1) / lines), 1, 1},
            {uint32_t((width + lines - 1) / lines), 1, 1},
            {uint32_t(height), 1, 1},
            {uint32_t((pixels + 127) / 128), 1, 1},
        }};

        vk::raii::CommandBuffers buffers(device_, vk::CommandBufferAllocateInfo(*pool_, vk::CommandBufferLevel::ePrimary, 1));
        plan->commands = std::move(buffers.front());
        auto& cmd = plan->commands;
        cmd.begin(vk::CommandBufferBeginInfo());
        vk::MemoryBarrier hostToShader(vk::AccessFlagBits::eHostWrite, vk::AccessFlagBits::eShaderRead);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eHost, vk::PipelineStageFlagBits::eComputeShader, {}, hostToShader,
                            {}, {});
        vk::MemoryBarrier shaderToShader(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead);
        for (size_t i = 0; i < stages.size(); ++i) {
            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *plan->pipelines[i]);
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *plan->pipelineLayouts[i], 0, *plan->sets[i], {});
            cmd.dispatch(groups[i][0], groups[i][1], groups[i][2]);
            if (i + 1 < stages.size())
                cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader,
                                    {}, shaderToShader, {}, {});
        }
        vk::MemoryBarrier shaderToHost(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eHostRead);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eHost, {}, shaderToHost,
                            {}, {});
        cmd.end();
        plan->fence = vk::raii::Fence(device_, vk::FenceCreateInfo());
        return plan;
    }

    int disparities_, P1_, P2_;
    float uniqueness_;
    String deviceName_;
    vk::raii::Context context_;
    vk::raii::Instance instance_ = nullptr;
    vk::raii::PhysicalDevice physical_ = nullptr;
    vk::raii::Device device_ = nullptr;
    vk::raii::Queue queue_ = nullptr;
    vk::raii::CommandPool pool_ = nullptr;
    vk::PhysicalDeviceMemoryProperties memoryProperties_;
    uint32_t queueFamily_ = 0;
    std::unique_ptr<Plan> plan_;
};

}  // namespace

Ptr<StereoSGM> StereoSGM::create(int numDisparities, int P1, int P2, float uniqueness) {
    try {
        return makePtr<StereoSGMImpl>(numDisparities, P1, P2, uniqueness);
    } catch (const vk::SystemError& e) {
        CV_Error(Error::StsError, String("Vulkan: ") + e.what());
    }
}

}  // namespace vkstereo
}  // namespace cv
