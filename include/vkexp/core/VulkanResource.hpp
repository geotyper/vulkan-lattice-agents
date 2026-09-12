#pragma once

#include <vulkan/vulkan.h>

#include <cstddef>
#include <string_view>
#include <utility>

namespace vkexp {

template <typename Handle, auto Destroy> class UniqueDeviceHandle {
public:
    UniqueDeviceHandle() = default;
    UniqueDeviceHandle(const VkDevice device, const Handle handle)
        : device_(device), handle_(handle) {}
    ~UniqueDeviceHandle() { reset(); }

    UniqueDeviceHandle(const UniqueDeviceHandle&) = delete;
    UniqueDeviceHandle& operator=(const UniqueDeviceHandle&) = delete;

    UniqueDeviceHandle(UniqueDeviceHandle&& other) noexcept
        : device_(std::exchange(other.device_, VK_NULL_HANDLE)),
          handle_(std::exchange(other.handle_, VK_NULL_HANDLE)) {}

    UniqueDeviceHandle& operator=(UniqueDeviceHandle&& other) noexcept {
        if (this != &other) {
            reset();
            device_ = std::exchange(other.device_, VK_NULL_HANDLE);
            handle_ = std::exchange(other.handle_, VK_NULL_HANDLE);
        }
        return *this;
    }

    void reset(const VkDevice device = VK_NULL_HANDLE, const Handle handle = VK_NULL_HANDLE) {
        if (handle_ != VK_NULL_HANDLE) {
            Destroy(device_, handle_, nullptr);
        }
        device_ = device;
        handle_ = handle;
    }

    [[nodiscard]] Handle* put(const VkDevice device) {
        reset();
        device_ = device;
        return &handle_;
    }

    [[nodiscard]] Handle get() const { return handle_; }
    [[nodiscard]] operator Handle() const { return handle_; }
    [[nodiscard]] explicit operator bool() const { return handle_ != VK_NULL_HANDLE; }

private:
    VkDevice device_{};
    Handle handle_{};
};

using UniquePipeline = UniqueDeviceHandle<VkPipeline, vkDestroyPipeline>;
using UniquePipelineLayout = UniqueDeviceHandle<VkPipelineLayout, vkDestroyPipelineLayout>;
using UniqueShaderModule = UniqueDeviceHandle<VkShaderModule, vkDestroyShaderModule>;
using UniqueDescriptorSetLayout =
    UniqueDeviceHandle<VkDescriptorSetLayout, vkDestroyDescriptorSetLayout>;
using UniqueDescriptorPool = UniqueDeviceHandle<VkDescriptorPool, vkDestroyDescriptorPool>;
using UniqueDeviceMemory = UniqueDeviceHandle<VkDeviceMemory, vkFreeMemory>;
using UniqueBuffer = UniqueDeviceHandle<VkBuffer, vkDestroyBuffer>;
using UniqueImage = UniqueDeviceHandle<VkImage, vkDestroyImage>;
using UniqueImageView = UniqueDeviceHandle<VkImageView, vkDestroyImageView>;
using UniqueSampler = UniqueDeviceHandle<VkSampler, vkDestroySampler>;
using UniqueCommandPool = UniqueDeviceHandle<VkCommandPool, vkDestroyCommandPool>;
using UniqueFence = UniqueDeviceHandle<VkFence, vkDestroyFence>;

struct BufferResourceConfig {
    VkDeviceSize size{};
    VkBufferUsageFlags usage{};
    VkMemoryPropertyFlags memoryProperties{VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT};
};

class BufferResource {
public:
    BufferResource() = default;
    ~BufferResource() = default;

    BufferResource(const BufferResource&) = delete;
    BufferResource& operator=(const BufferResource&) = delete;
    BufferResource(BufferResource&&) noexcept = default;
    BufferResource& operator=(BufferResource&&) noexcept = default;

    void create(VkPhysicalDevice physicalDevice, VkDevice device,
                const BufferResourceConfig& config);
    void reset();
    void write(const void* data, VkDeviceSize size, VkDeviceSize offset = 0) const;
    void read(void* data, VkDeviceSize size, VkDeviceSize offset = 0) const;

    [[nodiscard]] VkBuffer buffer() const { return buffer_.get(); }
    [[nodiscard]] VkDeviceMemory memory() const { return memory_.get(); }
    [[nodiscard]] VkDeviceSize size() const { return size_; }
    [[nodiscard]] VkBufferUsageFlags usage() const { return usage_; }
    [[nodiscard]] VkMemoryPropertyFlags memoryProperties() const { return memoryProperties_; }
    [[nodiscard]] explicit operator bool() const { return static_cast<bool>(buffer_); }

private:
    void validateHostAccess(VkDeviceSize size, VkDeviceSize offset) const;

    VkDevice device_{};
    UniqueDeviceMemory memory_;
    UniqueBuffer buffer_;
    VkDeviceSize size_{};
    VkBufferUsageFlags usage_{};
    VkMemoryPropertyFlags memoryProperties_{};
};

struct ImageResourceConfig {
    VkExtent2D extent{};
    VkFormat format{VK_FORMAT_UNDEFINED};
    VkImageUsageFlags usage{};
    VkMemoryPropertyFlags memoryProperties{VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT};
    VkFilter filter{VK_FILTER_LINEAR};
    // Which aspect the view names. Colour for everything the compute path makes;
    // a depth attachment is the one case where the view has to say so, and a
    // view created with the wrong aspect is rejected at attachment time rather
    // than at creation, which is a long way from the mistake.
    VkImageAspectFlags aspect{VK_IMAGE_ASPECT_COLOR_BIT};
};

class ImageResource {
public:
    ImageResource() = default;
    ~ImageResource() = default;

    ImageResource(const ImageResource&) = delete;
    ImageResource& operator=(const ImageResource&) = delete;
    ImageResource(ImageResource&&) noexcept = default;
    ImageResource& operator=(ImageResource&&) noexcept = default;

    void create(VkPhysicalDevice physicalDevice, VkDevice device,
                const ImageResourceConfig& config);
    void reset();

    [[nodiscard]] VkImage image() const { return image_.get(); }
    [[nodiscard]] VkImageView view() const { return view_.get(); }
    [[nodiscard]] VkSampler sampler() const { return sampler_.get(); }
    [[nodiscard]] VkExtent2D extent() const { return extent_; }
    [[nodiscard]] VkFormat format() const { return format_; }
    [[nodiscard]] VkImageUsageFlags usage() const { return usage_; }
    [[nodiscard]] explicit operator bool() const { return static_cast<bool>(image_); }

private:
    UniqueDeviceMemory memory_;
    UniqueImage image_;
    UniqueImageView view_;
    UniqueSampler sampler_;
    VkExtent2D extent_{};
    VkFormat format_{VK_FORMAT_UNDEFINED};
    VkImageUsageFlags usage_{};
};

[[nodiscard]] UniqueShaderModule loadShaderModule(VkDevice device, std::string_view path);

} // namespace vkexp
