#include "vkexp/core/VulkanResource.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace vkexp {
namespace {

std::uint32_t findMemoryType(const VkPhysicalDevice physicalDevice, const std::uint32_t typeFilter,
                             const VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memoryProperties{};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);
    for (std::uint32_t index = 0; index < memoryProperties.memoryTypeCount; ++index) {
        const bool supported = (typeFilter & (1U << index)) != 0U;
        const bool matches =
            (memoryProperties.memoryTypes[index].propertyFlags & properties) == properties;
        if (supported && matches) {
            return index;
        }
    }
    throw std::runtime_error("Unable to find a suitable Vulkan memory type");
}

} // namespace

void BufferResource::create(const VkPhysicalDevice physicalDevice, const VkDevice device,
                            const BufferResourceConfig& config) {
    reset();
    if (device == VK_NULL_HANDLE || physicalDevice == VK_NULL_HANDLE || config.size == 0 ||
        config.usage == 0) {
        throw std::invalid_argument("Invalid Vulkan buffer configuration");
    }

    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = config.size;
    bufferInfo.usage = config.usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer buffer{};
    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        throw std::runtime_error("Unable to create Vulkan buffer");
    }
    buffer_.reset(device, buffer);

    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device, buffer_.get(), &requirements);
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex =
        findMemoryType(physicalDevice, requirements.memoryTypeBits, config.memoryProperties);
    VkDeviceMemory memory{};
    if (vkAllocateMemory(device, &allocation, nullptr, &memory) != VK_SUCCESS) {
        throw std::runtime_error("Unable to allocate Vulkan buffer memory");
    }
    memory_.reset(device, memory);
    if (vkBindBufferMemory(device, buffer_.get(), memory_.get(), 0) != VK_SUCCESS) {
        throw std::runtime_error("Unable to bind Vulkan buffer memory");
    }

    device_ = device;
    size_ = config.size;
    usage_ = config.usage;
    memoryProperties_ = config.memoryProperties;
}

void BufferResource::reset() {
    buffer_.reset();
    memory_.reset();
    device_ = VK_NULL_HANDLE;
    size_ = 0;
    usage_ = 0;
    memoryProperties_ = 0;
}

void BufferResource::validateHostAccess(const VkDeviceSize size, const VkDeviceSize offset) const {
    constexpr VkMemoryPropertyFlags required =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if (!buffer_ || (memoryProperties_ & required) != required) {
        throw std::logic_error("Buffer is not host-visible coherent memory");
    }
    if (size == 0 || offset > size_ || size > size_ - offset) {
        throw std::out_of_range("Buffer host access is outside the allocation");
    }
}

void BufferResource::write(const void* data, const VkDeviceSize size,
                           const VkDeviceSize offset) const {
    validateHostAccess(size, offset);
    if (data == nullptr) {
        throw std::invalid_argument("Cannot write null data to a Vulkan buffer");
    }
    void* mapped{};
    if (vkMapMemory(device_, memory_.get(), offset, size, 0, &mapped) != VK_SUCCESS) {
        throw std::runtime_error("Unable to map Vulkan buffer memory");
    }
    std::memcpy(mapped, data, static_cast<std::size_t>(size));
    vkUnmapMemory(device_, memory_.get());
}

void BufferResource::read(void* data, const VkDeviceSize size, const VkDeviceSize offset) const {
    validateHostAccess(size, offset);
    if (data == nullptr) {
        throw std::invalid_argument("Cannot read Vulkan buffer data into null memory");
    }
    void* mapped{};
    if (vkMapMemory(device_, memory_.get(), offset, size, 0, &mapped) != VK_SUCCESS) {
        throw std::runtime_error("Unable to map Vulkan buffer memory");
    }
    std::memcpy(data, mapped, static_cast<std::size_t>(size));
    vkUnmapMemory(device_, memory_.get());
}

void ImageResource::create(const VkPhysicalDevice physicalDevice, const VkDevice device,
                           const ImageResourceConfig& config) {
    reset();

    VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = config.format;
    imageInfo.extent = {config.extent.width, config.extent.height, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = config.usage;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImage image{};
    if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
        throw std::runtime_error("Unable to create Vulkan image");
    }
    image_.reset(device, image);

    VkMemoryRequirements requirements{};
    vkGetImageMemoryRequirements(device, image_.get(), &requirements);
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex =
        findMemoryType(physicalDevice, requirements.memoryTypeBits, config.memoryProperties);
    VkDeviceMemory memory{};
    if (vkAllocateMemory(device, &allocation, nullptr, &memory) != VK_SUCCESS) {
        throw std::runtime_error("Unable to allocate Vulkan image memory");
    }
    memory_.reset(device, memory);
    if (vkBindImageMemory(device, image_.get(), memory_.get(), 0) != VK_SUCCESS) {
        throw std::runtime_error("Unable to bind Vulkan image memory");
    }

    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.image = image_.get();
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = config.format;
    viewInfo.subresourceRange.aspectMask = config.aspect;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.layerCount = 1;
    VkImageView view{};
    if (vkCreateImageView(device, &viewInfo, nullptr, &view) != VK_SUCCESS) {
        throw std::runtime_error("Unable to create Vulkan image view");
    }
    view_.reset(device, view);

    VkSamplerCreateInfo samplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    samplerInfo.magFilter = config.filter;
    samplerInfo.minFilter = config.filter;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.maxLod = 1.0F;
    VkSampler sampler{};
    if (vkCreateSampler(device, &samplerInfo, nullptr, &sampler) != VK_SUCCESS) {
        throw std::runtime_error("Unable to create Vulkan sampler");
    }
    sampler_.reset(device, sampler);
    extent_ = config.extent;
    format_ = config.format;
    usage_ = config.usage;
}

void ImageResource::reset() {
    sampler_.reset();
    view_.reset();
    image_.reset();
    memory_.reset();
    extent_ = {};
    format_ = VK_FORMAT_UNDEFINED;
    usage_ = 0;
}

UniqueShaderModule loadShaderModule(const VkDevice device, const std::string_view path) {
    std::ifstream file(std::string{path}, std::ios::ate | std::ios::binary);
    if (!file) {
        throw std::runtime_error("Unable to open shader: " + std::string{path});
    }
    const auto size = file.tellg();
    if (size <= 0 || size % 4 != 0) {
        throw std::runtime_error("Invalid SPIR-V file: " + std::string{path});
    }
    std::vector<std::byte> code(static_cast<std::size_t>(size));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(code.data()), size);
    VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    info.codeSize = code.size();
    info.pCode = reinterpret_cast<const std::uint32_t*>(code.data());
    VkShaderModule shader{};
    if (vkCreateShaderModule(device, &info, nullptr, &shader) != VK_SUCCESS) {
        throw std::runtime_error("Unable to create shader module: " + std::string{path});
    }
    return UniqueShaderModule{device, shader};
}

} // namespace vkexp
