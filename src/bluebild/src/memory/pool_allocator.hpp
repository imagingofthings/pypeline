#pragma once

#include <cassert>
#include <cstddef>
#include <map>
#include <memory>
#include <unordered_map>
#include <functional>
#include <mutex>

#include "bluebild/config.h"
#include "memory/allocator.hpp"
#include "bluebild/exceptions.hpp"

namespace bluebild {

class PoolAllocator : public Allocator {
public:
  PoolAllocator(std::function<void*(std::size_t)> allocateFunc,
                std::function<void(void*)> deallocateFunc)
      : allocateFunc_(std::move(allocateFunc)),
        deallocateFunc_(std::move(deallocateFunc)),
        lock_(new std::mutex()), memorySize_(0) {
    if (!allocateFunc_ || !deallocateFunc_) {
      throw InvalidAllocatorFunctionError();
    }
  }

  PoolAllocator(const PoolAllocator&) = delete;

  PoolAllocator(PoolAllocator&&) = default;

  auto operator=(const PoolAllocator&) -> PoolAllocator& = delete;

  auto operator=(PoolAllocator&&) -> PoolAllocator& = default;

  ~PoolAllocator() override {
    for (auto& pair : allocatedMem_) {
      assert(false);  // No allocated memory should still exist with correct usage
      deallocateFunc_(pair.first);
      memorySize_ -= pair.second;
    }
    for (auto& pair : freeMem_) {
      deallocateFunc_(pair.second);
      memorySize_ -= pair.first;
    }
  }

  auto allocate(std::size_t size) -> void* override {
    if (!size) return nullptr;
    std::lock_guard<std::mutex> guard(*lock_);

    void* ptr = nullptr;
    // find block which is greater or equal to size
    auto boundIt = freeMem_.lower_bound(size);

    if (boundIt == freeMem_.end()) {
      // No memory block is large enough. Free the largest one and allocate new size.
      if(!freeMem_.empty()) {
        auto backIt = --freeMem_.end();
        deallocateFunc_(backIt->second);
        memorySize_ -= backIt->first;
        freeMem_.erase(backIt);
      }
      ptr = allocateFunc_(size);
      memorySize_ += size;
      allocatedMem_.emplace(ptr, size);
    } else {
      // Use already allocated memory block.
      ptr = boundIt->second;
      allocatedMem_.emplace(boundIt->second, boundIt->first);
      freeMem_.erase(boundIt);
    }

    return ptr;
  }

  auto deallocate(void* ptr) -> void override {
    std::lock_guard<std::mutex> guard(*lock_);

    auto it = allocatedMem_.find(ptr);
    assert(it != allocatedMem_.end());  // avoid throwing exception when deallocating
    if (it != allocatedMem_.end()) {
      freeMem_.emplace(it->second, it->first);
      allocatedMem_.erase(it);
    }
  }

  auto size() -> std::uint_least64_t override {
    return memorySize_;
  }

  auto print() -> void override {
      printf("//content of freeMem_:\n");
      for (auto it = freeMem_.begin(); it != freeMem_.end(); ++it) {
          printf("freeMem_      %ld Bytes\n", it->first);
      }
      printf("//content of allocatedMem_:\n");
      for (auto it = allocatedMem_.begin(); it != allocatedMem_.end(); ++it) {
          printf("allocatedMem_ %ld Bytes\n", it->second);
      }
      fflush(stdout);
  }
    

private:
  std::function<void*(std::size_t)> allocateFunc_;
  std::function<void(void*)> deallocateFunc_;

  std::multimap<std::size_t, void*> freeMem_;
  std::unordered_map<void*, std::size_t> allocatedMem_;

  std::unique_ptr<std::mutex> lock_;
  std::uint_least64_t memorySize_;
};
}  // namespace bluebild
