#ifndef RPP_CPU_DEVICE_HPP
#define RPP_CPU_DEVICE_HPP


#include <algorithm>
#include <new>
#include <memory>
#include <string>
#include <functional>

#include <rpp/config.h>
#include <rpp/utility.hpp>
#include <rpp/support/error.hpp>
#include <rpp/support/arch_tagged_pointer.hpp>

namespace rpp::cpu {

template <typename Architecture_>
class DataMapper {
    struct AllocEntry {
        void* ptr = nullptr;
        std::function<void(void*)> deleter;

        ~AllocEntry() noexcept {
            deleter(ptr);
        }
    };

public:
    using Architecture = Architecture_;

    using Error = Error<std::string>;

    template <typename T>
    using Result = Result<T, Error>;

    template <typename T>
    using ArchPtr = Ptr<T, Architecture>;

private:
    std::vector<AllocEntry> allocations_;
    std::size_t alignment = 64;


    template <typename T>
    Result<ArchPtr<T>> allocate(size_t size) noexcept {
        T *ptr = ::new (std::nothrow, std::align_val_t{alignment}) T[size];
        if (ptr == nullptr) {
            return Error{ErrorCode::OutOfResources, "Failed to allocate memory"};
        }

        allocations_.emplace_back(ptr, [size](void* ptr) noexcept {
            auto* typed_ptr = static_cast<T*>(ptr);
            std::destroy_n(typed_ptr, size);
            ::delete[] typed_ptr;
        });

        return tag_pointer<Architecture>(ptr);
    }

public:

    template <typename T, typename S, size_t N>
    Result<ArchPtr<T>> copy(Span<S, N> data) noexcept {
        auto allocation = allocate<T>(data.size());
        if (!allocation) {
            return allocation;
        }

        if (auto err = catch_exceptions([&]{ std::copy_n(data.begin(), data.size(), allocation.value()); })) {
            return err;
        }

        return allocation;
    }

    template <typename T, typename It>
    Result<ArchPtr<T>> copy(It begin, It end) noexcept {
        auto allocation = allocate<T>(std::distance(begin, end));
        if (!allocation) { return allocation; }

        if (auto err = catch_exceptions([&]{ std::copy(begin, end, allocation.value()); })) {
            return err;
        }

        return allocation;
    }

    template <typename T, typename It>
    Result<ArchPtr<T>> copy_n(It ptr, size_t size) noexcept {
        auto allocation = allocate<T>(size);
        if (!allocation) { return allocation; }

        if (auto err = catch_exceptions([&]{ std::copy_n(ptr, size, allocation.value()); })) {
            return err;
        }

        return allocation;
    }


};



}// namespace rpp::cpu

#endif //RPP_CPU_DEVICE_HPP
