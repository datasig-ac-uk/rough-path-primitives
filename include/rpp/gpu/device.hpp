#ifndef RPP_GPU_DEVICE_HPP
#define RPP_GPU_DEVICE_HPP

#include <iterator>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include <rpp/config.h>
#include <rpp/support/arch_tagged_pointer.hpp>
#include <rpp/support/span.hpp>
#include <rpp/support/error.hpp>

namespace rpp::gpu {
struct DeviceLaunchConfig {
    cudaStream_t stream;
    std::vector<cudaLaunchAttribute> launch_attributes;
};


inline
Error<char const *> map_cuda_error(cudaError_t err) {
    ErrorCode code = ErrorCode::Unknown;

    switch (err) {
        case cudaSuccess:
            // code = ErrorCode::Ok;
            return Error<char const *>{ErrorCode::Ok, nullptr};
            break;

        case cudaErrorInvalidValue:
        case cudaErrorInvalidDevice:
        case cudaErrorInvalidSymbol:
        case cudaErrorInvalidPitchValue:
        case cudaErrorInvalidTexture:
        case cudaErrorInvalidTextureBinding:
        case cudaErrorInvalidChannelDescriptor:
        case cudaErrorInvalidMemcpyDirection:
        case cudaErrorInvalidFilterSetting:
        case cudaErrorInvalidNormSetting:
        case cudaErrorInvalidResourceHandle:
        case cudaErrorInvalidConfiguration:
        case cudaErrorInvalidSurface:
        case cudaErrorInvalidAddressSpace:
        case cudaErrorInvalidPtx:
        case cudaErrorInvalidGraphicsContext:
        case cudaErrorInvalidSource:
        case cudaErrorInvalidKernelImage:
        case cudaErrorInvalidClusterSize:
            code = ErrorCode::InvalidArgument;
            break;

        case cudaErrorMemoryAllocation:
        case cudaErrorLaunchOutOfResources:
        case cudaErrorOperatingSystem:
        case cudaErrorSetOnActiveProcess:
            code = ErrorCode::OutOfResources;
            break;

        case cudaErrorLaunchTimeout:
            code = ErrorCode::Timeout;
            break;

        case cudaErrorNotReady:
            code = ErrorCode::Cancelled;
            break;

        case cudaErrorNotSupported:
        case cudaErrorUnsupportedLimit:
        case cudaErrorUnsupportedPtxVersion:
        case cudaErrorJitCompilerNotFound:
        case cudaErrorCallRequiresNewerDriver:
        case cudaErrorCompatNotSupportedOnDevice:
        case cudaErrorStreamCaptureUnsupported:
            code = ErrorCode::NotImplemented;
            break;

        case cudaErrorMissingConfiguration:
        case cudaErrorLaunchFailure:
        case cudaErrorLaunchIncompatibleTexturing:
        case cudaErrorAssert:
        case cudaErrorTooManyPeers:
        case cudaErrorHostMemoryAlreadyRegistered:
        case cudaErrorHostMemoryNotRegistered:
        case cudaErrorPeerAccessAlreadyEnabled:
        case cudaErrorPeerAccessNotEnabled:
        case cudaErrorContextIsDestroyed:
        case cudaErrorIllegalState:
        case cudaErrorStreamCaptureInvalidated:
        case cudaErrorStreamCaptureMerge:
        case cudaErrorStreamCaptureUnmatched:
        case cudaErrorStreamCaptureUnjoined:
        case cudaErrorStreamCaptureIsolation:
        case cudaErrorStreamCaptureImplicit:
        case cudaErrorCapturedEvent:
        case cudaErrorStreamCaptureWrongThread:
        case cudaErrorIllegalAddress:
        case cudaErrorIllegalInstruction:
        case cudaErrorMisalignedAddress:
        case cudaErrorHardwareStackError:
            code = ErrorCode::ContractViolation;
            break;

        case cudaErrorMapBufferObjectFailed:
        case cudaErrorUnmapBufferObjectFailed:
        case cudaErrorArrayIsMapped:
        case cudaErrorAlreadyMapped:
        case cudaErrorNoKernelImageForDevice:
        case cudaErrorECCUncorrectable:
        case cudaErrorSharedObjectSymbolNotFound:
        case cudaErrorSharedObjectInitFailed:
        case cudaErrorDevicesUnavailable:
        case cudaErrorStartupFailure:
        case cudaErrorCudartUnloading:
        case cudaErrorSystemNotReady:
        case cudaErrorSystemDriverMismatch:
        case cudaErrorUnknown:
            code = ErrorCode::Internal;
            break;

        case cudaErrorInitializationError:
        case cudaErrorInsufficientDriver:
        case cudaErrorNoDevice:
        case cudaErrorDeviceAlreadyInUse:
        case cudaErrorProfilerDisabled:
            code = ErrorCode::OutOfResources;
            break;

        default:
            code = ErrorCode::Unknown;
            break;
    }

    return Error<char const *>(code, cudaGetErrorString(err));
}


template <typename Architecture_>
class DataMapper {
    cudaStream_t stream_;
    std::vector<void *> allocations_;

public:

    using Architecture = Architecture_;
    using Error = Error<char const *>;

    template <typename T>
    using Result = Result<T, Error>;

    template <typename T>
    using ArchPtr = Ptr<T, Architecture>;


    explicit constexpr DataMapper(cudaStream_t stream) : stream_(stream) {
    }

    ~DataMapper() noexcept {
        for (auto ptr: allocations_) {
            cudaFreeAsync(ptr, stream_);
        }
    }

private:
    template<typename T=std::byte>
    Result<ArchPtr<T>> allocate(size_t size) noexcept {
        T *ptr = nullptr;
        size_t size_bytes = size * sizeof(T);

        auto err = cudaMallocAsync(&ptr, size_bytes, stream_);
        if (err != cudaSuccess) {
            return map_cuda_error(err);
        }

        allocations_.push_back(ptr);
        return tag_pointer<Architecture>(ptr);
    }

public:
    template<typename T, size_t N>
    Result<ArchPtr<T>> copy(Span<const T, N> host_data) noexcept {
        auto allocation = allocate<T>(host_data.size());
        if (!allocation) {
            return allocation;
        }

        auto err = cudaMemcpyAsync(
            static_cast<void *>(allocation.value()),
            static_cast<void const *>(host_data.data()),
            host_data.size_bytes(),
            cudaMemcpyHostToDevice,
            stream_);

        if (err != cudaSuccess) {
            return map_cuda_error(err);
        }

        return allocation;
    }

    template <typename T, typename S, size_t N, typename=std::enable_if_t<!std::is_same_v<T, std::remove_cv_t<S>>>>
    Result<ArchPtr<T>> copy(Span<S, N> host_data) noexcept {
        std::vector<T> data(host_data.begin(), host_data.end());
        return copy(data);
    }

    template<typename T, typename It>
    Result<ArchPtr<T>> copy(It begin, It end) noexcept {
        using Traits = traits::IteratorTraits<It>;
        using Value = std::remove_cv_t<typename Traits::value_type>;

        if constexpr (std::is_pointer_v<It> && std::is_same_v<Value, T>) {
            return copy(Span<const T>{ begin, static_cast<size_t>(end - begin) });
        } else if constexpr (traits::is_arch_ptr_v<It> && std::is_same_v<Value, T>) {
            return copy(Span<const T>{ begin.raw_ptr(), static_cast<size_t>(end - begin.raw_ptr()) });
        } else {
            std::vector<T> data(begin, end);
            return copy(data);
        }
    }

    template <typename T, typename It>
    Result<ArchPtr<T>> copy_n(It begin, size_t count) noexcept {
        copy(begin, begin + count);
    }


};

} // namespace rpp::gpu

#endif //RPP_GPU_DEVICE_HPP
