#ifndef RPP_GPU_DEVICE_HPP
#define RPP_GPU_DEVICE_HPP

#include <vector>
#include <cuda_runtime.h>

#include <rpp/config.h>
#include <rpp/support/error.hpp>

namespace rpp::gpu {

struct DeviceLaunchConfig {
    cudaStream_t stream;
    std::vector<cudaLaunchAttribute> launch_attributes;
};


inline
Error<char const*> map_cuda_error(cudaError_t err) {
    ErrorCode code = ErrorCode::Unknown;

    switch (err) {
        case cudaSuccess:
            // code = ErrorCode::Ok;
            return Error<char const*>{ErrorCode::Ok, nullptr};
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

    return Error<char const*>(code, cudaGetErrorString(err));
}

} // namespace rpp::gpu

#endif //RPP_GPU_DEVICE_HPP
