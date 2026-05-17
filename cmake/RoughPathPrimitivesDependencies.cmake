

include(CMakeFindDependencyMacro)


macro(_rpp_check_cuda_dependencies _FPN _CPT)
    if (NOT DEFINED CMAKE_CUDA_COMPILER)
        set(${_FPN}_${_CPT}_AVAILABLE FALSE)
        set(${_FPN}_NOT_FOUND_MESSAGE
                "${_FPN}: ${_CPT} component requires the CUDA Compiler"
        )
        return()
    endif()

    find_package(CUDAToolkit CONFIG QUIET)
    if (NOT CUDAToolkit_FOUND)
        set(${_FPN}_${_CPT}_AVAILABLE FALSE)
        set(${_FPN}_NOT_FOUND_MESSAGE
                "${_FPN}: ${_CPT} component requires the CUDAToolkit"
        )
        return()
    endif()

    set(${_FPN}_${_CPT}_AVAILABLE TRUE)
endmacro()


macro(_rpp_check_dependencies _FPN _CPT)
    if (${_CPT} MATCHES "CUDA|GPU")
        _rpp_check_cuda_dependencies(${_FPN} ${_CPT})
    endif()
endmacro()