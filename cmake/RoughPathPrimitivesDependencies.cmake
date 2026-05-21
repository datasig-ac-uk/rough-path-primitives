

include(CMakeFindDependencyMacro)


macro(_rpp_check_cuda_dependencies _FPN _CPT)
    if (NOT DEFINED CMAKE_CUDA_COMPILER)
        set(${_FPN}_${_CPT}_AVAILABLE FALSE)
        set(${_FPN}_NOT_FOUND_MESSAGE
                "${_FPN}: ${_CPT} component requires the CUDA Compiler"
        )
    else ()
        find_package(CUDAToolkit CONFIG QUIET)
        if (NOT CUDAToolkit_FOUND)
            set(${_FPN}_${_CPT}_AVAILABLE FALSE)
            set(${_FPN}_NOT_FOUND_MESSAGE
                    "${_FPN}: ${_CPT} component requires the CUDAToolkit"
            )
        else ()
            set(${_FPN}_${_CPT}_AVAILABLE TRUE)
        endif ()
    endif ()

endmacro()


macro(_rpp_check_dependencies _FPN _CPT)
    if (${_CPT} MATCHES "CUDA|GPU")
        set(${_FPN}_${_CPT}_AVAILABLE FALSE)
        _rpp_check_cuda_dependencies(${_FPN} ${_CPT})
    endif()
endmacro()
