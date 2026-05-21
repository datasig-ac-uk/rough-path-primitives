

include(CMakeFindDependencyMacro)


macro(_rpp_check_cuda_dependencies _FPN _CPT)
    if (TARGET CUDA::cudart)
        set(${_FPN}_${_CPT}_AVAILABLE TRUE)
    else ()
        find_package(CUDAToolkit QUIET)
        if (TARGET CUDA::cudart)
            set(${_FPN}_${_CPT}_AVAILABLE TRUE)
        else ()
            set(${_FPN}_${_CPT}_AVAILABLE FALSE)
            set(${_FPN}_NOT_FOUND_MESSAGE
                    "${_FPN}: ${_CPT} component requires the CUDAToolkit"
            )
        endif ()
    endif ()

endmacro()


macro(_rpp_check_dependencies _FPN _CPT)
    if (${_CPT} MATCHES "CUDA|GPU")
        set(${_FPN}_${_CPT}_AVAILABLE FALSE)
        _rpp_check_cuda_dependencies(${_FPN} ${_CPT})
    endif()
endmacro()
