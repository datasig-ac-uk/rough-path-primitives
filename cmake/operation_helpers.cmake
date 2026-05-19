
macro(rpp_add_ops_cat _working _missing _comp _prefix _ops)

    foreach (op IN LISTS ${_ops})
        set(_path "${_prefix}/${_comp}/${op}.hpp")
        message(TRACE "Adding operation header file: ${_path}")
        if (NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/${_path}")
            list(APPEND ${_missing} "${_path}")
        endif ()
        list(APPEND ${_working} ${_path})
    endforeach ()
endmacro()

function(rpp_make_op_file_list _out)
    set(options PRIMARY)
    set(one_param_args PLATFORM STRATEGY)
    set(multi_param_args ADDITIONAL)
    cmake_parse_arguments(PARSE_ARGV 1 "ARG" "${options}" "${one_param_args}" "${multi_param_args}")

    set(_working)
    set(_missing)
    if (ARG_PRIMARY)
        set(_prefix "include/rpp/operations")
    else ()
        set(_prefix "include/rpp/${ARG_PLATFORM}/operations/${ARG_STRATEGY}")
    endif ()

    rpp_add_ops_cat(_working _missing linalg ${_prefix} RPP_LINALG_OPS)
    rpp_add_ops_cat(_working _missing basic ${_prefix} RPP_BASIC_OPS)
    rpp_add_ops_cat(_working _missing intermediate ${_prefix} RPP_INTERMEDIATE_OPS)


    foreach (addition IN LISTS ARG_ADDITIONAL)
        set(_path "${_prefix}/${addition}")
        message(TRACE "Adding additional operation header: ${_path}")
        if (NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/${_path}")
            list(APPEND _missing "${_path}")
        endif ()
        list(APPEND _working ${_path})
    endforeach ()

    if (NOT "${_missing}" STREQUAL "")
        list(JOIN _missing "\n" _missing_paths)
        message(WARNING
                "The following operation files could not be found:\n${_missing_paths}")
    endif ()

    set(${_out} ${_working} PARENT_SCOPE)
endfunction()
