# Fallback for source archives and builds without Git or release tags.
set(RPP_VERSION 1.0.2)

# Check for this project's metadata so vendored sources do not use a parent's tags.
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/../.git")
    find_package(Git QUIET)
    if(Git_FOUND)
        execute_process(
                COMMAND "${GIT_EXECUTABLE}" describe --tags
                        --match "releases/[0-9]*" --abbrev=0
                WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/.."
                RESULT_VARIABLE _rpp_git_result
                OUTPUT_VARIABLE _rpp_git_tag
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
        )
        if(_rpp_git_result EQUAL 0 AND
                _rpp_git_tag MATCHES "^releases/([0-9]+\\.[0-9]+\\.[0-9]+)$")
            set(RPP_VERSION "${CMAKE_MATCH_1}")
        endif()
        unset(_rpp_git_result)
        unset(_rpp_git_tag)
    endif()
endif()
