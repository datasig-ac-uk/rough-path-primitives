add_subdirectory(basis)
add_subdirectory(cpu)
add_subdirectory(dense)
add_subdirectory(sparse)
add_subdirectory(support)

if (RPP_ENABLE_CUDA)
    add_subdirectory(gpu)
endif()
