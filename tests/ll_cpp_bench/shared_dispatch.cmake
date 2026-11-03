# Share Dispatch compiler outputs across Combine experiments. Headers and source
# changes are tracked by the child build system, not by a source-only stamp.
function(ll_bench_shared_dispatch)
    set(_compiler "${ASCEND_HOME_PATH}/tools/bisheng_compiler/bin/bisheng")
    file(SHA256 "${_compiler}" _compiler_hash)
    file(SHA256 "${SHMEM_ROOT}/deepep-sdk.json" _sdk_hash)
    set(_args
        -G "${CMAKE_GENERATOR}"
        "-DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
        "-DCMAKE_CXX_COMPILER=${_compiler}"
        "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
        "-DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}"
        "-DCMAKE_SHARED_LINKER_FLAGS:STRING=${CMAKE_SHARED_LINKER_FLAGS}"
        "-DDEEPEP_SOURCE_DIR=${DEEPEP_SOURCE_DIR}"
        "-DASCEND_HOME_PATH=${ASCEND_HOME_PATH}" "-DSHMEM_ROOT=${SHMEM_ROOT}"
        "-DDEEPEP_LL_PHYSICAL_RANKS_PER_HOST=${DEEPEP_LL_PHYSICAL_RANKS_PER_HOST}"
        "-DDEEPEP_LL_METADATA_CORES=${DEEPEP_LL_METADATA_CORES}"
        -DDEEPEP_DEVICE_COMPONENT=dispatch
        -DDEEPEP_LL_REFERENCE_PROFILE=OFF -DDEEPEP_LL_COUNT_HISTOGRAM=OFF)
    string(SHA256 _key "${_args};${_compiler_hash};${_sdk_hash}")
    set(_build "${DEEPEP_SOURCE_DIR}/build/ll-cpp-dispatch/${_key}")
    file(MAKE_DIRECTORY "${_build}")
    # A standalone build.sh and build_all.sh may reach the same cache. Serialize
    # configuration/builds without disabling normal compiler dependency checks.
    file(LOCK "${_build}.lock" GUARD FUNCTION TIMEOUT 900)
    execute_process(COMMAND "${CMAKE_COMMAND}" -S "${DEEPEP_SOURCE_DIR}/cmake/device"
        -B "${_build}" ${_args} RESULT_VARIABLE _result)
    if(NOT _result STREQUAL "0")
        message(FATAL_ERROR "Shared Dispatch configuration failed: ${_result}")
    endif()
    set(DEEPEP_DISPATCH_OBJECTS_FILE "${_build}/dispatch-objects.txt" PARENT_SCOPE)
    add_custom_target(ll_bench_dispatch_build
        COMMAND "${CMAKE_COMMAND}" "-DBUILD_DIR=${_build}" "-DJOBS=${DEEPEP_DEVICE_JOBS}"
            -P "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/build_dispatch.cmake"
        VERBATIM)
    message(STATUS "Shared Dispatch objects: ${_build}")
endfunction()
