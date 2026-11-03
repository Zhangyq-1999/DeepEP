# Materialize the upstream kernel tree verbatim, outside the experimental source
# directory. The original Combine includes only headers from this frozen tree.
set(LL_COMBINE_BASELINE_REV "5b38bf2ea1f67f38c0c2d3662618ae1a0457bfc6")
find_package(Git REQUIRED)
set(LL_COMBINE_BASELINE_DIR "${CMAKE_BINARY_DIR}/original-${LL_COMBINE_BASELINE_REV}")
set(_baseline_archive "${CMAKE_BINARY_DIR}/original-kernels.tar")
execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${DEEPEP_SOURCE_DIR}" archive
    --format=tar "--output=${_baseline_archive}" "${LL_COMBINE_BASELINE_REV}" csrc/kernels
    RESULT_VARIABLE _baseline_status ERROR_VARIABLE _baseline_error)
if(NOT _baseline_status STREQUAL "0")
    message(FATAL_ERROR "Cannot read original Combine revision ${LL_COMBINE_BASELINE_REV}. Fetch that revision first. ${_baseline_error}")
endif()
file(MAKE_DIRECTORY "${LL_COMBINE_BASELINE_DIR}")
file(ARCHIVE_EXTRACT INPUT "${_baseline_archive}" DESTINATION "${LL_COMBINE_BASELINE_DIR}")
set(DEEPEP_LL_COMBINE_SOURCE "${LL_COMBINE_BASELINE_DIR}/csrc/kernels/combine/low_latency.cpp")
# Only the physical-host rank count changes for sparse-device tests. Keep all
# original Combine source and algorithm headers byte-for-byte intact.
if(DEFINED DEEPEP_LL_PHYSICAL_RANKS_PER_HOST AND NOT DEEPEP_LL_PHYSICAL_RANKS_PER_HOST STREQUAL "16")
    if(NOT DEEPEP_LL_PHYSICAL_RANKS_PER_HOST STREQUAL "4")
        message(FATAL_ERROR "Unsupported baseline host layout")
    endif()
    set(_policy "${LL_COMBINE_BASELINE_DIR}/csrc/kernels/dispatch/low_latency/common/ll_group_policy.h")
    file(READ "${_policy}" _policy_text)
    string(REPLACE "kLlPhysicalRanksPerHost = 16U" "kLlPhysicalRanksPerHost = 4U" _policy_text "${_policy_text}")
    file(CONFIGURE OUTPUT "${_policy}" CONTENT "${_policy_text}" @ONLY NEWLINE_STYLE UNIX)
endif()
message(STATUS "Original Combine: ${LL_COMBINE_BASELINE_REV} (unaltered Combine; physical ranks per host=${DEEPEP_LL_PHYSICAL_RANKS_PER_HOST})")
