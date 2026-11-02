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
message(STATUS "Original Combine: ${LL_COMBINE_BASELINE_REV} (unaltered kernel and headers)")
