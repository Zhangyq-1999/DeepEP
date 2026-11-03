# The configure path in shared_dispatch.cmake takes the same lock.
if(NOT IS_DIRECTORY "${BUILD_DIR}" OR NOT JOBS MATCHES "^[1-9][0-9]*$")
    message(FATAL_ERROR "Invalid shared Dispatch build directory or job count")
endif()
file(LOCK "${BUILD_DIR}.lock" GUARD PROCESS TIMEOUT 900)
execute_process(COMMAND "${CMAKE_COMMAND}" --build "${BUILD_DIR}" --parallel "${JOBS}"
    RESULT_VARIABLE _result)
if(NOT _result STREQUAL "0")
    message(FATAL_ERROR "Shared Dispatch build failed: ${_result}")
endif()
