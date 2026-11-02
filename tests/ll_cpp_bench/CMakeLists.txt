cmake_minimum_required(VERSION 3.20)
project(ll_combine_bench LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
get_filename_component(DEEPEP_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
find_package(MPI REQUIRED COMPONENTS CXX)
include("${DEEPEP_SOURCE_DIR}/cmake/Dependencies.cmake")
find_file(SHMEM_MPI_PLUGIN aclshmem_bootstrap_mpi.so PATHS "${SHMEM_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
option(DEEPEP_LL_COMBINE_SMALL_BATCH "Cache partial tasks and trim idle lanes" OFF)
option(DEEPEP_LL_COMBINE_EARLY_REDUCE "Reduce ready contributions for BS <= 72" OFF)
set(DEEPEP_LL_COMBINE_RECV_CORES "0" CACHE STRING "0=half; explicit receive cores in [8,36]")
if(NOT DEEPEP_LL_COMBINE_RECV_CORES MATCHES "^(0|[89]|[12][0-9]|3[0-6])$")
    message(FATAL_ERROR "Receive cores must be 0 or an integer in [8,36]")
endif()
set(DEEPEP_LL_METADATA_CORES 4)
set(DEEPEP_LL_REFERENCE_PROFILE OFF)
set(DEEPEP_LL_STAGE_TIMER OFF)
set(DEEPEP_LL_COUNT_HISTOGRAM OFF)
include("${DEEPEP_SOURCE_DIR}/cmake/Device.cmake")
find_path(DEEPEP_PLATFORM_INCLUDE tiling/platform/platform_ascendc.h
    HINTS "${ASCEND_HOME_PATH}/asc/include" "${ASCEND_HOME_PATH}/include/op_common"
          "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc/include" NO_DEFAULT_PATH REQUIRED)
add_executable(ll_combine_bench main.cpp "${DEEPEP_SOURCE_DIR}/csrc/runtime/session.cpp")
target_include_directories(ll_combine_bench PRIVATE "${DEEPEP_SOURCE_DIR}/csrc"
    "${DEEPEP_PLATFORM_INCLUDE}" "${ASCEND_HOME_PATH}/include" "${SHMEM_ROOT}/include")
target_link_directories(ll_combine_bench PRIVATE "${ASCEND_HOME_PATH}/lib64"
    "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64")
target_link_libraries(ll_combine_bench PRIVATE MPI::MPI_CXX ascendcl shmem deepep_device
    platform tiling_api unified_dlog ${CMAKE_DL_LIBS} pthread)
target_compile_definitions(ll_combine_bench PRIVATE DEEPEP_LL_RDMA=1 DEEPEP_LL_METADATA_CORES=4
    DEEPEP_LL_COMBINE_RECV_CORES=${DEEPEP_LL_COMBINE_RECV_CORES}
    $<$<BOOL:${DEEPEP_LL_COMBINE_SMALL_BATCH}>:DEEPEP_LL_COMBINE_SMALL_BATCH=1>
    $<$<BOOL:${DEEPEP_LL_COMBINE_EARLY_REDUCE}>:DEEPEP_LL_COMBINE_EARLY_REDUCE=1>)
# Hash the compiled source tree, including uncommitted edits, for cross-rank checks.
file(GLOB_RECURSE _identity_files CONFIGURE_DEPENDS
    "${DEEPEP_SOURCE_DIR}/csrc/*.h" "${DEEPEP_SOURCE_DIR}/csrc/*.hpp" "${DEEPEP_SOURCE_DIR}/csrc/*.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/*.hpp" "${CMAKE_CURRENT_LIST_DIR}/*.cpp")
list(SORT _identity_files)
set(_source_hashes "")
foreach(_file IN LISTS _identity_files)
    file(SHA256 "${_file}" _hash)
    string(APPEND _source_hashes "${_hash}")
endforeach()
string(SHA256 _source_id "${_source_hashes}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_identity_files})
target_compile_definitions(ll_combine_bench PRIVATE DEEPEP_BENCH_SOURCE_ID="${_source_id}")
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/bench-build.txt" CONTENT
    "source=${_source_id}\nsmall_batch=${DEEPEP_LL_COMBINE_SMALL_BATCH}\nrecv_cores=${DEEPEP_LL_COMBINE_RECV_CORES}\nearly_reduce=${DEEPEP_LL_COMBINE_EARLY_REDUCE}\nmpi=${MPI_CXX_COMPILER}\nshmem=${SHMEM_ROOT}\n")
