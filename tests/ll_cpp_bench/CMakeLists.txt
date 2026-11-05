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
option(DEEPEP_LL_COMBINE_PROFILE "Build optional Combine diagnostics" OFF)
option(DEEPEP_LL_OPS_BASE "Build the ops-transformer SHMEM adaptation" OFF)
option(DEEPEP_LL_COMBINE_ORIGINAL "Use the unaltered upstream Combine kernel" ON)
option(DEEPEP_LL_COMBINE_SMALL_BATCH "Cache partial tasks and trim idle lanes" OFF)
option(DEEPEP_LL_COMBINE_LOCAL_FIRST "Accumulate local-host returns before remote returns" OFF)
option(DEEPEP_LL_COMBINE_BATCH_FINAL "Batch vector readiness and prefetch final contributions" OFF)
option(DEEPEP_LL_COMBINE_EARLY_REDUCE "Reduce ready contributions for BS <= 72" OFF)
option(DEEPEP_LL_COMBINE_REMOTE_FIRST "Return cross-host results before local-host results" OFF)
option(DEEPEP_LL_COMBINE_DEFER_RETURN "Overlap return staging with next-packet preparation" OFF)
option(DEEPEP_LL_COMBINE_EARLY_ISSUE "Submit the preceding packet before preparing the next packet" OFF)
option(DEEPEP_LL_COMBINE_PROGRESS_PROBE "Remember confirmed result contributions" OFF)
option(DEEPEP_LL_COMBINE_SIMT_META "SIMT stable task compaction and batch metadata loads" OFF)
option(DEEPEP_LL_COMBINE_SIMT_FLAG "SIMT checks of complete UB flag snapshots" OFF)
option(DEEPEP_LL_COMBINE_PREFETCH_RETURN "Post the previous return after issuing next input reads" OFF)
set(DEEPEP_LL_COMBINE_READY "0" CACHE STRING "0=original, 1=task dependencies, 2=shared rank ready, 3=cached task dependencies, 4=route snapshot, 5=paired staging, 6=prefetch submission, 7=eager-first stream, 8=compact FIRST_HOP plan")
if(NOT DEEPEP_LL_COMBINE_READY MATCHES "^[012345678]$")
    message(FATAL_ERROR "ready must be an integer in [0,8]")
endif()
if(DEEPEP_LL_COMBINE_READY AND (DEEPEP_LL_COMBINE_ORIGINAL OR DEEPEP_LL_OPS_BASE OR DEEPEP_LL_COMBINE_SMALL_BATCH OR DEEPEP_LL_COMBINE_SIMT_META))
    message(FATAL_ERROR "Ready experiments require isolated opt builds")
endif()
set(DEEPEP_LL_COMBINE_NOTIFY "0" CACHE STRING "0=MTE rank, 1=SIMT rank, 2=SIMT per core")
if(NOT DEEPEP_LL_COMBINE_NOTIFY MATCHES "^[012]$")
    message(FATAL_ERROR "notify must be 0, 1 or 2")
endif()
if(DEEPEP_LL_COMBINE_NOTIFY AND (DEEPEP_LL_COMBINE_ORIGINAL OR DEEPEP_LL_OPS_BASE))
    message(FATAL_ERROR "Notification experiments require --variant opt")
endif()
set(DEEPEP_LL_COMBINE_COALESCE "0" CACHE STRING "0=off, 1=destination card, 2=server same-slot relay")
set(DEEPEP_LL_COMBINE_COALESCE_ROWS "0" CACHE STRING "0=one whole bucket WQE, 1=one hidden per WQE")
if(NOT DEEPEP_LL_COMBINE_COALESCE MATCHES "^[012]$" OR NOT DEEPEP_LL_COMBINE_COALESCE_ROWS MATCHES "^[01]$")
    message(FATAL_ERROR "Invalid coalesce mode or packet size")
endif()
if(DEEPEP_LL_COMBINE_READY AND (DEEPEP_LL_COMBINE_NOTIFY OR DEEPEP_LL_COMBINE_COALESCE))
    message(FATAL_ERROR "Test ready modes separately from notification and coalescing")
endif()
if(DEEPEP_LL_COMBINE_NOTIFY AND DEEPEP_LL_COMBINE_COALESCE)
    message(FATAL_ERROR "Test notification and coalescing experiments separately")
endif()
if(DEEPEP_LL_COMBINE_COALESCE AND (DEEPEP_LL_COMBINE_ORIGINAL OR DEEPEP_LL_OPS_BASE))
    message(FATAL_ERROR "Coalescing requires --variant opt")
endif()
if(DEEPEP_LL_COMBINE_COALESCE_ROWS AND NOT DEEPEP_LL_COMBINE_COALESCE)
    message(FATAL_ERROR "coalesce-rows requires a coalescing mode")
endif()
set(DEEPEP_LL_COMBINE_POLL_NOP "0" CACHE STRING "Nop cycles after a failed flag probe; 0 disables")
if(NOT DEEPEP_LL_COMBINE_POLL_NOP MATCHES "^(0|[1-9][0-9]*)$")
    message(FATAL_ERROR "poll-nop must be an integer in [0,65535]")
endif()
if(DEEPEP_LL_COMBINE_POLL_NOP GREATER 65535)
    message(FATAL_ERROR "poll-nop must be an integer in [0,65535]")
endif()
if(DEEPEP_LL_COMBINE_BATCH_FINAL AND NOT DEEPEP_LL_COMBINE_LOCAL_FIRST)
    message(FATAL_ERROR "batch-final requires local-first")
endif()
if(DEEPEP_LL_COMBINE_LOCAL_FIRST AND (DEEPEP_LL_COMBINE_EARLY_REDUCE OR DEEPEP_LL_COMBINE_PROGRESS_PROBE OR DEEPEP_LL_COMBINE_SIMT_FLAG))
    message(FATAL_ERROR "local-first cannot combine with early-reduce, progress-probe or simt-flag")
endif()
if(DEEPEP_LL_COMBINE_SIMT_FLAG AND (DEEPEP_LL_COMBINE_PROGRESS_PROBE OR DEEPEP_LL_COMBINE_EARLY_REDUCE))
    message(FATAL_ERROR "SIMT flag checks cannot be combined with progress-probe or early-reduce")
endif()
if(DEEPEP_LL_COMBINE_PREFETCH_RETURN AND DEEPEP_LL_COMBINE_EARLY_ISSUE)
    message(FATAL_ERROR "prefetch-return and early-issue select different submission points")
endif()
set(DEEPEP_LL_PHYSICAL_RANKS_PER_HOST "16" CACHE STRING "Participating ranks per physical host: 4 or 16")
if(NOT DEEPEP_LL_PHYSICAL_RANKS_PER_HOST MATCHES "^(4|16)$")
    message(FATAL_ERROR "ranks-per-host must be 4 or 16")
endif()
add_compile_definitions(DEEPEP_LL_PHYSICAL_RANKS_PER_HOST=${DEEPEP_LL_PHYSICAL_RANKS_PER_HOST})
set(DEEPEP_LL_COMBINE_RECV_CORES "0" CACHE STRING "0=half; explicit receive cores in [8,36]")
if(NOT DEEPEP_LL_COMBINE_RECV_CORES MATCHES "^(0|[89]|[12][0-9]|3[0-6])$")
    message(FATAL_ERROR "Receive cores must be 0 or an integer in [8,36]")
endif()
set(DEEPEP_LL_METADATA_CORES 4)
set(DEEPEP_LL_REFERENCE_PROFILE OFF)
set(DEEPEP_LL_STAGE_TIMER OFF)
set(DEEPEP_LL_COUNT_HISTOGRAM OFF)
if(DEEPEP_LL_OPS_BASE)
    if(NOT DEEPEP_LL_PHYSICAL_RANKS_PER_HOST STREQUAL "16")
        message(FATAL_ERROR "ops-shmem-base currently requires --ranks-per-host 16")
    endif()
    if(DEEPEP_LL_COMBINE_ORIGINAL OR DEEPEP_LL_COMBINE_PROFILE OR DEEPEP_LL_COMBINE_SMALL_BATCH OR
       DEEPEP_LL_COMBINE_EARLY_REDUCE OR DEEPEP_LL_COMBINE_REMOTE_FIRST OR DEEPEP_LL_COMBINE_DEFER_RETURN OR
       DEEPEP_LL_COMBINE_EARLY_ISSUE OR DEEPEP_LL_COMBINE_PROGRESS_PROBE OR
       DEEPEP_LL_COMBINE_SIMT_META OR DEEPEP_LL_COMBINE_SIMT_FLAG OR DEEPEP_LL_COMBINE_PREFETCH_RETURN OR
       DEEPEP_LL_COMBINE_LOCAL_FIRST OR DEEPEP_LL_COMBINE_POLL_NOP OR NOT DEEPEP_LL_COMBINE_RECV_CORES STREQUAL "0")
        message(FATAL_ERROR "ops-shmem-base has no DeepEP experiment flags or instrumentation")
    endif()
    include("${CMAKE_CURRENT_LIST_DIR}/ops_shmem/source.cmake")
elseif(DEEPEP_LL_COMBINE_ORIGINAL)
    if(DEEPEP_LL_COMBINE_SMALL_BATCH OR DEEPEP_LL_COMBINE_EARLY_REDUCE OR DEEPEP_LL_COMBINE_PROFILE OR
       DEEPEP_LL_COMBINE_REMOTE_FIRST OR DEEPEP_LL_COMBINE_DEFER_RETURN OR
       DEEPEP_LL_COMBINE_EARLY_ISSUE OR DEEPEP_LL_COMBINE_PROGRESS_PROBE OR
       DEEPEP_LL_COMBINE_SIMT_META OR DEEPEP_LL_COMBINE_SIMT_FLAG OR DEEPEP_LL_COMBINE_PREFETCH_RETURN OR
       DEEPEP_LL_COMBINE_LOCAL_FIRST OR DEEPEP_LL_COMBINE_POLL_NOP OR NOT DEEPEP_LL_COMBINE_RECV_CORES STREQUAL "0")
        message(FATAL_ERROR "Original Combine cannot enable experimental options or profiling; use --variant opt")
    endif()
    include("${CMAKE_CURRENT_LIST_DIR}/baseline.cmake")
else()
    set(DEEPEP_LL_COMBINE_SOURCE "${DEEPEP_SOURCE_DIR}/csrc/kernels/combine/low_latency.cpp")
endif()
string(SHA256 _variant_tag "${CMAKE_BINARY_DIR}")
string(SUBSTRING "${_variant_tag}" 0 12 _variant_tag)
set(DEEPEP_DEVICE_OUTPUT_NAME "deepep_bench_${_variant_tag}")
set(DEEPEP_DEVICE_LOCAL_SYMBOLS ON)
set(DEEPEP_LL_BENCH_ENTRY_SYNC ON)
set(DEEPEP_DEVICE_JOBS "4" CACHE STRING "Parallel Device compilation jobs")
if(NOT DEEPEP_LL_OPS_BASE)
    include("${CMAKE_CURRENT_LIST_DIR}/shared_dispatch.cmake")
    ll_bench_shared_dispatch()
    set(DEEPEP_DEVICE_COMPONENT combine)
else()
    set(DEEPEP_DEVICE_COMPONENT all)
endif()
include("${DEEPEP_SOURCE_DIR}/cmake/Device.cmake")
if(TARGET ll_bench_dispatch_build)
    add_dependencies(device_build ll_bench_dispatch_build)
endif()
find_path(DEEPEP_PLATFORM_INCLUDE tiling/platform/platform_ascendc.h
    HINTS "${ASCEND_HOME_PATH}/asc/include" "${ASCEND_HOME_PATH}/include/op_common"
          "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc/include" NO_DEFAULT_PATH REQUIRED)
if(DEEPEP_LL_OPS_BASE)
    set(_bench_main ops_shmem/main.cpp)
else()
    set(_bench_main main.cpp)
endif()
add_library(ll_bench_variant MODULE ${_bench_main})
set_target_properties(ll_bench_variant PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES
    NO_SONAME TRUE)
target_link_options(ll_bench_variant PRIVATE -Wl,-Bsymbolic)
if(DEEPEP_LL_OPS_BASE)
    target_compile_definitions(ll_bench_variant PRIVATE LL_OPS_REV="${LL_OPS_REV}")
endif()
if(DEEPEP_LL_COMBINE_ORIGINAL)
    target_sources(ll_bench_variant PRIVATE baseline_queries.cpp)
endif()
target_include_directories(ll_bench_variant PRIVATE "${DEEPEP_SOURCE_DIR}/csrc"
    "${DEEPEP_PLATFORM_INCLUDE}" "${ASCEND_HOME_PATH}/include" "${SHMEM_ROOT}/include")
target_link_directories(ll_bench_variant PRIVATE "${ASCEND_HOME_PATH}/lib64"
    "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64")
# tiling_api is static in CANN 9.2 and uses platform symbols. Keep its provider
# afterwards so --as-needed retains libplatform when linking the executable.
target_link_libraries(ll_bench_variant PRIVATE MPI::MPI_CXX ascendcl shmem deepep_device
    tiling_api platform unified_dlog ${CMAKE_DL_LIBS} pthread)
target_compile_definitions(ll_bench_variant PRIVATE DEEPEP_LL_RDMA=1 DEEPEP_LL_METADATA_CORES=4
    DEEPEP_LL_COMBINE_READY=${DEEPEP_LL_COMBINE_READY}
    DEEPEP_LL_COMBINE_NOTIFY=${DEEPEP_LL_COMBINE_NOTIFY}
    DEEPEP_LL_COMBINE_COALESCE=${DEEPEP_LL_COMBINE_COALESCE}
    DEEPEP_LL_COMBINE_COALESCE_ROWS=${DEEPEP_LL_COMBINE_COALESCE_ROWS}
    DEEPEP_LL_COMBINE_ORIGINAL=$<BOOL:${DEEPEP_LL_COMBINE_ORIGINAL}>
    DEEPEP_LL_COMBINE_RECV_CORES=${DEEPEP_LL_COMBINE_RECV_CORES}
    $<$<BOOL:${DEEPEP_LL_COMBINE_SMALL_BATCH}>:DEEPEP_LL_COMBINE_SMALL_BATCH=1>
    $<$<BOOL:${DEEPEP_LL_COMBINE_SIMT_META}>:DEEPEP_LL_COMBINE_SIMT_META=1>
    $<$<BOOL:${DEEPEP_LL_COMBINE_SIMT_FLAG}>:DEEPEP_LL_COMBINE_SIMT_FLAG=1>
    $<$<BOOL:${DEEPEP_LL_COMBINE_LOCAL_FIRST}>:DEEPEP_LL_COMBINE_LOCAL_FIRST=1>
    $<$<BOOL:${DEEPEP_LL_COMBINE_BATCH_FINAL}>:DEEPEP_LL_COMBINE_BATCH_FINAL=1>
    $<$<BOOL:${DEEPEP_LL_COMBINE_EARLY_REDUCE}>:DEEPEP_LL_COMBINE_EARLY_REDUCE=1>)
target_compile_definitions(ll_bench_variant PRIVATE
    DEEPEP_BENCH_READY=${DEEPEP_LL_COMBINE_READY}
    DEEPEP_BENCH_NOTIFY=${DEEPEP_LL_COMBINE_NOTIFY}
    DEEPEP_BENCH_COALESCE=${DEEPEP_LL_COMBINE_COALESCE}
    DEEPEP_BENCH_COALESCE_ROWS=${DEEPEP_LL_COMBINE_COALESCE_ROWS}
    DEEPEP_BENCH_REMOTE_FIRST=$<BOOL:${DEEPEP_LL_COMBINE_REMOTE_FIRST}>
    DEEPEP_BENCH_DEFER_RETURN=$<BOOL:${DEEPEP_LL_COMBINE_DEFER_RETURN}>
    DEEPEP_BENCH_EARLY_ISSUE=$<BOOL:${DEEPEP_LL_COMBINE_EARLY_ISSUE}>
    DEEPEP_BENCH_PROGRESS_PROBE=$<BOOL:${DEEPEP_LL_COMBINE_PROGRESS_PROBE}>
    DEEPEP_BENCH_SIMT_META=$<BOOL:${DEEPEP_LL_COMBINE_SIMT_META}>
    DEEPEP_BENCH_SIMT_FLAG=$<BOOL:${DEEPEP_LL_COMBINE_SIMT_FLAG}>
    DEEPEP_BENCH_PREFETCH_RETURN=$<BOOL:${DEEPEP_LL_COMBINE_PREFETCH_RETURN}>
    DEEPEP_BENCH_LOCAL_FIRST=$<BOOL:${DEEPEP_LL_COMBINE_LOCAL_FIRST}>
    DEEPEP_BENCH_BATCH_FINAL=$<BOOL:${DEEPEP_LL_COMBINE_BATCH_FINAL}>
    DEEPEP_BENCH_POLL_NOP=${DEEPEP_LL_COMBINE_POLL_NOP})
# Hash the compiled source tree, including uncommitted edits, for cross-rank checks.
file(GLOB_RECURSE _identity_files CONFIGURE_DEPENDS
    "${DEEPEP_SOURCE_DIR}/csrc/*.h" "${DEEPEP_SOURCE_DIR}/csrc/*.hpp" "${DEEPEP_SOURCE_DIR}/csrc/*.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/*.h" "${CMAKE_CURRENT_LIST_DIR}/*.hpp" "${CMAKE_CURRENT_LIST_DIR}/*.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/../ll_python_bench/*.hpp")
list(SORT _identity_files)
set(_source_hashes "${LL_OPS_REV}")
foreach(_file IN LISTS _identity_files)
    file(SHA256 "${_file}" _hash)
    string(APPEND _source_hashes "${_hash}")
endforeach()
string(SHA256 _source_id "${_source_hashes}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_identity_files})
target_compile_definitions(ll_bench_variant PRIVATE DEEPEP_BENCH_SOURCE_ID="${_source_id}")
add_executable(ll_combine_bench runner.cpp "${DEEPEP_SOURCE_DIR}/csrc/runtime/session.cpp")
target_include_directories(ll_combine_bench PRIVATE "${DEEPEP_SOURCE_DIR}/csrc"
    "${DEEPEP_PLATFORM_INCLUDE}" "${ASCEND_HOME_PATH}/include" "${SHMEM_ROOT}/include")
target_link_directories(ll_combine_bench PRIVATE "${ASCEND_HOME_PATH}/lib64"
    "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64")
target_link_libraries(ll_combine_bench PRIVATE MPI::MPI_CXX ascendcl shmem
    tiling_api platform unified_dlog ${CMAKE_DL_LIBS} pthread)
target_compile_definitions(ll_combine_bench PRIVATE DEEPEP_LL_RDMA=1 DEEPEP_LL_METADATA_CORES=4)
add_dependencies(ll_combine_bench ll_bench_variant)
option(DEEPEP_LL_PYTHON_BENCH "Build the PyTorch-controlled benchmark extension" OFF)
if(DEEPEP_LL_PYTHON_BENCH)
    if(DEEPEP_LL_OPS_BASE OR DEEPEP_LL_COMBINE_PROFILE)
        message(FATAL_ERROR "Python timing currently uses clean DeepEP variants")
    endif()
    find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(_ll_py_combine main.cpp "${DEEPEP_SOURCE_DIR}/csrc/runtime/session.cpp")
    # Same case generator, Host options and Device library as the C++ variant.
    foreach(_prop INCLUDE_DIRECTORIES LINK_DIRECTORIES COMPILE_DEFINITIONS)
        get_target_property(_value ll_bench_variant ${_prop})
        set_property(TARGET _ll_py_combine PROPERTY ${_prop} "${_value}")
    endforeach()
    get_target_property(_libs ll_bench_variant LINK_LIBRARIES)
    target_link_libraries(_ll_py_combine PRIVATE ${_libs})
    target_compile_definitions(_ll_py_combine PRIVATE LL_BENCH_PYTHON=1)
    if(DEEPEP_LL_COMBINE_ORIGINAL)
        target_sources(_ll_py_combine PRIVATE baseline_queries.cpp)
    endif()
    target_link_options(_ll_py_combine PRIVATE -Wl,-Bsymbolic)
    set(DEEPEP_LL_PYTHON_VARIANT "base" CACHE STRING "Installed Python benchmark variant")
    set(_python_destination "${Python_SITEARCH}/deepep_ll_bench/${DEEPEP_LL_PYTHON_VARIANT}")
    # The installed extension uses its colocated kernel, not the build-tree copy.
    set_target_properties(_ll_py_combine PROPERTIES INSTALL_RPATH "$ORIGIN")
    install(TARGETS _ll_py_combine LIBRARY DESTINATION "${_python_destination}" COMPONENT ll_python)
    install(FILES "${DEEPEP_DEVICE_LIBRARY}" "${CMAKE_BINARY_DIR}/bench-build.txt"
        DESTINATION "${_python_destination}" COMPONENT ll_python)
endif()
if(DEEPEP_LL_OPS_BASE)
    file(WRITE "${CMAKE_BINARY_DIR}/ops-baseline.txt"
        "implementation=ops-shmem-base\nupstream=${LL_OPS_REV}\ntransport=SHMEM QP Put; quiet at upstream batch boundary\n")
endif()
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/bench-build.txt" CONTENT
    "source=${_source_id}\noriginal=${DEEPEP_LL_COMBINE_ORIGINAL}\nbaseline_revision=${LL_COMBINE_BASELINE_REV}\ncombine_source=${DEEPEP_LL_COMBINE_SOURCE}\nsmall_batch=${DEEPEP_LL_COMBINE_SMALL_BATCH}\nrecv_cores=${DEEPEP_LL_COMBINE_RECV_CORES}\nearly_reduce=${DEEPEP_LL_COMBINE_EARLY_REDUCE}\nremote_first=${DEEPEP_LL_COMBINE_REMOTE_FIRST}\ndefer_return=${DEEPEP_LL_COMBINE_DEFER_RETURN}\nearly_issue=${DEEPEP_LL_COMBINE_EARLY_ISSUE}\nprogress_probe=${DEEPEP_LL_COMBINE_PROGRESS_PROBE}\nsimt_meta=${DEEPEP_LL_COMBINE_SIMT_META}\nsimt_flag=${DEEPEP_LL_COMBINE_SIMT_FLAG}\nlocal_first=${DEEPEP_LL_COMBINE_LOCAL_FIRST}\nbatch_final=${DEEPEP_LL_COMBINE_BATCH_FINAL}\npoll_nop=${DEEPEP_LL_COMBINE_POLL_NOP}\nprefetch_return=${DEEPEP_LL_COMBINE_PREFETCH_RETURN}\nready=${DEEPEP_LL_COMBINE_READY}\nnotify=${DEEPEP_LL_COMBINE_NOTIFY}\ncoalesce=${DEEPEP_LL_COMBINE_COALESCE}\ncoalesce_rows=${DEEPEP_LL_COMBINE_COALESCE_ROWS}\nranks_per_host=${DEEPEP_LL_PHYSICAL_RANKS_PER_HOST}\nprofile=${DEEPEP_LL_COMBINE_PROFILE}\nmpi=${MPI_CXX_COMPILER}\nshmem=${SHMEM_ROOT}\n")
