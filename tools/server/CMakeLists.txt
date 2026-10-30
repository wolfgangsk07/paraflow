include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

# server-context containing the core server logic, used by llama-server and CLI

set(TARGET server-context)

add_library(${TARGET} STATIC
    server-chat.cpp
    server-chat.h
    server-task.cpp
    server-task.h
    server-queue.cpp
    server-queue.h
    server-common.cpp
    server-common.h
    server-decision.cpp
    server-decision.h
    server-context.cpp
    server-context.h
    server-stream.cpp
    server-stream.h
    server-tools.cpp
    server-tools.h
    server-mcp.cpp
    server-mcp.h
    server-schema.cpp
    server-schema.h
)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(${TARGET} PRIVATE ../mtmd)
target_include_directories(${TARGET} PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC llama-common mtmd ${CMAKE_THREAD_LIBS_INIT})

# Optional, process-local concurrent CUDA activity timing. The default build has
# no CUPTI/driver dependency and retains the same server/CLI interface.
option(LLAMA_SERVER_CUPTI_ACTIVITY "Collect process-local CUDA kernel/copy timing" OFF)
if (LLAMA_SERVER_CUPTI_ACTIVITY)
    find_package(CUDAToolkit REQUIRED)
    if (CUDAToolkit_VERSION VERSION_LESS 12.6 OR NOT CUDAToolkit_VERSION VERSION_LESS 13.0)
        message(FATAL_ERROR "CUPTI activity record ABI is currently verified for CUDA 12.6-12.x")
    endif()
    find_path(LLAMA_CUPTI_INCLUDE_DIR cupti.h
        HINTS "${CUDAToolkit_LIBRARY_ROOT}/extras/CUPTI/include"
              "${CUDAToolkit_ROOT}/extras/CUPTI/include"
              "/usr/local/cuda/extras/CUPTI/include")
    if (NOT LLAMA_CUPTI_INCLUDE_DIR)
        message(FATAL_ERROR "LLAMA_SERVER_CUPTI_ACTIVITY requires CUDA CUPTI headers")
    endif()
    target_sources(${TARGET} PRIVATE gpu-activity.cpp gpu-activity.h activity-intervals.h)
    target_include_directories(${TARGET} PRIVATE ${LLAMA_CUPTI_INCLUDE_DIR} ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(${TARGET} PRIVATE ${CMAKE_DL_LIBS})
else()
    target_sources(${TARGET} PRIVATE gpu-activity-stub.cpp gpu-activity.h)
endif()


# llama-server-impl: server logic, reusable by app

set(TARGET llama-server-impl)

add_library(${TARGET}
    server.cpp
    server-http.cpp
    server-http.h
    server-models.cpp
    server-models.h
)
set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${TARGET} PRIVATE ../mtmd ${PROJECT_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC server-context llama-ui cpp-httplib ${CMAKE_THREAD_LIBS_INIT})

add_dependencies(${TARGET} llama-ui-assets)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} LIBRARY)
endif()

# llama-server executable

set(TARGET llama-server)

add_executable(${TARGET} main.cpp)
install(TARGETS ${TARGET} RUNTIME)

target_link_libraries(${TARGET} PRIVATE llama-server-impl)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
