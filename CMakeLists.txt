cmake_minimum_required(VERSION 2.8.7)

project(pslite C CXX)

if(NOT MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

include(ExternalProject) 
set(pslite_LINKER_LIBS_L "" )
set(pslite_INCLUDE_DIR_L "" )

# ---[ zmq
include("cmake/External/zmq.cmake")
include_directories(pslite ${ZMQ_INCLUDE_DIRS})
list(APPEND pslite_LINKER_LIBS_L ${ZMQ_LIBRARIES})

# ---[ eigen
include("cmake/External/eigen.cmake")

# ---[ Google-protobuf
include(cmake/ProtoBuf.cmake)

# generate protobuf sources
set(proto_gen_folder "${PROJECT_BINARY_DIR}/src")
file(GLOB_RECURSE proto_files "src/*.proto")
pslite_protobuf_generate_cpp_py(${proto_gen_folder} proto_srcs proto_hdrs proto_python "${PROJECT_SOURCE_DIR}" "src" ${proto_files})
include_directories(pslite "${PROJECT_SOURCE_DIR}/include/")
include_directories(pslite "${PROJECT_BINARY_DIR}/include/")
include_directories(pslite "${PROJECT_BINARY_DIR}/src/")
list(APPEND pslite_INCLUDE_DIR_L "${PROJECT_BINARY_DIR}/include/")

#FILE(COPY DIRECTORY "${PROJECT_BINARY_DIR}/include" DESTINATION "${PROJECT_SOURCE_DIR}/" FILES_MATCHING PATTERN "*.pb.h")
FILE(GLOB SOURCE "src/*.cc")

if(MSVC)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
FILE(GLOB getopt_SOURCE "src/windows/getopt.c")
list(APPEND SOURCE ${getopt_SOURCE}) 
add_definitions(-DSTATIC_GETOPT)
include_directories(pslite "${PROJECT_SOURCE_DIR}/src/windows")
list(APPEND pslite_LINKER_LIBS_L "ipHlpApi.lib" "ws2_32.lib")
  foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
    if(${flag_var} MATCHES "/MD")
      string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
    endif(${flag_var} MATCHES "/MD")
  endforeach(flag_var)
endif()

list(APPEND SOURCE ${proto_srcs})
add_library(pslite ${SOURCE})

target_link_libraries(pslite ${pslite_LINKER_LIBS})
list(APPEND pslite_LINKER_LIBS_L ${PROTOBUF_LIBRARIES})
list(APPEND pslite_INCLUDE_DIR_L "${PROJECT_SOURCE_DIR}/include")
list(APPEND pslite_INCLUDE_DIR_L ${PROTOBUF_INCLUDE_DIR})

set(pslite_LINKER_LIBS ${pslite_LINKER_LIBS_L} PARENT_SCOPE)
set(pslite_INCLUDE_DIR ${pslite_INCLUDE_DIR_L} PARENT_SCOPE)

macro(requires_core_deps NAME)
  target_link_libraries(${NAME}
          ${ZMQ_LIBRARIES})
  add_dependencies(${NAME} ZMQ)
endmacro(requires_core_deps)

macro(add_pslite_executable NAME)
  message("add_executable(${NAME} ${ARGN})")
  add_executable(${NAME} ${ARGN})
  target_link_libraries(${NAME} pslite)
  add_dependencies(${NAME} pslite)
  requires_core_deps(${NAME})
endmacro(add_pslite_executable)

subdirs(examples)