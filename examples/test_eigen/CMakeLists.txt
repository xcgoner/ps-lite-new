project(test_eigen)

include_directories(
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

find_package(Threads REQUIRED)

file(GLOB SOURCE_FILES test_eigen.cc)
add_pslite_executable(test_eigen ${SOURCE_FILES})
requires_eigen(test_eigen)
TARGET_LINK_LIBRARIES ( test_eigen ${CMAKE_THREAD_LIBS_INIT} )
