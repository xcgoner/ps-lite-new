project(test_eigen)

include_directories(
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
file(GLOB SOURCE_FILES test_eigen.cc)
add_pslite_executable(test_eigen ${SOURCE_FILES})
requires_eigen(test_eigen)
