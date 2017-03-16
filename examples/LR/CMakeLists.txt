project(LR)

include_directories(
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
file(GLOB SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc")
add_pslite_executable(distlr ${SOURCE_FILES})
