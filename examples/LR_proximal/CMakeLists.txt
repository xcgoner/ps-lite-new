project(LR_proximal)

include_directories(
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

set( LR_APP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/main" )

file(GLOB SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc")
set(MAIN_FILES "${LR_APP_DIR}/main.cc" ${SOURCE_FILES})
add_pslite_executable(lrprox ${MAIN_FILES})
add_pslite_executable(test_LR "${LR_APP_DIR}/test_LR.cc" ${SOURCE_FILES})
add_pslite_executable(test_util "${LR_APP_DIR}/test_util.cc")
add_pslite_executable(test_reader "${LR_APP_DIR}/test_reader.cc")

find_package(Threads REQUIRED)
requires_eigen(lrprox)
requires_eigen(test_LR)
TARGET_LINK_LIBRARIES ( lrprox ${CMAKE_THREAD_LIBS_INIT} )

