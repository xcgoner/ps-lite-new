project(LR_proximal)

include_directories(
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

set( LR_APP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/main" )

file(GLOB SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc")
#set(MAIN_FILES "${LR_APP_DIR}/main.cc" ${SOURCE_FILES})
add_pslite_executable(lrprox "${LR_APP_DIR}/lr_prox.cc" ${SOURCE_FILES})
add_pslite_executable(softmaxprox "${LR_APP_DIR}/softmax_prox.cc" ${SOURCE_FILES})
add_pslite_executable(test_LR "${LR_APP_DIR}/test_LR.cc" ${SOURCE_FILES})
add_pslite_executable(test_softmax "${LR_APP_DIR}/test_softmax.cc" ${SOURCE_FILES})
add_pslite_executable(test_util "${LR_APP_DIR}/test_util.cc")
add_pslite_executable(test_reader "${LR_APP_DIR}/test_reader.cc")

find_package(Threads REQUIRED)
requires_eigen(lrprox)
requires_eigen(softmaxprox)
requires_eigen(test_LR)
requires_eigen(test_softmax)
TARGET_LINK_LIBRARIES ( lrprox ${CMAKE_THREAD_LIBS_INIT} )
TARGET_LINK_LIBRARIES ( softmaxprox ${CMAKE_THREAD_LIBS_INIT} )

