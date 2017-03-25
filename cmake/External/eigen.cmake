# build directory
set(EIGEN_PREFIX ${CMAKE_BINARY_DIR}/external/eigen-prefix)
# install directory
set(EIGEN_INSTALL ${CMAKE_BINARY_DIR}/external/eigen-install)

set( Eigen3_VERSION "3.3.3" )

add_definitions(-DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE)
find_package(BLAS REQUIRED)
ExternalProject_Add(eigen
    PREFIX ${EIGEN_PREFIX}
    URL "http://bitbucket.org/eigen/eigen/get/${Eigen3_VERSION}.tar.bz2"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    BUILD_IN_SOURCE 1
    INSTALL_COMMAND cp -r Eigen unsupported <INSTALL_DIR>/
    INSTALL_DIR ${EIGEN_INSTALL})
add_definitions(-DHAS_EIGEN)

set(EIGEN_INCLUDE_DIRS ${EIGEN_INSTALL})

macro(requires_eigen NAME)
  include_directories(${NAME} ${EIGEN_INCLUDE_DIRS})
  target_link_libraries(${NAME} ${BLAS_LIBRARIES})
  add_dependencies(${NAME} eigen)
endmacro(requires_eigen)