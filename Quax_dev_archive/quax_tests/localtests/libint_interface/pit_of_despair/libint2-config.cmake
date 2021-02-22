# - CMAKE Config file for the Libint2 package
# This will define the following CMake cache variables
#
#    LIBINT2_FOUND           - true if libint2.h header and libint2 library were found
#    LIBINT2_VERSION         - the libint2 version
#    LIBINT2_EXT_VERSION     - the libint2 version including the (optional) buildid, such as beta.3
#    LIBINT2_INCLUDE_DIRS    - (deprecated: use the CMake IMPORTED targets listed below) list of libint2 include directories
#    LIBINT2_LIBRARIES       - (deprecated: use the CMake IMPORTED targets listed below) list of libint2 libraries
#
# and the following imported targets
#
#     Libint2::int2          - library only
#     Libint2::cxx           - (if Eigen + Boost was found at the library configure time) library + C++11 API
#
# Author: Eduard Valeyev - libint@valeyev.net

# Set package version
set(LIBINT2_VERSION "2.7.0")
set(LIBINT2_EXT_VERSION "2.7.0-beta.6")


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was libint2-config.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

set(LIBINT2_LIBRARIES Libint2::int2)
set(LIBINT2_INCLUDE_DIRS "${PACKAGE_PREFIX_DIR}/include")

# Import library targets
if(NOT TARGET Libint2::libint2)
  include("${CMAKE_CURRENT_LIST_DIR}/libint2-targets.cmake")
  if(NOT TARGET Libint2::libint2)
    message(FATAL_ERROR "expected Libint2::libint2 among imported Libint2 targets")
  endif()
endif()

# Need Threads::Threads
if (NOT TARGET Threads::Threads)
  find_package(Threads QUIET REQUIRED)
endif(NOT TARGET Threads::Threads)

# this aliases _target_name (if defined) to _alias_name
# this also sets IMPORTED_GLOBAL on _target_name to true
macro(alias_target _alias_name _target_name)
  if (TARGET ${_target_name} AND NOT TARGET ${_alias_name})
    get_property(${_target_name}_is_global_set TARGET ${_target_name} PROPERTY IMPORTED_GLOBAL SET)
    if (${_target_name}_is_global_set)
      get_property(${_target_name}_is_global TARGET ${_target_name} PROPERTY IMPORTED_GLOBAL)
      if (NOT ${_target_name}_is_global)
        set_property(TARGET ${_target_name} PROPERTY IMPORTED_GLOBAL TRUE)
      endif(NOT ${_target_name}_is_global)
    endif(${_target_name}_is_global_set)
    add_library(${_alias_name} ALIAS ${_target_name})
  endif(TARGET ${_target_name} AND NOT TARGET ${_alias_name})
endmacro(alias_target)

# alias new target names to old namespaced targets
alias_target(Libint2::int2 Libint2::libint2)
alias_target(Libint2::int2-static Libint2::libint2-static)
alias_target(Libint2::int2_static Libint2::libint2-static)
alias_target(Libint2::Eigen Libint2::libint2_Eigen)
alias_target(Libint2::cxx Libint2::libint2_cxx)

set(LIBINT2_FOUND TRUE)
