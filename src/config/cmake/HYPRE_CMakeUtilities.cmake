# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

# Function to set hypre build options
function(set_hypre_option category name description default_value)
  # Detect if the user explicitly set this option via -D on the command line
  if(DEFINED CACHE{${name}})
    set(HYPRE_USER_SET_${name} ON CACHE INTERNAL "User explicitly set ${name}")
  else()
    set(HYPRE_USER_SET_${name} OFF CACHE INTERNAL "User explicitly set ${name}")
  endif()

  option(${name} "${description}" ${default_value})
  if (${category} STREQUAL "CUDA" OR ${category} STREQUAL "HIP" OR ${category} STREQUAL "SYCL")
    if (HYPRE_ENABLE_${category} STREQUAL "ON")
      set(GPU_OPTIONS ${GPU_OPTIONS} ${name} PARENT_SCOPE)
    endif()
  else()
    set(${category}_OPTIONS ${${category}_OPTIONS} ${name} PARENT_SCOPE)
  endif()
endfunction()

# Function to set internal hypre build options
function(set_internal_hypre_option var_prefix var_name)
  if(HYPRE_ENABLE_${var_name})
    if(var_prefix STREQUAL "")
      set(HYPRE_${var_name} ON CACHE INTERNAL "")
    else()
      set(HYPRE_${var_prefix}_${var_name} ON CACHE INTERNAL "")
    endif()
  endif()
endfunction()

# Function to setup git version info
function(setup_git_version_info HYPRE_GIT_DIR)
  set(GIT_VERSION_FOUND FALSE PARENT_SCOPE)
  if (EXISTS "${HYPRE_GIT_DIR}")
    execute_process(COMMAND git -C ${HYPRE_GIT_DIR} describe --match v* --long --abbrev=9 --always
                    OUTPUT_VARIABLE develop_string
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    RESULT_VARIABLE git_result)
    if (git_result EQUAL 0)
      set(GIT_VERSION_FOUND TRUE PARENT_SCOPE)
      execute_process(COMMAND git -C ${HYPRE_GIT_DIR} describe --match v* --abbrev=0 --always
                      OUTPUT_VARIABLE develop_lastag
                      OUTPUT_STRIP_TRAILING_WHITESPACE)
      execute_process(COMMAND git -C ${HYPRE_GIT_DIR} rev-list --count ${develop_lastag}..HEAD
                      OUTPUT_VARIABLE develop_number
                      OUTPUT_STRIP_TRAILING_WHITESPACE)
      execute_process(COMMAND git -C ${HYPRE_GIT_DIR} rev-parse --abbrev-ref HEAD
                      OUTPUT_VARIABLE develop_branch
                      OUTPUT_STRIP_TRAILING_WHITESPACE)
      set(HYPRE_DEVELOP_STRING  ${develop_string} PARENT_SCOPE)
      set(HYPRE_DEVELOP_NUMBER  ${develop_number} PARENT_SCOPE)
      set(HYPRE_BRANCH_NAME     ${develop_branch} PARENT_SCOPE)
      if (develop_branch MATCHES "master")
        set(HYPRE_DEVELOP_BRANCH  ${develop_branch} PARENT_SCOPE)
      endif ()
    endif()
  endif()
endfunction()

# Function to check if two options have the same value
function(ensure_options_match option1 option2)
  if(DEFINED CACHE{${option1}} AND DEFINED CACHE{${option2}})
    #if ((${option1} AND NOT ${option2}) OR (NOT ${option1} AND ${option2}))
    if(NOT (${option1} STREQUAL ${option2}) AND NOT (${option1} STREQUAL "OFF" AND ${option2} STREQUAL "OFF"))
      # Save the value of the conflicting options
      set(saved_value1 "${${option1}}")
      set(saved_value2 "${${option2}}")

      # Unset conflicting options
      unset(${option1} CACHE)
      unset(${option2} CACHE)
      message(FATAL_ERROR "Incompatible options: ${option1} (${saved_value1}) and ${option2} (${saved_value2}) must have the same value. Unsetting both options.")
    endif()
  endif()
endfunction()

# Function to check if two options have different values
function(ensure_options_differ option1 option2)
  if(DEFINED HYPRE_ENABLE_${option1} AND DEFINED HYPRE_ENABLE_${option2})
    if(HYPRE_ENABLE_${option1} AND ${HYPRE_ENABLE_${option2}})
      # Save the value of the conflicting options
      set(saved_value1 "${HYPRE_ENABLE_${option1}}")
      set(saved_value2 "${HYPRE_ENABLE_${option2}}")

      # Unset conflicting options
      unset(HYPRE_ENABLE_${option1} CACHE)
      unset(HYPRE_ENABLE_${option2} CACHE)

      message(FATAL_ERROR "Error: HYPRE_ENABLE_${option1} (${saved_value1}) and HYPRE_ENABLE_${option2} (${saved_value2}) are mutually exclusive. Only one can be set to ON. Unsetting both options.")
    endif()
  endif()
endfunction()

# Helper function to process Fortran mangling scheme
function(process_fmangling_scheme varname description)
  set(mangling_map
    UNSPECIFIED 0
    NONE 1
    ONE_UNDERSCORE 2
    TWO_UNDERSCORES 3
    CAPS 4
    PRE_POST_UNDERSCORE 5
  )
  list(LENGTH mangling_map map_length)
  math(EXPR last_index "${map_length} - 1")

  # Check if varname is a numeric value
  if (HYPRE_ENABLE_${varname} MATCHES "^[0-9]+$")
    foreach(i RANGE 0 ${last_index} 2)
      math(EXPR next_index "${i} + 1")
      list(GET mangling_map ${next_index} value)
      if (HYPRE_ENABLE_${varname} STREQUAL ${value})
        list(GET mangling_map ${i} key)
        message(STATUS "HYPRE_ENABLE_${varname} corresponds to Fortran ${description} mangling scheme: ${key}")
        set(HYPRE_${varname} ${value} CACHE INTERNAL "Set the Fortran ${description} mangling scheme")
        return()
      endif()
    endforeach()
  endif()

  # Check if varname matches any string key
  foreach(i RANGE 0 ${last_index} 2)
    list(GET mangling_map ${i} key)
    math(EXPR next_index "${i} + 1")
    list(GET mangling_map ${next_index} value)
    if (HYPRE_ENABLE_${varname} MATCHES ${key})
      if (NOT HYPRE_ENABLE_${varname} MATCHES "UNSPECIFIED")
        message(STATUS "Using Fortran ${description} mangling scheme: ${key}")
      endif()
      set(HYPRE_${varname} ${value} CACHE INTERNAL "Set the Fortran ${description} mangling scheme")
      return()
    endif()
  endforeach()

  # Default case
  message(STATUS "Unknown value for HYPRE_ENABLE_${varname}. Defaulting to UNSPECIFIED (0)")
  set(HYPRE_${varname} 0 CACHE INTERNAL "Set the Fortran ${description} mangling scheme")
endfunction()

# Function to configure MPI target
function(configure_mpi_target)
  find_package(MPI REQUIRED)
  target_link_libraries(${PROJECT_NAME} PUBLIC MPI::MPI_C)

  # Determine the correct MPI include directory
  if(MPI_CXX_INCLUDE_DIR)
    set(MPI_INCLUDE_DIR ${MPI_CXX_INCLUDE_DIR})
  elseif(MPI_CXX_INCLUDE_PATH)
    set(MPI_INCLUDE_DIR ${MPI_CXX_INCLUDE_PATH})
  elseif(MPI_CXX_COMPILER_INCLUDE_DIR)
    set(MPI_INCLUDE_DIR ${MPI_CXX_COMPILER_INCLUDE_DIR})
  elseif(MPI_C_COMPILER_INCLUDE_DIR)
    set(MPI_INCLUDE_DIR ${MPI_C_COMPILER_INCLUDE_DIR})
  elseif(MPI_C_INCLUDE_DIR)
    set(MPI_INCLUDE_DIR ${MPI_C_INCLUDE_DIR})
  elseif(MPI_C_INCLUDE_PATH)
    set(MPI_INCLUDE_DIR ${MPI_C_INCLUDE_PATH})
  elseif(MPI_INCLUDE_PATH)
    set(MPI_INCLUDE_DIR ${MPI_INCLUDE_PATH})
  elseif(MPICH_DIR)
    set(MPI_INCLUDE_DIR ${MPICH_DIR}/include)
  elseif(DEFINED ENV{MPICH_DIR})
    set(MPI_INCLUDE_DIR $ENV{MPICH_DIR}/include)
  else()
    message(WARNING "MPI include directory not found. Please specify -DMPI_INCLUDE_DIR or the compilation may fail.")
  endif()

  if (HYPRE_ENABLE_CUDA OR HYPRE_ENABLE_HIP OR HYPRE_ENABLE_SYCL)
    message(STATUS "Adding MPI include directory: ${MPI_INCLUDE_DIR}")
    target_include_directories(${PROJECT_NAME} PUBLIC ${MPI_INCLUDE_DIR})
  endif ()
  message(STATUS "MPI execution command: ${MPIEXEC_EXECUTABLE}")

  set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES} MPI::MPI_C)

  # Check if MPI supports the MPI_Comm_f2c function
  include(CheckCSourceCompiles)
  check_c_source_compiles("
    #include <mpi.h>
    int main() {
      MPI_Comm c = MPI_Comm_f2c(0);
      return 0;
    }
  " HYPRE_HAVE_MPI_COMM_F2C)

  # Define a pattern for LTO-related flags (compiler-specific)
  set(LTO_FLAG_PATTERNS ".*lto.*" ".*ipo.*" ".*-fthinlto.*" ".*fat-lto-objects.*")

  # Remove LTO-related flags from MPI target properties if applicable
  foreach (mpi_target MPI::MPI_C MPI::MPI_CXX)
    if (TARGET ${mpi_target})
      get_target_property(target_options ${mpi_target} INTERFACE_COMPILE_OPTIONS)
      if (target_options)
        #message(STATUS "target_options: ${target_options}")
        set(original_options "${target_options}") # Save for comparison
        list(APPEND target_options) # Ensure it's treated as a list

        # Remove matching flags
        set(removed_flags "")
        list(APPEND removed_flags)
        foreach (pattern IN LISTS LTO_FLAG_PATTERNS)
          foreach (flag IN LISTS target_options)
            if("${flag}" MATCHES "${pattern}")
              list(REMOVE_ITEM target_options "${flag}")
              list(APPEND removed_flags "${flag}")
            endif()
          endforeach()
        endforeach()
        #message(STATUS "removed_flags: ${removed_flags}")
        list(LENGTH removed_flags removed_flags_length)
        if (removed_flags_length GREATER 0)
          set(target_options "${target_options}" CACHE STRING "Updated ${target_options} without LTO-related flags" FORCE)
          set_target_properties(${mpi_target} PROPERTIES INTERFACE_COMPILE_OPTIONS "${target_options}")
          message(STATUS "Removed LTO-related flags from ${mpi_target}: ${removed_flags}")
        endif()
      endif()
    endif()
  endforeach()
endfunction()

# Function to get dependency library version
function(get_library_version LIBNAME)
  if(TARGET ${LIBNAME}::${LIBNAME})
    get_target_property(LIB_VERSION ${LIBNAME}::${LIBNAME} VERSION)
  endif()
  if(NOT LIB_VERSION)
    if(DEFINED ${LIBNAME}_VERSION)
      set(LIB_VERSION "${${LIBNAME}_VERSION}")
    elseif(DEFINED ${LIBNAME}_VERSION_STRING)
      set(LIB_VERSION "${${LIBNAME}_VERSION_STRING}")
    elseif(DEFINED ${LIBNAME}_VERSION_MAJOR AND DEFINED ${LIBNAME}_VERSION_MINOR)
      set(LIB_VERSION "${${LIBNAME}_VERSION_MAJOR}.${${LIBNAME}_VERSION_MINOR}")
      if(DEFINED ${LIBNAME}_VERSION_PATCH)
        set(LIB_VERSION "${LIB_VERSION}.${${LIBNAME}_VERSION_PATCH}")
      endif()
    endif()
  endif()
  if(LIB_VERSION)
    message(STATUS "  ${LIBNAME} version: ${LIB_VERSION}")
  else()
    message(STATUS "  ${LIBNAME} version: unknown")
  endif()
endfunction()

# Macro to extract language flags
macro(get_language_flags in_var out_var lang_type)
  string(REGEX MATCHALL "\\$<\\$<COMPILE_LANGUAGE:${lang_type}>:([^>]*)>" matches "${in_var}")
  if(matches)
    string(REGEX REPLACE "\\$<\\$<COMPILE_LANGUAGE:${lang_type}>:" "" temp "${matches}")
    string(REGEX REPLACE ">" "" temp "${temp}")
    set(${out_var} ${temp})
  else()
    set(${out_var} "")
  endif()
endmacro()

# Function to print the INTERFACE properties of a target in a readable format.
function(pretty_print_target_interface TARGET_NAME)
    message(STATUS "--- INTERFACE properties for target: [${TARGET_NAME}] ---")

    # Loop over all property names passed to the function (stored in ARGN)
    foreach(PROP ${ARGN})
        get_target_property(PROP_VALUE ${TARGET_NAME} ${PROP})
        message(STATUS "  [${PROP}]") # Print the property name

        if(PROP_VALUE)
            # Loop over each item in the semicolon-separated list
            foreach(ITEM ${PROP_VALUE})
                message(STATUS "    - ${ITEM}") # Print each item indented
            endforeach()
        else()
            message(STATUS "    <NOTFOUND>")
        endif()
    endforeach()
    #message(STATUS "--- End of INTERFACE properties for [${TARGET_NAME}] ---")
endfunction()

# Function to handle TPL (Third-Party Library) setup
function(setup_tpl LIBNAME)
  string(TOUPPER ${LIBNAME} LIBNAME_UPPER)

  # Note we need to check for "USING" instead of "WITH" because
  # we want to allow for post-processing of build options via cmake
  if(HYPRE_USING_${LIBNAME_UPPER})
    # If the TPL was already added as a subproject, prefer using the existing target
    if(${LIBNAME_UPPER} STREQUAL "UMPIRE")
      # Check if we are auto-fetching Umpire
      maybe_build_umpire()

      if(TARGET umpire::umpire)
        # Link privately but propagate include directories so dependents see headers
        target_link_libraries(${PROJECT_NAME} PRIVATE umpire::umpire)
        get_target_property(_UMPIRE_INCLUDES umpire::umpire INTERFACE_INCLUDE_DIRECTORIES)
        if(_UMPIRE_INCLUDES)
          target_include_directories(${PROJECT_NAME} PUBLIC ${_UMPIRE_INCLUDES})
        endif()
        # Ensure C++ standard library is linked for non-MSVC toolchains
        if(UNIX)
          target_link_libraries(${PROJECT_NAME} PUBLIC stdc++)
        endif()
        fixup_umpire_cuda_runtime()
        message(STATUS "Found existing Umpire target: umpire::umpire")
        set(${LIBNAME_UPPER}_FOUND TRUE PARENT_SCOPE)
        set(HYPRE_NEEDS_CXX TRUE PARENT_SCOPE)
        message(STATUS "Enabled support for using ${LIBNAME_UPPER}")
        # Verify C interface headers are present
        check_umpire_c_interface()
        return()
      elseif(TARGET umpire)
        # Provide the standardized namespace alias if missing
        add_library(umpire::umpire ALIAS umpire)
        target_link_libraries(${PROJECT_NAME} PRIVATE umpire::umpire)
        get_target_property(_UMPIRE_INCLUDES umpire INTERFACE_INCLUDE_DIRECTORIES)
        if(_UMPIRE_INCLUDES)
          target_include_directories(${PROJECT_NAME} PUBLIC ${_UMPIRE_INCLUDES})
        endif()
        if(UNIX)
          target_link_libraries(${PROJECT_NAME} PUBLIC stdc++)
        endif()
        fixup_umpire_cuda_runtime()
        message(STATUS "Found existing Umpire target: umpire")
        set(${LIBNAME_UPPER}_FOUND TRUE PARENT_SCOPE)
        set(HYPRE_NEEDS_CXX TRUE PARENT_SCOPE)
        message(STATUS "Enabled support for using ${LIBNAME_UPPER}")
        # Verify C interface headers are present
        check_umpire_c_interface()
        return()
      endif()
    endif()
    if(TPL_${LIBNAME_UPPER}_LIBRARIES AND TPL_${LIBNAME_UPPER}_INCLUDE_DIRS)
      # Use specified TPL libraries and include dirs
      foreach(dir ${TPL_${LIBNAME_UPPER}_INCLUDE_DIRS})
        if(NOT EXISTS ${dir})
          message(FATAL_ERROR "${LIBNAME_UPPER} include directory not found: ${dir}")
        endif()
      endforeach()

      set(_tpl_libdirs)
      foreach(lib ${TPL_${LIBNAME_UPPER}_LIBRARIES})
        if(EXISTS ${lib})
          message(STATUS "${LIBNAME_UPPER} library found: ${lib}")
          get_filename_component(LIB_DIR "${lib}" DIRECTORY)
          set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY BUILD_RPATH "${LIB_DIR}")
          set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY INSTALL_RPATH "${LIB_DIR}")
          list(APPEND _tpl_libdirs "${LIB_DIR}")
        else()
          message(WARNING "${LIBNAME_UPPER} library not found at specified path: ${lib}")
        endif()
      endforeach()

      target_link_libraries(${PROJECT_NAME} PUBLIC ${TPL_${LIBNAME_UPPER}_LIBRARIES})
      target_include_directories(${PROJECT_NAME} PUBLIC ${TPL_${LIBNAME_UPPER}_INCLUDE_DIRS})
      # Record dependency dirs for export hints (include likely under <prefix>/include; libdir likely under <prefix>/lib)
      foreach(_d IN LISTS _tpl_libdirs TPL_${LIBNAME_UPPER}_INCLUDE_DIRS)
        if(_d)
          list(APPEND HYPRE_DEPENDENCY_DIRS "${_d}")
          # Also add the parent directory as a candidate install prefix
          get_filename_component(_parent "${_d}" DIRECTORY)
          list(APPEND HYPRE_DEPENDENCY_DIRS "${_parent}")
        endif()
      endforeach()
      list(REMOVE_DUPLICATES HYPRE_DEPENDENCY_DIRS)
      set(HYPRE_DEPENDENCY_DIRS "${HYPRE_DEPENDENCY_DIRS}" CACHE INTERNAL "" FORCE)
      if(${LIBNAME_UPPER} STREQUAL "UMPIRE")
        fixup_umpire_cuda_runtime()
        check_umpire_c_interface()
      endif()
    else()
      # Use find_package (prefer CONFIG). Provide clearer error for libraries when missing.
      find_package(${LIBNAME} CONFIG)
      if(${LIBNAME}_FOUND)
        list(APPEND HYPRE_DEPENDENCY_DIRS "${${LIBNAME}_ROOT}")
        set(HYPRE_DEPENDENCY_DIRS "${HYPRE_DEPENDENCY_DIRS}" CACHE INTERNAL "" FORCE)

        if(${LIBNAME} STREQUAL "caliper")
          set(HYPRE_NEEDS_CXX TRUE PARENT_SCOPE)
        elseif(${LIBNAME} STREQUAL "umpire")
          set(HYPRE_NEEDS_CXX TRUE PARENT_SCOPE)
        endif()

        if(TARGET ${LIBNAME}::${LIBNAME})
          if(${LIBNAME_UPPER} STREQUAL "UMPIRE")
            target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBNAME}::${LIBNAME})
            get_target_property(_UMPIRE_INCLUDES ${LIBNAME}::${LIBNAME} INTERFACE_INCLUDE_DIRECTORIES)
            if(_UMPIRE_INCLUDES)
              target_include_directories(${PROJECT_NAME} PUBLIC ${_UMPIRE_INCLUDES})
            endif()
            fixup_umpire_cuda_runtime()
          else()
            target_link_libraries(${PROJECT_NAME} PUBLIC ${LIBNAME}::${LIBNAME})
          endif()
          message(STATUS "Found ${LIBNAME} target: ${LIBNAME}::${LIBNAME}")
        elseif(TARGET ${LIBNAME})
          if(${LIBNAME_UPPER} STREQUAL "UMPIRE")
            target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBNAME})
            get_target_property(_UMPIRE_INCLUDES ${LIBNAME} INTERFACE_INCLUDE_DIRECTORIES)
            if(_UMPIRE_INCLUDES)
              target_include_directories(${PROJECT_NAME} PUBLIC ${_UMPIRE_INCLUDES})
            endif()
            fixup_umpire_cuda_runtime()
          else()
            target_link_libraries(${PROJECT_NAME} PUBLIC ${LIBNAME})
          endif()
          message(STATUS "Found ${LIBNAME} target: ${LIBNAME}")
        else()
          message(FATAL_ERROR "${LIBNAME} target not found. Please check your ${LIBNAME} installation")
        endif()
      else()
        # If a CMake package is not found, try pkg-config as a fallback
        set(_found_pkg FALSE)
        find_package(PkgConfig QUIET)
        if(PKG_CONFIG_FOUND)
          # If the user provided an install prefix (e.g., -DDSUPERLU_DIR=/prefix),
          # augment PKG_CONFIG_PATH with common subdirs so pkg-config can find the .pc
          set(_old_pkg_config_path "$ENV{PKG_CONFIG_PATH}")
          set(_pc_hint_dirs)
          foreach(_var IN ITEMS ${LIBNAME_UPPER}_DIR ${LIBNAME_UPPER}_ROOT TPL_${LIBNAME_UPPER}_DIR TPL_${LIBNAME_UPPER}_ROOT)
            if(DEFINED ${_var})
              set(_root "${${_var}}")
              if(EXISTS "${_root}")
                list(APPEND _pc_hint_dirs
                  "${_root}"
                  "${_root}/lib/pkgconfig"
                  "${_root}/lib64/pkgconfig"
                  "${_root}/share/pkgconfig")
              endif()
            endif()
          endforeach()
          if(_pc_hint_dirs)
            # Prepend hints to PKG_CONFIG_PATH for this configure step
            list(REMOVE_DUPLICATES _pc_hint_dirs)
            string(JOIN ":" _pc_hint_path ${_pc_hint_dirs})
            if(_old_pkg_config_path)
              set(ENV{PKG_CONFIG_PATH} "${_pc_hint_path}:$ENV{PKG_CONFIG_PATH}")
            else()
              set(ENV{PKG_CONFIG_PATH} "${_pc_hint_path}")
            endif()
            message(STATUS "Augmented PKG_CONFIG_PATH with hints for ${LIBNAME}: ${_pc_hint_path}")
          endif()
          # Allow user to override the pkg-config name
          set(_pc_names)
          if(TPL_${LIBNAME_UPPER}_PKGCONFIG_NAME)
            list(APPEND _pc_names "${TPL_${LIBNAME_UPPER}_PKGCONFIG_NAME}")
          endif()
          # Common sensible defaults
          string(TOLOWER ${LIBNAME} _lib_lower)
          list(APPEND _pc_names "${_lib_lower}")
          if(${LIBNAME_UPPER} STREQUAL "DSUPERLU")
            list(INSERT _pc_names 0 "superlu_dist")
          elseif(${LIBNAME_UPPER} STREQUAL "SUPERLU")
            list(INSERT _pc_names 0 "superlu")
          elseif(${LIBNAME_UPPER} STREQUAL "PARMETIS")
            list(INSERT _pc_names 0 "parmetis")
          elseif(${LIBNAME_UPPER} STREQUAL "METIS")
            list(INSERT _pc_names 0 "metis")
          endif()

          foreach(_pc IN LISTS _pc_names)
            if(NOT _pc)
              continue()
            endif()
            # Use a distinct prefix to avoid clobbering variables
            pkg_check_modules(PC_${LIBNAME_UPPER} QUIET IMPORTED_TARGET ${_pc})
            if(PC_${LIBNAME_UPPER}_FOUND)
              # Prefer linking the generated IMPORTED target when available
              if(TARGET PkgConfig::PC_${LIBNAME_UPPER})
                target_link_libraries(${PROJECT_NAME} PUBLIC PkgConfig::PC_${LIBNAME_UPPER})
              else()
                target_link_libraries(${PROJECT_NAME} PUBLIC ${PC_${LIBNAME_UPPER}_LINK_LIBRARIES})
              endif()
              if(PC_${LIBNAME_UPPER}_INCLUDE_DIRS)
                target_include_directories(${PROJECT_NAME} PUBLIC ${PC_${LIBNAME_UPPER}_INCLUDE_DIRS})
              endif()
              if(PC_${LIBNAME_UPPER}_LIBDIR)
                foreach(_dir IN LISTS PC_${LIBNAME_UPPER}_LIBDIR)
                  set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY BUILD_RPATH "${_dir}")
                  set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY INSTALL_RPATH "${_dir}")
                endforeach()
              endif()
              if(PC_${LIBNAME_UPPER}_VERSION)
                message(STATUS "Found ${LIBNAME} via pkg-config (${_pc}), version: ${PC_${LIBNAME_UPPER}_VERSION}")
              else()
                message(STATUS "Found ${LIBNAME} via pkg-config (${_pc})")
              endif()
              set(${LIBNAME_UPPER}_FOUND TRUE PARENT_SCOPE)
              set(_found_pkg TRUE)
              break()
            endif()
          endforeach()
          # Restore PKG_CONFIG_PATH if we modified it
          if(DEFINED _old_pkg_config_path)
            set(ENV{PKG_CONFIG_PATH} "${_old_pkg_config_path}")
          endif()
        endif()

        if(NOT _found_pkg)
          if(${LIBNAME_UPPER} STREQUAL "UMPIRE")
            message(FATAL_ERROR
              "===============================================================\n"
              "Umpire was requested but could not be found by CMake or pkg-config.\n"
              "Try one of the following options (in this order):\n"
              "  1) Auto-build Umpire (recommended):\n"
              "     -DHYPRE_BUILD_UMPIRE=ON\n\n"
              "  2) Provide a CMake package config for Umpire:\n"
              "     -Dumpire_ROOT=\"/path-to-umpire-install\"   (or)\n"
              "     -Dumpire_DIR=\"/path-to-umpire-install/lib/cmake/umpire\"\n\n"
              "  3) Provide explicit include and library paths:\n"
              "     -DTPL_UMPIRE_INCLUDE_DIRS=\"/path-to-umpire-install/include\"\n"
              "     -DTPL_UMPIRE_LIBRARIES=\"/path-to-umpire-install/lib/libumpire.so;...\"\n\n"
              "  4) Provide a pkg-config name:\n"
              "     -DTPL_UMPIRE_PKGCONFIG_NAME=umpire\n\n"
              "To opt out (not recommended for GPU builds), set:\n"
              "  -DHYPRE_ENABLE_UMPIRE=OFF\n"
              "==============================================================="
            )
          else()
            message(FATAL_ERROR "${LIBNAME_UPPER} not found via CMake package or pkg-config. Provide TPL_${LIBNAME_UPPER}_LIBRARIES/TPL_${LIBNAME_UPPER}_INCLUDE_DIRS or TPL_${LIBNAME_UPPER}_PKGCONFIG_NAME.")
          endif()
        endif()
        # When found via pkg-config, record dependency dirs for export hints
        if(_found_pkg)
          set(_dep_dirs)
          if(PC_${LIBNAME_UPPER}_LIBDIRS)
            list(APPEND _dep_dirs ${PC_${LIBNAME_UPPER}_LIBDIRS})
          elseif(PC_${LIBNAME_UPPER}_LIBDIR)
            list(APPEND _dep_dirs ${PC_${LIBNAME_UPPER}_LIBDIR})
          endif()
          if(PC_${LIBNAME_UPPER}_INCLUDE_DIRS)
            list(APPEND _dep_dirs ${PC_${LIBNAME_UPPER}_INCLUDE_DIRS})
          endif()
          foreach(_d IN LISTS _dep_dirs)
            if(_d)
              list(APPEND HYPRE_DEPENDENCY_DIRS "${_d}")
              get_filename_component(_parent "${_d}" DIRECTORY)
              list(APPEND HYPRE_DEPENDENCY_DIRS "${_parent}")
            endif()
          endforeach()
          list(REMOVE_DUPLICATES HYPRE_DEPENDENCY_DIRS)
          set(HYPRE_DEPENDENCY_DIRS "${HYPRE_DEPENDENCY_DIRS}" CACHE INTERNAL "" FORCE)
        endif()
      endif()

      # Display library info
      get_library_version(${LIBNAME})
      if(DEFINED ${LIBNAME}_DIR)
        message(STATUS "  Config directory: ${${LIBNAME}_DIR}")
      endif()
    endif()

    message(STATUS "Enabled support for using ${LIBNAME_UPPER}")

    if(${LIBNAME_UPPER} STREQUAL "SUPERLU" OR ${LIBNAME_UPPER} STREQUAL "DSUPERLU" OR ${LIBNAME_UPPER} STREQUAL "UMPIRE")
      target_link_libraries(${PROJECT_NAME} PUBLIC stdc++)
    endif()

    set(${LIBNAME_UPPER}_FOUND TRUE PARENT_SCOPE)

    # Run C interface check when Umpire is enabled and found via any path
    if(${LIBNAME_UPPER} STREQUAL "UMPIRE")
      check_umpire_c_interface()
    endif()
  endif()
endfunction()

# Function to setup TPL or internal library implementation
function(setup_tpl_or_internal LIB_NAME)
  string(TOUPPER ${LIB_NAME} LIB_NAME_UPPER)

  if(HYPRE_USING_HYPRE_${LIB_NAME_UPPER})
    # Use internal library
    add_subdirectory(${LIB_NAME})
    message(STATUS "Using internal ${LIB_NAME_UPPER}")
    target_include_directories(${PROJECT_NAME} PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/${LIB_NAME}>
      $<INSTALL_INTERFACE:include>
    )
  else()
    # Use external library
    if(TPL_${LIB_NAME_UPPER}_LIBRARIES)
      # Use specified TPL libraries
      message(STATUS "Enabled support for using external ${LIB_NAME_UPPER}.")
      foreach(lib ${TPL_${LIB_NAME_UPPER}_LIBRARIES})
        if(EXISTS ${lib})
          message(STATUS "${LIB_NAME_UPPER} library found: ${lib}")
        else()
          message(WARNING "${LIB_NAME_UPPER} library not found at specified path: ${lib}")
        endif()
      endforeach()
      target_link_libraries(${PROJECT_NAME} PUBLIC ${TPL_${LIB_NAME_UPPER}_LIBRARIES})
    else()
      # Find system library
      find_package(${LIB_NAME_UPPER} REQUIRED)
      if(${LIB_NAME_UPPER}_FOUND)
        message(STATUS "Using system ${LIB_NAME_UPPER}")
        if(TARGET ${LIB_NAME_UPPER}::${LIB_NAME_UPPER})
          target_link_libraries(${PROJECT_NAME} PUBLIC ${LIB_NAME_UPPER}::${LIB_NAME_UPPER})
        else()
          target_link_libraries(${PROJECT_NAME} PUBLIC ${${LIB_NAME_UPPER}_LIBRARIES})
        endif()
      else()
        message(FATAL_ERROR "${LIB_NAME_UPPER} not found")
      endif()
    endif()
  endif()
endfunction()

# Verify that Umpire provides the C interface headers by compiling a tiny C program
function(check_umpire_c_interface)
  if(NOT HYPRE_ENABLE_UMPIRE)
    return()
  endif()

  # Gather include directories for Umpire
  set(_umpire_includes)
  if(TARGET umpire::umpire)
    get_target_property(_umpire_includes umpire::umpire INTERFACE_INCLUDE_DIRECTORIES)
  elseif(TARGET umpire)
    get_target_property(_umpire_includes umpire INTERFACE_INCLUDE_DIRECTORIES)
  elseif(TPL_UMPIRE_INCLUDE_DIRS)
    set(_umpire_includes ${TPL_UMPIRE_INCLUDE_DIRS})
  endif()

  if(NOT _umpire_includes)
    # Try to locate the header path as a last resort
    find_path(_umpire_hdr_dir
      NAMES umpire/interface/c_fortran/umpire.h
      HINTS ${HYPRE_DEPENDENCY_DIRS}
    )
    if(_umpire_hdr_dir)
      list(APPEND _umpire_includes ${_umpire_hdr_dir})
    endif()

  endif()


  include(CheckCSourceCompiles)

  # Preserve and set required includes for the compile test
  set(_old_required_includes "${CMAKE_REQUIRED_INCLUDES}")
  set(CMAKE_REQUIRED_INCLUDES ${_umpire_includes})

  set(_code "#include \"umpire/interface/c_fortran/umpire.h\"\nint main(void) { umpire_resourcemanager rm; (void)rm; return 0; }")
  check_c_source_compiles("${_code}" UMPIRE_HAS_C_INTERFACE)

  # Restore CMAKE_REQUIRED_INCLUDES
  set(CMAKE_REQUIRED_INCLUDES "${_old_required_includes}")

  if(NOT UMPIRE_HAS_C_INTERFACE)
    message(FATAL_ERROR
      "Umpire does not appear to provide the C interface headers.\n"
      "Failed to compile a test including 'umpire/interface/c_fortran/umpire.h'.\n"
      "Ensure Umpire is built with its C interface enabled (e.g., -DUMPIRE_ENABLE_C=ON) and that headers are visible in the include path.\n"
      "For auto-build, try enabling the hypre build option -DHYPRE_BUILD_UMPIRE=ON.\n"
      "For manual Umpire builds, see https://hypre.readthedocs.io/en/latest/ch-misc.html#building-umpire\n")
  else()
    message(STATUS "Verified Umpire C interface headers are available.")
  endif()
endfunction()

# Fix up BLT/Umpire CUDA runtime linkage to use CUDA::cudart instead of legacy cuda_runtime
function(fixup_umpire_cuda_runtime)
  # Ensure BLT 'cuda_runtime' interface resolves to CUDA::cudart so shared links do not emit legacy -lcuda_runtime
  if(HYPRE_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    if(TARGET cuda_runtime)
      get_target_property(_iface cuda_runtime INTERFACE_LINK_LIBRARIES)
      if(_iface)
        set(_fixed_iface)
        foreach(_lib IN LISTS _iface)
          if(_lib STREQUAL "cuda_runtime")
            list(APPEND _fixed_iface CUDA::cudart)
          else()
            list(APPEND _fixed_iface ${_lib})
          endif()
        endforeach()
        set_target_properties(cuda_runtime PROPERTIES INTERFACE_LINK_LIBRARIES "${_fixed_iface}")
      else()
        target_link_libraries(cuda_runtime INTERFACE CUDA::cudart)
      endif()
      if(NOT TARGET blt::cuda_runtime)
        add_library(blt::cuda_runtime ALIAS cuda_runtime)
      endif()
    else()
      add_library(cuda_runtime INTERFACE IMPORTED)
      target_link_libraries(cuda_runtime INTERFACE CUDA::cudart)
      add_library(blt::cuda_runtime ALIAS cuda_runtime)
    endif()

    # Replace any legacy 'cuda_runtime' link items on umpire/camp targets with CUDA::cudart
    foreach(_tgt IN ITEMS camp umpire umpire_resource umpire_strategy umpire_op umpire_event umpire_util umpire_interface)
      if(TARGET ${_tgt})
        get_target_property(_ll ${_tgt} LINK_LIBRARIES)
        if(_ll)
          set(_new_ll)
          foreach(_l IN LISTS _ll)
            if(_l STREQUAL "cuda_runtime")
              list(APPEND _new_ll CUDA::cudart)
            else()
              list(APPEND _new_ll ${_l})
            endif()
          endforeach()
          set_target_properties(${_tgt} PROPERTIES LINK_LIBRARIES "${_new_ll}")
        endif()

        get_target_property(_ill ${_tgt} INTERFACE_LINK_LIBRARIES)
        if(_ill)
          set(_new_ill)
          foreach(_l IN LISTS _ill)
            if(_l STREQUAL "cuda_runtime")
              list(APPEND _new_ill CUDA::cudart)
            else()
              list(APPEND _new_ill ${_l})
            endif()
          endforeach()
          set_target_properties(${_tgt} PROPERTIES INTERFACE_LINK_LIBRARIES "${_new_ill}")
        endif()
      endif()
    endforeach()
  endif()
endfunction()

# Optionally fetch and build Umpire prior to configuring hypre's TPLs
function(maybe_build_umpire)
  if(NOT HYPRE_BUILD_UMPIRE)
    return()
  endif()

  # Only auto-build Umpire when a GPU backend is enabled, unless the user explicitly enabled it
  if(NOT (HYPRE_ENABLE_CUDA OR HYPRE_ENABLE_HIP OR HYPRE_ENABLE_SYCL))
    if(NOT HYPRE_ENABLE_UMPIRE)
      message(STATUS "Skipping Umpire auto-build: GPU backend is not enabled and HYPRE_ENABLE_UMPIRE is OFF")
      return()
    endif()
  endif()

  # If user already provided Umpire or it was added, skip
  if(TPL_UMPIRE_LIBRARIES OR TPL_UMPIRE_INCLUDE_DIRS OR TARGET umpire::umpire OR TARGET umpire)
    message(STATUS "Umpire already provided. Skipping auto-build.")
    return()
  endif()

  # Respect explicit user disabling of Umpire; otherwise enable it for GPU builds only
  if(NOT HYPRE_ENABLE_UMPIRE)
    if(HYPRE_ENABLE_CUDA OR HYPRE_ENABLE_HIP OR HYPRE_ENABLE_SYCL)
      if(NOT HYPRE_USER_SET_HYPRE_ENABLE_UMPIRE)
        set(HYPRE_ENABLE_UMPIRE ON CACHE BOOL "Use Umpire Allocator" FORCE)
        set(HYPRE_USING_UMPIRE ON CACHE INTERNAL "")
      else()
        message(WARNING "HYPRE_BUILD_UMPIRE=ON but HYPRE_ENABLE_UMPIRE was explicitly set by user to OFF. Proceeding will enable it due to GPU build.")
        set(HYPRE_ENABLE_UMPIRE ON CACHE BOOL "Use Umpire Allocator" FORCE)
        set(HYPRE_USING_UMPIRE ON CACHE INTERNAL "")
      endif()
    endif()
  endif()

  include(FetchContent)

  # Determine version/tag for Umpire
  set(_umpire_tag "${HYPRE_UMPIRE_VERSION}")
  if(_umpire_tag STREQUAL "latest")
    # Default to a recent release if auto-detection is not desired here
    set(_umpire_tag "v2025.09.0")
  endif()

  # Configure Umpire build options according to hypre needs (one canonical block)
  set(UMPIRE_ENABLE_C              ON  CACHE BOOL "Enable C interface in Umpire" FORCE)
  set(UMPIRE_ENABLE_TOOLS          OFF CACHE BOOL "Disable Umpire tools" FORCE)
  set(ENABLE_BENCHMARKS            OFF CACHE BOOL "Disable Umpire benchmarks" FORCE)
  set(ENABLE_EXAMPLES              OFF CACHE BOOL "Disable Umpire examples" FORCE)
  set(ENABLE_DOCS                  OFF CACHE BOOL "Disable Umpire docs" FORCE)
  set(ENABLE_TESTS                 OFF CACHE BOOL "Disable Umpire tests" FORCE)
  set(ENABLE_CUDA ${HYPRE_ENABLE_CUDA} CACHE BOOL "Enable CUDA in Umpire" FORCE)
  set(ENABLE_HIP  ${HYPRE_ENABLE_HIP}  CACHE BOOL "Enable HIP in Umpire" FORCE)
  set(ENABLE_SYCL ${HYPRE_ENABLE_SYCL} CACHE BOOL "Enable SYCL in Umpire" FORCE)

  # Ensure Umpire installs to the same prefix as hypre
  set(CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}" CACHE PATH "Install prefix" FORCE)

  # Ensure CUDA version is visible to BLT/camp in the subproject when CUDA is enabled
  if(HYPRE_ENABLE_CUDA)
    if(DEFINED CUDAToolkit_VERSION)
      set(CUDA_VERSION "${CUDAToolkit_VERSION}" CACHE STRING "CUDA toolkit version for subprojects" FORCE)
      set(CUDA_VERSION_STRING "${CUDAToolkit_VERSION}" CACHE STRING "CUDA toolkit version string for subprojects" FORCE)
    elseif(DEFINED CMAKE_CUDA_COMPILER_VERSION)
      set(CUDA_VERSION "${CMAKE_CUDA_COMPILER_VERSION}" CACHE STRING "CUDA toolkit version for subprojects" FORCE)
      set(CUDA_VERSION_STRING "${CMAKE_CUDA_COMPILER_VERSION}" CACHE STRING "CUDA toolkit version string for subprojects" FORCE)
    endif()
    # Also ensure CUDA include dirs are visible to subprojects that compile host C++ with CUDA headers
    if(DEFINED CUDAToolkit_INCLUDE_DIRS)
      include_directories(BEFORE SYSTEM ${CUDAToolkit_INCLUDE_DIRS})
    endif()
  endif()

  # Fetch Umpire with its submodules using FetchContent (populate only)
  set(FETCHCONTENT_QUIET OFF)
  FetchContent_Declare(
    umpire
    GIT_REPOSITORY https://github.com/LLNL/Umpire.git
    GIT_TAG        ${_umpire_tag}
    GIT_SHALLOW    TRUE
    GIT_SUBMODULES blt;src/tpl/umpire/camp;src/tpl/umpire/fmt
    GIT_PROGRESS   TRUE
  )
  FetchContent_Populate(umpire)

  # Sanitize version placeholders in config.hpp.in to avoid leading-zero octal (e.g., 09)
  set(_src_dir "${umpire_SOURCE_DIR}")
  set(_bld_dir "${CMAKE_BINARY_DIR}/_deps/umpire-build")
  file(MAKE_DIRECTORY "${_bld_dir}")
  set(_umpire_cfg_in "${_src_dir}/src/umpire/config.hpp.in")
  if(EXISTS "${_umpire_cfg_in}")
    string(REGEX MATCH "^v?([0-9]+)\.([0-9]+)\.([0-9]+)" _ver_match "${_umpire_tag}")
    if(CMAKE_MATCH_COUNT GREATER 0)
      set(_umaj "${CMAKE_MATCH_1}")
      set(_umin "${CMAKE_MATCH_2}")
      set(_upat "${CMAKE_MATCH_3}")
      string(REGEX REPLACE "^0+" "" _umin "${_umin}")
      if(_umin STREQUAL "")
        set(_umin "0")
      endif()
      string(REGEX REPLACE "^0+" "" _upat "${_upat}")
      if(_upat STREQUAL "")
        set(_upat "0")
      endif()
      file(READ "${_umpire_cfg_in}" _cfg_content)
      string(REPLACE "@Umpire_VERSION_MAJOR@" "${_umaj}" _cfg_content "${_cfg_content}")
      string(REPLACE "@Umpire_VERSION_MINOR@" "${_umin}" _cfg_content "${_cfg_content}")
      string(REPLACE "@Umpire_VERSION_PATCH@" "${_upat}" _cfg_content "${_cfg_content}")
      file(WRITE "${_umpire_cfg_in}" "${_cfg_content}")
      message(STATUS "Sanitized Umpire config.hpp.in version placeholders: ${_umaj}.${_umin}.${_upat}")
    endif()
  endif()

  # Add Umpire as a subproject now that sources are sanitized
  add_subdirectory("${_src_dir}" "${_bld_dir}")

  # Fix up CUDA runtime linkage to use CUDA::cudart instead of legacy cuda_runtime
  fixup_umpire_cuda_runtime()

  # Create the namespaced alias if necessary for consistent linkage
  if(TARGET umpire AND NOT TARGET umpire::umpire)
    add_library(umpire::umpire ALIAS umpire)
  endif()

  message(STATUS "Umpire will be built from sources (tag: ${_umpire_tag}) and installed into: ${CMAKE_INSTALL_PREFIX}")
endfunction()

# Function to setup FEI (to be phased out)
function(setup_fei)
  if (HYPRE_USING_FEI)
    set(HYPRE_NEEDS_CXX TRUE PARENT_SCOPE)

    if (NOT TPL_FEI_INCLUDE_DIRS)
      message(FATAL_ERROR "TPL_FEI_INCLUDE_DIRS option should be set for FEI support.")
    endif ()

    foreach (dir ${TPL_FEI_INCLUDE_DIRS})
      if (NOT EXISTS ${dir})
        message(FATAL_ERROR "FEI include directory not found: ${dir}")
      endif ()
      target_compile_options(${PROJECT_NAME} PUBLIC -I${dir})
    endforeach ()

    message(STATUS "Enabled support for using FEI.")
    set(FEI_FOUND TRUE PARENT_SCOPE)
    target_include_directories(${PROJECT_NAME} PUBLIC ${TPL_FEI_INCLUDE_DIRS})
  endif()
endfunction()

# A handy function to add the current source directory to a local
# filename. To be used for creating a list of sources.
function(convert_filenames_to_full_paths NAMES)
  unset(tmp_names)
  foreach(name ${${NAMES}})
    list(APPEND tmp_names ${CMAKE_CURRENT_SOURCE_DIR}/${name})
  endforeach()
  set(${NAMES} ${tmp_names} PARENT_SCOPE)
endfunction()

# A function to add hypre subdirectories to the build
function(add_hypre_subdirectories DIRS)
  foreach(DIR IN LISTS DIRS)
    add_subdirectory(${DIR})
    target_include_directories(${PROJECT_NAME} PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/${DIR}>)
  endforeach()
endfunction()

# A function to add an executable to the build with the correct flags, includes, and linkage.
function(add_hypre_executable SRC_FILE DEP_SRC_FILE)
  get_filename_component(SRC_FILENAME ${SRC_FILE} NAME)
  if (DEP_SRC_FILE)
    get_filename_component(DEP_SRC_FILENAME ${DEP_SRC_FILE} NAME)
  endif ()

  # If CUDA is enabled, tag source files to be compiled with nvcc.
  if (HYPRE_USING_CUDA)
    set_source_files_properties(${SRC_FILENAME} PROPERTIES LANGUAGE CUDA)
    if (DEP_SRC_FILE)
      set_source_files_properties(${DEP_SRC_FILENAME} PROPERTIES LANGUAGE CUDA)
    endif ()
  endif ()

  # If HIP is enabled, tag source files to be compiled with hipcc/clang
  if (HYPRE_USING_HIP)
    set_source_files_properties(${SRC_FILENAME} PROPERTIES LANGUAGE HIP)
    if (DEP_SRC_FILE)
       set_source_files_properties(${DEP_SRC_FILENAME} PROPERTIES LANGUAGE HIP)
    endif ()
  endif ()

  # If SYCL is enabled, tag source files to be compiled with dpcpp.
  if (HYPRE_USING_SYCL)
    set_source_files_properties(${SRC_FILENAME} PROPERTIES LANGUAGE CXX)
    if (DEP_SRC_FILE)
       set_source_files_properties(${DEP_SRC_FILENAME} PROPERTIES LANGUAGE CXX)
    endif ()
  endif ()

  # Get executable name
  string(REPLACE ".c" "" EXE_NAME ${SRC_FILENAME})

  # Add the executable, including DEP_SRC_FILE if provided
  if (DEP_SRC_FILE)
    add_executable(${EXE_NAME} ${SRC_FILE} ${DEP_SRC_FILE})
  else ()
    add_executable(${EXE_NAME} ${SRC_FILE})
  endif ()

  # Link with HYPRE and inherit its compile properties
  target_link_libraries(${EXE_NAME} PUBLIC HYPRE)

  # For Unix systems, also link with math library
  if (UNIX)
    target_link_libraries(${EXE_NAME} PUBLIC m)
  endif ()

  # Explicitly specify the linker
  if ((HYPRE_USING_CUDA AND NOT HYPRE_ENABLE_LTO) OR HYPRE_USING_HIP OR HYPRE_USING_SYCL)
    set_target_properties(${EXE_NAME} PROPERTIES LINKER_LANGUAGE CXX)
  endif ()

  # Turn on LTO if requested
  if (HYPRE_ENABLE_LTO)
    set_target_properties(${EXE_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
  endif ()

  # Inherit compile definitions and options from HYPRE target
  get_target_property(HYPRE_COMPILE_OPTS HYPRE COMPILE_OPTIONS)
  if (HYPRE_COMPILE_OPTS)
    if (HYPRE_USING_CUDA OR HYPRE_USING_HIP OR HYPRE_USING_SYCL)
      get_language_flags("${HYPRE_COMPILE_OPTS}" CXX_OPTS "CXX")
      target_compile_options(${EXE_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${CXX_OPTS}>)
    else ()
      get_language_flags("${HYPRE_COMPILE_OPTS}" C_OPTS "C")
      target_compile_options(${EXE_NAME} PRIVATE $<$<COMPILE_LANGUAGE:C>:${C_OPTS}>)
    endif ()
  endif ()

  # Copy executable to original source directory
  add_custom_command(TARGET ${EXE_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${EXE_NAME}> ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Copied ${EXE_NAME} to ${CMAKE_CURRENT_SOURCE_DIR}"
  )
endfunction()

# Function to process a list of executable source files
function(add_hypre_executables EXE_SRCS)
  # Support both usage styles:
  #  - add_hypre_executables(EXAMPLE_SRCS)     -> variable name
  #  - add_hypre_executables("${TEST_SRCS}")   -> expanded list content
  set(_HYPRE_EXE_SRC_LIST)
  if(EXE_SRCS MATCHES "\\.(c|cc|cxx|cpp|cu|cuf|f|f90)(;|$)")
    list(APPEND _HYPRE_EXE_SRC_LIST ${EXE_SRCS})
  else()
    if(DEFINED ${EXE_SRCS})
      list(APPEND _HYPRE_EXE_SRC_LIST ${${EXE_SRCS}})
    else()
      list(APPEND _HYPRE_EXE_SRC_LIST ${EXE_SRCS})
    endif()
  endif()

  foreach(SRC_FILE IN LISTS _HYPRE_EXE_SRC_LIST)
    add_hypre_executable(${SRC_FILE} "")
  endforeach()
endfunction()

# Function to add a tags target if Universal Ctags is found
function(add_hypre_target_tags)
  find_program(CTAGS_EXECUTABLE ctags)
  if(CTAGS_EXECUTABLE)
    add_custom_target(tags
      COMMAND ${CTAGS_EXECUTABLE} -e -R
              --languages=C,C++,CUDA
              --langmap=C++:+.hip
              --c-kinds=+p
              --c++-kinds=+p
              --extras=+q
              --exclude=.git
              --exclude=build
              -o TAGS .
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      COMMENT "Generating TAGS file with Universal Ctags"
      VERBATIM
    )
  endif()
endfunction()

# Function to add a distclean target
function(add_hypre_target_distclean)
  set(DISTCLEAN_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/DistcleanScript.cmake")

  file(WRITE ${DISTCLEAN_SCRIPT} "
  # Remove everything in the build directory except .git, .gitignore, and this script
  file(GLOB build_items RELATIVE \"${CMAKE_BINARY_DIR}\" \"${CMAKE_BINARY_DIR}/*\")
  foreach(item \${build_items})
    if(NOT item STREQUAL \".git\" AND
       NOT item STREQUAL \".gitignore\" AND
       NOT item STREQUAL \"${CMAKE_MATCH_1}\")
      if(NOT \"${DISTCLEAN_SCRIPT}\" STREQUAL \"${CMAKE_BINARY_DIR}/\${item}\")
        file(REMOVE_RECURSE \"${CMAKE_BINARY_DIR}/\${item}\")
      endif()
    endif()
  endforeach()

  # Remove build artifacts in the source tree
  set(patterns
    \"*.o\" \"*.mod\" \"*~\"
    \"test/*.out*\" \"test/*.err*\"
    \"examples/ex[0-9]\" \"examples/ex1[0-9]\"
    \"test/ij\" \"test/struct\" \"test/structmat\"
    \"test/sstruct\" \"test/ams_driver\"
    \"test/struct_migrate\" \"test/ij_assembly\"
  )
  foreach(pat \${patterns})
    file(GLOB_RECURSE matches RELATIVE \"${CMAKE_SOURCE_DIR}\" \"${CMAKE_SOURCE_DIR}/\${pat}\")
    foreach(m \${matches})
      file(REMOVE_RECURSE \"${CMAKE_SOURCE_DIR}/\${m}\")
    endforeach()
  endforeach()

  # Remove the script itself
  file(REMOVE \"${DISTCLEAN_SCRIPT}\")
  ")

  add_custom_target(distclean
    COMMAND ${CMAKE_COMMAND} -P ${DISTCLEAN_SCRIPT}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Removing all build artifacts and generated files"
    VERBATIM
  )
endfunction()

# Function to add an uninstall target
function(add_hypre_target_uninstall)
  add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -E remove_directory "${CMAKE_INSTALL_PREFIX}"
    COMMAND ${CMAKE_COMMAND} -E echo "Removed installation directory: ${CMAKE_INSTALL_PREFIX}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Uninstalling HYPRE"
    VERBATIM
  )
endfunction()

# Function to print the status of build options
function(print_option_status)
  # Define column widths
  set(COLUMN1_WIDTH 40)
  set(COLUMN2_WIDTH 10)
  math(EXPR HEADER1_PAD "${COLUMN1_WIDTH} - 3")
  math(EXPR HEADER2_PAD "${COLUMN2_WIDTH} - 1")

  # Create separator line
  string(REPEAT "-" ${HEADER1_PAD} SEPARATOR1)
  string(REPEAT "-" ${HEADER2_PAD} SEPARATOR2)
  set(separator "+${SEPARATOR1}+${SEPARATOR2}+")

  # Function to print a block of options
  function(print_option_block title options)
    message(STATUS "")
    message(STATUS " ${title}:")
    message(STATUS " ${separator}")
    message(STATUS " | Option                              | Status  |")
    message(STATUS " ${separator}")

    foreach(opt ${options})
      if(${${opt}})
        set(status "ON")
      else()
        set(status "OFF")
      endif()

      string(LENGTH "${opt}" opt_length)
      math(EXPR padding "${COLUMN1_WIDTH} - ${opt_length} - 5")
      if(${padding} GREATER 0)
        string(REPEAT " " ${padding} pad_spaces)
      else()
        set(pad_spaces "")
      endif()

      string(LENGTH "${status}" status_length)
      math(EXPR status_padding "${COLUMN2_WIDTH} - ${status_length} - 3")
      if(${status_padding} GREATER 0)
        string(REPEAT " " ${status_padding} status_pad_spaces)
      else()
        set(status_pad_spaces "")
      endif()

      message(STATUS " | ${opt}${pad_spaces} | ${status}${status_pad_spaces} |")
    endforeach()

    message(STATUS " ${separator}")
  endfunction()

  message(STATUS "")
  message(STATUS "HYPRE Configuration Summary:")

  # Print BASE_OPTIONS
  print_option_block("Base Options" "${BASE_OPTIONS}")

  # Print GPU_OPTIONS
  if(HYPRE_ENABLE_CUDA OR HYPRE_ENABLE_HIP OR HYPRE_ENABLE_SYCL)
    print_option_block("GPU Options" "${GPU_OPTIONS}")
  endif()

  # Print TPL_OPTIONS
  print_option_block("Third-Party Library Options" "${TPL_OPTIONS}")

  message(STATUS "")
endfunction()

# Macro for setting up mixed precision compilation (must be defined before subdirectories)
macro(setup_mixed_precision_compilation module_name)
  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs SRCS)
  cmake_parse_arguments(REGULAR "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(NOT REGULAR_SRCS)
    message(FATAL_ERROR "SRCS argument is required for setup_mixed_precision_compilation")
  endif()

  # Create object libraries for each precision
  add_library(${module_name}_flt  OBJECT ${REGULAR_SRCS})
  add_library(${module_name}_dbl  OBJECT ${REGULAR_SRCS})
  add_library(${module_name}_ldbl OBJECT ${REGULAR_SRCS})

  # Set precision-specific compile definitions
  target_compile_definitions(${module_name}_flt  PRIVATE MP_BUILD_SINGLE=1)
  target_compile_definitions(${module_name}_dbl  PRIVATE MP_BUILD_DOUBLE=1)
  target_compile_definitions(${module_name}_ldbl PRIVATE MP_BUILD_LONGDOUBLE=1)

  # Set include directories and link libraries for all precision variants
  foreach(precision IN ITEMS flt dbl ldbl)
    target_include_directories(${module_name}_${precision} PRIVATE
      ${CMAKE_SOURCE_DIR}
      ${CMAKE_BINARY_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_SOURCE_DIR}/utilities
      ${CMAKE_SOURCE_DIR}/blas
      ${CMAKE_SOURCE_DIR}/lapack
      ${CMAKE_SOURCE_DIR}/seq_mv
      ${CMAKE_SOURCE_DIR}/seq_block_mv
      ${CMAKE_SOURCE_DIR}/parcsr_mv
      ${CMAKE_SOURCE_DIR}/parcsr_block_mv
      ${CMAKE_SOURCE_DIR}/parcsr_ls
      ${CMAKE_SOURCE_DIR}/IJ_mv
      ${CMAKE_SOURCE_DIR}/krylov
      ${CMAKE_SOURCE_DIR}/struct_mv
      ${CMAKE_SOURCE_DIR}/sstruct_mv
      ${CMAKE_SOURCE_DIR}/struct_ls
      ${CMAKE_SOURCE_DIR}/sstruct_ls
      ${CMAKE_SOURCE_DIR}/distributed_matrix
      ${CMAKE_SOURCE_DIR}/matrix_matrix
      ${CMAKE_SOURCE_DIR}/multivector
    )
    # Link to MPI if it's enabled
    if(HYPRE_ENABLE_MPI)
      target_link_libraries(${module_name}_${precision} PRIVATE MPI::MPI_C)
    endif()
  endforeach()

  # Add the precision object files to the main target
  target_sources(${PROJECT_NAME} PRIVATE
    $<TARGET_OBJECTS:${module_name}_flt>
    $<TARGET_OBJECTS:${module_name}_dbl>
    $<TARGET_OBJECTS:${module_name}_ldbl>
  )
endmacro()
