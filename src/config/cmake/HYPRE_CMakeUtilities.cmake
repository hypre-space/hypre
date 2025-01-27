# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

# Function to set hypre build options
function(set_hypre_option category name description default_value)
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
    execute_process(COMMAND git -C ${HYPRE_GIT_DIR} describe --match v* --long --abbrev=9
                    OUTPUT_VARIABLE develop_string
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    RESULT_VARIABLE git_result)
    if (git_result EQUAL 0)
      set(GIT_VERSION_FOUND TRUE PARENT_SCOPE)
      execute_process(COMMAND git -C ${HYPRE_GIT_DIR} describe --match v* --abbrev=0
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
  if(DEFINED ${option1} AND DEFINED ${option2})
    if(${option1} AND ${${option2}})
      # Save the value of the conflicting options
      set(saved_value1 "${${option1}}")
      set(saved_value2 "${${option2}}")

      # Unset conflicting options
      unset(${option1} CACHE)
      unset(${option2} CACHE)

      message(FATAL_ERROR "Error: ${option1} (${saved_value1}) and ${option2} (${saved_value2}) are mutually exclusive. Only one can be set to ON. Unsetting both options.")
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

# Function to handle TPL (Third-Party Library) setup
function(setup_tpl LIBNAME)
  string(TOUPPER ${LIBNAME} LIBNAME_UPPER)

  # Note we need to check for "USING" instead of "WITH" because
  # we want to allow for post-processing of build options via cmake
  if(HYPRE_USING_${LIBNAME_UPPER})
    if(TPL_${LIBNAME_UPPER}_LIBRARIES AND TPL_${LIBNAME_UPPER}_INCLUDE_DIRS)
      # Use specified TPL libraries and include dirs
      foreach(dir ${TPL_${LIBNAME_UPPER}_INCLUDE_DIRS})
        if(NOT EXISTS ${dir})
          message(FATAL_ERROR "${LIBNAME_UPPER} include directory not found: ${dir}")
        endif()
      endforeach()

      foreach(lib ${TPL_${LIBNAME_UPPER}_LIBRARIES})
        if(EXISTS ${lib})
          message(STATUS "${LIBNAME_UPPER} library found: ${lib}")
          get_filename_component(LIB_DIR "${lib}" DIRECTORY)
          set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY INSTALL_RPATH "${LIB_DIR}")
        else()
          message(WARNING "${LIBNAME_UPPER} library not found at specified path: ${lib}")
        endif()
      endforeach()

      target_link_libraries(${PROJECT_NAME} PUBLIC ${TPL_${LIBNAME_UPPER}_LIBRARIES})
      target_include_directories(${PROJECT_NAME} PUBLIC ${TPL_${LIBNAME_UPPER}_INCLUDE_DIRS})
    else()
      # Use find_package
      find_package(${LIBNAME} REQUIRED CONFIG)
      if(${LIBNAME}_FOUND)
        list(APPEND HYPRE_DEPENDENCY_DIRS "${${LIBNAME}_ROOT}")
        set(HYPRE_DEPENDENCY_DIRS "${HYPRE_DEPENDENCY_DIRS}" CACHE INTERNAL "" FORCE)

        if(${LIBNAME} STREQUAL "caliper")
          set(HYPRE_NEEDS_CXX TRUE PARENT_SCOPE)
        endif()

        if(TARGET ${LIBNAME}::${LIBNAME})
          target_link_libraries(${PROJECT_NAME} PUBLIC ${LIBNAME}::${LIBNAME})
          message(STATUS "Found ${LIBNAME} target: ${LIBNAME}::${LIBNAME}")
        elseif(TARGET ${LIBNAME})
          target_link_libraries(${PROJECT_NAME} PUBLIC ${LIBNAME})
          message(STATUS "Found ${LIBNAME} target: ${LIBNAME}")
        else()
          message(FATAL_ERROR "${LIBNAME} target not found. Please check your ${LIBNAME} installation")
        endif()
      else()
        message(FATAL_ERROR "${LIBNAME_UPPER} target not found. Please check your ${LIBNAME_UPPER} installation")
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

# A function to add each executable in the list to the build with the
# correct flags, includes, and linkage.
function(add_hypre_executables EXE_SRCS)
  # Add one executable per cpp file
  foreach(SRC_FILE IN LISTS ${EXE_SRCS})
    get_filename_component(SRC_FILENAME ${SRC_FILE} NAME)

    # If CUDA is enabled, tag source files to be compiled with nvcc.
    if (HYPRE_USING_CUDA)
      set_source_files_properties(${SRC_FILENAME} PROPERTIES LANGUAGE CUDA)
    endif ()

    # If HIP is enabled, tag source files to be compiled with hipcc/clang
    if (HYPRE_USING_HIP)
      set_source_files_properties(${SRC_FILENAME} PROPERTIES LANGUAGE HIP)
    endif ()

    # If SYCL is enabled, tag source files to be compiled with dpcpp.
    if (HYPRE_USING_SYCL)
      set_source_files_properties(${SRC_FILENAME} PROPERTIES LANGUAGE CXX)
    endif ()

    # Get executable name
    string(REPLACE ".c" "" EXE_NAME ${SRC_FILENAME})

    # Add the executable
    add_executable(${EXE_NAME} ${SRC_FILE})

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
    #message(STATUS "${EXE_NAME}: ${HYPRE_COMPILE_OPTS}")
    if (HYPRE_COMPILE_OPTS)
      if (HYPRE_USING_CUDA OR HYPRE_USING_HIP OR HYPRE_USING_SYCL)
        get_language_flags("${HYPRE_COMPILE_OPTS}" CXX_OPTS "CXX")
        target_compile_options(${EXE_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${CXX_OPTS}>)
        #message(STATUS "Added CXX compile options: ${CXX_OPTS} to ${EXE_NAME}")
      else ()
        get_language_flags("${HYPRE_COMPILE_OPTS}" C_OPTS "C")
        target_compile_options(${EXE_NAME} PRIVATE $<$<COMPILE_LANGUAGE:C>:${C_OPTS}>)
        #message(STATUS "Added C compile options: ${C_OPTS} to ${EXE_NAME}")
      endif ()
    endif ()

    # Copy executable to original source directory
    add_custom_command(TARGET ${EXE_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${EXE_NAME}> ${CMAKE_CURRENT_SOURCE_DIR}
      COMMENT "Copied ${EXE_NAME} to ${CMAKE_CURRENT_SOURCE_DIR}"
    )
  endforeach (SRC_FILE)
endfunction ()

# Function to add a tags target if etags is found
function(add_hypre_target_tags)
  find_program(ETAGS_EXECUTABLE etags)
  if(ETAGS_EXECUTABLE)
    add_custom_target(tags
      COMMAND find ${CMAKE_CURRENT_SOURCE_DIR}
              -type f
              "(" -name "*.h" -o -name "*.c" -o -name "*.cpp"
              -o -name "*.hpp" -o -name "*.cxx"
              -o -name "*.f" -o -name "*.f90" ")"
              -not -path "*/build/*"
              -print | ${ETAGS_EXECUTABLE}
              --declarations
              --ignore-indentation
              --no-members
              -
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMENT "Generating TAGS file with etags"
      VERBATIM
    )
  endif()
endfunction()

# Function to add a distclean target
function(add_hypre_target_distclean)
  add_custom_target(distclean
    COMMAND find ${CMAKE_BINARY_DIR} -mindepth 1 -delete
    COMMAND find ${CMAKE_SOURCE_DIR} -name "*.o" -type f -delete
    COMMAND find ${CMAKE_SOURCE_DIR} -name "*.mod" -type f -delete
    COMMAND find ${CMAKE_SOURCE_DIR} -name "*~" -type f -delete
    COMMAND find ${CMAKE_SOURCE_DIR}/test -name "*.out*" -type f -delete
    COMMAND find ${CMAKE_SOURCE_DIR}/test -name "*.err*" -type f -delete
    COMMAND find ${CMAKE_SOURCE_DIR}/examples -type f -name "ex[0-9]" -name "ex[10-19]" -delete
    COMMAND find ${CMAKE_SOURCE_DIR}/test -type f -name "ij|struct|sstruct|ams_driver|maxwell_unscalled|struct_migrate|sstruct_fac|ij_assembly" -delete
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
