# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

# Function to set hypre build options
function(set_hypre_option category name description default_value)
  option(${name} "${description}" ${default_value})
  if (${category} STREQUAL "CUDA" OR ${category} STREQUAL "HIP" OR ${category} STREQUAL "SYCL")
    if (HYPRE_WITH_${category} STREQUAL "ON")
      set(GPU_OPTIONS ${GPU_OPTIONS} ${name} PARENT_SCOPE)
    endif()
  else()
    set(${category}_OPTIONS ${${category}_OPTIONS} ${name} PARENT_SCOPE)
  endif()
endfunction()

# Function to set generic variables based on condition
function(set_conditional_var condition var_name)
  if(${condition})
    #set(${var_name} ON CACHE INTERNAL "${var_name} set to ON because ${condition} is true")
    set(${var_name} ON CACHE BOOL "" FORCE)
  endif()
endfunction()

# Function to set conditional hypre build options
function(set_conditional_hypre_option condition_prefix var_prefix var_name)
  if(HYPRE_${condition_prefix}_${var_name})
    if(var_prefix STREQUAL "")
      set(HYPRE_${var_name} ON CACHE BOOL "" FORCE)
    else()
      set(HYPRE_${var_prefix}_${var_name} ON CACHE BOOL "" FORCE)
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
  else()
    message(WARNING "MPI include directory not found. Please specify -DMPI_INCLUDE_DIR or the compilation may fail.")
  endif()

  if (HYPRE_WITH_CUDA OR HYPRE_WITH_HIP OR HYPRE_WITH_SYCL)
    message(STATUS "Adding MPI include directory: ${MPI_INCLUDE_DIR}")
    target_include_directories(${PROJECT_NAME} PUBLIC ${MPI_INCLUDE_DIR})
  endif ()
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
        message(STATUS "Found ${LIBNAME_UPPER} library")
        list(APPEND HYPRE_DEPENDENCY_DIRS "${${LIBNAME}_ROOT}")
        set(HYPRE_DEPENDENCY_DIRS "${HYPRE_DEPENDENCY_DIRS}" CACHE INTERNAL "" FORCE)

        if(${LIBNAME} STREQUAL "caliper")
          set(HYPRE_NEEDS_CXX TRUE PARENT_SCOPE)
        endif()

        if(TARGET ${LIBNAME}::${LIBNAME})
          target_link_libraries(${PROJECT_NAME} PUBLIC ${LIBNAME}::${LIBNAME})
        elseif(TARGET ${LIBNAME})
          target_link_libraries(${PROJECT_NAME} PUBLIC ${LIBNAME})
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

  if(HYPRE_USING_INTERNAL_${LIB_NAME_UPPER})
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
      message(STATUS "Enabled support for using ${LIB_NAME_UPPER}.")
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
    endif()

    # If HIP is enabled, tag source files to be compiled with hipcc/clang
    if (HYPRE_USING_HIP)
      set_source_files_properties(${SRC_FILENAME} PROPERTIES LANGUAGE HIP)
    endif()

    # If SYCL is enabled, tag source files to be compiled with dpcpp.
    if (HYPRE_USING_SYCL)
      set_source_files_properties(${SRC_FILENAME} PROPERTIES LANGUAGE CXX)
    endif()

    # Get executable name
    string(REPLACE ".c" "" EXE_NAME ${SRC_FILENAME})

    # Add the executable
    add_executable(${EXE_NAME} ${SRC_FILE})

    # Explicitly specify the linker
    if (HYPRE_USING_CUDA OR HYPRE_USING_HIP OR HYPRE_USING_SYCL)
      set_target_properties(${EXE_NAME} PROPERTIES LINKER_LANGUAGE CXX)
    endif()

    # Link libraries
    set(HYPRE_LIBS "HYPRE")

    # Link libraries for Unix systems
    if (UNIX)
      list(APPEND HYPRE_LIBS m)
    endif (UNIX)

    # Append the additional libraries and options
    target_link_libraries(${EXE_NAME} PRIVATE "${HYPRE_LIBS}")
  endforeach(SRC_FILE)
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
  if(HYPRE_WITH_CUDA OR HYPRE_WITH_HIP OR HYPRE_WITH_SYCL)
    print_option_block("GPU Options" "${GPU_OPTIONS}")
  endif()

  # Print TPL_OPTIONS
  print_option_block("Third-Party Library Options" "${TPL_OPTIONS}")

  message(STATUS "")
endfunction()