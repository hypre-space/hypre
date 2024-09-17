# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

# Function to set conditional variables
function(set_conditional_var condition var_name)
  if(${condition})
    set(${var_name} ON CACHE BOOL "" FORCE)
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

# Function to handle TPL (Third-Party Library) setup
function(setup_tpl NAME)
  if(HYPRE_USING_${NAME})
    if(NOT TPL_${NAME}_LIBRARIES OR NOT TPL_${NAME}_INCLUDE_DIRS)
      message(FATAL_ERROR "Both TPL_${NAME}_LIBRARIES and TPL_${NAME}_INCLUDE_DIRS must be set for ${NAME} support.")
    endif()

    foreach(dir ${TPL_${NAME}_INCLUDE_DIRS})
      if(NOT EXISTS ${dir})
        message(FATAL_ERROR "${NAME} include directory not found: ${dir}")
      endif()
    endforeach()

    message(STATUS "Enabled support for using ${NAME}.")
    
    foreach(lib ${TPL_${NAME}_LIBRARIES})
      if(EXISTS ${lib})
        message(STATUS "${NAME} library found: ${lib}")
      else()
        message(WARNING "${NAME} library not found at specified path: ${lib}")
      endif()
    endforeach()

    target_link_libraries(${PROJECT_NAME} PUBLIC ${TPL_${NAME}_LIBRARIES})
    target_include_directories(${PROJECT_NAME} PUBLIC ${TPL_${NAME}_INCLUDE_DIRS})
    
    if(${NAME} STREQUAL "SUPERLU" OR ${NAME} STREQUAL "DSUPERLU" OR ${NAME} STREQUAL "UMPIRE")
      target_link_libraries(${PROJECT_NAME} PUBLIC stdc++)
    endif()

    set(${NAME}_FOUND TRUE PARENT_SCOPE)
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
    
    # Add USE_VENDOR_BLAS definition for BLAS
    if(${LIB_NAME_UPPER} STREQUAL "BLAS")
      target_compile_definitions(${PROJECT_NAME} PUBLIC "USE_VENDOR_BLAS")
    endif()
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

  # Create header and separator
  message(STATUS "")
  message(STATUS "HYPRE Configuration Summary:")
  message(STATUS "${separator}")
  message(STATUS "| Option                              | Status  |")
  message(STATUS "${separator}")

  # Iterate through each option and display its status
  foreach(opt ${ARGN})
    # Determine the status string
    if(${${opt}})
      set(status "ON")
    else()
      set(status "OFF")
    endif()

    # Calculate padding for the option name
    string(LENGTH "${opt}" opt_length)
    math(EXPR padding "${COLUMN1_WIDTH} - ${opt_length} - 5") # 5 accounts for "| Option | Status |"
    if(${padding} GREATER 0)
      string(REPEAT " " ${padding} pad_spaces)
    else()
      set(pad_spaces "")
    endif()

    # Calculate padding for the status
    string(LENGTH "${status}" status_length)
    math(EXPR status_padding "${COLUMN2_WIDTH} - ${status_length} - 3") # 3 accounts for "| " and space
    if(${status_padding} GREATER 0)
      string(REPEAT " " ${status_padding} status_pad_spaces)
    else()
      set(status_pad_spaces "")
    endif()

    # Print the formatted row
    message(STATUS "| ${opt}${pad_spaces} | ${status}${status_pad_spaces} |")
  endforeach()

  # Print the footer separator
  message(STATUS "${separator}")
  message(STATUS "")
endfunction()
