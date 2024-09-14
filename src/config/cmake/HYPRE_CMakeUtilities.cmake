# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

function(configure_mpi_target)
  find_package(MPI REQUIRED)
  target_link_libraries(${PROJECT_NAME} PUBLIC MPI::MPI_C)

  # Determine the correct MPI include directory
  if(MPI_CXX_INCLUDE_DIRS)
    set(MPI_INCLUDE_DIRS ${MPI_CXX_INCLUDE_DIRS})
  elseif(MPI_CXX_INCLUDE_PATH)
    set(MPI_INCLUDE_DIRS ${MPI_CXX_INCLUDE_PATH})
  elseif(MPI_CXX_COMPILER_INCLUDE_DIRS)
    set(MPI_INCLUDE_DIRS ${MPI_CXX_COMPILER_INCLUDE_DIRS})
  elseif(MPI_C_COMPILER_INCLUDE_DIRS)
    set(MPI_INCLUDE_DIRS ${MPI_C_COMPILER_INCLUDE_DIRS})
  elseif(MPI_C_INCLUDE_DIRS)
    set(MPI_INCLUDE_DIRS ${MPI_C_INCLUDE_DIRS})
  elseif(MPI_C_INCLUDE_PATH)
    set(MPI_INCLUDE_DIRS ${MPI_C_INCLUDE_PATH})
  elseif(MPI_INCLUDE_PATH)
    set(MPI_INCLUDE_DIRS ${MPI_INCLUDE_PATH})
  else()
    message(WARNING "MPI include directory not found. Compilation may fail.")
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

# A function to add each executable in the list to the build with the
# correct flags, includes, and linkage.
function(add_hypre_executables EXE_SRCS)
  # Add one executable per cpp file
  foreach(SRC_FILE IN LISTS ${EXE_SRCS})
    get_filename_component(SRC_FILENAME ${SRC_FILE} NAME)

    if (HYPRE_USING_CUDA)
      # If CUDA is enabled, tag source files to be compiled with nvcc.
      set_source_files_properties(${SRC_FILENAME} PROPERTIES LANGUAGE CUDA)
    endif (HYPRE_USING_CUDA)

    if (HYPRE_USING_SYCL)
      # If SYCL is enabled, tag source files to be compiled with dpcpp.
      set_source_files_properties(${SRC_FILENAME} PROPERTIES LANGUAGE CXX)
    endif (HYPRE_USING_SYCL)


    string(REPLACE ".c" "" EXE_NAME ${SRC_FILENAME})
    # Actually add the exe
    add_executable(${EXE_NAME} ${SRC_FILE})

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