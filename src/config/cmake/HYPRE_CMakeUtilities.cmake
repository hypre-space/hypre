# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

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
