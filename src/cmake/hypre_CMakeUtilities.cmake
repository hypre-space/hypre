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
# correct flags, includes, and linkage
function(add_hypre_c_executables EXE_SRCS)
  # Add one executable per cpp file
  foreach(SRC_FILE IN LISTS ${EXE_SRCS})
    get_filename_component(SRC_FILENAME ${SRC_FILE} NAME)

    string(REPLACE ".c" "" EXE_NAME ${SRC_FILENAME})
    add_hypre_executable(${EXE_NAME}
      MAIN ${SRC_FILE}
      LIBRARIES HYPRE)
  endforeach(SRC_FILE)
endfunction()

# A function to an individual executable 
function(add_hypre_executable HYPRE_EXE_NAME)
  # Parse the input arguments looking for the things we need
  set(POSSIBLE_ARGS "MAIN" "EXTRA_SOURCES" "EXTRA_HEADERS" "EXTRA_OPTIONS" "EXTRA_DEFINES" "LIBRARIES")
  set(CURRENT_ARG)
  foreach(arg ${ARGN})
    list(FIND POSSIBLE_ARGS ${arg} is_arg_name)
    if (${is_arg_name} GREATER -1)
      set(CURRENT_ARG ${arg})
      set(${CURRENT_ARG}_LIST)
    else()
      list(APPEND ${CURRENT_ARG}_LIST ${arg})
    endif()
  endforeach()

  # Actually add the exe
  add_executable(${HYPRE_EXE_NAME} ${MAIN_LIST}
    ${EXTRA_SOURCES_LIST} ${EXTRA_HEADERS_LIST})

  # Append the additional libraries and options
  target_link_libraries(${HYPRE_EXE_NAME} PRIVATE "${LIBRARIES_LIST}")
  target_compile_options(${HYPRE_EXE_NAME} PRIVATE ${EXTRA_OPTIONS_LIST})
  target_compile_definitions(${HYPRE_EXE_NAME} PRIVATE ${EXTRA_DEFINES_LIST})
    
  # Handle the MPI separately
  target_link_libraries(${HYPRE_EXE_NAME} PRIVATE ${MPI_C_LIBRARIES})

  target_include_directories(${HYPRE_EXE_NAME} PRIVATE ${MPI_C_INCLUDE_PATH})
  if (MPI_C_COMPILE_FLAGS)
    target_compile_options(${HYPRE_EXE_NAME} PRIVATE ${MPI_C_COMPILE_FLAGS})
  endif()

  if (MPI_C_LINK_FLAGS)
    set_target_properties(${HYPRE_EXE_NAME} PROPERTIES
      LINK_FLAGS "${MPI_C_LINK_FLAGS}")
  endif()

endfunction()
