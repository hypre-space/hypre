# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

add_definitions (-DHYPRE_TIMING)

if (MSVC)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif ()

# Set default tolerance if not defined
if(NOT DEFINED HYPRE_CHECK_TOL)
  set(HYPRE_CHECK_TOL 1.0e-6)
endif()

# Define list of executables
set(TEST_SRCS
  sstruct.c
  struct.c
  structmat.c
  ams_driver.c
  struct_migrate.c
)

# Mixed precision test executables (only built when mixed precision is enabled)
if(HYPRE_ENABLE_MIXED_PRECISION)
  set(MP_TEST_SRCS
    test_mp_pcg.c
    test_mp_pcg_3d.c
  )
  add_hypre_executables("${MP_TEST_SRCS}")
endif()

# Build each executable
add_hypre_executables("${TEST_SRCS}")

# Special treatment for sstructmat as it depends on sstruct_helpers (not an executable)
add_hypre_executable("sstructmat.c" "sstruct_helpers.c")

# Special treatment for ij drivers as they depend on ij_helpers (not an executable)
add_hypre_executable("ij.c" "ij_helpers.c")
add_hypre_executable("ij_assembly.c" "ij_helpers.c")

# Copy test input files
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/TEST_sstruct/sstruct.in.default
               ${CMAKE_CURRENT_BINARY_DIR}/sstruct.in.default COPYONLY)

# Install test drivers in ${CMAKE_INSTALL_PREFIX}/bin when tests are enabled
if(HYPRE_BUILD_TESTS)
  # Collect targets derived from source file names
  set(HYPRE_TEST_DRIVER_TARGETS)
  foreach(src_file ${TEST_SRCS})
    get_filename_component(exec_name ${src_file} NAME_WE)
    list(APPEND HYPRE_TEST_DRIVER_TARGETS ${exec_name})
  endforeach()

  # Mixed precision drivers when enabled
  if(HYPRE_ENABLE_MIXED_PRECISION)
    foreach(src_file ${MP_TEST_SRCS})
      get_filename_component(exec_name ${src_file} NAME_WE)
      list(APPEND HYPRE_TEST_DRIVER_TARGETS ${exec_name})
    endforeach()
  endif()

  install(TARGETS ${HYPRE_TEST_DRIVER_TARGETS}
          RUNTIME DESTINATION bin)
endif()

# Add check target
find_program(BASH_PROGRAM bash)

# Run check target
add_custom_target(check
  COMMAND ${CMAKE_COMMAND} -E echo "Checking the library..."
  COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 $<TARGET_FILE:ij> 2> ij.err
  COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 $<TARGET_FILE:struct> 2> struct.err
  COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 $<TARGET_FILE:sstruct> 2> sstruct.err
  COMMAND ${CMAKE_COMMAND} -E echo "Test error files:"
  COMMAND ${CMAKE_COMMAND} -E cat ij.err struct.err sstruct.err
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ij struct sstruct
  COMMENT "Running basic library check"
  VERBATIM
)

# Add checkpar target for parallel testing
add_custom_target(checkpar
  COMMAND ${CMAKE_COMMAND} -E echo "Checking the library..."

  COMMAND ${CMAKE_COMMAND} -E echo_append "Testing IJ... "
  COMMAND ${BASH_PROGRAM} ${CMAKE_CURRENT_SOURCE_DIR}/runtest.sh
          -atol ${HYPRE_CHECK_TOL}
          -mpi "${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG}"
          ${CMAKE_CURRENT_SOURCE_DIR}/TEST_ij/solvers.sh
  COMMAND ${BASH_PROGRAM} ${CMAKE_CURRENT_SOURCE_DIR}/checktest.sh
  COMMAND ${BASH_PROGRAM} ${CMAKE_CURRENT_SOURCE_DIR}/cleantest.sh
  COMMAND echo ""

  COMMAND ${CMAKE_COMMAND} -E echo_append "Testing Struct... "
  COMMAND ${BASH_PROGRAM} ${CMAKE_CURRENT_SOURCE_DIR}/runtest.sh
          -atol ${HYPRE_CHECK_TOL}
          -mpi "${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG}"
          ${CMAKE_CURRENT_SOURCE_DIR}/TEST_struct/solvers.sh
  COMMAND ${BASH_PROGRAM} ${CMAKE_CURRENT_SOURCE_DIR}/checktest.sh
  COMMAND ${BASH_PROGRAM} ${CMAKE_CURRENT_SOURCE_DIR}/cleantest.sh
  COMMAND echo ""

  COMMAND ${CMAKE_COMMAND} -E echo_append "Testing SStruct... "
  COMMAND ${BASH_PROGRAM} ${CMAKE_CURRENT_SOURCE_DIR}/runtest.sh
          -atol ${HYPRE_CHECK_TOL}
          -mpi "${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG}"
          ${CMAKE_CURRENT_SOURCE_DIR}/TEST_sstruct/solvers.sh
  COMMAND ${BASH_PROGRAM} ${CMAKE_CURRENT_SOURCE_DIR}/checktest.sh
  COMMAND ${BASH_PROGRAM} ${CMAKE_CURRENT_SOURCE_DIR}/cleantest.sh
  COMMAND echo ""

  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${TEST_SRCS}
  COMMENT "Running parallel tests"
  VERBATIM
)

# Add CTest support
if(BUILD_TESTING)
  message(STATUS "Building tests with CTest support")

  # IJ tests
  add_test(NAME test_ij_solvers
    COMMAND ${BASH_PROGRAM} -c "\
      ${BASH_PROGRAM} ${CMAKE_CURRENT_SOURCE_DIR}/runtest.sh \
        -atol ${HYPRE_CHECK_TOL} \
        -mpi \"${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG}\" \
        ${CMAKE_CURRENT_SOURCE_DIR}/TEST_ij/solvers.sh && \
      ${BASH_PROGRAM} ${CMAKE_CURRENT_SOURCE_DIR}/checktest.sh && \
      ${BASH_PROGRAM} ${CMAKE_CURRENT_SOURCE_DIR}/cleantest.sh"
  )
  set_tests_properties(test_ij_solvers PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ij
    FAIL_REGULAR_EXPRESSION "Failed"
  )

  # Struct tests
  add_test(NAME test_struct_solvers
    COMMAND ${BASH_PROGRAM} -c "\
      ${BASH_PROGRAM} ${CMAKE_CURRENT_SOURCE_DIR}/runtest.sh \
        -atol ${HYPRE_CHECK_TOL} \
        -mpi \"${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG}\" \
        ${CMAKE_CURRENT_SOURCE_DIR}/TEST_struct/solvers.sh && \
      ${BASH_PROGRAM} ${CMAKE_CURRENT_SOURCE_DIR}/checktest.sh && \
      ${BASH_PROGRAM} ${CMAKE_CURRENT_SOURCE_DIR}/cleantest.sh"
  )
  set_tests_properties(test_struct_solvers PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS struct
    FAIL_REGULAR_EXPRESSION "Failed"
  )

  # SStruct tests
  add_test(NAME test_sstruct_solvers
    COMMAND ${BASH_PROGRAM} -c "\
      ${BASH_PROGRAM} ${CMAKE_CURRENT_SOURCE_DIR}/runtest.sh \
        -atol ${HYPRE_CHECK_TOL} \
        -mpi \"${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG}\" \
        ${CMAKE_CURRENT_SOURCE_DIR}/TEST_sstruct/solvers.sh && \
      ${BASH_PROGRAM} ${CMAKE_CURRENT_SOURCE_DIR}/checktest.sh && \
      ${BASH_PROGRAM} ${CMAKE_CURRENT_SOURCE_DIR}/cleantest.sh"
  )
  set_tests_properties(test_sstruct_solvers PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS sstruct
    FAIL_REGULAR_EXPRESSION "Failed"
  )
endif()

# Optional Fortran-based drivers
if(HYPRE_ENABLE_FORTRAN)
  # Build Fortran wrapper object libraries to be linked into C drivers
  add_library(hypre_fstruct_wrappers OBJECT fstruct_mv.f fstruct_ls.f)
  add_library(hypre_fsstruct_wrappers OBJECT fsstruct_mv.f fsstruct_ls.f)
  add_library(hypre_fparcsr_wrappers OBJECT fparcsr_mv.f fparcsr_ls.f)
  add_library(hypre_fij_wrappers OBJECT fij_mv.f)

  # Common include dirs for mixed-language linking
  foreach(wlib IN ITEMS hypre_fstruct_wrappers hypre_fsstruct_wrappers hypre_fparcsr_wrappers hypre_fij_wrappers)
    target_include_directories(${wlib} PRIVATE
      ${CMAKE_SOURCE_DIR}
      ${CMAKE_BINARY_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}
    )
    if(HYPRE_ENABLE_MPI)
      target_link_libraries(${wlib} PRIVATE MPI::MPI_Fortran)
    endif()
  endforeach()

  # Pure Fortran test drivers
  add_executable(f77_ij f77_ij.f)
  target_link_libraries(f77_ij PUBLIC HYPRE)
  if(HYPRE_ENABLE_MPI)
    target_link_libraries(f77_ij PUBLIC MPI::MPI_Fortran)
  endif()

  add_executable(f77_struct f77_struct.f)
  target_link_libraries(f77_struct PUBLIC HYPRE)
  if(HYPRE_ENABLE_MPI)
    target_link_libraries(f77_struct PUBLIC MPI::MPI_Fortran)
  endif()
endif()
