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

set(TEST_SRCS
  ij.c
  sstruct.c
  struct.c
  ams_driver.c
  maxwell_unscaled.c
  struct_migrate.c
  sstruct_fac.c
  ij_assembly.c
)

add_hypre_executables(TEST_SRCS)

# Copy test input files
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/TEST_sstruct/sstruct.in.default
               ${CMAKE_CURRENT_BINARY_DIR}/sstruct.in.default COPYONLY)

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
