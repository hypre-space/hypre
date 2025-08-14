# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

set(HDRS
  HYPRE_error_f.h
  HYPRE_utilities.h
  _hypre_utilities.h
  _hypre_utilities.hpp
  fortran.h
  fortran_matrix.h
  multiprecision.h
)

# Add MUP headers
if(HYPRE_ENABLE_MIXED_PRECISION)
  list(APPEND HDRS
    HYPRE_utilities_mup.h
    _hypre_mup_def.h
    _hypre_utilities_mup.h
    _hypre_utilities_mup_def.h
    _hypre_utilities_mup_undef.h
  )
endif()

# Regular sources
set(REGULAR_SRCS
  F90_HYPRE_error.c
  F90_HYPRE_general.c
  HYPRE_handle.c
  HYPRE_version.c
  amg_linklist.c
  binsearch.c
  exchange_data.c
  fortran_matrix.c
  ap.c
  log.c
  complex.c
  error.c
  int_array.c
  hopscotch_hash.c
  matrix_stats.c
  magma.c
  memory_tracker.c
  merge_sort.c
  mmio.c
  mpi_comm_f2c.c
  nvtx.c
  omp_device.c
  prefix_sum.c
  qsort.c
  unique.c
  utilities.c
  mpistubs.c
  qsplit.c
  random.c
  state.c
  threading.c
  timer.c
  timing.c
  device_utils.c
  general.c
  handle.c
  int_array_device.c
  memory.c
  omp_device.c
  nvtx.c
  stl_ops.c
  caliper_instrumentation.c
)

# Mixed precision sources
set(MUP_SRCS
  multiprecision.c
  mup_fixed.c
  mup_functions.c
  mup_pre.c
  printf.c
)

if (HYPRE_ENABLE_MIXED_PRECISION)
  setup_mixed_precision_compilation("utilities" SRCS "${REGULAR_SRCS}")
  target_sources(${PROJECT_NAME} PRIVATE ${MUP_SRCS} ${HDRS})
else()
  list(APPEND REGULAR_SRCS
    multiprecision.c
    printf.c
  )
  target_sources(${PROJECT_NAME} PRIVATE ${REGULAR_SRCS} ${HDRS})
endif()

if (HYPRE_USING_GPU)
  set(GPU_SRCS
    device_utils.c
    general.c
    handle.c
    int_array_device.c
    memory.c
    memory_tracker.c
    nvtx.c
    omp_device.c
    stl_ops.c
    HYPRE_handle.c
  )
  convert_filenames_to_full_paths(GPU_SRCS)
  set(HYPRE_GPU_SOURCES ${HYPRE_GPU_SOURCES} ${GPU_SRCS} PARENT_SCOPE)
endif ()

convert_filenames_to_full_paths(HDRS)
set(HYPRE_HEADERS ${HYPRE_HEADERS} ${HDRS} PARENT_SCOPE)
