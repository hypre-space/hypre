# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

set(HDRS
  HYPRE_utilities.h
  _hypre_utilities.h
  _hypre_utilities.hpp
  HYPRE_error_f.h
  fortran.h
  fortran_matrix.h
)

set(SRCS
  HYPRE_handle.c
  HYPRE_version.c
  amg_linklist.c
  binsearch.c
  exchange_data.c
  F90_HYPRE_error.c
  F90_HYPRE_general.c
  fortran_matrix.c
  ap.c
  log.c
  complex.c
  device_utils.c
  error.c
  general.c
  handle.c
  int_array.c
  int_array_device.c
  hopscotch_hash.c
  matrix_stats.c
  magma.c
  memory.c
  memory_tracker.c
  merge_sort.c
  mmio.c
  mpi_comm_f2c.c
  nvtx.c
  omp_device.c
  prefix_sum.c
  printf.c
  qsort.c
  utilities.c
  mpistubs.c
  qsplit.c
  random.c
  state.c
  threading.c
  timer.c
  timing.c
)

target_sources(${PROJECT_NAME}
  PRIVATE ${SRCS}
          ${HDRS}
)

if (HYPRE_USING_CUDA OR HYPRE_USING_SYCL)
  set(GPU_SRCS
    device_utils.c
    general.c
    handle.c
    int_array_device.c
    memory.c
    memory_tracker.c
    nvtx.c
    omp_device.c
    HYPRE_handle.c
  )
  convert_filenames_to_full_paths(GPU_SRCS)
  set(HYPRE_GPU_SOURCES ${HYPRE_GPU_SOURCES} ${GPU_SRCS} PARENT_SCOPE)
endif ()

convert_filenames_to_full_paths(HDRS)
set(HYPRE_HEADERS ${HYPRE_HEADERS} ${HDRS} PARENT_SCOPE)
