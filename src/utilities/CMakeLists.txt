# Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

set(HDRS
  HYPRE_utilities.h
  _hypre_utilities.h
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
  cuda_utils.c
  error.c
  general.c
  handle.c
  hopscotch_hash.c
  memory.c
  merge_sort.c
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
  threading.c
  timer.c
  timing.c
)

target_sources(${PROJECT_NAME}
  PRIVATE ${SRCS}
          ${HDRS}
)

if (HYPRE_USING_CUDA)
  set(CUDA_SRCS
    HYPRE_handle.c
    cuda_utils.c
    handle.c
    general.c
    memory.c
    omp_device.c
    nvtx.c
  )
  convert_filenames_to_full_paths(CUDA_SRCS)
  set(HYPRE_CUDA_SOURCES ${HYPRE_CUDA_SOURCES} ${CUDA_SRCS} PARENT_SCOPE)
endif ()

convert_filenames_to_full_paths(HDRS)
set(HYPRE_HEADERS ${HYPRE_HEADERS} ${HDRS} PARENT_SCOPE)
