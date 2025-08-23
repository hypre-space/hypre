# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

set(HDRS
  HYPRE_parcsr_mv.h
  _hypre_parcsr_mv.h
)

# Add MUP headers
if(HYPRE_ENABLE_MIXED_PRECISION)
  list(APPEND HDRS
    HYPRE_parcsr_mv_mup.h
    _hypre_parcsr_mv_mup.h
    _hypre_parcsr_mv_mup_def.h
    _hypre_parcsr_mv_mup_undef.h
  )
endif()

# Regular sources
set(REGULAR_SRCS
  communicationT.c
  F90_HYPRE_parcsr_matrix.c
  F90_HYPRE_parcsr_vector.c
  F90_parcsr_matrix.c
  F90_par_vector.c
  HYPRE_parcsr_matrix.c
  HYPRE_parcsr_vector.c
  gen_fffc.c
  new_commpkg.c
  numbers.c
  par_csr_aat.c
  par_csr_assumed_part.c
  par_csr_bool_matop.c
  par_csr_bool_matrix.c
  par_csr_communication.c
  par_csr_filter.c
  par_csr_matop.c
  par_csr_matrix.c
  par_csr_matrix_stats.c
  par_csr_matmat.c
  par_csr_matvec.c
  par_csr_matop_marked.c
  par_csr_triplemat.c
  par_make_system.c
  par_vector.c
  par_vector_batched.c
  par_csr_fffc_device.c
  par_csr_filter_device.c
  par_csr_matop_device.c
  par_csr_matmat_device.c
  par_csr_matvec_device.c
  par_csr_triplemat_device.c
  par_vector_device.c
)

# Mixed precision sources
set(MUP_SRCS
  mup_fixed.c
  mup_functions.c
  mup_pre.c
  parcsr_mv_mp.c
  HYPRE_parcsr_mv_mp.c
)

if(HYPRE_ENABLE_MIXED_PRECISION)
  setup_mixed_precision_compilation("parcsr_mv" SRCS "${REGULAR_SRCS}")
  target_sources(${PROJECT_NAME} PRIVATE ${MUP_SRCS} ${HDRS})
else()
  target_sources(${PROJECT_NAME} PRIVATE ${REGULAR_SRCS} ${HDRS})
endif()

if (HYPRE_USING_GPU)
  set(GPU_SRCS
    par_csr_matmat_device.c
    par_csr_matvec_device.c
    par_csr_fffc_device.c
    par_csr_filter_device.c
    par_csr_matop_device.c
    par_csr_triplemat_device.c
    par_vector_device.c
  )
  convert_filenames_to_full_paths(GPU_SRCS)
  set(HYPRE_GPU_SOURCES ${HYPRE_GPU_SOURCES} ${GPU_SRCS} PARENT_SCOPE)
endif ()

convert_filenames_to_full_paths(HDRS)
set(HYPRE_HEADERS ${HYPRE_HEADERS} ${HDRS} PARENT_SCOPE)
