# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

set(HDRS
  _hypre_parcsr_block_mv.h
)

# Add MUP headers
if(HYPRE_ENABLE_MIXED_PRECISION)
  list(APPEND HDRS
    _hypre_parcsr_block_mv_mup_def.h
  )
endif()

# Regular sources
set(REGULAR_SRCS
  csr_block_matrix.c
  csr_block_matvec.c
  par_csr_block_matrix.c
  par_csr_block_matvec.c
  par_csr_block_comm.c
  par_csr_block_rap.c
  par_csr_block_rap_communication.c
  par_csr_block_interp.c
  par_csr_block_relax.c
  par_block_nodal_systems.c
)

if(HYPRE_ENABLE_MIXED_PRECISION)
  setup_mixed_precision_compilation("parcsr_block_mv" SRCS "${REGULAR_SRCS}")
  target_sources(${PROJECT_NAME} PRIVATE ${HDRS})
else()
  target_sources(${PROJECT_NAME} PRIVATE ${REGULAR_SRCS} ${HDRS})
endif()

convert_filenames_to_full_paths(HDRS)
set(HYPRE_HEADERS ${HYPRE_HEADERS} ${HDRS} PARENT_SCOPE)
