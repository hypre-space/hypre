# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

set(HDRS
  _hypre_lobpcg_interpreter.h
  _hypre_lobpcg_multivector.h
  _hypre_lobpcg_temp_multivector.h
  _hypre_lobpcg_csr_matmultivec.h
  _hypre_lobpcg_par_csr_matmultivec.h
  _hypre_lobpcg_par_csr_pmvcomm.h
  _hypre_lobpcg_par_multivector.h
  _hypre_lobpcg_seq_multivector.h
)

# Add MUP headers
if(HYPRE_ENABLE_MIXED_PRECISION)
  list(APPEND HDRS
    _hypre_multivector_mup_def.h
  )
endif()

# Regular sources
set(REGULAR_SRCS
  multivector.c
  temp_multivector.c
)

if (HYPRE_ENABLE_MIXED_PRECISION)
  setup_mixed_precision_compilation("multivector" SRCS "${REGULAR_SRCS}")
endif()
target_sources(${PROJECT_NAME} PRIVATE ${REGULAR_SRCS} ${HDRS})

convert_filenames_to_full_paths(HDRS)
set(HYPRE_HEADERS ${HYPRE_HEADERS} ${HDRS} PARENT_SCOPE)
