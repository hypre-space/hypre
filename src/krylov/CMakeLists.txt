# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

set(HDRS
  HYPRE_krylov.h
  HYPRE_lobpcg.h
  _hypre_krylov.h
  _hypre_lobpcg.h
)

# Add MUP headers when mixed precision is enabled
if(HYPRE_ENABLE_MIXED_PRECISION)
  list(APPEND HDRS
    HYPRE_krylov_mup.h
    HYPRE_lobpcg_mup.h
    _hypre_krylov_mup.h
    _hypre_lobpcg_mup.h
    _hypre_krylov_mup_def.h
    _hypre_krylov_mup_undef.h
  )
endif()

# Regular sources
set(REGULAR_SRCS
  bicgstab.c
  cgnr.c
  cogmres.c
  gmres.c
  flexgmres.c
  lgmres.c
  HYPRE_bicgstab.c
  HYPRE_cgnr.c
  HYPRE_gmres.c
  HYPRE_cogmres.c
  HYPRE_lgmres.c
  HYPRE_flexgmres.c
  HYPRE_pcg.c
  pcg.c
  HYPRE_lobpcg.c
  lobpcg.c
)

# Mixed precision sources
set(MUP_SRCS
  mup_fixed.c
  mup_functions.c
  mup_pre.c
  mup_lobpcg_fixed.c
  mup_lobpcg_functions.c
  mup_lobpcg_pre.c
)

if(HYPRE_ENABLE_MIXED_PRECISION)
  setup_mixed_precision_compilation("krylov" SRCS "${REGULAR_SRCS}")
  target_sources(${PROJECT_NAME} PRIVATE ${MUP_SRCS} ${HDRS})
else()
  target_sources(${PROJECT_NAME} PRIVATE ${REGULAR_SRCS} ${HDRS})
endif()

convert_filenames_to_full_paths(HDRS)
set(HYPRE_HEADERS ${HYPRE_HEADERS} ${HDRS} PARENT_SCOPE)
