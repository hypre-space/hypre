# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

set(HDRS
  HYPRE_krylov.h
  HYPRE_lobpcg.h
  HYPRE_MatvecFunctions.h
  krylov.h
  lobpcg.h
)

set(SRCS
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

target_sources(${PROJECT_NAME}
  PRIVATE ${SRCS}
          ${HDRS}
)

convert_filenames_to_full_paths(HDRS)
set(HYPRE_HEADERS ${HYPRE_HEADERS} ${HDRS} PARENT_SCOPE)
