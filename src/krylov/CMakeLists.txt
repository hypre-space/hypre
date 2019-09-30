# Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
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
  
convert_filenames_to_full_paths(HDRS)
convert_filenames_to_full_paths(SRCS)

set(HYPRE_HEADERS ${HYPRE_HEADERS} ${HDRS} PARENT_SCOPE)
set(HYPRE_SOURCES ${HYPRE_SOURCES} ${SRCS} PARENT_SCOPE)


