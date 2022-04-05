# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

set(HDRS
  _hypre_blas.h
  f2c.h
  hypre_blas.h
)

set(SRCS
  dasum.c
  daxpy.c
  dcopy.c
  ddot.c
  dgemm.c
  dgemv.c
  dger.c
  dnrm2.c
  drot.c
  dscal.c
  dswap.c
  dsymm.c
  dsymv.c
  dsyr2.c
  dsyr2k.c
  dsyrk.c
  dtrmm.c
  dtrmv.c
  dtrsm.c
  dtrsv.c
  f2c.c
  idamax.c
  lsame.c
  xerbla.c
)

target_sources(${PROJECT_NAME}
  PRIVATE ${SRCS}
          ${HDRS}
)

convert_filenames_to_full_paths(HDRS)
set(HYPRE_HEADERS ${HYPRE_HEADERS} ${HDRS} PARENT_SCOPE)

