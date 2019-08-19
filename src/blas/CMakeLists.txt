# Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

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

convert_filenames_to_full_paths(SRCS)

set(HYPRE_SOURCES ${HYPRE_SOURCES} ${SRCS} PARENT_SCOPE)


