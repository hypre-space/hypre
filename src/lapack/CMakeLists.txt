# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

set(HDRS
  _hypre_lapack.h
)

set(SRCS
  dbdsqr.c
  dgebd2.c
  dgebrd.c
  dgelq2.c
  dgelqf.c
  dgels.c
  dgeqr2.c
  dgeqrf.c
  dgesvd.c
  dgetrf.c
  dgetri.c
  dgetrs.c
  dgetf2.c
  dlabad.c
  dlabrd.c
  dlacpy.c
  dlae2.c
  dlaev2.c
  dlamch.c
  dlange.c
  dlanst.c
  dlansy.c
  dlapy2.c
  dlarfb.c
  dlarf.c
  dlarfg.c
  dlarft.c
  dlartg.c
  dlas2.c
  dlascl.c
  dlaset.c
  dlasq1.c
  dlasq2.c
  dlasq3.c
  dlasq4.c
  dlasq5.c
  dlasq6.c
  dlasr.c
  dlasrt.c
  dlassq.c
  dlaswp.c
  dlasv2.c
  dlatrd.c
  dorg2l.c
  dorg2r.c
  dorgbr.c
  dorgl2.c
  dorglq.c
  dorgql.c
  dorgqr.c
  dorgtr.c
  dorm2r.c
  dormbr.c
  dorml2.c
  dormlq.c
  dormqr.c
  dpotf2.c
  dpotrf.c
  dpotrs.c
  dsteqr.c
  dsterf.c
  dsyev.c
  dsygs2.c
  dsygst.c
  dsygv.c
  dsytd2.c
  dsytrd.c
  dtrti2.c
  dtrtri.c
  ieeeck.c
  ilaenv.c
  lsame.c
  xerbla.c
)

# Turn optimization off for this file
if (MSVC)
  set_source_files_properties (dlamch.c PROPERTIES COMPILE_FLAGS /Od)
else ()
  set_source_files_properties (dlamch.c PROPERTIES COMPILE_FLAGS -O0)
endif ()

target_sources(${PROJECT_NAME}
  PRIVATE ${SRCS}
          ${HDRS}
)

convert_filenames_to_full_paths(HDRS)
set(HYPRE_HEADERS ${HYPRE_HEADERS} ${HDRS} PARENT_SCOPE)
