## Process this file with automake to create Makefile.in

if STATICONLY
 lib_LIBRARIES = libHYPRE_lapack.a
 libHYPRE_lapack_a_SOURCES = $(FILES)
else !STATICONLY
 lib_LTLIBRARIES = libHYPRE_lapack.la
 libHYPRE_lapack_la_SOURCES = $(FILES)
 libHYPRE_lapack_la_LDFLAGS = -avoid-version
endif !STATICONLY

#
FILES = \
 dbdsqr.c\
 dgebd2.c\
 dgebrd.c\
 dgelq2.c\
 dgelqf.c\
 dgels.c\
 dgeqr2.c\
 dgeqrf.c\
 dgesvd.c\
 dlabad.c\
 dlabrd.c\
 dlacpy.c\
 dlae2.c\
 dlaev2.c\
 dlamch.c\
 dlange.c\
 dlanst.c\
 dlansy.c\
 dlapy2.c\
 dlarf.c\
 dlarfb.c\
 dlarfg.c\
 dlarft.c\
 dlartg.c\
 dlascl.c\
 dlaset.c\
 dlas2.c\
 dlasq1.c\
 dlasq2.c\
 dlasq3.c\
 dlasq4.c\
 dlasq5.c\
 dlasq6.c\
 dlasr.c\
 dlasrt.c\
 dlassq.c\
 dlasv2.c\
 dlatrd.c\
 dorg2l.c\
 dorg2r.c\
 dorgbr.c\
 dorgl2.c\
 dorglq.c\
 dorgql.c\
 dorgqr.c\
 dorgtr.c\
 dorm2r.c\
 dormbr.c\
 dorml2.c\
 dormlq.c\
 dormqr.c\
 dpotf2.c\
 dpotrf.c\
 dpotrs.c\
 dsteqr.c\
 dsterf.c\
 dsyev.c\
 dsygs2.c\
 dsygst.c\
 dsygv.c\
 dsytd2.c\
 dsytrd.c\
 ieeeck.c\
 ilaenv.c\
 lapack_utils.c\
 lsame.c\
 xerbla.c\
 f2c.h\
 hypre_lapack.h
