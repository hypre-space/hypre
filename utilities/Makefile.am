## Process this file with automake to create Makefile.in

if STATICONLY
 lib_LIBRARIES = libHYPRE_utilities.a 
 libHYPRE_utilities_a_SOURCES = $(FILES)
else !STATICONLY
 lib_LTLIBRARIES = libHYPRE_utilities.la 
 libHYPRE_utilities_la_SOURCES = $(FILES)
 libHYPRE_utilities_la_LDFLAGS = -avoid-version
endif !STATICONLY

include_HEADERS = utilities.h HYPRE_utilities.h

AM_CPPFLAGS = -DHYPRE_TIMING

#
FILES = \
 amg_linklist.c\
 binsearch.c\
 hypre_memory.c\
 hypre_qsort.c\
 memory_dmalloc.c\
 mpistubs.c\
 qsplit.c\
 random.c\
 thread_mpistubs.c\
 threading.c\
 timer.c\
 timing.c\
 umalloc_local.c\
 HYPRE_utilities.h\
 amg_linklist.h\
 fortran.h\
 general.h\
 hypre_memory.h\
 hypre_smp_forloop.h\
 mpistubs.h\
 thread_mpistubs.h\
 threading.h\
 timing.h\
 umalloc_local.h\
 utilities.h

