############################################################################
#
#  Program:         ARPACK
#
#  Module:          Makefile
#
#  Purpose:         Sources Makefile
#
#  Creation date:   February 22, 1996
#
#  Modified:        September 6, 1996
#
#  Send bug reports, comments or suggestions to arpack.caam.rice.edu
#
############################################################################
#\SCCS Information: @(#)
# FILE: Makefile   SID: 2.3   DATE OF SID: 9/24/96   RELEASE: 2
 
#include ../ARmake.inc
 
############################################################################
#  To create or add to the library, enter make followed by one or
#  more of the precisions desired.  Targets sdrv, ddrv, cdrv,
#  zdrv are used to add to the ARPACK library those BLAS routines needed by 
#  driver programs in the EXAMPLES directory.
#
#  Some examples:
#       make single
#       make single complex
#       make single sdrv
#       make single double complex complex16
#  Alternatively, the command
#       make
#  without any arguments creates a library of all four precisions.
#  The library also contains all extra BLAS routines used by driver
#  programs in the EXAMPLES directory.
#
#  The name of the library is defined by $(ARPACKLIB) in
#  ../ARmake.inc and is created at the next higher directory level.
#
#
SOBJ  = isamax.o sasum.o saxpy.o scopy.o sdot.o  snrm2.o srot.o  srotg.o\
        sscal.o  sswap.o sgemv.o sger.o  ssymv.o ssyr2.o strmm.o sgbmv.o

DOBJ  = idamax.o dasum.o daxpy.o dcopy.o ddot.o  dnrm2.o drot.o  drotg.o\
        dscal.o  dswap.o dgemv.o dger.o  dsymv.o dsyr2.o dtrmm.o dgbmv.o

COBJ  = icamax.o caxpy.o ccopy.o cdotc.o cdotu.o cgbmv.o  cgemv.o cgerc.o\
        cgeru.o  cscal.o csscal.o cswap.o ctrmm.o ctrsv.o scasum.o scnrm2.o\
        isamax.o sscal.o 

ZOBJ  = izamax.o zaxpy.o zcopy.o zdotc.o zdotu.o zgbmv.o  zgemv.o zgerc.o\
        zgeru.o zscal.o zdscal.o zswap.o ztrmm.o ztrsv.o dzasum.o dznrm2.o\
        idamax.o dscal.o

# The following BLAS routine are not required by ARPACK source code.
# They are called by the driver programs in the EXAMPLES directory.
#
SDRVOBJ = sgemm.o strsm.o stbsv.o 
DDRVOBJ = dgemm.o dtrsm.o dtbsv.o
CDRVOBJ = ccopy.o cgemv.o cgeru.o cgemm.o cscal.o cswap.o ctrsm.o ctbsv.o\
          icamax.o 
ZDRVOBJ = zcopy.o zgemv.o zgeru.o zgemm.o zscal.o zswap.o ztrsm.o ztbsv.o\
          izamax.o

.SUFFIXES:      .o .F .f

.f.o:
	$(FC) $(FFLAGS) -c $<
all: single double complex complex16 sdrv ddrv cdrv zdrv
 
single: $(SOBJ)
	$(AR) $(ARFLAGS) $(ARPACKLIB) $(SOBJ)
	$(RANLIB) $(ARPACKLIB)
 
double: $(DOBJ)
	$(AR) $(ARFLAGS) $(ARPACKLIB) $(DOBJ)
	$(RANLIB) $(ARPACKLIB)

complex: $(COBJ)
	$(AR) $(ARFLAGS) $(ARPACKLIB) $(COBJ)
	$(RANLIB) $(ARPACKLIB)

complex16: $(ZOBJ)
	$(AR) $(ARFLAGS) $(ARPACKLIB) $(ZOBJ)
	$(RANLIB) $(ARPACKLIB)
#
# Add routines needed by driver programs (in the EXAMPLES
# directory) to $(ARPACKLIB).
#
sdrv: $(SDRVOBJ) $(CDRVOBJ)
	$(AR) $(ARFLAGS) $(ARPACKLIB) $(SDRVOBJ) $(CDRVOBJ)
	$(RANLIB) $(ARPACKLIB)

ddrv: $(DDRVOBJ) $(ZDRVOBJ)
	$(AR) $(ARFLAGS) $(ARPACKLIB) $(DDRVOBJ) $(ZDRVOBJ)
	$(RANLIB) $(ARPACKLIB)

cdrv: $(CDRVOBJ)
	$(AR) $(ARFLAGS) $(ARPACKLIB) $(CDRVOBJ)
	$(RANLIB) $(ARPACKLIB)

zdrv: $(ZDRVOBJ)
	$(AR) $(ARFLAGS) $(ARPACKLIB) $(ZDRVOBJ)
	$(RANLIB) $(ARPACKLIB)
#
#  clean	- remove all object files
#
clean:
	rm -f *.o a.out core

