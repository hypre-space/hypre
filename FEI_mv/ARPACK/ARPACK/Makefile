############################################################################
#
#  Program:         ARPACK
#
#  Module:          Makefile
#
#  Purpose:         Top-level Makefile
#
#  Creation date:   February 22, 1996
#
#  Modified:
#
#  Send bug reports, comments or suggestions to arpack@caam.rice.edu
#
############################################################################

include ARmake.inc

PRECISIONS = double  ddrv
#PRECISIONS = single double complex complex16
#PRECISIONS = double ddrv
#  If LAPACK and BLAS are not installed on you system, and you are going 
#  to use the driver programs in the EXAMPLES directory,  you need to add 
#  sdrv(for single precision), ddrv(for double precision), 
#  cdrv(for single precision complex), or zdrv(for double precision complex) 
#  to the definition of PRECISION.

############################################################################
#
#  The library can be set up to include routines for any combination of the
#  four PRECISIONS.  First, modify the definitions in ARmake.inc to match 
#  your library archiver, compiler, and the options to be used.
#
#  Sample ARmake.inc's can be found in the directory ARMAKES
#  
#  Then to create or add to the library, enter make lib after having
#  modified the environment variable PRECISIONS defined in this Makefile.
#
#       make lib
#  creates the library for serial ARPACK,
#
#  The name of the libraries are defined in the file called ARmake.inc and
#  are created at this directory level.
#
#  To remove the object files after the libraries and testing executables
#  are created, enter
#       make clean
#
############################################################################

all: lib 

lib: arpacklib

clean: cleanlib

arpacklib:
	@( \
	for f in $(DIRS); \
	do \
		$(CD) $$f; \
		$(ECHO) Making lib in $$f; \
		$(MAKE) $(PRECISIONS); \
		$(CD) ..; \
	done );
	$(RANLIB) $(ARPACKLIB)

cleantest:

cleanlib:
	( cd $(BLASdir); $(MAKE) clean )
	( cd $(LAPACKdir); $(MAKE) clean )
	( cd $(UTILdir); $(MAKE) clean )
	( cd $(SRCdir); $(MAKE) clean )

help:
    @$(ECHO) "usage: make ?"
