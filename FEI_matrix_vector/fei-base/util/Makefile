#######################################################################
#
#	Second-level makefile for the ISIS++ sparse matrix solver utilities.
#
#######################################################################

SHELL	= /bin/sh

include ../make.options

utils:
	-$(MKDIR) $(MATRIX_DATA_bin)
	-$(MKDIR $(MATRIX_DATA_bin)/$(ISIS_ARCH)
	cd convert; make $@ ISIS_ROOT="$(ISIS_ROOT)"

arpack:
	cd ARPACK; make $@ ISIS_ROOT="$(ISIS_ROOT)"

clean:
	@touch erase.ii
	rm -f *.ii *~ #*

