#BHEADER***********************************************************************
# (c) 1997   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision$
#EHEADER***********************************************************************


.SUFFIXES:
.SUFFIXES: .c .f .o

##################################################################
# Default
##################################################################

DRIVER = StructMatrixAndyDriver
#DRIVER = PETScSolverParILUTDriver

default: $(DRIVER)

##################################################################
# Main rules
##################################################################

# Declarations for this installation site
include ../bmakes/base

$(DRIVER): $(DRIVER).o
	@echo  "Linking" $@ "... "
	${LD} -o $(DRIVER) $(DRIVER).o ${LFLAGS} 

link:
	rm $(DRIVER).o 
	make $(DRIVER)

##################################################################
# Generic rules
##################################################################

.c.o:
	@echo "Making (c) " $@
	${CC} -o $@ -c ${CFLAGS} $<

#	@${CC} -o $@ -c ${CFLAGS} $<

.f.${AMG_ARCH}.o:
	@echo "Making (f) " $@
	@${F77} -o $@ -c ${FFLAGS} $<

##################################################################
# Miscellaneous rules
##################################################################

veryclean: clean
	@rm -f driver libHYPRE.a

clean:
	@rm -f *.o

