#BHEADER***********************************************************************
# (c) 1996   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision$
#EHEADER***********************************************************************

.SUFFIXES:
.SUFFIXES: .c .f .${AMG_ARCH}.o

AMG_OBJS1 = ${AMG_FILES:.c=.${AMG_ARCH}.o}
AMG_OBJS  = ${AMG_OBJS1:.f=.${AMG_ARCH}.o}

##################################################################
# Rules
##################################################################

${AMG_LIBRARY}.${AMG_ARCH}.a: ${AMG_OBJS}
	@echo  "Building ${AMG_LIBRARY}.${AMG_ARCH}.a ... "
	@${AMG_AR} ${AMG_LIBRARY}.${AMG_ARCH}.a ${AMG_OBJS}
	@${AMG_RANLIB} ${AMG_LIBRARY}.${AMG_ARCH}.a

.c.${AMG_ARCH}.o:
	@echo "Making (c) " $@
	@${AMG_CC} -o $@ -c ${CFLAGS} $<

.f.${AMG_ARCH}.o:
	@echo "Making (f) " $@
	@${AMG_F77} -o $@ -c ${FFLAGS} $<

