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

${AMG_FDRIVER}.${AMG_ARCH}: ${AMG_OBJS}
	@echo  "Linking" ${AMG_FDRIVER} "... "
	@${AMG_LINK} -o ${AMG_FDRIVER}.${AMG_ARCH} \
		${LFLAGS} ${AMG_OBJS} -L${AMG_DIR}/lib -lamg \
	        ${AMG_LIBS}

.c.${AMG_ARCH}.o:
	@echo "Making (c) " $@
	@${AMG_CC} -o $@ -c ${CFLAGS} $<

.f.${AMG_ARCH}.o:
	@echo "Making (f) " $@
	@${AMG_F77} -o $@ -c ${FFLAGS} $<

