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

##################################################################
# All
##################################################################

all: convert_icf3d.${AMG_ARCH} convert_ale3d.${AMG_ARCH}

##################################################################
# convert_icf3d
##################################################################

CONVERT_ICF3D_HDRS = general.h matrix.h io.h
CONVERT_ICF3D_FILES = convert_icf3d.c matrix.c write.c
CONVERT_ICF3D_OBJS = ${CONVERT_ICF3D_FILES:.c=.${AMG_ARCH}.o}

convert_icf3d.${AMG_ARCH}: ${CONVERT_ICF3D_OBJS}
	@echo  "Linking" $@ "... "
	@${AMG_LINK} -o $@ ${CONVERT_ICF3D_OBJS} ${LFLAGS}

${CONVERT_ICF3D_OBJS}: ${CONVERT_ICF3D_HDRS}

##################################################################
# convert_ale3d
##################################################################

CONVERT_ALE3D_HDRS = general.h matrix.h io.h
CONVERT_ALE3D_FILES = convert_ale3d.c matrix.c write.c
CONVERT_ALE3D_OBJS = ${CONVERT_ALE3D_FILES:.c=.${AMG_ARCH}.o}

convert_ale3d.${AMG_ARCH}: ${CONVERT_ALE3D_OBJS}
	@echo  "Linking" $@ "... "
	@${AMG_LINK} -o $@ ${CONVERT_ALE3D_OBJS} ${LFLAGS}

${CONVERT_ALE3D_OBJS}: ${CONVERT_ALE3D_HDRS}

##################################################################
# Generic rules
##################################################################

.c.${AMG_ARCH}.o:
	@echo "Making (c) " $@
	@${AMG_CC} -o $@ -c ${CFLAGS} $<

.f.${AMG_ARCH}.o:
	@echo "Making (f) " $@
	@${AMG_F77} -o $@ -c ${FFLAGS} $<

##################################################################
# Miscellaneous rules
##################################################################

install:
	@${AMG_INSTALL} convert_icf3d.${AMG_ARCH} ${AMG_BIN_DIR}/convert_icf3d
	@${AMG_INSTALL} convert_ale3d.${AMG_ARCH} ${AMG_BIN_DIR}/convert_ale3d

veryclean: clean
	@rm -f convert_icf3d.${AMG_ARCH}
	@rm -f convert_ale3d.${AMG_ARCH}

clean:
	@rm -f *.${AMG_ARCH}.o

