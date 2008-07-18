#BHEADER**********************************************************************
# Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# This file is part of HYPRE.  See file COPYRIGHT for details.
#
# HYPRE is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
# $Revision$
#EHEADER**********************************************************************








.SUFFIXES:
.SUFFIXES: .c .f .${AMG_ARCH}.o

AMG_OBJS1 = ${AMG_FILES:.c=.${AMG_ARCH}.o}
AMG_OBJS  = ${AMG_OBJS1:.f=.${AMG_ARCH}.o}

##################################################################
# Main rules
##################################################################

${AMG_DRIVER}.${AMG_ARCH}: ${AMG_OBJS}
	@echo  "Linking" $@ "... "
	@${AMG_LINK} -o $@ ${AMG_OBJS} ${LFLAGS}

${AMG_OBJS}: ${AMG_HEADERS}

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
	@${AMG_INSTALL} ${AMG_DRIVER}.${AMG_ARCH} ${AMG_BIN_DIR}/${AMG_DRIVER}
	@${AMG_INSTALL} amgrun ${AMG_BIN_DIR}/.

veryclean: clean
	@rm -f ${AMG_DRIVER}.${AMG_ARCH}

clean:
	@rm -f *.${AMG_ARCH}.o

