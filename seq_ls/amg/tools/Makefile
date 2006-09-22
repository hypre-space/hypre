#BHEADER**********************************************************************
# Copyright (c) 2006   The Regents of the University of California.
# Produced at the Lawrence Livermore National Laboratory.
# Written by the HYPRE team. UCRL-CODE-222953.
# All rights reserved.
#
# This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
# Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
# disclaimer, contact information and the GNU Lesser General Public License.
#
# HYPRE is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free Software 
# Foundation) version 2.1 dated February 1999.
#
# HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
# WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
# FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
#
# $Revision$
#EHEADER**********************************************************************




.SUFFIXES:
.SUFFIXES: .c .f .${AMG_ARCH}.o

##################################################################
# All
##################################################################

all: convert_icf3d.${AMG_ARCH} convert_ale3d.${AMG_ARCH} \
     kill_ones.${AMG_ARCH}

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
# kill_ones
##################################################################

KILL_ONES_HDRS = general.h matrix.h io.h
KILL_ONES_FILES = kill_ones.c matrix.c write.c
KILL_ONES_OBJS = ${KILL_ONES_FILES:.c=.${AMG_ARCH}.o}

kill_ones.${AMG_ARCH}: ${KILL_ONES_OBJS}
	@echo  "Linking" $@ "... "
	@${AMG_LINK} -o $@ ${KILL_ONES_OBJS} ${LFLAGS}

${KILL_ONES_OBJS}: ${KILL_ONES_HDRS}

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
	@${AMG_INSTALL} kill_ones.${AMG_ARCH} ${AMG_BIN_DIR}/kill_ones

veryclean: clean
	@rm -f convert_icf3d.${AMG_ARCH}
	@rm -f convert_ale3d.${AMG_ARCH}

clean:
	@rm -f *.${AMG_ARCH}.o

