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

HEADERS =\
 box.h\
 struct_grid.h\
 grid_to_coord.h\
 headers.h\
 struct_stencil.h\
 struct_matrix.h\
 struct_vector.h
 
LOCALFILES =\
 box.c\
 struct_grid.c\
 grid_to_coord.c\
 struct_stencil.c\
 struct_matrix.c\
 struct_matrix_PETSc.c\
 struct_vector.c\
 struct_vector_PETSc.c

HYPREFILES =\
 HYPRE_struct_grid.c\
 HYPRE_struct_stencil.c\
 HYPRE_struct_matrix.c\
 HYPRE_struct_vector.c

FILES = $(HYPREFILES) $(LOCALFILES)

OBJS = ${FILES:.c=.o}

# Declarations for this installation site
include ../bmakes/base

##################################################################
# Main rules
##################################################################

driver: driver.o lib
	@echo  "Linking" $@ "... "
	${LD} -o driver driver.o ${LFLAGS} 

link:
	rm driver
	make driver

lib: libHYPRE_StructMatrix.a

libHYPRE_StructMatrix.a: $(PROTOS) ${OBJS}
	@echo  "Building $@ ... "
	@ar -ru libHYPRE_StructMatrix.a ${OBJS}
	@ranlib libHYPRE_StructMatrix.a

${OBJS}: ${HEADERS}

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

