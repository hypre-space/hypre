# Generated automatically from Makefile.in by configure.
#BHEADER***********************************************************************
# (c) 1998   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision$
#EHEADER***********************************************************************

.SUFFIXES:
.SUFFIXES: .c .f .o

srcdir = .
 
#CC = purify cc
CC = mpicc

C_COMPILE_FLAGS= -g  
CXX_COMPILE_FLAGS= -g  
F77_COMPILE_FLAGS=   
CINCLUDES=  -I/usr/local/mpi/mpich-1.2.0//include
CDEFS =

CFLAGS =\
 ${C_COMPILE_FLAGS}\
 -I$(srcdir)\
 -I$(srcdir)/../..\
 -I$(srcdir)/../../utilities\
 -I$(srcdir)/../../seq_mv\
 ${CINCLUDES}\
 ${CDEFS}
 
RANLIB= ranlib
 
MPILIBFLAGS =  -L/usr/local/mpi/mpich-1.2.0//lib  -lmpich -lsocket -lnsl  
LIBFLAGS =   -lm
LDLIBFLAGS =  
 
LFLAGS =\
 -L.\
 -lX11\
 -L../../seq_mv\
 -L../../utilities\
 -lHYPRE_amge_ls\
 -lHYPRE_seq_mv\
 -lHYPRE_utilities\
 ${MPILIBFLAGS} ${LIBFLAGS} ${LDLIBFLAGS}
 
EISPACK = /home/panayot/eispack/eispack/rs.o /home/panayot/eispack/eispack/tql1.o /home/panayot/eispack/eispack/tql2.o /home/panayot/eispack/eispack/tred1.o /home/panayot/eispack/eispack/tred2.o /home/panayot/eispack/eispack/pythag.o

HEADERS =\
 headers.h\
 AMGe_matrix_topology.h\
 HYPRE_amge.h

FILES =\
 driver_spectral_amge_triangles.c\
 amge_initial_graphs.c\
 amge_topology_setup.c\
 amge_spectral_interpolation_setup.c\
 matrix_topology.c\
 AMGe_matrix_topology.c\
 amge_coarsen_topology.c\
 transpose_matrix_create.c\
 matrix_matrix_product.c\
 create_maximal_intersection_sets.c\
 matrix_matrix_t_product.c\
 agglomerate.c\
 coarse_node_selection.c\
 build_spectral_amge_interpolation.c\
 Schur_complement.c\
 create_element_matrices_dofs.c\
 domain_assemble.c\
 compute_dof_function.c\
 compute_dof_node.c\
 amge_elmmat_read.c\
 boundarysurfaces.c\
 assemble.c\
 rap.c\
 compute_dof_on_boundary.c\
 sym_GS_solve.c\
 sym_GS_pcg.c\
 V_cycle_GS_smoothing.c\
 V_cycle_GS_pcg.c\
 matrix_inverse.c\
 graphics_2d.c\
 xgraph0.c



OBJS = ${FILES:.c=.o}




##################################################################
# Targets
##################################################################

all: libHYPRE_amge_ls.a driver

install: all

clean:
	@rm -f *.o

veryclean: clean
	@rm -f driver
	@rm -f libHYPRE_amge_ls.a

##################################################################
# Rules
##################################################################

driver: driver_spectral_amge_triangles.o libHYPRE_amge_ls.a
	@echo  "Linking" $@ "... "
	${CC} -o driver driver_spectral_amge_triangles.o ${EISPACK} ${LFLAGS}



libHYPRE_amge_ls.a: ${OBJS}
	@echo  "Building $@ ... "
	ar -rcu $@ ${OBJS}
	${RANLIB} $@

${OBJS}: ${HEADERS}

##################################################################
# Generic rules
##################################################################

.c.o:
	${CC} -o $@ -c ${CFLAGS} $<
