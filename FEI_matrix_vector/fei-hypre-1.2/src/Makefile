FEI_ROOT=..
FEI_DIR=src

default:
	cd $(FEI_ROOT); make $(FEI_DIR) FEI_DIR=$(FEI_DIR)

clean:
	cd $(FEI_ROOT); make clean.$(FEI_DIR) FEI_DIR=$(FEI_DIR)

##################################################################
#### Newly added for testing the HYPRE_LinSysCore
##################################################################

CXX = cxx

CXXFLAGS = -w -DSUPERLU -DMLPACK \
 -I../../isis-mv\
 -I../../../IJ_matrix_vector\
 -I../../../utilities\
 -I../../../parcsr_matrix_vector\
 -I../../../parcsr_linear_solvers\
 -I../../../seq_matrix_vector\
 -I/g/g18/chtong/SuperLU/SRC\
 -I../../..

MPILIBFLAGS =  -L/usr/local/mpich-1.1.2shm/build/alpha/ch_shmem/lib -lpmpich \
               -lmpich -lrpc -lgs -lpthread
LIBFLAGS =   -lm
LDLIBFLAGS =

LFLAGS =\
 -L.\
 -L../lib/alpha\
 -L../../isis-mv\
 -L../../../IJ_matrix_vector\
 -L../../../utilities\
 -L../../../parcsr_matrix_vector\
 -L../../../parcsr_linear_solvers\
 -L../../../seq_matrix_vector\
 -L../../../distributed_matrix\
 -L../../../distributed_linear_solvers/pilut\
 -L../../../distributed_linear_solvers/ParaSails\
 -L../../../matrix_matrix\
 -L/g/g18/chtong/SuperLU\
 -L/g/g18/chtong/ML/ml/src\
 -lfei_isis\
 -lHYPRE_parcsr_ls \
 -lHYPRE_ParaSails \
 -lHYPRE_FEI_isismv \
 -lHYPRE_IJ_mv \
 -lml \
 -lHYPRE_utilities \
 -lHYPRE_parcsr_mv \
 -lHYPRE_MatrixMatrix \
 -lHYPRE_DistributedMatrix \
 -lHYPRE_DistributedMatrixPilutSolver \
 -lsuperlu_alpha \
 -lHYPRE_seq_mv \
 ${MPILIBFLAGS} ${LIBFLAGS} ${LDLIBFLAGS}\
 -ldxml -lm

.SUFFIXES: .cc .c .f .o

FILES = test.cc

OBJS = ${FILES:.cc=.o}

test: $(OBJS) ../lib/alpha/libfei_isis.a
	@echo  "Linking" $@ "... "
	${CXX} -o $@ test.o ${LFLAGS}

.cc.o:
	${CXX} -o $@ -c ${CXXFLAGS} $<


