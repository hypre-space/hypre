CC = gcc
# CC = mpicc
CLASS_DIR = /home/painter/linear_solvers/babel/cfiles
HYPRE_INCLUDES = -I/home/casc/software/hypre/internal/include -I/home/painter/linear_solvers/struct_matrix_vector
BABEL_INCLUDE = /home/painter/babel-0.1/babel_runtime
SKEL_DIR = /home/painter/babel-0.1/BABEL/babel__proxies
BABEL_DIR = /home/painter/babel-0.1/BABEL
BABEL_SKEL_DIR = /home/painter/babel-0.1/BABEL/babel__proxies
BABEL_STRUCT_DIR = /home/painter/babel-0.1/GOV/llnl/babel
HYPRE_LIBFLAGS = -L/home/painter/linear_solvers/utilities -L/home/painter/linear_solvers/struct_matrix_vector -L/home/painter/linear_solvers/struct_linear_solvers
OPT = -g
PURIFY =
# PURIFY = purify

# MPILIBS not needed with mpicc...
MPILIBS = -L/usr/local/mpi/mpich/lib/solaris/ch_p4 -lmpich  -lsocket -lnsl -lpthread -lnsl 

OBJS = \
${CLASS_DIR}/*.o \
${SKEL_DIR}/*.o \
${BABEL_DIR}/*.o \
${BABEL_SKEL_DIR}/*.o

objects:
	${CC} -DHYPRE_SEQUENTIAL -c $(OPT) *.c -I${BABEL_INCLUDE} ${HYPRE_INCLUDES} -I${CLASS_DIR} -I${CLASS_DIR} -I${SKEL_DIR} -I${BABEL_DIR} -I${BABEL_SKEL_DIR} -I${BABEL_STRUCT_DIR} 
	ar ru libhypre_class.a *.o

testum.o:
	${CC} -DHYPRE_SEQUENTIAL -c $(OPT) testum.c -I${BABEL_INCLUDE} ${HYPRE_INCLUDES} -I${CLASS_DIR} -I${CLASS_DIR} -I${SKEL_DIR} -I${BABEL_DIR} -I${BABEL_SKEL_DIR} -I${BABEL_STRUCT_DIR} 

stub.o:
	${CC} -DHYPRE_SEQUENTIAL -c $(OPT) stub.c -I${BABEL_INCLUDE} ${HYPRE_INCLUDES} -I${CLASS_DIR} -I${CLASS_DIR} -I${SKEL_DIR} -I${BABEL_DIR} -I${BABEL_SKEL_DIR} -I${BABEL_STRUCT_DIR} 

testum: testum.o objects
	${PURIFY} ${CC} testum.o -otestum -L. -lhypre_class -L${BABEL_DIR} -lbabel -L/home/painter/babel-0.1/babel_runtime -lbabel_env -L${BABEL_DIR}/babel__proxies -lbabel_skel ${HYPRE_LIBFLAGS} -lHYPRE_ls -lHYPRE_mv -lHYPRE_utilities ${MPILIBS} -lm

test: testum

stub: stub.o
	${CC} stub.o -ostub -L. -lhypre_class -L${BABEL_DIR} -lbabel -L/home/painter/babel-0.1/babel_runtime -lbabel_env -L${BABEL_DIR}/babel__proxies -lbabel_skel ${HYPRE_LIBFLAGS} -lHYPRE_mv -lHYPRE_utilities -L/usr/local/lib/gcc-lib/sparc-sun-solaris2.5/2.7.2/ -lgcc

all: objects test

