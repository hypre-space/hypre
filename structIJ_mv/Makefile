CC= mpicc

CDEFS = 

INCLUDEDIRS = \
  -I../hypre/include

CFLAGS =  -g ${INCLUDEDIRS} ${CDEFS}

.c.o:
	${CC} -o $@ -c ${CFLAGS} $<

OBJS = \
 HYPRE_structIJ_matrix.o \
 HYPRE_structIJ_vector.o \
 structIJ_matrix.o \
 structIJ_vector.o 

MPILIBFLAGS =    
LIBFLAGS =   -lm
LDLIBFLAGS =  
BLASLIBFLAGS =  -lsunperf -L/opt/SUNWspro/lib -R /opt/SUNWspro/lib -L/usr/lib -L/usr/ccs/lib -L/opt/SUNWspro/SC4.2/lib -L/opt/SUNWspro/lib -L/usr/local/mpi/mpich-1.1.1/lib/solaris/ch_p4 -lpmpich -lmpich -lsocket -lnsl -lpthread -laio /opt/SUNWspro/SC4.2/lib/libM77.a -lF77 -lsunmath -lm -lSUNWPro_lic

LFLAGS =\
 -L../hypre/lib\
 -L../CI_struct_matrix_vector\
 -lHYPRE_CI_mv \
 -lHYPRE_ls\
 -lHYPRE_mv\
 -lHYPRE_parcsr_ls\
 -lHYPRE_ParaSails \
 -lHYPRE_MatrixMatrix \
 -lHYPRE_DistributedMatrix \
 -lHYPRE_IJ_mv\
 -lHYPRE_parcsr_mv\
 -lHYPRE_seq_mv\
 -lHYPRE_utilities\
 ${BLASLIBFLAGS} \
 ${MPILIBFLAGS} ${LIBFLAGS} ${LDLIBFLAGS}

a.out: driver.o ${OBJS}
	${CC} driver.o ${OBJS} ${LFLAGS}
