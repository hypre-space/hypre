include $(PCPACK_DIR)/bmake/bmake_${WHICH_MAKE}/common

CFLAGS = ${CFLAGS_DH} -DPCPACK_PETSC_LIB=${DH_PETSC_LIB}
FFLAGS = ${FFLAGS_DH}

objs:  Mem_dh.o Parser_dh.o Timer_dh.o globalObjects.o  \
       blas_dh.o krylov_dh.o TimeLog_dh.o io_dh.o shellSort_dh.o \
       Hash_dh.o Numbering_dh.o SortedList_dh.o Vec_dh.o    \
       getRow_dh.o petsc_euclid.o MatGenFD.o Mat_dh.o \
       ilu_seq.o Euclid_all.o Euclid_mpi.o Euclid_seq.o \
       Euclid_apply.o ilu_mpi.o fake_mpi.o
