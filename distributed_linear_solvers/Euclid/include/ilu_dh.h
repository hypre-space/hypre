#ifndef ILU_MPI_DH
#define ILU_MPI_DH

#include "euclid_common.h"

void reallocate_private(int row, int newEntries, int *nzHave,
                int **rp, int **cval, float **aval, double **avalD, int **fill);

extern void iluk_mpi(Euclid_dh ctx);

extern void iluk_seq(Euclid_dh ctx);
extern void iluk_seq_D(Euclid_dh ctx);
  /* for sequential or parallel block jacobi.  If used
     for block jacobi, column indices are referenced to 0
     on return; make sure and add beg_row to these values
     before printing the matrix!

     1st version is for single precision, 2nd is for double.
   */


#endif

