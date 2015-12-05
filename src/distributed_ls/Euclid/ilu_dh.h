#ifndef ILU_MPI_DH
#define ILU_MPI_DH

#include "euclid_common.h"

void reallocate_private(int row, int newEntries, int *nzHave,
                int **rp, int **cval, float **aval, double **avalD, int **fill);

extern void ilu_mpi_pilu(Euclid_dh ctx);
  /* driver for comms intermingled with factorization */


extern void iluk_mpi_pilu(Euclid_dh ctx);
  /* the factorization algorithm */

extern void compute_scaling_private(int row, int len, double *AVAL, Euclid_dh ctx);

extern void iluk_mpi_bj(Euclid_dh ctx);

extern void iluk_seq(Euclid_dh ctx);
extern void iluk_seq_block(Euclid_dh ctx);
  /* for sequential or parallel block jacobi.  If used
     for block jacobi, column indices are referenced to 0
     on return; make sure and add beg_row to these values
     before printing the matrix!

     1st version is for single precision, 2nd is for double.
   */

extern void ilut_seq(Euclid_dh ctx);


#endif

