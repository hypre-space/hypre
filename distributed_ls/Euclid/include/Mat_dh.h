#ifndef MAT_DH_DH
#define MAT_DH_DH

#include "euclid_common.h"

struct _mat_dh {
  int m, n;    /* dimensions of local rectangular submatrix;
                * the global matrix is n by n.
                */
  int beg_row;   /* global number of 1st locally owned row */

  /* sparse row-oriented storage for locally owned submatrix */
  int *rp;       
  int *len;   /* length of each row; only used for MPI triangular solves */
  int *cval;
  int *fill;
  int *diag;
  double *aval;
  bool owner;  /* for MPI triangular solves */

  /* these are used for parallel MatVec, and solves */
  int num_recv, num_send;  /* needed for destructor! */
  MPI_Request *recv_req;
  MPI_Request *send_req;
  MPI_Status *status;
  double *recvbuf, *sendbuf;
  int *sendind;
  int sendlen;
  Numbering_dh numb;
  bool matvecIsSetup;
};

extern void Mat_dhCreate(Mat_dh *mat);
extern void Mat_dhDestroy(Mat_dh mat);

  /* adopted from ParaSails, by Edmond Chow */
extern void Mat_dhMatVecSetup(Mat_dh mat);
extern void Mat_dhMatVecSetdown(Mat_dh mat);
extern void Mat_dhMatVec(Mat_dh mat, double *lhs, double *rhs);

  /* either avalD or avalF should be NULL; or, if both are null,
   * 1s are printed instead of values.
   */

extern int Mat_dhReadNz(Mat_dh mat);

extern void Mat_dhPrintTriples(Mat_dh mat, char *filename);

#endif
