/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

#include "HYPRE_distributed_matrix_types.h"
#include "HYPRE_distributed_matrix_protos.h"
#include "HYPRE_IJ_mv.h"
#include "mpi.h"

#ifndef _PARASAILS_H
#define _PARASAILS_H

// LAPACK prototype
//extern "C" void dgels_
//    (char *trans, int *m, int *n, int *nrhs, double *A, int *lda, 
//    double *B, int *ldb, double *work, int *lwork, int *info);

class ParaSAILS
{
public:

    ParaSAILS(const HYPRE_DistributedMatrix& mat);

   ~ParaSAILS() {}

    void calculate();
    int nnz();

    void set_level(int level) {nlevels = level;}
    void set_thresh(double threshold) {thresh = threshold;}
    void set_lfil(int fill) {lfil = fill;}
    void set_prune_alg(int p) {prune_alg = p;}
    void set_dump(int d) {dump = d;}
    // perhaps set_prune_alg should take thresh and lfil params

    void dump_pattern();

    // some undone functions
    // void calculate_with_same_pattern(const RowMatrix& mat);
    // void compute_additional_level();

// used to be private:
// why are they public?

    HYPRE_DistributedMatrix A; // coefficient matrix
    HYPRE_IJMatrix M;          // preconditioner
    int npes;
    int myid;

    int nlevels;
    double thresh;
    int lfil;
    int prune_alg;
    int dump;

// following are used by SharedData class

    int    my_start_row;
    int    my_end_row;

    int    *start_rows;
    int    *end_rows;

// may be used by nonmember functions
    MPI_Comm comm; 

private:

    int n;

};

#endif /* _PARASAILS_H */
