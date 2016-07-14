/* 
 * -- Auxiliary routine in distributed SuperLU (version 1.0) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * Xiaoye S. Li
 * April 16, 2002
 *
 */

#include "superlu_ddefs.h"

void GetDiagU(int_t n, LUstruct_t *LUstruct, gridinfo_t *grid, double *diagU)
{
  /*
   * Purpose
   * =======
   *
   * GetDiagU extracts the main diagonal of matrix U of the LU factorization.
   *  
   * Arguments
   * =========
   *
   * n        (input) int
   *          Dimension of the matrix.
   *
   * LUstruct (input) LUstruct_t*
   *          The data structures to store the distributed L and U factors.
   *          see superlu_ddefs.h for its definition.
   *
   * grid     (input) gridinfo_t*
   *          The 2D process mesh. It contains the MPI communicator, the number
   *          of process rows (NPROW), the number of process columns (NPCOL),
   *          and my process rank. It is an input argument to all the
   *          parallel routines.
   *
   * diagU    (output) double*, dimension (n)
   *          The main diagonal of matrix U.
   *          On exit, it is available on all processes.
   *
   *
   * Note
   * ====
   *
   * The diagonal blocks of the L and U matrices are stored in the L
   * data structures, and are on the diagonal processes of the
   * 2D process grid.
   *
   * This routine is modified from gather_diag_to_all() in pdgstrs_Bglobal.c.
   *
   */
    int_t *xsup;
    int iam, knsupc, pkk;
    int nsupr; /* number of rows in the block L(:,k) (LDA) */
    int_t i, j, jj, k, lk, lwork, nsupers, p;
    int_t num_diag_procs, *diag_procs, *diag_len;
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    LocalLU_t *Llu = LUstruct->Llu;
    double *dblock, *dwork, *lusup;

    iam = grid->iam;
    nsupers = Glu_persist->supno[n-1] + 1;
    xsup = Glu_persist->xsup;

    get_diag_procs(n, Glu_persist, grid, &num_diag_procs,
		   &diag_procs, &diag_len);
    jj = diag_len[0];
    for (j = 1; j < num_diag_procs; ++j) jj = SUPERLU_MAX( jj, diag_len[j] );
    if ( !(dwork = doubleMalloc_dist(jj)) ) ABORT("Malloc fails for dwork[]");

    for (p = 0; p < num_diag_procs; ++p) {
	pkk = diag_procs[p];
	if ( iam == pkk ) {
	    /* Copy diagonal into buffer dwork[]. */
	    lwork = 0;
	    for (k = p; k < nsupers; k += num_diag_procs) {
		knsupc = SuperSize( k );
		lk = LBj( k, grid );
		nsupr = Llu->Lrowind_bc_ptr[lk][1]; /* LDA of lusup[] */
		lusup = Llu->Lnzval_bc_ptr[lk];
		for (i = 0; i < knsupc; ++i) /* Copy the diagonal. */
		    dwork[lwork+i] = lusup[i*(nsupr+1)];
		lwork += knsupc;
	    }
	    MPI_Bcast( dwork, lwork, MPI_DOUBLE, pkk, grid->comm );
	} else {
	    MPI_Bcast( dwork, diag_len[p], MPI_DOUBLE, pkk, grid->comm );
	}

	/* Scatter dwork[] into global diagU vector. */
	lwork = 0;
	for (k = p; k < nsupers; k += num_diag_procs) {
	    knsupc = SuperSize( k );
	    dblock = &diagU[FstBlockC( k )];
	    for (i = 0; i < knsupc; ++i) dblock[i] = dwork[lwork+i];
	    lwork += knsupc;
	}
    } /* for p = ... */

    SUPERLU_FREE(diag_procs);
    SUPERLU_FREE(diag_len);
    SUPERLU_FREE(dwork);
}
