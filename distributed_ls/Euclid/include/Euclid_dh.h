#ifndef EUCLID_MPI_INTERFACE_DH
#define EUCLID_MPI_INTERFACE_DH

#include "euclid_common.h"


/*======================================================================
 * Naming convention: functions ending in _mpi are located in
 * src/Euclid_mpi.c; those ending in _seq are in src/Euclid_seq.c;
 * most others should be in Euclid_all.c.
 *
 * Exceptions: all Apply() (triangular solves) are in src/Euclid_apply.c
 *
 * Users should only need to call functions with names of the form
 * Euclid_dhXXX (public functions). 
 *
 * Some of the functions whose names are of the form XXX_private_XXX,
 * as could easily be static functions; similarly, the enums and
 * structs do need to be public.  They are, primarily, for ease in
 * debugging and ready reference.
 *
 * Exceptions: the apply_private functions aren't listed here --- they're
 * all static in src/Euclid_apply.c
 *======================================================================*/

/*-----------------------------------------------------------------------
 * public functions 
 *-----------------------------------------------------------------------*/

extern void Euclid_dhCreate(Euclid_dh *ctxOUT);
extern void Euclid_dhDestroy(Euclid_dh ctx);
extern void Euclid_dhSetup(Euclid_dh ctx);
extern void Euclid_dhPrintParams(Euclid_dh ctx, FILE *fp);

  /* void pointers used in Apply to enable compatibility with PETSc, etc. */
extern int Euclid_dhApply(void *ctx, void *xx, void *yy);

/*-----------------------------------------------------------------------
 * public support functions for setting matrices 
 *-----------------------------------------------------------------------*/

 /* The following are located in src/getRow.c.
    When Euclid is used only for preconditioning, all inputs in
    the following are treated as const.  When used as a solver,
    the csr structures <rp, cval, aval> in Euclid_dhInputCSRMat()
    and Euclid_dhInputEuclidMat() are treated as const in a loose
    sense: after calling Euclid_dhDestroy() their contents will
    be identical to what they were. 

    For Euclid_dhInputCSRMat, the csr representation must be 0
    based.
*/

#ifdef EUCLID_GET_ROW
extern void Euclid_dhInputCSRMat(Euclid_dh ctx, int globalRows, 
                                int localRows, int beg_row, 
                                int *rp, int *cval, double *aval);

extern void Euclid_dhInputEuclidMat(Euclid_dh ctx, Mat_dh A);
#endif

#ifdef PETSC_MODE
extern void Euclid_dhInputPetscMat(Euclid_dh ctx, Mat A);
#endif

#ifdef HYPRE_MODE
extern void Euclid_dhInputHypreMat(Euclid_dh ctx, HYPRE_ParCSRMatrix A);
#endif


/*-----------------------------------------------------------------------
 * private functions 
 *-----------------------------------------------------------------------*/

extern void Euclid_dhApply_private(Euclid_dh ctx, double *xx, double *yy);
extern void profile_private(char *msg, int n, int *rp, int *cval, float *aval, double *avalD);
extern void order_interiors_private(Euclid_dh ctx);
extern void find_nzA_private(Euclid_dh ctx);
extern void find_nzF_private(Euclid_dh ctx);

extern void invert_diagonals_private(Euclid_dh ctx);
extern void get_runtime_params_private(Euclid_dh ctx);
extern void print_factor_private(Euclid_dh ctx, char *filename);
extern void factor_private(Euclid_dh ctx);
extern void order_bdry_nodes_private_mpi(Euclid_dh ctx);
extern void euclid_setup_private_mpi(Euclid_dh ctx);
extern void setup_pilu_private_mpi(Euclid_dh ctx);
extern void find_nabors_private_mpi(Euclid_dh ctx);
extern int find_owner_private_mpi(Euclid_dh ctx, int index);
extern void exchange_permutations_private_mpi(Euclid_dh ctx);
extern void exchange_bdry_rows_private_mpi(Euclid_dh ctx);


extern void euclid_setup_private_seq(Euclid_dh ctx);
extern void order_bdry_nodes_private_seq(Euclid_dh ctx);
extern void partition_private_seq(Euclid_dh ctx);
extern void metis_order_private_seq(Euclid_dh ctx);

#ifdef USING_METIS
static void metis_order_private_seq(Euclid_dh ctx);
#endif

extern void print_triples_to_file_private(int globalRows, int localRows, int begRow,
                            int *rp, int *cval, float *avalF, double *avalD,
                            int *n2o_row, int *n2o_col, Hash_dh o2n_globalCol,
                            char *filename);

/*-------------------------------------------------------------------
 * private enums
 *-------------------------------------------------------------------*/

/* maintainers note: if you change the next two enums, also
   change "static char *algo_par_strings[]" and 
   "static char *algo_ilu_strings[]" in src/Euclid_all.c
 */

enum{ NONE_PAR, BJILU_PAR, PILU_PAR, GRAPHCOLOR_PAR };
  /* choices for algo_par (parallelization method) */

enum{ NONE_ILU, ILUK_ILU, ILUT_ILU };
  /* choices for algo_ilu (factorization method) */

enum{ SIMPLE_PART, METIS_PART };
  /* partitioning methods */

enum{ EUCLID_NONE, EUCLID_CG, EUCLID_BICGSTAB};
  /* krylov solvers */

enum{ NATURAL_ORDER };
  /* methods for ordering subdomain interiors */

/*----------------------------------------------------------------------
 * Private data structures
 *----------------------------------------------------------------------*/

 /* for use when a single mpi task owns multiple subdomains */
typedef struct {
    int beg_row;    /* global number of 1st locally owned row */
    int end_row;    /* 1+global number of last locally owned row */
    int first_bdry; /* global number of 1st bdry row (used for PILU) */
} PartNode;


/* primary data structure: this is monstrously long; but it works. 
   Users must ensure the following fields are initialized prior
   to calling Euclid_dhSetup():

     m
     n
     beg_row
     A
     ownsAstruct

  These fields are most easily initialized by calling the appropriate
  Euclid_dhInputXxxMat() function.
*/
struct _mpi_interface_dh {
  int nzA;       /* nonzeros in local input matrix A */
  int nzF;       /* nonzeros in local factor */
  int nzAglobal; /* nonzeros in global input matrix A */
  int nzFglobal; /* nonzeros in global input matrix F */
  double rho_init;  /* guess for memory allocation for factor; will
                       initially allocate space for rho_init*nzA nonzeros.
                     */
  double rho_final; /* what rho_init it should have been, 
                       for best memory usage 
                     */

  int m;  /* number of local rows */
  int n;  /* number of local columns; also, number of global rows */
  int beg_row; /* first locally owned row */
  int first_bdry; /* local number of 1st bdry row (used for PILU) */

  bool ownsAstruct;
  void *A;  /* the input matrix, which may be a PETSc, HYPRE, Euclid,
               or other matrix object. 
             */

  /* data structure for the factor, F = L+U-I */
  bool isSinglePrecision;
  int *rpF;
  int *cvalF;
  float  *avalF; 
  double *avalFD; 
  int *diagF;
  int *fillF;
  int allocF;  /* allocated lengths of cvalF and avalF */

  /* partitioning */
  int partMethod;
  int blockCount;  /* number of blocks into which the local graph is partitioned */
  PartNode block[MAX_SUBDOMAINS]; /* there's one of these for each block. */
  int *colorCounter; /* for graph-color ILU; after Setup, 
                        rows colorCounter[j] through colorCounter[j+1] - 1
                        are similarly colored.
                      */

  /* permutations */
  int *n2o_row;
  int *n2o_col;
  Hash_dh o2n_nonLocal;  /* for non-local boundary nodes; for PILU only  */
  Hash_dh n2o_nonLocal;
  bool isNaturallyOrdered;
  bool isSymOrdered; /* vectors are equal: n2o_row == n2o_col */
  int orderMethod;   /* how to order subdomain interiors */

  /* stuph needed for PILU */
  /* for mapping row numbers to processors; don't confuse "beg_row" with "beg_rows" (ugh!) */ 
  int *beg_rows; /* P_i owns rows starting at global row number beg_rowspi] */
  int *end_rows; /* last global row, +1, owned by P[i] */
  int *bdryNodeCounts; /* subdomain(i) contains bdryNodeCounts[i] boundary nodes */
  int *bdryRowNzCounts; 
  int *nabors;     /* list of task IDs of neighboring subdomains */
  int naborCount;  /* number of neighboring subdomains */
  Hash_dh externalRows; 

  /* row scaling vector */
  float   *scale;
  double  *scaleD;
  bool    isScaled;

  /* workspace for triangular solves */
  float *work;
  double *workD;

  /* used for factorization and triangular solves */
  int from;
  int to;

  /* runtime parameters (mostly) */
  int algo_par; /* parallelization strategy (see enum above) */
  int algo_ilu; /* ILU factorization method (see enum above) */
  int level;      /* for ILU(k) */
  int cLevel;     /* for graph coloring */
  double droptol;     /* for ILUT */
  double sparseTolA;  /* for sparsifying A */
  double sparseTolF;  /* for sparsifying the factors, F */
  double pivotMin;    /* if pivots are <= to this value, fix 'em */
  double pivotFix;    /* multiplier for adjusting small pivots */
  double maxVal;      /* largest abs. value in matrix */

  /* these are for use with Euclid's solvers */
  int krylovMethod;
  int maxIts;
  double rtol;
  double atol;
  int    itsOUT; /* number of iterations at exit */

  /* for statistical recording */
  int zeroDiags; /* number of zero diagonals before factorization */
  int zeroPivots; /* number of pivots that are <= pivotMin, and
                   * hence were "fixed" during numeric factorization.
                   */
  int symbolicZeroDiags; /* number of diags inserted during symbolic
                          * factorization.
                          */
  bool   printProfile;
 
  int    logging;  /* added in support of Hypre; not sure what this does yet . . . */
}; 


#endif /*  #ifndef EUCLID_MPI_INTERFACE_DH */
