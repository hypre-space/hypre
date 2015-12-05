/*BHEADER**********************************************************************
 * HYPRE_lobpcg.h
 *
 * $Revision: 1.7 $
 * Date: 03/12/2004
 * Authors: M. Argentati and A. Knyazev
 *********************************************************************EHEADER*/

#ifndef HYPRE_LOBPCG_HEADER
#define HYPRE_LOBPCG_HEADER

#ifdef __cplusplus
extern "C" {
#endif

/*  matrix types */
enum en1 {NONE1,DENSE,HYPRE_MATRIX,HYPRE_VECTORS};
enum en2 {NONE2,GENERAL,SYMMETRIC};
typedef enum en1 mst;
typedef enum en2 mt;

/*
   DENSE   - regular dense m x n matrix
   HYPRE_MATRIX - hypre ParCSR matrix
   HYPRE_VECTORS - array of hypre Par vectors
   GENERAL - general matrix
   SYMMETRIC - symmetric matrix
*/

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

/* matrix data structure - sparse or dense */
typedef struct {
  double **val;              /* storage for m x n dense matrix */
  double *val1;              /* non-zero data for sparse matrix */
  int    m,n,nz;             /* m x n matrix with nz non-zero values */
  mst    mat_storage_type;   /* storage type */
  mt     mat_type;           /* matrix type */
  int    numb_par_vectors_alloc; /* used to specify number of parallel vectors */
                             /* that are allocated so that vectors can be reused */
  HYPRE_ParCSRMatrix MPar;   /* hypre parallel csr matrix */
  HYPRE_ParVector *vsPar;    /* array of parallel csr vectors */
} Matx;

typedef struct {
  int flag_A;         /* ain - A matrix */
  char Ain[128];      /* file name for A which is in matrix market format */
  int flag_B;         /* bin - A matrix */
  char Bin[128];      /* file name for A which is in binary format */
  int flag_V;         /* vin - initial guess at eigenvectors */
  char Vin[128];      /* file name for X which is in matrix market format */
  int flag_T;         /* tin - precondioning matrix */
  char Tin[128];      /* file name for T which is in matrix market format */
  int flag_precond;   /* use matrix A itself as preconditioner */
  int Vrand;          /* vrand - use vrand random vectors for initial eigenvectors */
  int Veye;           /* veye - use identity vectors for initial eigenvectors */
  int flag_orth_check; /* chk - check orthogonality of eigenvectors */
  int verbose;        /* =0 (no output), =1 (standard output), =2 (detailed output) */
  int flag_feig;      /* f - print eigenvectors to matrix market file */
  int flag_itr;       /* itr - iterationoverride flag */
  int max_iter_count; /* itr - maximum iteration count */
  int flag_tol;       /* tol - tolerance override flag */
  double tol;         /* tolerance override value */
  int flag_f;         /* print eigenvalues and eigenvectors to a file */
  int pcg_max_itr;    /* maximum iterations for pcg solve */
  int pcg_max_flag;   /* flag for option if specified for pcg solve */
  double pcg_tol;     /* convergence tolerence for pcg solve */
  int printA;         /* printA - print input matrix A */
} lobpcg_options;

struct hypre_LobpcgData_struct;
typedef struct hypre_LobpcgData_struct *HYPRE_LobpcgData;

int HYPRE_LobpcgSolve(HYPRE_LobpcgData lobpcgdata,
    int (*FunctA)(HYPRE_ParVector x,HYPRE_ParVector y),
    HYPRE_ParVector *v,double **eigval);

int HYPRE_LobpcgCreate(HYPRE_LobpcgData *lobpcg);
int HYPRE_LobpcgSetup(HYPRE_LobpcgData lobpcg);
int HYPRE_LobpcgDestroy(HYPRE_LobpcgData lobpcg);
int HYPRE_LobpcgSetVerbose(HYPRE_LobpcgData lobpcg,int verbose);
int HYPRE_LobpcgSetRandom(HYPRE_LobpcgData lobpcg);
int HYPRE_LobpcgSetEye(HYPRE_LobpcgData lobpcg);
int HYPRE_LobpcgSetOrthCheck(HYPRE_LobpcgData lobpcg);
int HYPRE_LobpcgSetMaxIterations(HYPRE_LobpcgData lobpcg,int max_iter);
int HYPRE_LobpcgSetTolerance(HYPRE_LobpcgData lobpcg,double tol);
int HYPRE_LobpcgSetBlocksize(HYPRE_LobpcgData lobpcg,int bsize);
int HYPRE_LobpcgSetSolverFunction(HYPRE_LobpcgData lobpcg,
    int (*FunctSolver)(HYPRE_ParVector x,HYPRE_ParVector y));
int HYPRE_LobpcgGetSolverFunction(HYPRE_LobpcgData lobpcg,
   int (**FunctSolver)(HYPRE_ParVector x,HYPRE_ParVector y));
int HYPRE_LobpcgGetMaxIterations(HYPRE_LobpcgData lobpcg,int *max_iter);
int HYPRE_LobpcgGetTolerance(HYPRE_LobpcgData lobpcg,double *tol);
int HYPRE_LobpcgGetVerbose(HYPRE_LobpcgData lobpcg,int *verbose);
int HYPRE_LobpcgGetRandom(HYPRE_LobpcgData lobpcg,int *rand_vec);
int HYPRE_LobpcgGetEye(HYPRE_LobpcgData lobpcg,int *eye_vec);
int HYPRE_LobpcgGetOrthCheckNorm(HYPRE_LobpcgData lobpcg,double *orth_frob_norm);
int HYPRE_LobpcgGetIterations(HYPRE_LobpcgData lobpcg,int *iterations);
int HYPRE_LobpcgGetEigval(HYPRE_LobpcgData lobpcg,double **eigval);
int HYPRE_LobpcgGetResvec(HYPRE_LobpcgData lobpcg,double ***resvec);
int HYPRE_LobpcgGetEigvalHistory(HYPRE_LobpcgData lobpcg,double ***eigvalhistory);
int HYPRE_LobpcgGetBlocksize(HYPRE_LobpcgData lobpcg,int *bsize);

int *CopyPartition(int *partition);

#ifdef __cplusplus
}
#endif

#endif
