/*BHEADER**********************************************************************
 * lobpcg.h
 *
 * $Revision$
 * Date: 10/7/2002
 * Authors: M. Argentati and A. Knyazev
 *********************************************************************EHEADER*/

/*------------------------------------------------------------------------*/
/* HYPRE includes                                                         */
/*------------------------------------------------------------------------*/
#include "HYPRE.h"
#include "seq_mv.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_mv.h"
#include "IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "krylov.h"
#include "HYPRE_lobpcg.h"

#define MM_MAX_LINE_LENGTH 1025
#define MM_PREMATURE_EOF 12
#define MAX_NUMBER_VECTORS 50
#define MAX_ITERATIONS 20
#define MATRIX_INPUT_PLANE_IJ 1
#define MATRIX_INPUT_MTX      2
#define MATRIX_INPUT_BIN      3
#define TRUE  1
#define FALSE 0
#define LOBPCG_DEFAULT_MAXITR     500
#define LOBPCG_DEFAULT_TOL        1E-6
#define LOBPCG_DEFAULT_BSIZE      1
#define LOBPCG_DEFAULT_VERBOSE    1
#define LOBPCG_DEFAULT_RANDOM     FALSE
#define LOBPCG_DEFAULT_EYE        FALSE
#define LOBPCG_DEFAULT_ORTH_CHECK FALSE

#define MAX_NUMBER_COUNTS 25
#define HYPRE_ParVectorInnerProd_Data 0
#define HYPRE_ParVectorCreate_Data    1
#define HYPRE_ParVectorCopy_Data      2
#define hypre_ParVectorAxpy_Data      3
#define HYPRE_ParVectorDestroy_Data   4
#define HYPRE_ParVectorSetConstantValues_Data  5
#define NUMBER_A_MULTIPLIES           6
#define NUMBER_SOLVES                 7

/* Aij format input data */
typedef struct {
  double row;            /* row number */
  double col;            /* column number */
  double val;            /* value */
  double index;          /* computed index */
} input_data;

typedef struct {
  int      verbose;            /* =0,1 or 2 to control output */
  int      rand_vec;         /* =1 randomize input vectors */
  int      eye_vec;         /* =1 set input vectors to n x bsize identity */
  int      orth_check;          /* =1 check orthoganality of eigenvectors */
  double   orth_frob_norm;      /* Frobenius norm of V'V-I, if requested */
  int      max_iter;         /* maximum number of iterations */
  int      iterations;         /* actual number of iterations */
  double   tol;          /* tolerance on max residual for convergence */
  double   *eigval;         /* pointer to array to store eigenvalues */
  double   **resvec;         /* pointer to array to store residual vectors */
  double   **eigvalhistory; /* pointer to array to store eigenvalue history */
  int      *partition;         /* partition for parallel vectors */
  int      bsize;         /* block size of eigenvectors */
  int (*FunctSolver)(HYPRE_ParVector x,HYPRE_ParVector y);
} hypre_LobpcgData;

/*--------------------------------------------------------------------------
* Macro for assert
*--------------------------------------------------------------------------*/
#define assert2(ierr)                           assert(ierr==0)

/*--------------------------------------------------------------------------
 * Accessor functions for the lobpcg structure
 *--------------------------------------------------------------------------*/
#define hypre_LobpcgVerbose(lobpcgdata)         ((lobpcgdata) -> verbose)
#define hypre_LobpcgRandom(lobpcgdata)          ((lobpcgdata) -> rand_vec)
#define hypre_LobpcgEye(lobpcgdata)             ((lobpcgdata) -> eye_vec)
#define hypre_LobpcgOrthCheck(lobpcgdata)       ((lobpcgdata) -> orth_check)
#define hypre_LobpcgOrthFrobNorm(lobpcgdata)    ((lobpcgdata) -> orth_frob_norm)
#define hypre_LobpcgMaxIterations(lobpcgdata)   ((lobpcgdata) -> max_iter)
#define hypre_LobpcgIterations(lobpcgdata)      ((lobpcgdata) -> iterations)
#define hypre_LobpcgTol(lobpcgdata)             ((lobpcgdata) -> tol)
#define hypre_LobpcgEigval(lobpcgdata)          ((lobpcgdata) -> eigval)
#define hypre_LobpcgResvec(lobpcgdata)          ((lobpcgdata) -> resvec)
#define hypre_LobpcgEigvalHistory(lobpcgdata)   ((lobpcgdata) -> eigvalhistory)
#define hypre_LobpcgPartition(lobpcgdata)       ((lobpcgdata) -> partition)
#define hypre_LobpcgBlocksize(lobpcgdata)       ((lobpcgdata) -> bsize)
#define hypre_LobpcgFunctSolver(lobpcgdata)     ((lobpcgdata) -> FunctSolver)

/* function prototypes */
double Mat_Norm_Inf(Matx *A);
double Mat_Norm_Frob(Matx *A);
double Max_Vec(double *y,int n);
double **Mymalloc(int m,int n);
int Assemble_DENSE(Matx *A,input_data *input1,int mA,int nA,int nzA,mt
 mat_type);
int comp(const void *p1,const void *p2);
void expect(FILE *fid,int i);
int Get_Rank();
int lobpcg(Matx *, int (*)(), int (*)(), double, int*, int, Matx *, Matx *,
Matx *);
int Mat_Add(Matx *A,Matx *B,double alpha,Matx *C);
Matx *Mat_Alloc1();
int Mat_Copy_Cols(Matx *A,Matx *B,int col1,int col2);
int Mat_Copy(Matx *A,Matx *B);
int Mat_Copy_MN(Matx *A,Matx *B,int row_offset,int col_offset);
int Mat_Copy_Rows(Matx *A,Matx *B,int row1,int row2);
int Mat_Diag(double *d,int n,Matx *A);
int Mat_Eye(int n,Matx *A);
int Mat_Free(Matx *A);
int Mat_Get_Col(Matx *A,Matx *B,int *idxA);
int Mat_Get_Col2(Matx *A,int *idxA);
int Mat_Init1(Matx *A);
int Mat_Init_Dense(Matx *A,int m,int n,mt mat_type);
int Mat_Init(Matx *A,int m,int n,int nz,mst mat_storage_type,mt mat_type);
int Mat_Inv_Triu(Matx *A,Matx *B);
int Mat_Mult(Matx *A,Matx *B,Matx *C);
int Mat_Mult2(Matx *A,Matx *B,int *idx);
int Mat_Norm2_Col(Matx *A,double *y);
int Mat_Put_Col(Matx *A,Matx *B,int *idxB);
int Mat_Size(Matx *A,int rc);
int Mat_Size_Mtx(char *file_name,int rc);
int Mat_Size_Bin(char *file_name,int rc);
int Mat_Sym(Matx *A);
int Mat_Trans_Idx(Matx *A,Matx *B,Matx *C,int *idxA,int *idxB);
int Mat_Trans_Mult2(Matx *A,int *idxA,Matx *B,int *idxB,Matx *C);
int Mat_Trans(Matx *A,Matx *B);
int Mat_Trans_Mult(Matx *A,Matx *B,Matx *C);
int MatViewUtil(Matx *A);
int myeig1(Matx *A, Matx *B, Matx *X,double *lambda);
int myqr1(Matx *A,Matx *Q,Matx *R);
int Qr2(Matx *V,Matx *R,int *idxB);
int PrintArray(Matx *A,char fname[]);
int Print_Par_Matrix_To_Mtx(HYPRE_ParCSRMatrix A,char *filename);
int PrintPartitioning(int *partitioning,char *name);
void PrintMatrixParameters(Matx *A);
int readmatrix(char Ain[],Matx *A,mst mat_storage_type,int *partitioning);
int rr(Matx *U,Matx *LU,Matx *R,Matx *LR,Matx *P,Matx *LP,
       double *lambda,int *idx,int bsize,int k,int last_flag);
int VecViewUtil2(int *b,int n);
int VecViewUtil(double *b,int n);
void display_input(input_data *input1,int n);

void Get_IJAMatrixFromFileStandard(double **val, int **ia,
     int **ja, int *N, char *matfile);
void Get_IJAMatrixFromFileMtx(double **val, int **ia,
     int **ja, int *m, int *n, char *matfile);
int  Get_CRSMatrixFromFileBinary(double **val, int **ia,
     int **ja, int *m, int *n, char *file_name);
int IJAMatrixiToSymmetric(double **val, int **ia,
    int **ja, int *m);
void HYPRE_Load_IJAMatrix(Matx *A, int matrix_input_type, char *matfile,int
*partitioning);
void HYPRE_Load_IJAMatrix2(HYPRE_ParCSRMatrix  *A_ptr,
     int matrix_input_type, char *matfile,int *partitioning);
void HYPRE_LoadMatrixVectorMtx(Matx *A,char *matfile,int *partitioning);
int Mat_Init_Identity(Matx *A,int m,int n,mst mat_storage_type,int
*partitioning);
void PrintVector(double *data,int n,char fname[]);
void PrintMatrix(double **data,int m,int n,char fname[]);

int hypre_LobpcgSetGetPartition(int action,int **part);
hypre_Vector *hypre_ParVectorToVector(MPI_Comm comm,hypre_ParVector *v);
int Init_Rand_Vectors(HYPRE_ParVector *v_ptr,int *partitioning, int m,int n);
int Init_Eye_Vectors(HYPRE_ParVector *v_ptr,int *partitioning, int m,int n);
int verbose2(int action);
int total_numb_vectors_alloc(int count);
int Trouble_Check(int mode,int test);
int misc_flags(int setget,int flag);
int collect_data(int state,int counter_type,int phase);
void Display_Execution_Statistics();
int time_functions(int set_run,int ftype_in,int numb_rows,int count_in,
    int (*FunctA)(HYPRE_ParVector x,HYPRE_ParVector y));
