/*BHEADER**********************************************************************
 * lobpcg.c
 *
 * $Revision$
 * Date: 10/7/2002
 * Authors: M. Argentati and A. Knyazev
 *********************************************************************EHEADER*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <stdarg.h>
#include <float.h>
#include "lobpcg.h"

/*------------------------------------------------------------------------*/
/* HYPRE includes                                                         */
/*------------------------------------------------------------------------*/
#include <utilities/fortran.h>

/* function prototypes for functions that are passed to lobpcg */
/*void dsygv_(int *itype, char *jobz, char *uplo, int *n,
        double *a, int *lda, double *b, int *ldb, double *w, 
        double *work, int *lwork, int *info);*/
void hypre_F90_NAME_BLAS(dsygv, DSYGV)(int *itype, char *jobz, char *uplo, int *n,
        double *a, int *lda, double *b, int *ldb, double *w, 
        double *work, int *lwork, /*@out@*/ int *info);

int Func_AMult(Matx *B,Matx *C,int *idx); /* added by MEA 9/23/02 */
int Func_TPrec(Matx *R,int *idx);

int (*FunctA_ptr)(HYPRE_ParVector x,HYPRE_ParVector y);
int (*FunctPrec_ptr)(HYPRE_ParVector x,HYPRE_ParVector y);
HYPRE_ParVector temp_global_vector; /* this needs to be available to other program modules */
Matx *temp_global_data; /* this needs to be available to other program modules */
static Matx *TMP_Global1;
static Matx *TMP_Global2;
static int rowcount_global;

int HYPRE_LobpcgSolve(HYPRE_LobpcgData lobpcgdata,
    int (*FunctA)(HYPRE_ParVector x,HYPRE_ParVector y),
    HYPRE_ParVector *v,double **eigval1)
{

  Matx *X,*eigval,*resvec,*eigvalhistory; /* Matrix pointers */
  Matx *TMP,*TMP1,*TMP2; /* Matrices for misc calculations */
  extern HYPRE_ParVector temp_global_vector;
  int i,j,max_iter,bsize,verbose,rand_vec,eye_vec;
  double tol;
  double minus_one=-1;
  int (*FuncT)(Matx *B,int *idx)=NULL;
  int *partitioning,*part2;
  int mypid,nprocs;

  MPI_Comm_rank(MPI_COMM_WORLD, &mypid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  /* store function pointers */
  FunctA_ptr=FunctA;
  HYPRE_LobpcgGetSolverFunction(lobpcgdata,&FunctPrec_ptr);

  /* get lobpcg parameters */  
  HYPRE_LobpcgGetMaxIterations(lobpcgdata,&max_iter);
  HYPRE_LobpcgGetTolerance(lobpcgdata,&tol);
  HYPRE_LobpcgGetVerbose(lobpcgdata,&verbose);
  HYPRE_LobpcgGetRandom(lobpcgdata,&rand_vec);
  HYPRE_LobpcgGetEye(lobpcgdata,&eye_vec);
  HYPRE_LobpcgGetBlocksize(lobpcgdata,&bsize);

  /* initialize detailed verbose status for data collection */
  if (verbose==2)
  {
    verbose2(0);    /* turn off */
    collect_data(3,0,0);
  }
  else verbose2(2); /* turn off */

  /* derive partitioning from v */
  partitioning=hypre_ParVectorPartitioning((hypre_ParVector *)  v[0]);
  hypre_LobpcgSetGetPartition(0,&partitioning);
  rowcount_global=partitioning[nprocs];

  /* check time test */
  time_functions(1,0,rowcount_global,0,FunctA);

  /* create global temp vector to use in solve function */
  part2=CopyPartition(partitioning);
  if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorCreate_Data,0);
  HYPRE_ParVectorCreate(MPI_COMM_WORLD,rowcount_global,
    part2,&temp_global_vector);
  HYPRE_ParVectorInitialize(temp_global_vector);

  /* create global temporary data containing bsize parallel hypre vectors */
  /* these may be used in other programs for lobpcg */
  temp_global_data=Mat_Alloc1();
  Mat_Init(temp_global_data,rowcount_global,bsize,rowcount_global*bsize,HYPRE_VECTORS,GENERAL);

  /* check to see if we need to randomize v or set to identity */
  if (rand_vec==TRUE)
  {
    Init_Rand_Vectors(v,partitioning,rowcount_global,bsize);
  }
  else if (eye_vec==TRUE)
  {
    Init_Eye_Vectors(v,partitioning,rowcount_global,bsize);

  }

  /* Setup X array for initial eigenvectors */
  X=Mat_Alloc1();
  Mat_Init(X,rowcount_global,bsize,rowcount_global*bsize,HYPRE_VECTORS,GENERAL);
  for (i=0; i<bsize; i++){
     if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorCopy_Data,0);
     HYPRE_ParVectorCopy(v[i],X->vsPar[i]);
     
  }

  TMP_Global1=Mat_Alloc1();
  TMP_Global2=Mat_Alloc1();
  resvec=Mat_Alloc1();
  eigval=Mat_Alloc1();
  eigvalhistory=Mat_Alloc1();
  TMP=Mat_Alloc1();
  TMP1=Mat_Alloc1();
  TMP2=Mat_Alloc1();

  if (FunctPrec_ptr!=NULL) FuncT=Func_TPrec;

  /* call main lobpcg solver */
  lobpcg(X,Func_AMult,FuncT,tol,&max_iter,verbose,eigval,eigvalhistory,resvec);

  /* check orthogonality of eigenvectors */
  if (hypre_LobpcgOrthCheck((hypre_LobpcgData *) lobpcgdata)==TRUE)
  {
    Mat_Trans_Mult(X,X,TMP);
    Mat_Eye(bsize,TMP1); 
    Mat_Add(TMP,TMP1,minus_one,TMP2);
    hypre_LobpcgOrthFrobNorm((hypre_LobpcgData *) lobpcgdata)=Mat_Norm_Frob(TMP2);
  }

  /* get v vectors back */
  for (i=0; i<bsize; i++){
     if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorCopy_Data,0);
     HYPRE_ParVectorCopy(X->vsPar[i],v[i]);
     
  }

  /* store output */
  /* max_iter is returned from lobpcg as the actual number of iterations */ 
  hypre_LobpcgIterations((hypre_LobpcgData *) lobpcgdata)=max_iter;
  hypre_LobpcgEigval((hypre_LobpcgData *) lobpcgdata)=(double *)calloc(bsize,sizeof(double)); 
  hypre_LobpcgResvec((hypre_LobpcgData *) lobpcgdata)=Mymalloc(bsize,max_iter); 
  hypre_LobpcgEigvalHistory((hypre_LobpcgData *) lobpcgdata)=Mymalloc(bsize,max_iter); 

  for (i=0; i<bsize; i++){
    ((hypre_LobpcgData *) lobpcgdata)->eigval[i]=eigval->val[i][0];
    for (j=0; j<max_iter; j++){
      ((hypre_LobpcgData *) lobpcgdata)->resvec[i][j]=resvec->val[i][j];
      ((hypre_LobpcgData *) lobpcgdata)->eigvalhistory[i][j]=eigvalhistory->val[i][j];
    }
  }

  /* save eigenvalues to output */
  HYPRE_LobpcgGetEigval(lobpcgdata,eigval1);

  /* free all memory associated with these pointers */
  if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorDestroy_Data,0);
  HYPRE_ParVectorDestroy(temp_global_vector);

  Mat_Free(temp_global_data);free(temp_global_data); 
  Mat_Free(X);free(X); 
  Mat_Free(resvec);free(resvec);
  Mat_Free(eigvalhistory);free(eigvalhistory);
  Mat_Free(eigval);free(eigval);
  Mat_Free(TMP);free(TMP); 
  Mat_Free(TMP1);free(TMP1); 
  Mat_Free(TMP2);free(TMP2); 
  Mat_Free(TMP_Global1);free(TMP_Global1);
  Mat_Free(TMP_Global2);free(TMP_Global2);

  /* print out execution statistics */
  if (verbose==2 && Get_Rank()==0)
    Display_Execution_Statistics();

  return 0;
}

/*****************************************************************************/
int Func_AMult(Matx *B,Matx *C,int *idx)
{
  /* return C=A*B according to index */
  int i;
  for (i=0; i<B->n; i++) {
    if (idx[i]>0){
      FunctA_ptr(B->vsPar[i],C->vsPar[i]);
      if (verbose2(1)==TRUE) collect_data(0,NUMBER_A_MULTIPLIES,0);
    }
  }
  return 0;
}

/*****************************************************************************/
int Func_TPrec(Matx *R,int *idx)
{
  /* Solve A*X=R, return R=X using  */
  int i;
  extern HYPRE_ParVector temp_global_vector;

  for (i=0; i<R->n; i++) {
    if (idx[i]>0)
    {
      /* this next copy is not required, but improves performance substantially */
      if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorCopy_Data,0);
      HYPRE_ParVectorCopy(R->vsPar[i],temp_global_vector);
      FunctPrec_ptr(R->vsPar[i],temp_global_vector);
      if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorCopy_Data,0);
      HYPRE_ParVectorCopy(temp_global_vector,R->vsPar[i]);
      if (verbose2(1)==TRUE) collect_data(0,NUMBER_SOLVES,0);
    }
  }
  
  return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_LobpcgCreate
 *--------------------------------------------------------------------------*/
int
HYPRE_LobpcgCreate(HYPRE_LobpcgData *lobpcg)
{
   HYPRE_LobpcgData lobpcg2;

   /* allocate memory */
   if (!(*lobpcg=(HYPRE_LobpcgData) malloc(sizeof(hypre_LobpcgData)))) {
     fprintf(stderr, "Out of memory\n");
     abort();
   }

   lobpcg2=*lobpcg;

   /* set defaults */
   HYPRE_LobpcgSetMaxIterations(lobpcg2,LOBPCG_DEFAULT_MAXITR);
   HYPRE_LobpcgSetTolerance(lobpcg2,LOBPCG_DEFAULT_TOL);
   HYPRE_LobpcgSetBlocksize(lobpcg2,LOBPCG_DEFAULT_BSIZE);
   HYPRE_LobpcgSetSolverFunction(lobpcg2,NULL);
   hypre_LobpcgVerbose((hypre_LobpcgData *) lobpcg2)=LOBPCG_DEFAULT_VERBOSE;
   hypre_LobpcgRandom((hypre_LobpcgData *) lobpcg2)=LOBPCG_DEFAULT_RANDOM;
   hypre_LobpcgEye((hypre_LobpcgData *) lobpcg2)=LOBPCG_DEFAULT_EYE;
   hypre_LobpcgOrthCheck((hypre_LobpcgData *) lobpcg2)=LOBPCG_DEFAULT_ORTH_CHECK;

   /* initialize pointers */
   hypre_LobpcgEigval((hypre_LobpcgData *) lobpcg2)=NULL;
   hypre_LobpcgResvec((hypre_LobpcgData *) lobpcg2)=NULL;
   hypre_LobpcgEigvalHistory((hypre_LobpcgData *) lobpcg2)=NULL;
   hypre_LobpcgPartition((hypre_LobpcgData *) lobpcg2)=NULL;

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_LobpcgSetup
 *--------------------------------------------------------------------------*/
int
HYPRE_LobpcgSetup(HYPRE_LobpcgData lobpcg)
{
  if(!lobpcg) abort();
  /* future use */
  return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_LobpcgDestroy
 *--------------------------------------------------------------------------*/
int
HYPRE_LobpcgDestroy(HYPRE_LobpcgData lobpcg)
{

   /* free memory */
   if (((hypre_LobpcgData *) lobpcg)->eigval!=NULL)
      free(((hypre_LobpcgData *) lobpcg)->eigval);
   if (((hypre_LobpcgData *) lobpcg)->resvec != NULL){
      if (((hypre_LobpcgData *) lobpcg)->resvec[0] != NULL)
         free(((hypre_LobpcgData *) lobpcg)->resvec[0]);
      free(((hypre_LobpcgData *) lobpcg)->resvec);
   }
   if (((hypre_LobpcgData *) lobpcg)->eigvalhistory != NULL){
      if (((hypre_LobpcgData *) lobpcg)->eigvalhistory[0] != NULL) 
         free(((hypre_LobpcgData *) lobpcg)->eigvalhistory[0]);
      free(((hypre_LobpcgData *) lobpcg)->eigvalhistory);
   }
   if (((hypre_LobpcgData *) lobpcg)->partition!=NULL)
      free(((hypre_LobpcgData *) lobpcg)->partition);
   free((hypre_LobpcgData *) lobpcg);

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_LobpcgSetVerbose 
 *--------------------------------------------------------------------------*/
int
HYPRE_LobpcgSetVerbose(HYPRE_LobpcgData lobpcg,int verbose)
{
   hypre_LobpcgVerbose((hypre_LobpcgData *) lobpcg)=verbose;
   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_LobpcgSetRandom 
 *--------------------------------------------------------------------------*/
int
HYPRE_LobpcgSetRandom(HYPRE_LobpcgData lobpcg)
{
   hypre_LobpcgRandom((hypre_LobpcgData *) lobpcg)=TRUE;
   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_LobpcgSetEye
 *--------------------------------------------------------------------------*/
int
HYPRE_LobpcgSetEye(HYPRE_LobpcgData lobpcg)
{
   hypre_LobpcgEye((hypre_LobpcgData *) lobpcg)=TRUE;
   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_LobpcgSetOrthCheck
 *--------------------------------------------------------------------------*/
int
HYPRE_LobpcgSetOrthCheck(HYPRE_LobpcgData lobpcg)
{
   hypre_LobpcgOrthCheck((hypre_LobpcgData *) lobpcg)=TRUE;
   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_LobpcgSetMaxIterations 
 *--------------------------------------------------------------------------*/
int
HYPRE_LobpcgSetMaxIterations(HYPRE_LobpcgData lobpcg,int max_iter)
{
   hypre_LobpcgMaxIterations((hypre_LobpcgData *) lobpcg)=max_iter;
   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_LobpcgSetTolerance 
 *--------------------------------------------------------------------------*/
int
HYPRE_LobpcgSetTolerance(HYPRE_LobpcgData lobpcg,double tol)
{
   hypre_LobpcgTol((hypre_LobpcgData *) lobpcg)=tol;
   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_LobpcgSetBlocksize 
 *--------------------------------------------------------------------------*/
int
HYPRE_LobpcgSetBlocksize(HYPRE_LobpcgData lobpcg,int bsize)
{
   hypre_LobpcgBlocksize((hypre_LobpcgData *) lobpcg)=bsize;
   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_LobpcgSetSolverFunction 
 *--------------------------------------------------------------------------*/
int
HYPRE_LobpcgSetSolverFunction(HYPRE_LobpcgData lobpcg,
   int (*FunctSolver)(HYPRE_ParVector x,HYPRE_ParVector y))
{
   hypre_LobpcgFunctSolver((hypre_LobpcgData *) lobpcg)=FunctSolver;
   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_LobpcgGetSolverFunction
 *--------------------------------------------------------------------------*/
int
HYPRE_LobpcgGetSolverFunction(HYPRE_LobpcgData lobpcg,
   int (**FunctSolver)(HYPRE_ParVector x,HYPRE_ParVector y))
{
   *FunctSolver=hypre_LobpcgFunctSolver((hypre_LobpcgData *) lobpcg);
   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_LobpcgGetMaxIterations 
 *--------------------------------------------------------------------------*/
int
HYPRE_LobpcgGetMaxIterations(HYPRE_LobpcgData lobpcg,int *max_iter)
{
   *max_iter=hypre_LobpcgMaxIterations((hypre_LobpcgData *) lobpcg);
   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_LobpcgGetOrthCheckNorm
 *--------------------------------------------------------------------------*/
int
HYPRE_LobpcgGetOrthCheckNorm(HYPRE_LobpcgData lobpcg,double *orth_frob_norm)
{
   *orth_frob_norm=hypre_LobpcgOrthFrobNorm((hypre_LobpcgData *) lobpcg);
   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_LobpcgGetIterations 
 *--------------------------------------------------------------------------*/
int
HYPRE_LobpcgGetIterations(HYPRE_LobpcgData lobpcg,int *iterations)
{
   *iterations=hypre_LobpcgIterations((hypre_LobpcgData *) lobpcg);
   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_LobpcgGetTolerance 
 *--------------------------------------------------------------------------*/
int
HYPRE_LobpcgGetTolerance(HYPRE_LobpcgData lobpcg,double *tol)
{
   *tol=hypre_LobpcgTol((hypre_LobpcgData *) lobpcg);
   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_LobpcgGetVerbose 
 *--------------------------------------------------------------------------*/
int
HYPRE_LobpcgGetVerbose(HYPRE_LobpcgData lobpcg,int *verbose)
{
   *verbose=hypre_LobpcgVerbose((hypre_LobpcgData *) lobpcg);
   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_LobpcgGetRandom
 *--------------------------------------------------------------------------*/
int
HYPRE_LobpcgGetRandom(HYPRE_LobpcgData lobpcg,int *rand_vec)
{
   *rand_vec=hypre_LobpcgRandom((hypre_LobpcgData *) lobpcg);
   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_LobpcgGetEye
 *--------------------------------------------------------------------------*/
int
HYPRE_LobpcgGetEye(HYPRE_LobpcgData lobpcg,int *eye_vec)
{
   *eye_vec=hypre_LobpcgEye((hypre_LobpcgData *) lobpcg);
   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_LobpcgGetEigval 
 *--------------------------------------------------------------------------*/
int
HYPRE_LobpcgGetEigval(HYPRE_LobpcgData lobpcg,double **eigval)
{
   *eigval=hypre_LobpcgEigval((hypre_LobpcgData *) lobpcg);
   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_LobpcgGetResvec 
 *--------------------------------------------------------------------------*/
int
HYPRE_LobpcgGetResvec(HYPRE_LobpcgData lobpcg,double ***resvec)
{
   *resvec=hypre_LobpcgResvec((hypre_LobpcgData *) lobpcg);
   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_LobpcgGetEigvalHistory 
 *--------------------------------------------------------------------------*/
int
HYPRE_LobpcgGetEigvalHistory(HYPRE_LobpcgData lobpcg,double ***eigvalhistory)
{
   *eigvalhistory=hypre_LobpcgEigvalHistory((hypre_LobpcgData *) lobpcg);
   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_LobpcgGetBlocksize 
 *--------------------------------------------------------------------------*/
int
HYPRE_LobpcgGetBlocksize(HYPRE_LobpcgData lobpcg,int *bsize)
{
   *bsize=hypre_LobpcgBlocksize((hypre_LobpcgData *) lobpcg);
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_LobpcgSetGetPartitioning 
 * Store or retrieve partitioning to use in lobpcg for parallel vectors
 *--------------------------------------------------------------------------*/
int
hypre_LobpcgSetGetPartition(int action,int **part) 
{
static int *partitioning;

   /* store partitioning */
   if (action==0)
   {
      partitioning=*part;
      return 0;
   }
   /* get partitioning */
   else if (action==1)
   {
      *part=partitioning;
      if (part==NULL) {
		fprintf(stderr, "hypre_LobpcgSetGetPartition function failed");
		abort();
	  }
      return 0;
   }
   else
   {
	 fprintf(stderr, "hypre_LobpcgSetGetPartition function failed");
	 abort();
	 return(1);
   } 
}


/******************************************************************************
* LOBPCG.C - eigenvalue solver implemented using C-language
* and LAPACK Fotran calls. By M.E. Argentati and Andrew Knyazev
*
* Revision: 1.00-C03 Date: 2002/7/8
* Modified by AK, 2002/7/14
*
* This implementation is a simplified (for B=I) version of the 
* preconditioned conjugate gradient method (Algorithm 5.1) described in
* A. V. Knyazev, Toward the Optimal Preconditioned Eigensolver:
* Locally Optimal Block Preconditioned Conjugate Gradient Method,
* SIAM Journal on Scientific Computing 23 (2001), no. 2, pp. 517-541. 
* See http://epubs.siam.org/sam-bin/getfile/SISC/articles/36612.pdf.
*
* It is also logically similar to a MATLAB inplementation - LOBPCG.m 
* Revision: 4.0 Beta 4 $  $Date: 2001/8/25, by A. V. Knyazev, which
* can be found at http://www-math.cudenver.edu/~aknyazev/software/CG/.
*
* This program also uses some available software including several 
* Fortran routines from LAPACK (http://www.netlib.org/lapack/),
* The author would like to acknowledge use of this software and
* thank the authors.
*
*Required input/output: 
*   X - initial approximation to eigenvectors, colunm-matrix n-by-bsize
        Eigenvectors are returned in X.
*   FuncA - name of function that multiplies a matrix A times a vector
*           FuncA(Matx *B, Matx *C, int *idx) for C=A*B using index idx.
*   nargs - the number of optional parameters (0-7)
*   mytol - tolerance, by default, mytol=n*sqrt(eps)
*   maxit - max number of iterations, by default, maxit = min(n,20)
*   verbose - =0 (no output), =1 (standard output) default, =2 (detailed output)
*   lambda - Matrix (vector) of eigenvalues (bsize)
*   lambdahistory - history of eigenvalues iterates (bsize x maxit)
*   resvec - history of residuals (bsize x maxit)
*
*Optional function input:
*   FuncT - preconditioner, is the name of a function, default T=I
*           Funct(Matx *R,int *idx), usually solve T*X=R, return R=X
*           using index idx.
*
* Examples:
* All parameters:
* lobpcg(X,FuncA,7,FuncT,mytol,maxit,verbose,lambda,lambdahistory,resvec);
*
* All parameters, but no preconditioner:
* lobpcg(X,FuncA,NULL,mytol,maxit,verbose,lambda,lambdahistory,resvec);
******************************************************************************/

int
lobpcg(X, FuncA, FuncT, mytol, maxit_ptr, verbose, lambda_out, lambdahistory, resvec)
	 Matx *X, *lambda_out, *lambdahistory, *resvec;
	 int (*FuncA)(Matx *B, Matx *c, int *idx), (*FuncT)(Matx *B,int *idx), *maxit_ptr, verbose;
	 double mytol;
{
  /* initialize variables */
  double minus_one=-1;
  int maxit=0;
  /* storage for the following matrices must be allocated
     before the call to lobpcg */
  Matx    *AX,*R,*AR,*P,*AP;                    /* Matrix pointers */
  Matx    *RR1,*D,*TMP,*TMP1,*TMP2;
  Matx    *Temp_Dense;
  extern Matx *temp_global_data;

  int    i,j,k,n;
  int    bsize=0,bsizeiaf=0;
  int    Amult_count=0; /* count matrix A multiplies */
  int    Tmult_solve_count=0; /* count T matrix multiplies of prec. solves */
  int    *idx;       /* index to keep track of kept vectors */

  double *lambda, *y,*normR;

  /* allocate storage */
  AX=   Mat_Alloc1();
  R=    Mat_Alloc1();
  AR=   Mat_Alloc1();
  P=    Mat_Alloc1();
  AP=   Mat_Alloc1();
  TMP1= Mat_Alloc1();
  TMP2= Mat_Alloc1();
  TMP=  Mat_Alloc1();
  RR1=  Mat_Alloc1();
  D=    Mat_Alloc1();
  Temp_Dense=Mat_Alloc1();

  /* get size of A from X */
  n=Mat_Size(X,1);
  bsize=Mat_Size(X,2);

  /* check bsize */
  if (n<5*bsize && Get_Rank()==0){
    fprintf(stderr, "The problem size is too small compared to the block size for LOBPCG.\n");
    exit(EXIT_FAILURE);
  }

  /* set defaults */
  if (mytol<DBL_EPSILON) mytol=sqrt(DBL_EPSILON)*n;
  if (maxit_ptr != NULL) maxit=*maxit_ptr;
  if (maxit==0) maxit=n<20 ? n:20;

  Mat_Init_Dense(lambda_out,bsize,1,GENERAL);
  Mat_Init_Dense(lambdahistory,bsize,maxit+1,GENERAL);
  Mat_Init_Dense(resvec,bsize,maxit+1,GENERAL);

  if (bsize > 0) {
    /* allocate lambda */
    lambda=(double *)calloc((long)3*bsize,sizeof(double));
    y=(double *)calloc((long)3*bsize,sizeof(double));
    /* allocate norm of R vector */
    normR=(double *)calloc((long)bsize,sizeof(double));
    /* allocate index */
    idx=(int *)calloc((long)bsize,sizeof(int));
    for (i=0; i<bsize; i++) idx[i]=1; /* initialize index */
  }
  else {
    fprintf(stderr, "The block size is wrong.\n");
    exit(EXIT_FAILURE);
  }
		     
  /* perform orthonormalization of X */
  Qr2(X,RR1,idx);

  /* generate AX */
  Mat_Copy(X,AX);     /* initialize AX */
  if (misc_flags(1,0)!=TRUE)
  {
    FuncA(X,AX,idx);    /* AX=A*X */
    Amult_count=Amult_count+bsize;
  }

  /* initialize global data */
  Mat_Copy(X,TMP_Global1); 
  Mat_Copy(X,TMP_Global2); 

  /* compute initial eigenvalues */
  Mat_Trans_Mult(X,AX,TMP1);     /* X'*AX */
  Mat_Sym(TMP1);                 /* (TMP1+TMP1')/2 */
  Mat_Eye(bsize,TMP2);           /* identity */
  myeig1(TMP1,TMP2,TMP,lambda);  /* TMP has eignevectors */
  assert(lambda!=NULL);

  Mat_Mult2(X,TMP,idx);        /* X=X(idx)*TMP */
  Mat_Mult2(AX,TMP,idx);       /* AX=AX(idx)*TMP */

  /* compute initial residuals */
  Mat_Diag(lambda,bsize,TMP1);   /* diagonal matrix of eigenvalues */
  Mat_Mult(X,TMP1,TMP_Global1);
  Mat_Add(AX,TMP_Global1,minus_one,R);  /* R=AX-A*eigs */
  Mat_Norm2_Col(R,normR);
  assert(normR!=NULL); 
  
  /* initialize AR  and P and AP using the same size and format as X */
  Mat_Init(AR,X->m,X->n,X->nz,X->mat_storage_type,X->mat_type);
  Mat_Init(P,X->m,X->n,X->nz,X->mat_storage_type,X->mat_type);
  Mat_Init(AP,X->m,X->n,X->nz,X->mat_storage_type,X->mat_type);

  k=1;         /* iteration count */
  
  /* store auxillary information */
  if (resvec != NULL){
    for (i=0; i<bsize; i++) resvec->val[i][k-1]=normR[i];
  }
  if (lambdahistory != NULL){
    for (i=0; i<bsize; i++) lambdahistory->val[i][k-1]=lambda[i];
  }

  printf("\n");
  if (verbose==2 && Get_Rank()==0){
    for (i=0; i<bsize; i++) printf("Initial eigenvalues lambda %22.16e\n",lambda[i]);
    for (i=0; i<bsize; i++) printf("Initial residuals %12.6e\n",normR[i]);
  }
  else if (verbose==1 && Get_Rank()==0)
    printf("Initial Max. Residual %12.6e\n",Max_Vec(normR,bsize));

  /* increment data collection mode */
  if (verbose2(1)==TRUE) collect_data(1,0,0);

  /* main loop of CG method */
  while (Max_Vec(normR,bsize) - mytol >  DBL_EPSILON && k<maxit+1)
  {
    /* increment data collection mode */
    if ((verbose2(1)==TRUE) && (k==2)) collect_data(1,0,0);
  
    /* generate index of vectors to keep */
    bsizeiaf=0;
    for (i=0; i<bsize; i++){
      if (normR[i] - mytol > DBL_EPSILON){
        idx[i]=1;
        ++bsizeiaf;
      }
      else idx[i]=0;
    }
    assert(idx!=NULL); 
    
    /* check for precondioner */
    if ((FuncT != NULL) && (misc_flags(1,1)!=TRUE))
    {
      FuncT(R,idx);
      Tmult_solve_count=Tmult_solve_count+bsizeiaf;
    }

    /* orthonormalize R to increase stability  */
    Qr2(R,RR1,idx);

    /* compute AR */
    if (misc_flags(1,0)!=TRUE)
    {
      FuncA(R,AR,idx);
      Amult_count=Amult_count+bsizeiaf;
    }
    else Mat_Copy(R,AR);

    /* compute AP */
    if (k>1){
      Qr2(P,RR1,idx);
      Mat_Inv_Triu(RR1,Temp_Dense); /* TMP=inv(RR1) */
      Mat_Mult2(AP,Temp_Dense,idx);   /* AP(idx)=AP(idx)*TMP */
    }

    /* Raleigh-Ritz proceedure */
    if (bsize != bsizeiaf)
    {
      Mat_Get_Col(R,TMP_Global2,idx);  
      Mat_Get_Col(AR,temp_global_data,idx);
      Mat_Get_Col(P,TMP_Global1,idx); 
      Mat_Copy(TMP_Global1,P);
      Mat_Get_Col(AP,TMP_Global1,idx); 
      Mat_Copy(TMP_Global1,AP);
      rr(X,AX,TMP_Global2,temp_global_data,P,AP,lambda,idx,bsize,k,0);
    }
    else rr(X,AX,R,AR,P,AP,lambda,idx,bsize,k,0);

    /* get eigenvalues corresponding to index */
    j=0;
    for (i=0; i<bsize; i++){
      if (idx[i]>0){ 
        y[j]=lambda[i];
        ++j;
      }
    }
    assert(j==bsizeiaf);
    assert(y!=NULL);
    
    /* compute residuals */
    Mat_Diag(y,bsizeiaf,Temp_Dense);   
    Mat_Get_Col(X,TMP_Global1,idx);   
    Mat_Mult(TMP_Global1,Temp_Dense,TMP_Global2);
    Mat_Get_Col(AX,TMP_Global1,idx);      
    Mat_Add(TMP_Global1,TMP_Global2,minus_one,temp_global_data);
    Mat_Put_Col(temp_global_data,R,idx);       
    Mat_Norm2_Col(R,normR);

    /* store auxillary information */
    if (resvec != NULL){
      for (i=0; i<bsize; i++) resvec->val[i][k]=normR[i];
    }
    if (lambdahistory != NULL){
      for (i=0; i<bsize; i++) lambdahistory->val[i][k]=lambda[i];
    }

    if (verbose==2 && Get_Rank()==0){
      printf("Iteration %d \tbsize %d\n",k,bsizeiaf);
      for (i=0; i<bsize; i++) printf("Eigenvalue lambda %22.16e\n",lambda[i]);
      for (i=0; i<bsize; i++) printf("Residual %12.6e\n",normR[i]);
    }
    else if (verbose==1 && Get_Rank()==0) printf("Iteration %d \tbsize %d \tmaxres %12.6e\n",
      k,bsizeiaf,Max_Vec(normR,bsize));

    ++k;
  }

  /* increment data collection mode if only one iteration */
  if ((verbose2(1)==TRUE) && (k==2)) collect_data(1,0,0);

  /* increment data collection mode */
  if (verbose2(1)==TRUE) collect_data(1,0,0);
  
  /* call rr once more to release memory */
  rr(X,AX,R,AR,P,AP,lambda,idx,bsize,k,1);

  /* return actual number of iterations */
  if (maxit_ptr != NULL) *maxit_ptr=k;
  else {
    fprintf(stderr, "The number of iterations is empty.\n");
    exit(EXIT_FAILURE);
  }
	
  if (verbose==1 && Get_Rank()==0){
    printf("\n");
    for (i=0; i<bsize; i++) printf("Eigenvalue lambda %22.16e\n",lambda[i]);
    for (i=0; i<bsize; i++) printf("Residual %12.6e\n",normR[i]);
  }

  /* free all memory associated with these pointers */
  free(lambda);
  free(y);
  free(normR);
  free(idx);
  Mat_Free(AX);free(AX);
  Mat_Free(R);free(R);
  Mat_Free(AR);free(AR);
  Mat_Free(P);free(P);
  Mat_Free(AP);free(AP);
  Mat_Free(TMP1);free(TMP1);
  Mat_Free(TMP2);free(TMP2);
  Mat_Free(TMP);free(TMP);
  Mat_Free(RR1);free(RR1);
  Mat_Free(D);free(D);
  Mat_Free(Temp_Dense);free(Temp_Dense);

  return 0;
}

/*****************************************************************************/
int rr(Matx *U,Matx *LU,Matx *R,Matx *LR,Matx *P,Matx *LP,
       double *lambda,int *idx,int bsize,int k,int last_flag)
{
  /* The Rayleigh-Ritz method for triplets U, R, P */

  static Matx *GL;
  static Matx *GM;
  static Matx *GL12;
  static Matx *GL13;
  static Matx *GL22;
  static Matx *GL23;
  static Matx *GL33;
  static Matx *GM12;
  static Matx *GM13;
  static Matx *GM23;
  static Matx *GU;
  static Matx *D;
  static Matx *Temp_Dense;

  int i,n;
  int bsizeU,bsizeR,bsizeP;
  int restart;
  
  static int exec_first=0; 
  
  if (exec_first==0){
    GL=   Mat_Alloc1(); 
    GM=   Mat_Alloc1();
    GL12= Mat_Alloc1();
    GL13= Mat_Alloc1();
    GL22= Mat_Alloc1();
    GL23= Mat_Alloc1();
    GL33= Mat_Alloc1();
    GM12= Mat_Alloc1();
    GM13= Mat_Alloc1();
    GM23= Mat_Alloc1();
    GU=   Mat_Alloc1();
    D=    Mat_Alloc1();
    Temp_Dense=Mat_Alloc1();
    exec_first=1;
  }

  /* cleanup */
  if ((last_flag !=0) && (exec_first==1))
  {
    Mat_Free(GL);free(GL);
    Mat_Free(GM);free(GM);
    Mat_Free(GL12);free(GL12);
    Mat_Free(GL13);free(GL13);
    Mat_Free(GL23);free(GL23);
    Mat_Free(GL33);free(GL33);
    Mat_Free(GM12);free(GM12);
    Mat_Free(GM13);free(GM13);
    Mat_Free(GM23);free(GM23);
    Mat_Free(GU);free(GU);
    Mat_Free(D);free(D);
    Mat_Free(Temp_Dense);free(Temp_Dense);
    return 0;
  }

  /* setup and get sizes */
  bsizeU=Mat_Size(U,2);
  assert(bsize==bsizeU);

  bsizeR=0;
  for (i=0; i<bsize; i++){
    if (idx[i]>0) ++bsizeR; 
  }

  if (k==1){
    bsizeP=0;
    restart=1;
  }
  else {
    bsizeP=bsizeR; /* these must be the same size */
    restart=0;
  }

  Mat_Trans_Mult(LU,R,GL12);
  Mat_Trans_Mult(LR,R,GL22);
  Mat_Trans_Mult(U,R,GM12);
  Mat_Sym(GL22);
  
  if (restart==0){
    /* form GL */
    Mat_Trans_Mult(LU,P,GL13);
    Mat_Trans_Mult(LR,P,GL23);
    Mat_Trans_Mult(LP,P,GL33);
    Mat_Sym(GL33);
    Mat_Diag(lambda,bsizeU,D);
    
    n=bsize+bsizeR+bsizeP;
    Mat_Init_Dense(GL,n,n,SYMMETRIC);
     
    Mat_Copy_MN(D,GL,0,0); 
    Mat_Copy_MN(GL12,GL,0,bsizeU); 
    Mat_Copy_MN(GL13,GL,0,bsizeU+bsizeR); 
    Mat_Copy_MN(GL22,GL,bsizeU,bsizeU); 
    Mat_Copy_MN(GL23,GL,bsizeU,bsizeU+bsizeR); 
    Mat_Copy_MN(GL33,GL,bsizeU+bsizeR,bsizeU+bsizeR); 
    Mat_Trans(GL12,D);
    Mat_Copy_MN(D,GL,bsizeU,0); 
    Mat_Trans(GL13,D);
    Mat_Copy_MN(D,GL,bsizeU+bsizeR,0); 
    Mat_Trans(GL23,D);
    Mat_Copy_MN(D,GL,bsizeU+bsizeR,bsizeU); 

    /* form GM */
    Mat_Trans_Mult(U,P,GM13);
    Mat_Trans_Mult(R,P,GM23);
    Mat_Init_Dense(GM,n,n,SYMMETRIC);
    
    Mat_Eye(bsizeU,D);
    Mat_Copy_MN(D,GM,0,0);
    Mat_Eye(bsizeR,D);
    Mat_Copy_MN(D,GM,bsizeU,bsizeU);
    Mat_Eye(bsizeP,D);
    Mat_Copy_MN(D,GM,bsizeU+bsizeR,bsizeU+bsizeR);
    Mat_Copy_MN(GM12,GM,0,bsizeU);
    Mat_Copy_MN(GM13,GM,0,bsizeU+bsizeR);
    Mat_Copy_MN(GM23,GM,bsizeU,bsizeU+bsizeR);
    Mat_Trans(GM12,D);
    Mat_Copy_MN(D,GM,bsizeU,0);
    Mat_Trans(GM13,D);
    Mat_Copy_MN(D,GM,bsizeU+bsizeR,0);
    Mat_Trans(GM23,D);
    Mat_Copy_MN(D,GM,bsizeU+bsizeR,bsizeU);
  }
  else
  {
    /* form GL */
    n=bsizeU+bsizeR;
    Mat_Init_Dense(GL,n,n,SYMMETRIC);
    Mat_Diag(lambda,bsizeU,D);
    Mat_Copy_MN(D,GL,0,0);
    Mat_Copy_MN(GL12,GL,0,bsizeU);
    Mat_Copy_MN(GL22,GL,bsizeU,bsizeU);
    Mat_Trans(GL12,D);
    Mat_Copy_MN(D,GL,bsizeU,0);

    /* form GM */
    Mat_Init_Dense(GM,n,n,SYMMETRIC);
    Mat_Eye(bsizeU,D);
    Mat_Copy_MN(D,GM,0,0);
    Mat_Eye(bsizeR,D);
    Mat_Copy_MN(D,GM,bsizeU,bsizeU);
    Mat_Copy_MN(GM12,GM,0,bsizeU);
    Mat_Trans(GM12,D);
    Mat_Copy_MN(D,GM,bsizeU,0);

  }

  /* solve generalized eigenvalue problem */
  myeig1(GL,GM,GU,lambda);
  Mat_Copy_Cols(GU,Temp_Dense,0,bsizeU-1);
  Mat_Copy(Temp_Dense,GU);

  Mat_Copy_Rows(GU,Temp_Dense,bsizeU,bsizeU+bsizeR-1);
  Mat_Mult(R,Temp_Dense,TMP_Global1);
  Mat_Copy(TMP_Global1,R);
  Mat_Mult(LR,Temp_Dense,TMP_Global1);
  Mat_Copy(TMP_Global1,LR);

  if (restart==0){
    Mat_Copy_Rows(GU,Temp_Dense,bsizeU+bsizeR,bsizeU+bsizeR+bsizeP-1);
    Mat_Mult(P,Temp_Dense,TMP_Global1);
    Mat_Add(R,TMP_Global1,1,P);

    Mat_Mult(LP,Temp_Dense,TMP_Global1);
    Mat_Add(LR,TMP_Global1,1,LP);

    Mat_Copy_Rows(GU,Temp_Dense,0,bsizeU-1);
    Mat_Mult(U,Temp_Dense,TMP_Global1);
    Mat_Add(P,TMP_Global1,1,U);

    Mat_Mult(LU,Temp_Dense,TMP_Global1);
    Mat_Add(LP,TMP_Global1,1,LU);
  }
  else
  {
    Mat_Copy(R,P);   
    Mat_Copy(LR,LP);   

    Mat_Copy_Rows(GU,Temp_Dense,0,bsizeU-1);
    Mat_Mult(U,Temp_Dense,TMP_Global1);
    Mat_Add(R,TMP_Global1,1,U);

    Mat_Mult(LU,Temp_Dense,TMP_Global1);
    Mat_Add(LR,TMP_Global1,1,LU);
  }

  return 0;
} 

/*------------------------------------------------------------------------*/
/* myeig1                                                                 */
/*------------------------------------------------------------------------*/
int myeig1(Matx *A, Matx *B, Matx *X,double *lambda)
{
/*--------------------------------------------------------------------------
 * We deal with a blas portability issue (it's actually a 
 * C-calling-fortran issue) by using the following macro to call the blas:
 * hypre_F90_NAME_BLAS(name, NAME)();
 * So, for dsygv, we use:
 *
 *  hypre_F90_NAME_BLAS(dsygv, DSYGV)();
 *
 * This helps to get portability on some other platforms that 
 * such as on Cray computers.
 * The include file fortran.h is needed.
 *--------------------------------------------------------------------------*/

  int i,j,n,lda,lwork,info,itype,ldb;
  char jobz,uplo;
  double *a,*b,*work,temp;



  /* do some checks */
  assert(A->m==A->n);
  assert(B->m==B->n);
  assert(A->m==B->m);
  assert(A->mat_storage_type==DENSE);
  assert(B->mat_storage_type==DENSE);
  
  n=A->n; 
  lda=n;
  ldb=n;
  jobz='V';
  uplo='U';
  lwork=10*n;
  itype=1;

  Mat_Init_Dense(X,n,n,GENERAL);

  /* allocate memory */
    a=(double *)calloc((long)n*n,sizeof(double));
    b=(double *)calloc((long)n*n,sizeof(double));
    work=(double *)calloc((long)lwork,sizeof(double));
  if (a==NULL || b==NULL || work==NULL){
    fprintf(stderr, "Out of memory.\n");
    abort();
  }

  /* convert C-style to Fortran-style storage */
  /* column major order */
  for(i=0;i<n;++i){
    for(j=0;j<n;++j){
      a[j+n*i]=A->val[j][i];
      b[j+n*i]=B->val[j][i];
    }
  }
  assert(a!=NULL);assert(b!=NULL);
  
  /* compute generalized eigenvalues and eigenvectors of A*x=lambda*B*x */
  /*dsygv_(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb,
     lambda, &work[0], &lwork, &info);a */
  hypre_F90_NAME_BLAS(dsygv, DSYGV)(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb,
     lambda, &work[0], &lwork, &info);

  /* compute transpose of A */
  for (i=0;i<n;i++){
    for (j=0;j<i;j++){
    temp=A->val[i][j];
    A->val[i][j]=A->val[j][i];
    A->val[j][i]=temp;
    }
  }

  /* load X */
  /* convert Fortran-style to C-style storage */
  /* row  major order */
  for(j=0;j<n;++j){
    for(i=0;i<n;++i){
      X->val[i][j]=a[i+n*j];
    }
  }

  /* check error condition */
  if (info!=0) fprintf(stderr, "problem in dsygv eigensolver, info=%d\n",info);
  Trouble_Check(0,info);

  free(a);
  free(b);
  free(work);
  return info;
}

/******************************************************************************
*     SUBROUTINE DSYGV( ITYPE, JOBZ, UPLO, N, A, LDA, B, LDB, W, WORK,
     $                  LWORK, INFO )
*
*  -- LAPACK driver routine (version 3.0) --
*     Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
*     Courant Institute, Argonne National Lab, and Rice University
*     June 30, 1999
*
*     .. Scalar Arguments ..
      CHARACTER          JOBZ, UPLO
      INTEGER            INFO, ITYPE, LDA, LDB, LWORK, N
*     ..
*     .. Array Arguments ..
      DOUBLE PRECISION   A( LDA, * ), B( LDB, * ), W( * ), WORK( * )
*     ..
*
*  Purpose
*  =======
*
*  DSYGV computes all the eigenvalues, and optionally, the eigenvectors
*  of a real generalized symmetric-definite eigenproblem, of the form
*  A*x=(lambda)*B*x,  A*Bx=(lambda)*x,  or B*A*x=(lambda)*x.
*  Here A and B are assumed to be symmetric and B is also
*  positive definite.
*
*  Arguments
*  =========
*
*  ITYPE   (input) INTEGER
*          Specifies the problem type to be solved:
*          = 1:  A*x = (lambda)*B*x
*          = 2:  A*B*x = (lambda)*x
*          = 3:  B*A*x = (lambda)*x
*
*  JOBZ    (input) CHARACTER*1
*          = 'N':  Compute eigenvalues only;
*          = 'V':  Compute eigenvalues and eigenvectors.
*
*  UPLO    (input) CHARACTER*1
*          = 'U':  Upper triangles of A and B are stored;
*          = 'L':  Lower triangles of A and B are stored.
*
*  N       (input) INTEGER
*          The order of the matrices A and B.  N >= 0.
*
*  A       (input/output) DOUBLE PRECISION array, dimension (LDA, N)
*          On entry, the symmetric matrix A.  If UPLO = 'U', the
*          leading N-by-N upper triangular part of A contains the
*          upper triangular part of the matrix A.  If UPLO = 'L',
*          the leading N-by-N lower triangular part of A contains
*          the lower triangular part of the matrix A.
*
*          On exit, if JOBZ = 'V', then if INFO = 0, A contains the
*          matrix Z of eigenvectors.  The eigenvectors are normalized
*          as follows:
*          if ITYPE = 1 or 2, Z**T*B*Z = I;
*          if ITYPE = 3, Z**T*inv(B)*Z = I.
*          If JOBZ = 'N', then on exit the upper triangle (if UPLO='U')
*          or the lower triangle (if UPLO='L') of A, including the
*          diagonal, is destroyed.
*
*  LDA     (input) INTEGER
*          The leading dimension of the array A.  LDA >= max(1,N).
*
*  B       (input/output) DOUBLE PRECISION array, dimension (LDB, N)
*          On entry, the symmetric positive definite matrix B.
*          If UPLO = 'U', the leading N-by-N upper triangular part of B
*          contains the upper triangular part of the matrix B.
*          If UPLO = 'L', the leading N-by-N lower triangular part of B
*          contains the lower triangular part of the matrix B.
*
*          On exit, if INFO <= N, the part of B containing the matrix is
*          overwritten by the triangular factor U or L from the Cholesky
*          factorization B = U**T*U or B = L*L**T.
*
*  LDB     (input) INTEGER
*          The leading dimension of the array B.  LDB >= max(1,N).
*
*  W       (output) DOUBLE PRECISION array, dimension (N)
*          If INFO = 0, the eigenvalues in ascending order.
*
*  WORK    (workspace/output) DOUBLE PRECISION array, dimension (LWORK)
*          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
*
*  LWORK   (input) INTEGER
*          The length of the array WORK.  LWORK >= max(1,3*N-1).
*          For optimal efficiency, LWORK >= (NB+2)*N,
*          where NB is the blocksize for DSYTRD returned by ILAENV.
*
*          If LWORK = -1, then a workspace query is assumed; the routine
*          only calculates the optimal size of the WORK array, returns
*          this value as the first entry of the WORK array, and no error
*          message related to LWORK is issued by XERBLA.
*
*  INFO    (output) INTEGER
*          = 0:  successful exit
*          < 0:  if INFO = -i, the i-th argument had an illegal value
*          > 0:  DPOTRF or DSYEV returned an error code:
*             <= N:  if INFO = i, DSYEV failed to converge;
*                    i off-diagonal elements of an intermediate
*                    tridiagonal form did not converge to zero;
*             > N:   if INFO = N + i, for 1 <= i <= N, then the leading
*                    minor of order i of B is not positive definite.
*                    The factorization of B could not be completed and
*                    no eigenvalues or eigenvectors were computed.
*
*  =====================================================================
******************************************************************************/

/*****************************************************************************/
int Trouble_Check(int mode,int test)
{
  static int trouble=0;
  if (mode==0)
  {
     if (test!=0) trouble=1;
     else trouble=0;
  }
  return trouble;
}

