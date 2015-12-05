/*BHEADER**********************************************************************
 * lobpcg_matrix.c
 *
 * $Revision: 1.6 $
 * Date: 10/7/2002
 * Authors: M. Argentati and A. Knyazev
 *********************************************************************EHEADER*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <float.h>
#include <math.h>
#include <time.h>
#include "lobpcg.h"

/* glabal variables */

/*****************************************************************************/
int Assemble_DENSE(Matx *A,input_data *input1,int mA,int nA,int nzA,mt mat_type)
{
  /* Assemble mA x nA matrix with nzA non-zeros into a dense matrix */

  int i,j,row,col;

  Mat_Init(A,mA,nA,nzA,DENSE,mat_type);

  /* zero out matrix A */
  for (i=0;i<mA;++i){
    for (j=0;j<nA;++j){
      A->val[i][j]=0;
    }
  }
 
  /* define nonzero values for matrix A */
  /* handle SYMMETRIC case */
  if (mat_type==SYMMETRIC){
    for (i=0;i<nzA;++i){
      row=(int)input1[i].row;
      col=(int)input1[i].col;
      A->val[row][col]=input1[i].val;
      if (row>col){ /* fill in upper triangular part */
        A->val[col][row]=input1[i].val;
      }
    }
  }
  /* handle GENERAL case */
  else { 
    for (i=0;i<nzA;++i){
      row=(int)input1[i].row;
      col=(int)input1[i].col;
      A->val[row][col]=input1[i].val;
    }
  }
  return 0;
}

/*****************************************************************************/
int Mat_Mult(Matx *A,Matx *B,Matx *C)
{
  /* Compute C=A x B */
  int i,j,k;
  double sum,zero=0,temp2;

  if ((A->mat_storage_type==DENSE) && (B->mat_storage_type==DENSE)){
    /* check for compatable dimensions */
    assert(A->n == B->m);

    /* assume C is dense */
    Mat_Init_Dense(C,A->m,B->n,GENERAL);

    for (i=0;i<C->m;++i){
      for (j=0;j<C->n;++j){
        sum=0;
        for (k=0;k<A->n;++k){
          sum=sum+A->val[i][k]*B->val[k][j];
        }
        C->val[i][j]=sum;
      }
    }
    return 0;
  }
  else if ((A->mat_storage_type==HYPRE_VECTORS) && (B->mat_storage_type==DENSE)){
    /* multiply a set of HYPRE vectors A times a dense matrix B
       to form a set of HYPRE vectors C */
    assert(A->n == B->m);

    Mat_Init(C,A->m,B->n,A->m*B->n,HYPRE_VECTORS,GENERAL);

    for (j=0; j<C->n; j++) {
      /*ierr=VecSet(&zero,C->X[j]);CHKERRQ(ierr);*/
      if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorSetConstantValues_Data,0);
      HYPRE_ParVectorSetConstantValues(C->vsPar[j],zero);
      

      for (i=0; i<B->m; i++){
        temp2 = B->val[i][j];
        /*ierr=VecAXPY(&temp2,A->X[i],C->X[j]);CHKERRQ(ierr);*/
        if (verbose2(1)==TRUE) collect_data(0,hypre_ParVectorAxpy_Data,0);
        hypre_ParVectorAxpy(temp2,(hypre_ParVector *) A->vsPar[i],
         (hypre_ParVector *) C->vsPar[j]);
        
      }
    }
    return 0;
  }

  fprintf(stderr, "Mat_Mult function failed.");
  abort();
  return 1;
}

/*****************************************************************************/
int Mat_Mult2(Matx *A,Matx *B,int *idx)
{
  /* Multiply a set of HYPRE vectors A times a dense matrix B
     to form a set of HYPRE vectors A. Multiply on the right
     and only use indexed vectors. Store in same index set of 
     vectors. */

  int i,j,n,*idx2;
  double zero=0,temp2;
  extern Matx *temp_global_data;

  if ((A->mat_storage_type==HYPRE_VECTORS) && (B->mat_storage_type==DENSE)){
    /* get index count */
    n=0;
    for (i=0;i<A->n;++i){
      if (idx[i]>0) n++;
    }
    if (n==0) {
	  fprintf(stderr, "Mat_Mult2 function failed.");
	  abort();
	}

    /* allocate memory */
    if (!(idx2=(int *)malloc(n*sizeof(int)))) {
      fprintf(stderr, "Out of memory\n");
      abort();
    }

    /* build direct index */
    j=0;
    for (i=0;i<A->n;++i){
      if (idx[i]>0)
      {
        idx2[j]=i;
        j++;
      }
    }
    assert(n == B->m);

    for (j=0; j<n; j++) {
      if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorSetConstantValues_Data,0);
      HYPRE_ParVectorSetConstantValues(temp_global_data->vsPar[j],zero);
      for (i=0; i<n; i++){
        temp2 = B->val[i][j];
        if (verbose2(1)==TRUE) collect_data(0,hypre_ParVectorAxpy_Data,0);
        hypre_ParVectorAxpy(temp2,(hypre_ParVector *) A->vsPar[idx2[i]],
         (hypre_ParVector *) temp_global_data->vsPar[j]);
      }
    }
    for (j=0; j<n; j++) {
      if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorCopy_Data,0);
      HYPRE_ParVectorCopy(temp_global_data->vsPar[j],A->vsPar[idx2[j]]);
    }
    free(idx2);
    return 0;
  }
  fprintf(stderr, "Mat_Mult2 function failed.");
  abort();
  return 1;
}


/*****************************************************************************/
int Mat_Add(Matx *A,Matx *B,double alpha,Matx *C)
{
  /* Compute C=A + alpha*B */
  int i,j;
  double temp;

  if ((A->mat_storage_type==DENSE) && (B->mat_storage_type==DENSE)){
    assert(A->m == B->m);
    assert(A->n == B->n);

    /* C is dense */
    Mat_Init_Dense(C,A->m,A->n,GENERAL);

    for (i=0;i<A->m;++i){
      for (j=0;j<A->n;++j){
        C->val[i][j]=A->val[i][j]+alpha*B->val[i][j];
      }
    }
    return 0;
  }
  else if ((A->mat_storage_type==HYPRE_VECTORS) && 
          (B->mat_storage_type==HYPRE_VECTORS)){
    assert(A->m == B->m);
    assert(A->n == B->n);

    Mat_Copy(B,C);
    temp=alpha;
    for (i=0; i<C->n; i++) {
      if (verbose2(1)==TRUE) collect_data(0,hypre_ParVectorAxpy_Data,0);
      hypre_ParVectorAxpy(temp,(hypre_ParVector *) A->vsPar[i],
         (hypre_ParVector *) C->vsPar[i]);
      
    }
    return 0;
  }
  fprintf(stderr, "Mat_Add function failed.");
  abort();
  return 1;
}

/*****************************************************************************/
int Mat_Copy(Matx *A,Matx *B)
{
  /* Compute B=A */
  int i,j;
  
  if (A->mat_storage_type==DENSE){
  
    /* A is dense */
    Mat_Init_Dense(B,A->m,A->n,A->mat_type);

    for (i=0;i<A->m;++i){
      for (j=0;j<A->n;++j){
        B->val[i][j]=A->val[i][j];
      }
    }
    return 0;
  }
  else if (A->mat_storage_type==HYPRE_VECTORS){
    /* copy a hypre set of vectors to a hypre set of vectors */
    Mat_Init(B,A->m,A->n,A->nz,HYPRE_VECTORS,GENERAL);
    for (i=0; i<B->n; i++){
       if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorCopy_Data,0);
       HYPRE_ParVectorCopy(A->vsPar[i],B->vsPar[i]);
       
    }
    return 0;
  }
  fprintf(stderr, "Mat_Copy function failed.");
  abort();
  return 1;
}

/*****************************************************************************/
int Mat_Copy_Rows(Matx *A,Matx *B,int row1,int row2)
{
  /* Copy row1 through row2 of matrix A to B */
  int i,j;
 
  if (A->mat_storage_type==DENSE){
 
    /* check dimensions */
    assert(row2>=row1);
    assert(row2<A->m);

    /* A is dense */
    Mat_Init_Dense(B,row2-row1+1,A->n,GENERAL);

    for (i=row1;i<=row2;++i){
      for (j=0;j<A->n;++j){
        B->val[i-row1][j]=A->val[i][j];
      }
    }
    return 0;
  }
  assert(A->mat_storage_type==DENSE);
  return 1;
}

/*****************************************************************************/
int Mat_Copy_Cols(Matx *A,Matx *B,int col1,int col2)
{
  /* Copy col1 through col2 of matrix A to B */
  int i,j;

  if (A->mat_storage_type==DENSE){

    /* check dimensions */
    assert(col2>=col1);
    assert(col2<A->n);

    /* A is dense */
    Mat_Init_Dense(B,A->m,col2-col1+1,GENERAL);

    for (i=0;i<A->m;++i){
      for (j=col1;j<=col2;++j){
        B->val[i][j-col1]=A->val[i][j];
      }
    }
    return 0;
  }
  assert(A->mat_storage_type==DENSE);
  return 1;
}


/*****************************************************************************/
int Mat_Copy_MN(Matx *A,Matx *B,int row_offset,int col_offset)
{
  /* Copy matrix A to the location in B at row_offset, col_offset
     The matrix B is left untouched otherwise  */

  int i,j;
 
  if (A->mat_storage_type==DENSE){
 
    /* check to see if it will fit */
    assert((row_offset+A->m-1)<B->m);
    assert((col_offset+A->n-1)<B->n);

    for (i=0;i<A->m;++i){
      for (j=0;j<A->n;++j){
        B->val[row_offset+i][col_offset+j]=A->val[i][j];
      }
    }
    return 0;
  }
  assert(A->mat_storage_type==DENSE);
  return 1;
}

/*****************************************************************************/
int Mat_Inv_Triu(Matx *A,Matx *B)
{
  /* Compute inverse of upper triangular matrix B=inv(A)
     See page 91 of Numerical Recipes in Fortran 77 */

  int i,j,k,n;
  double sum,*p,min,max;

  assert(A->mat_storage_type==DENSE);
  assert(A->m == A->n);

  n=A->n;
  Mat_Copy(A,B);
  p=(double *)calloc((size_t) n,sizeof(double));
  if (p==NULL && Get_Rank()==0){
	fprintf(stderr, "Out of memory.\n");
	abort();
  }

     
  min=fabs(B->val[0][0]);
  max=fabs(B->val[0][0]);
  for (i=0;i<n;i++){
    p[i]=B->val[i][i];
    if (fabs(p[i])<min) min=fabs(p[i]);
    if (fabs(p[i])>max) max=fabs(p[i]);
    B->val[i][i]=1.0/p[i];
  }
  if ((min/max)<DBL_EPSILON ) printf("Mat_Inv_Ut: matrix is nearly singular\n");

  for (i=n-2;i>-1;i--){
    for (j=n-1;j>i;j--){
      sum=0;
      for (k=i+1;k<j+1;++k){
        sum=sum-B->val[i][k]*B->val[k][j];
      }
      B->val[i][j]=sum/p[i];
    }
  }

  free(p);
  return 0;
}

/*****************************************************************************/
int Mat_Get_Col(Matx *A,Matx *B,int *idxA)
{
  /* Compute B=A(idxA) */
  int i,j,k,count;

  if (A->mat_storage_type==DENSE){
    /* get index count */
    count=0;
    for (i=0;i<A->n;++i){
      if (idxA[i]>0) count++;
    }
    assert(count>0);
    Mat_Init_Dense(B,A->m,count,GENERAL);

    j=0;
    for (k=0;k<A->n;++k){
      if (idxA[k]>0){
        for (i=0;i<A->m;++i){
          B->val[i][j]=A->val[i][k];
        }
        j++;
      }
    }
    return 0;
  }
  else if (A->mat_storage_type==HYPRE_VECTORS){
    /* get index count */
    count=0;
    for (i=0;i<A->n;++i){
      if (idxA[i]>0) count++;
    }
    assert(count>0);
    Mat_Init(B,A->m,count,A->m*count,HYPRE_VECTORS,GENERAL);

    j=0;
    for (k=0;k<A->n;++k){
      if (idxA[k]>0){
        /*VecCopy(A->X[k],B->X[j]);CHKERRQ(ierr);*/
        if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorCopy_Data,0);
        HYPRE_ParVectorCopy(A->vsPar[k],B->vsPar[j]);
        
        j++;
      }
    }
    return 0;
  }
  fprintf(stderr, "Mat_Get_Col function failed.");
  abort();
  return 1;
}

/*****************************************************************************/
int Mat_Get_Col2(Matx *A,int *idxA)
{
  /* Compute A=A(idxA). Copy only indexed vectors back into A */
  int i,j,k,count;

  if (A->mat_storage_type==HYPRE_VECTORS){
    /* get index count */
    count=0;
    for (i=0;i<A->n;++i){
      if (idxA[i]>0) count++;
    }
    assert(count>0);
    Mat_Init(A,A->m,count,A->m*count,HYPRE_VECTORS,GENERAL);

    j=0;
    for (k=0;k<A->n;++k){
      if (idxA[k]>0){
        if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorCopy_Data,0);
        HYPRE_ParVectorCopy(A->vsPar[k],A->vsPar[j]);
        j++;
      }
    }
    return 0;
  }
  fprintf(stderr, "Mat_Get_Col2 function failed.");
  abort();
  return 1;
}

/*****************************************************************************/
int Mat_Put_Col(Matx *A,Matx *B,int *idxB)
{
  /* Compute B(idxB)=A. Only columns defined by idx are updated */
  int i,j,k,count;

  if ((A->mat_storage_type==DENSE) && (B->mat_storage_type==DENSE)){
    /* get index count */
    count=0;
    for (i=0;i<B->n;++i){
      if (idxB[i]>0) count++;
    }
    assert(count>0);
    assert(A->n==count);
    assert(B->n>=count);

    k=0;
    for (j=0;j<B->n;++j){
      if (idxB[j]>0){
        for (i=0;i<A->m;++i){
          B->val[i][j]=A->val[i][k];
        }
        k++;
      }
    }
    return 0;
  }
  else if ((A->mat_storage_type==HYPRE_VECTORS) && 
          (B->mat_storage_type==HYPRE_VECTORS)){
    /* get index count */
    count=0;
    for (i=0;i<B->n;++i){
      if (idxB[i]>0) count++;
    }
    assert(count>0);
    assert(A->n==count);
    assert(B->n>=count);
    k=0;
    for (j=0;j<B->n;++j){
      if (idxB[j]>0){
        /*VecCopy(A->X[k],B->X[j]);CHKERRQ(ierr);*/
        if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorCopy_Data,0);
        HYPRE_ParVectorCopy(A->vsPar[k],B->vsPar[j]);
        
        k++;
      }
    }
    return 0;
  }
  fprintf(stderr, "Mat_Put_Col function failed.");
  abort();
  return 1;
}

/*****************************************************************************/
int Mat_Diag(double *d,int n,Matx *A)
{
  /* Compute A=diag(d) */
  int i;

  /* A is dense */
  Mat_Init_Dense(A,n,n,SYMMETRIC);

  for (i=0;i<A->m;++i){
    A->val[i][i]=d[i];
  }

  return 0;
}

/*****************************************************************************/
int Mat_Eye(int n,Matx *A)
{
  /* Compute A=identity (n x n) */
  int i;

  /* A is dense */
  Mat_Init_Dense(A,n,n,SYMMETRIC);

  for (i=0;i<A->m;++i){
    A->val[i][i]=1.0;
  }

  return 0;
}

/*****************************************************************************/
int Mat_Trans(Matx *A,Matx *B)
{
  /* Compute B=A' */
  int i,j;

  if (A->mat_storage_type==DENSE){
    /* A is dense */
    Mat_Init_Dense(B,A->n,A->m,A->mat_type);

    for (i=0;i<B->m;++i){
      for (j=0;j<B->n;++j){
        B->val[i][j]=A->val[j][i];
      }
    }
    B->mat_type=A->mat_type;
    return 0;
  }
  fprintf(stderr, "Mat_Trans function failed.");
  abort();
  return 1;
}

/*****************************************************************************/
int Mat_Trans_Idx(Matx *A,Matx *B,Matx *C,int *idxA,int *idxB)
{
  /* Compute C=A(idxA)'*B(idxB) */
  int i,j,k,row=0,col=0,nidxA=0,nidxB=0;
  double sum;

  assert(A->mat_storage_type==DENSE);
  assert(B->mat_storage_type==DENSE);

  /* check for compatable dimensions */
  assert(A->m == B->m);

  /* get index count */
  for (i=0;i<A->n;++i){
    if (idxA[i]>0) nidxA++;
  }
  for (i=0;i<B->n;++i){
    if (idxB[i]>0) nidxB++;
  }
  assert(nidxA>0);
  assert(nidxB>0);

  Mat_Init_Dense(C,nidxA,nidxB,GENERAL);

  for (i=0;i<A->n;++i){
    for (j=0;j<B->n;++j){
      if (idxA[i]>0 && idxB[j]>0){
        sum=0;
        for (k=0;k<A->m;++k){
          sum=sum+A->val[k][i]*B->val[k][j];
        }
        C->val[row][col]=sum;
      }
      if (idxB[j]>0) row++;
    }
    if (idxA[i]>0) col++;
  }

  /* check counts */
  assert(col==nidxA);
  assert(row==nidxB);

  return 0;
}

/*****************************************************************************/
int Mat_Trans_Mult(Matx *A,Matx *B,Matx *C)
{
  /* Compute C=A' x B (transpose of A) x B */
  int i,j,k;
  register double sum;
  double temp;

  if ((A->mat_storage_type==DENSE) && (B->mat_storage_type==DENSE)){
    /* check for compatable dimensions */
    assert(A->m == B->m);

    /* assume C is dense */
    Mat_Init_Dense(C,A->n,B->n,GENERAL);

    for (i=0;i<A->n;++i){
      for (j=0;j<B->n;++j){
        sum=0;
        for (k=0;k<A->m;++k){
          sum=sum+A->val[k][i]*B->val[k][j];
        }
        C->val[i][j]=sum;
      }
    }
    return 0;
  }
  else if ((A->mat_storage_type==HYPRE_VECTORS) && 
          (B->mat_storage_type==HYPRE_VECTORS)){
    /* multiply the transpose of a set of HYPRE vectors A
       times a set of HYPRE vectors B to make a dense matrix C */
    assert(A->m == B->m);
    Mat_Init(C,A->n,B->n,A->n*B->n,DENSE,GENERAL);

    for (i=0;i<A->n;++i){
      for (j=0;j<B->n;++j){
        if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorInnerProd_Data,0);
        HYPRE_ParVectorInnerProd(A->vsPar[i],B->vsPar[j],&temp);
        
        C->val[i][j]=temp;
      }
    }
    return 0;
  }

  fprintf(stderr, "Incompatable matrix types in Mat_Trans_Mult\n");
  abort();
  return 1;
}

/*****************************************************************************/
int Mat_Trans_Mult2(Matx *A,int *idxA,Matx *B,int *idxB,Matx *C)
{
  /* Compute C=A' x B (transpose of A) x B based on indexes */
  int i,j,nA=0,nB=0;
  double temp;
  int *idxA2,*idxB2;

  /* setup index for A */
  for (i=0;i<A->n;++i){
    if (idxA[i]>0) nA++;
  }
  if (nA==0) {
	fprintf(stderr, "Mat_Trans_Mult2 function failed.");
	abort();
  }
  /* allocate memory */
  if (!(idxA2=(int *)malloc(nA*sizeof(int)))) {
    fprintf(stderr, "Out of memory\n");
    abort();
  }
  /* build direct index */
  j=0;
  for (i=0;i<A->n;++i){
    if (idxA[i]>0)
    {
      idxA2[j]=i;
      j++;
    }
  }

  /* setup index for B */
  for (i=0;i<B->n;++i){
    if (idxB[i]>0) nB++;
  }
  if (nB==0) {
	fprintf(stderr, "Mat_Trans_Mult2 function failed.");
	abort();
  }
  /* allocate memory */
  if (!(idxB2=(int *)malloc(nB*sizeof(int)))) {
	fprintf(stderr, "Out of memory\n");
    abort();
  }
  /* build direct index */
  j=0;
  for (i=0;i<B->n;++i){
    if (idxB[i]>0)
    {
      idxB2[j]=i;
      j++;
    }
  }

  if ((A->mat_storage_type==HYPRE_VECTORS) &&
          (B->mat_storage_type==HYPRE_VECTORS)){
    /* multiply the transpose of a set of HYPRE vectors A
       times a set of HYPRE vectors B to make a dense matrix C */
    assert(A->m == B->m);

    Mat_Init(C,nA,nB,nA*nB,DENSE,GENERAL);

    for (i=0;i<nA;++i){
      for (j=0;j<nB;++j){
        if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorInnerProd_Data,0);
        HYPRE_ParVectorInnerProd(A->vsPar[idxA2[i]],B->vsPar[idxB2[j]],&temp);
        
        C->val[i][j]=temp;
      }
    }
    free(idxA2);
    free(idxB2);
    return 0;
  }

  fprintf(stderr, "Incompatable matrix types in Mat_Trans_Mult\n");
  abort();
  return 1;
}

/*****************************************************************************/
int Mat_Sym(Matx *A)
{
  /* Symmetrize A=(A+A')/2 */
  int i,j;
  Matx *TMP;

  /* must be dense */
  assert(A->mat_storage_type==DENSE);

  /* check for compatable dimensions */
  assert(A->m == A->n);

  TMP=Mat_Alloc1();
  Mat_Init_Dense(TMP,A->m,A->n,SYMMETRIC);

  for (i=0;i<A->m;++i){
    for (j=0;j<A->n;++j){
      TMP->val[i][j]=(A->val[i][j]+A->val[j][i])/2.0;
    }
  }

  Mat_Copy(TMP,A);
  Mat_Free(TMP);
  free(TMP);
  return 0;
}

/*****************************************************************************/
int Mat_Norm2_Col(Matx *A,double *y)
{
  /* take the 2-norm of the columns of A and store in y */
  int i,j;
  double temp;

  if (A->mat_storage_type==DENSE){
    /* check for non-zero dimensions */
    assert(A->m>0 && A->n>0);
    for (j=0;j<A->n;++j){
      y[j]=0;
      for (i=0;i<A->m;++i){
        y[j]=y[j]+A->val[i][j]*A->val[i][j];
      }
      y[j]=sqrt(y[j]);
    }
    return 0;
  }
  else if (A->mat_storage_type==HYPRE_VECTORS){
    /* check for non-zero dimensions */
    assert(A->m>0 && A->n>0);
    for (j=0;j<A->n;++j){
      if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorInnerProd_Data,0);
      HYPRE_ParVectorInnerProd(A->vsPar[j],A->vsPar[j],&temp);
      
      y[j]=sqrt(temp);
    }
    return 0;
  }
  fprintf(stderr, "Mat_Norm2_Col function failed.");
  abort();
  return 1;
}

/*****************************************************************************/
int Mat_Size(Matx *A,int rc)
{
  if (rc==1) return A->m;
  else return A->n;
}

/*****************************************************************************/
int Mat_Init1(Matx *A)
{
  /* initialize matrix to well defined initial state */
  A->m=0;
  A->n=0;
  A->nz=0;
  A->mat_storage_type=NONE1;
  A->mat_type=NONE2;
  A->numb_par_vectors_alloc=0;
  return 0;
}

/*****************************************************************************/
Matx *Mat_Alloc1()
{
  /* allocate initialize matrix and set to well defined initial state */
  Matx *A;

  if (!(A=(Matx *) malloc(sizeof(Matx)))) {
    fprintf(stderr, "Out of memory\n");
    abort();
  }
  Mat_Init1(A);

  return A;
}

/*****************************************************************************/
int Mat_Init_Dense(Matx *A,int m,int n,mt mat_type)
{
  /* allocate and initialize dense matrix of size m x n */

  assert(n>0 && m>0);
   Mat_Free(A);
  A->val=Mymalloc(m,n);
  A->m=m;
  A->n=n;
  A->nz=n*m;
  A->mat_storage_type=DENSE;
  A->mat_type=mat_type;

  return 0;
}

/*****************************************************************************/
int Mat_Init(Matx *A,int m,int n,int nz,mst mat_storage_type,mt mat_type)
{
  int i,*partitioning,*part2;

  int nprocs;

  /* allocate and initialize a matrix */
  if (mat_storage_type==DENSE){
    assert(n>0 && m>0);
    /* don't reallocate if matrix exists and is same size */
    if ((A->mat_storage_type==DENSE) && (A->m==m)
       && (A->n==n))
    {
      A->nz=nz;
      A->mat_type=mat_type;
      return 0;
    }
    if (A->mat_storage_type!=0) Mat_Free(A);
    A->val=Mymalloc(m,n);
    A->m=m;
    A->n=n;
    A->nz=n*m;
    A->mat_storage_type=mat_storage_type;
    A->mat_type=mat_type;
  }
  else if (mat_storage_type==HYPRE_MATRIX){
	fprintf(stderr, "Mat_Init function failed.");
	abort();
	assert(n>0 && m>0 && nz>0);
	if (A->mat_storage_type!=0) Mat_Free(A);
	A->m=m;
	A->n=n;
	A->nz=nz;
	A->mat_storage_type=mat_storage_type;
	A->mat_type=mat_type;
  }
  else if (mat_storage_type==HYPRE_VECTORS){
	assert(n>0 && m>0 && nz>0);
	
	/* don't reallocate if these vectors already exist */
	/* reuse existing vectors */
	if ((A->mat_storage_type==HYPRE_VECTORS) && (A->m==m)
        && (n<=A->numb_par_vectors_alloc))
	  {
        A->n=n;
        A->nz=nz;
        A->mat_type=mat_type;
        return 0; 
	  }
	
	if (A->mat_storage_type!=0) Mat_Free(A);
	A->m=m;
	A->n=n;
	A->nz=nz;
	A->mat_storage_type=mat_storage_type;
	A->mat_type=mat_type;
	
	/* allocate memory */
	if ((A->vsPar=(HYPRE_ParVector *) malloc(A->n*sizeof(HYPRE_ParVector)))==NULL){
	  fprintf(stderr, "Could not allocate memory.\n");
	  abort();
	}
	A->numb_par_vectors_alloc=A->n;
	
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	hypre_LobpcgSetGetPartition(1,&partitioning);
	for (i=0; i<A->n; i++){
	  part2=CopyPartition(partitioning);
	  if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorCreate_Data,0);
	  HYPRE_ParVectorCreate(MPI_COMM_WORLD,A->m,part2,&A->vsPar[i]);
	  HYPRE_ParVectorInitialize(A->vsPar[i]);
	  
	}
  }
  else {
	fprintf(stderr, "Mat_Init function failed.");
	abort();
  }
  
  return 0;
}

/*****************************************************************************/
int Init_Rand_Vectors(HYPRE_ParVector *v_ptr,int *partitioning, int m,int n)
{
  /* randomize an array of n existing parallel vectors each of length m */

  int i,j;
  double temp;
  hypre_Vector  *v_temp;
  double  *vector_data;
  int   size,mypid;
  int *part2;
  extern HYPRE_ParVector temp_global_vector;

  /* initialize random number generator */
  srand((unsigned int) time(0));

  MPI_Comm_rank(MPI_COMM_WORLD, &mypid);

  v_temp=hypre_SeqVectorCreate(m);
  hypre_SeqVectorInitialize(v_temp);
  vector_data = hypre_VectorData(v_temp);
  size=hypre_VectorSize(v_temp);
  for (i=0; i<n; i++) {
    if (mypid == 0){
      for (j = 0; j < size; j++){
        temp=rand();
        vector_data[j] = temp/RAND_MAX;
      }
    }
    part2=CopyPartition(partitioning);
    HYPRE_VectorToParVector(MPI_COMM_WORLD,(HYPRE_Vector) v_temp,
      part2,&temp_global_vector);
    if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorCopy_Data,0);
    HYPRE_ParVectorCopy(temp_global_vector,v_ptr[i]);
  }
  hypre_SeqVectorDestroy(v_temp);
  return 0;
}

/*****************************************************************************/
int Mat_Init_Identity(Matx *A,int m,int n,mst mat_storage_type,int *partitioning)
{
  /* allocate and initialize set of hypre vectors equal to the identity */
  int i,j;
  hypre_Vector  *v_temp;
  double  *vector_data;
  int   size,mypid,nprocs;
  int *part2;

  if (mat_storage_type==HYPRE_VECTORS){
    assert(n>0 && m>0);
    if (A->mat_storage_type!=0) Mat_Free(A);
    A->m=m;
    A->n=n;
    A->nz=m*n;
    A->mat_storage_type=HYPRE_VECTORS;
    A->mat_type=GENERAL;

    /* allocate memory */
    if ((A->vsPar=(HYPRE_ParVector *) malloc(A->n*sizeof(HYPRE_ParVector)))==NULL){
       fprintf(stderr, "Could not allocate memory.\n");
       abort();
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &mypid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    v_temp=hypre_SeqVectorCreate(m);
    hypre_SeqVectorInitialize(v_temp);
    vector_data = hypre_VectorData(v_temp);
    size=hypre_VectorSize(v_temp);
    for (i=0; i<n; i++) {
       if (mypid == 0){
         for (j = 0; j < size; j++){
           if (j==i) vector_data[j] = 1.0;
           else vector_data[j]=0.0;
         }
       }
       part2=CopyPartition(partitioning);
       HYPRE_VectorToParVector(MPI_COMM_WORLD,(HYPRE_Vector) v_temp,part2,&A->vsPar[i]);
       
    }
    hypre_SeqVectorDestroy(v_temp);
    return 0;
  }
  fprintf(stderr, "Mat_Init_Identity function failed.");
  abort();
  return 1;
}

/*****************************************************************************/
int Init_Eye_Vectors(HYPRE_ParVector *v_ptr,int *partitioning, int m,int n)
{
  /* set an array of n existing parallel vectors each of length m
     to an m x n identity  */

  int i,j;
  hypre_Vector  *v_temp;
  HYPRE_ParVector vpar;
  double  *vector_data;
  int   size,mypid;
  int *part2;

  MPI_Comm_rank(MPI_COMM_WORLD, &mypid);

  v_temp=hypre_SeqVectorCreate(m);
  hypre_SeqVectorInitialize(v_temp);
  vector_data = hypre_VectorData(v_temp);
  size=hypre_VectorSize(v_temp);
  for (i=0; i<n; i++) {
    if (mypid == 0){
      for (j = 0; j < size; j++){
        if (i==j) vector_data[j] = 1.0;
        else vector_data[j] = 0.0;
      }
    }
    part2=CopyPartition(partitioning);
    HYPRE_VectorToParVector(MPI_COMM_WORLD,(HYPRE_Vector) v_temp,
      part2,&vpar);
    if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorCopy_Data,0);
    HYPRE_ParVectorCopy(vpar,v_ptr[i]);
  }
  hypre_SeqVectorDestroy(v_temp);
  if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorDestroy_Data,0);
  HYPRE_ParVectorDestroy(vpar);
  return 0;
}


/*--------------------------------------------------------------------------
 * hypre_ParVectorToVector:
 * generates a Vector from a ParVector on proc 0 and obtains the pieces
 * from the other procs in comm
 *--------------------------------------------------------------------------*/
hypre_Vector *
hypre_ParVectorToVector (MPI_Comm comm, hypre_ParVector *v)
{
   int			i,j;
   int 			global_size,local_size,*partition;
   int  		num_procs, my_id;
   double               *data1,*data2;
   hypre_Vector     	*local_vector1=NULL,*local_vector2=NULL;
   MPI_Status		status;

   MPI_Comm_size(comm,&num_procs);
   MPI_Comm_rank(comm,&my_id);

   partition=hypre_ParVectorPartitioning(v);

   if (my_id == 0){
     global_size=partition[num_procs];
     local_vector1=hypre_SeqVectorCreate(global_size);
     hypre_SeqVectorInitialize(local_vector1);
     data1 = hypre_VectorData(local_vector1);
     for (i=0;i<num_procs;++i){
       local_size = partition[i+1] - partition[i];
       if (i>0){
          /* receive data from other processors */
          MPI_Recv(&data1[partition[i]],local_size,MPI_DOUBLE,i,0,comm,&status);
       }
       else {
         /* handle data already on processor 0 */
         local_vector2=hypre_ParVectorLocalVector(v);
         data2=hypre_VectorData(local_vector2);
         for (j=0;j<local_size;++j) data1[j]=data2[j];
       }
     }
   }
   else {
     local_vector2=hypre_ParVectorLocalVector(v);
     data2=hypre_VectorData(local_vector2);
     local_size = partition[my_id+1] - partition[my_id];
     /* send data to processor 0 */
     MPI_Send(data2,local_size,MPI_DOUBLE, 0,0,comm);
   }
   return local_vector1;
}

/*****************************************************************************/
double **Mymalloc(int m,int n)
{
  /* Allocate memory for a m x n  double array */

  int i,j;
  double **a;

  if (!(a=(double **)malloc(m*sizeof(double*)))) {
    fprintf(stderr, "Out of memory\n");
    abort();
  }
  if (!(a[0]=(double *)malloc(m*n*sizeof(double)))) {
    fprintf(stderr, "Out of memory\n");
    abort();
  }

  for (i=1;i<m;i++) a[i]=a[i-1]+n;

  /* initialize to zero */
  for (i=0;i<m;i++){
    for (j=0;j<n;j++){
      a[i][j]=0;
    }
  }

  return a;
}

/*****************************************************************************/
int Mat_Free(Matx *A)
{
  int i;

  /* check for error */
  if (A->mat_storage_type<0 || A->mat_storage_type>5){
    abort();
  }

  /* free all memory that has been previously allocated */
  if (A->mat_storage_type==DENSE){
    if (A->val != NULL){
      if (A->val[0] != NULL) free(A->val[0]);
      free(A->val);
    }
    Mat_Init1(A);
  }
  else if (A->mat_storage_type==HYPRE_MATRIX){
    HYPRE_ParCSRMatrixDestroy(A->MPar);
    Mat_Init1(A);
  }
  else if (A->mat_storage_type==HYPRE_VECTORS){
    if (A->vsPar != NULL){
      for (i=0;i<A->numb_par_vectors_alloc;++i){
        if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorDestroy_Data,0);
        HYPRE_ParVectorDestroy(A->vsPar[i]);
      }
      free(A->vsPar);
    } 
    Mat_Init1(A);
  }

  return 0;
}

/*****************************************************************************/
int Qr1(Matx *U,Matx *V,double **rrq, int n)
{
  /* hypre: Orthonormalize the vectors u and store in the  vectors v
     using the Modified Gram-Schmidt algorithm */

  int i,j;
  double rr,temp;

  for (i=0;i<n;i++){
    if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorCopy_Data,0);
    HYPRE_ParVectorCopy(U->vsPar[i],V->vsPar[i]);
    for (j=0;j<i;j++){
      if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorInnerProd_Data,0);
      HYPRE_ParVectorInnerProd( V->vsPar[j],V->vsPar[i],&rr);
      rrq[j][i]=rr;
      rr=-rr;
      if (verbose2(1)==TRUE) collect_data(0,hypre_ParVectorAxpy_Data,0);
      hypre_ParVectorAxpy(rr,(hypre_ParVector *) V->vsPar[j],
         (hypre_ParVector *) V->vsPar[i]);
    }
    /* compute 2-norm */
    if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorInnerProd_Data,0);
    HYPRE_ParVectorInnerProd(V->vsPar[i],V->vsPar[i],&rr);

    rr=sqrt(rr);
    if (fabs(rr)<DBL_EPSILON) printf("Qr1: rr is small, vectors almost linearly dependent\n");
    rrq[i][i]=rr;
    temp=1.0000/rr;
    rr=temp;
    HYPRE_ParVectorScale(rr,V->vsPar[i]);
  }
  return 0;
}

/*****************************************************************************/
int Qr2(Matx *V,Matx *R,int *idx) 
{
  /* Orthonormalize the vectors v and store in the  vectors v
     using the Modified Gram-Schmidt algorithm.
     Only include vectors in index. */

  int i,j,n;
  int *idx2;
  double **rrq,rr,temp;

  /* get index count */
  n=0;
  for (i=0;i<V->n;++i){
    if (idx[i]>0) n++;
  }
  if (n==0) {
	fprintf(stderr, "Qr2 function failed.");
	abort();
  }

  /* allocate memory */
  if (!(idx2=(int *)malloc(n*sizeof(int)))) {
    fprintf(stderr, "Out of memory\n");
    abort();
  }

  /* build direct index */
  j=0;
  for (i=0;i<V->n;++i){
    if (idx[i]>0)
    {
      idx2[j]=i;
      j++;
    }
  }

  /* initialize R */
  Mat_Init(R,n,n,n*n,DENSE,GENERAL);
  rrq=R->val;

  for (i=0;i<n;i++){
    for (j=0;j<i;j++){
      if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorInnerProd_Data,0);
      HYPRE_ParVectorInnerProd( V->vsPar[idx2[j]],V->vsPar[idx2[i]],&rr);
      rrq[j][i]=rr;
      rr=-rr;
      if (verbose2(1)==TRUE) collect_data(0,hypre_ParVectorAxpy_Data,0);
      hypre_ParVectorAxpy(rr,(hypre_ParVector *) V->vsPar[idx2[j]],
         (hypre_ParVector *) V->vsPar[idx2[i]]);
    }
    /* compute 2-norm */
    if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorInnerProd_Data,0);
    HYPRE_ParVectorInnerProd(V->vsPar[idx2[i]],V->vsPar[idx2[i]],&rr);
    rr=sqrt(rr);
    if (fabs(rr)<DBL_EPSILON) printf("Qr2: rr is small, vectors almost linearly dependent\n");
    rrq[i][i]=rr;
    temp=1.0000/rr;
    rr=temp;
    HYPRE_ParVectorScale(rr,V->vsPar[idx2[i]]);
  }
  free(idx2);
  return 0;
}

/*****************************************************************************/
double Mat_Norm_Inf(Matx *A)
{
  /* compute infinity norm of matrix A */
  int i,j;
  double sum1=0,sum2=0;

  if(A->mat_storage_type == DENSE){
    assert(A->m>0 && A->n>0);
    for (i=0;i<A->m;i++){
      /* check row sum of absolute values */
      for (j=0;j<A->n;j++){
        if (fabs(A->val[i][j])>sum2) sum2=fabs(A->val[i][j]);
      }
      if (sum2>sum1) sum1=sum2;
    }
    return sum1;
  }
  fprintf(stderr, "Mat_Norm_Inf function failed.");
  abort();
  return(0.0);
}

/*****************************************************************************/
double Mat_Norm_Frob(Matx *A)
{
  /* compute Frobenius norm of matrix A */
  int i,j;
  double sum1=0;

  if(A->mat_storage_type == DENSE){
    assert(A->m>0 && A->n>0);
    for (i=0;i<A->m;i++){
      for (j=0;j<A->n;j++){
        sum1=sum1+A->val[i][j]*A->val[i][j];
      }
    }
    return sqrt(sum1);
  }
  fprintf(stderr, "Mat_Norm_Frob function failed.");
  abort();
  return(0.0);
}

/*****************************************************************************/
double Max_Vec(double *y,int n)
{
  /* Take the max of a 1-D vector  */
  int i;
  double temp;

  temp=y[0];
  for (i=0;i<n;i++){
    if (y[i]>temp) temp=y[i];
  }
  return temp;
}

/*****************************************************************************/
int Get_Rank()
{
  int mypid;
  MPI_Comm_rank(MPI_COMM_WORLD, &mypid);
  return mypid;
}

/*****************************************************************************/
int myqr1(Matx *A,Matx *Q,Matx *R)
{
  /* compute qr factorization */
  if (A->mat_storage_type==HYPRE_VECTORS){
    Mat_Init(Q,A->m,A->n,A->nz,HYPRE_VECTORS,GENERAL);
    Mat_Init(R,A->n,A->n,A->n*A->n,DENSE,GENERAL);
    Qr1(A,Q,R->val,Q->n);
    return 0;
  }
  fprintf(stderr, "myqr1 function failed.");
  abort();
  return 1;
}

/*****************************************************************************/
int verbose2(int action)
{
   /* this function is used for testing to print out misc stuff */
   static int flag;

   /* set verbose2 mode  */
   if (action==0)
   {
      flag=TRUE;
      return 0;
   }
   /* get verbose2 mode */
   else if (action==1)
   {
     if (flag==TRUE) return TRUE;
     else return FALSE; 
   }
   else if (action==2)
   {
     flag=FALSE;
   }
   return(0);
}

/*****************************************************************************/
int total_numb_vectors_alloc(int count)
{
   /* this function is used to count the total number of allocated parallel vectors  */
   static int total;
   static int first=0;

   if (first==0)
   {
     first=1;
     total=0;
   }
   total=total+count;
   printf("Total number of parallel vectors allocated=%d\n",total);
   return(0);
}

/*****************************************************************************/
int *CopyPartition(int *partition)
{
  /* declare storage and copy partition into an exact copy */
  int i,*part_temp,nprocs;

  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  /* allocate memory */
  if (!(part_temp=(int *)malloc((nprocs+1)*sizeof(int)))) {
    fprintf(stderr, "Out of memory\n");
    abort();
  }

  for (i=0;i<=nprocs;++i) part_temp[i]=partition[i];
  return part_temp;
}

/*****************************************************************************/
int misc_flags(int setget,int flag)
{
   /* This function sets some misc flags for testing.
      flag=0 - use identity for A in A multiple
      flag=1 - use identity for T solver */

   static int flag_value[2]={FALSE,FALSE};

   if (flag<0 || flag>1) {
	 fprintf(stderr, "misc_flags function failed.");
	 abort();
   }
   if (setget==0)
   {
     flag_value[flag]=TRUE;
   }
   return(flag_value[flag]);
}

/*****************************************************************************/
int collect_data(int state,int counter_type,int phase)
{
   /* This function collects data on execution related counts
      for various function calls */

   int i,j;
   static int first=0;
   static int phase_in=0;
   static int counts[MAX_NUMBER_COUNTS][4];

   /* phase_in  = 0 lobpcg setup
      phase_in  = 1 iteration 1 (k=1)
      phase_in  = 2 k>1 and k<<last iteration
      phase_in  = 3 cleanup */

   /* initialize */
   if (first==0)
   {
     for (i=0;i<MAX_NUMBER_COUNTS;++i){
       for (j=0;j<4;++j) counts[i][j]=0;
     }
     first=1;
   }  

   switch (state)
   {
     /* increment counter */
     case 0:
       if (state==0) ++counts[counter_type][phase_in];
       break;
     /* check phase */
     case 1:
       if (state==1) ++phase_in;
       break;
     /* return data */
     case 2:
       return counts[counter_type][phase];
     /* intialize data */
     case 3:
       for (i=0;i<MAX_NUMBER_COUNTS;++i){
         for (j=0;j<4;++j) counts[i][j]=0;
       }
       phase_in=0;
       break;
     /* error leg */
     default:
	   fprintf(stderr, "collect_data function failed.");
       abort();
   }

   return 0;
}


/*****************************************************************************/
int time_functions(int set_run,int ftype_in,int numb_rows,int count_in,
    int (*FunctA)(HYPRE_ParVector x,HYPRE_ParVector y))
{
   /* This function generates repeated calls to a subset of 
      HYPRE functions that deal with parallel vectors.
      set_run==0 setup and get ftype and count_in,
      set_run!=0 run test using numb_rows. */

   static int state=0;
   static int ftype,count;
   HYPRE_ParVector u,v;
   int *partitioning,*part2;
   int i;
   double temp;

   /* setup data */
   if (set_run==0){
     state=1;
     ftype=ftype_in;
     if (ftype<1) {
	   fprintf(stderr, "time_functions function failed.");
	   abort();
	 }
     count=count_in;
     if (count<1) {
	   fprintf(stderr, "time_functions function failed.");
	   abort();
	 }
     return 0;
   }
   else if (state==0) return 0;

   switch (ftype)
   {

   /* test create/destroy vectors */
   case 1:
     hypre_LobpcgSetGetPartition(1,&partitioning);
     /* run test */
     for (i=0; i<count; i++)
     {
       part2=CopyPartition(partitioning);
       if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorCreate_Data,0);
       HYPRE_ParVectorCreate(MPI_COMM_WORLD,numb_rows,part2,&u);
       HYPRE_ParVectorInitialize(u);
       if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorDestroy_Data,0);
       HYPRE_ParVectorDestroy(u);
     }
     break;

   /* test inner products */
   case 2:
     /* create vectors */
     hypre_LobpcgSetGetPartition(1,&partitioning);
     part2=CopyPartition(partitioning);
     if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorCreate_Data,0);
     HYPRE_ParVectorCreate(MPI_COMM_WORLD,numb_rows,part2,&u);
     HYPRE_ParVectorInitialize(u);
     part2=CopyPartition(partitioning);
     if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorCreate_Data,0);
     HYPRE_ParVectorCreate(MPI_COMM_WORLD,numb_rows,part2,&v);
     HYPRE_ParVectorInitialize(v);

     /* set to value */
     temp=1.0;
     if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorSetConstantValues_Data,0);
     HYPRE_ParVectorSetConstantValues(u,temp);
     if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorSetConstantValues_Data,0);
     HYPRE_ParVectorSetConstantValues(v,temp);

     /* run test */
     for (i=0; i<count; i++)
     {
        if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorInnerProd_Data,0);
        HYPRE_ParVectorInnerProd(u,v,&temp);
     }

     /* destroy vectors */
     if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorDestroy_Data,0);
     HYPRE_ParVectorDestroy(u);
     if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorDestroy_Data,0);
     HYPRE_ParVectorDestroy(v);
     break;

   /* test vector copy */
   case 3:
     /* create vectors */
     hypre_LobpcgSetGetPartition(1,&partitioning);
     part2=CopyPartition(partitioning);
     if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorCreate_Data,0);
     HYPRE_ParVectorCreate(MPI_COMM_WORLD,numb_rows,part2,&u);
     HYPRE_ParVectorInitialize(u);
     part2=CopyPartition(partitioning);
     if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorCreate_Data,0);
     HYPRE_ParVectorCreate(MPI_COMM_WORLD,numb_rows,part2,&v);
     HYPRE_ParVectorInitialize(v);

     /* set to value */
     temp=1.0;
     if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorSetConstantValues_Data,0);
     HYPRE_ParVectorSetConstantValues(u,temp);
     if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorSetConstantValues_Data,0);
     HYPRE_ParVectorSetConstantValues(v,temp);

     /* run test */
     for (i=0; i<count; i++)
     {
       if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorCopy_Data,0);
       HYPRE_ParVectorCopy(u,v);
     }

     /* destroy vectors */
     if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorDestroy_Data,0);
     HYPRE_ParVectorDestroy(u);
     if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorDestroy_Data,0);
     HYPRE_ParVectorDestroy(v);
     break;

   /* test vector add */
   case 4:
     /* create vectors */
     hypre_LobpcgSetGetPartition(1,&partitioning);
     part2=CopyPartition(partitioning);
     if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorCreate_Data,0);
     HYPRE_ParVectorCreate(MPI_COMM_WORLD,numb_rows,part2,&u);
     HYPRE_ParVectorInitialize(u);
     part2=CopyPartition(partitioning);
     if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorCreate_Data,0);
     HYPRE_ParVectorCreate(MPI_COMM_WORLD,numb_rows,part2,&v);
     HYPRE_ParVectorInitialize(v);

     /* set to value */
     temp=1.0;
     if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorSetConstantValues_Data,0);
     HYPRE_ParVectorSetConstantValues(u,temp);
     if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorSetConstantValues_Data,0);
     HYPRE_ParVectorSetConstantValues(v,temp);

     /* run test */
     for (i=0; i<count; i++)
     {
       if (verbose2(1)==TRUE) collect_data(0,hypre_ParVectorAxpy_Data,0);
       hypre_ParVectorAxpy(temp,(hypre_ParVector *) u,
         (hypre_ParVector *) v);
     }

     /* destroy vectors */
     if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorDestroy_Data,0);
     HYPRE_ParVectorDestroy(u);
     if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorDestroy_Data,0);
     HYPRE_ParVectorDestroy(v);
     break;

   /* set a vector to a constant value */
   case 5:
     /* create vectors */
     hypre_LobpcgSetGetPartition(1,&partitioning);
     part2=CopyPartition(partitioning);
     if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorCreate_Data,0);
     HYPRE_ParVectorCreate(MPI_COMM_WORLD,numb_rows,part2,&u);
     HYPRE_ParVectorInitialize(u);

     /* run test */
     temp=1.0;
     for (i=0; i<count; i++)
     {
       if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorSetConstantValues_Data,0);
       HYPRE_ParVectorSetConstantValues(u,temp);
     }

     /* destroy vectors */
     if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorDestroy_Data,0);
     HYPRE_ParVectorDestroy(u);
     break;

   /* test matrix-vector multiply */
   case 6:
     /* create vectors */
     hypre_LobpcgSetGetPartition(1,&partitioning);
     part2=CopyPartition(partitioning);
     if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorCreate_Data,0);
     HYPRE_ParVectorCreate(MPI_COMM_WORLD,numb_rows,part2,&u);
     HYPRE_ParVectorInitialize(u);
     part2=CopyPartition(partitioning);
     if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorCreate_Data,0);
     HYPRE_ParVectorCreate(MPI_COMM_WORLD,numb_rows,part2,&v);
     HYPRE_ParVectorInitialize(v);

     /* set to value */
     temp=1.0;
     if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorSetConstantValues_Data,0);
     HYPRE_ParVectorSetConstantValues(u,temp);
     if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorSetConstantValues_Data,0);
     HYPRE_ParVectorSetConstantValues(v,temp);

     /* run test */
     for (i=0; i<count; i++)
     {
       if (verbose2(1)==TRUE) collect_data(0,NUMBER_A_MULTIPLIES,0);
       FunctA(u,v);
     }

     /* destroy vectors */
     if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorDestroy_Data,0);
     HYPRE_ParVectorDestroy(u);
     if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorDestroy_Data,0);
     HYPRE_ParVectorDestroy(v);
     break;

   default:
	 fprintf(stderr, "time_functions function failed.");
     abort();
   }

   return state;
}


