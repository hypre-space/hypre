#ifdef PETSC_MODE

#include "petsc_euclid.h"
#include "src/mat/matimpl.h"
#include "Euclid_dh.h"
#include "Mat_dh.h"


#if 0

  }

  /* get row and column permutation vectors, if they exist. */
  ierr = OptionsGetString(PETSC_NULL, "-mat_ordering_type", orderingName, 80, &flag); CHKERRQ(ierr);
  if (flag) {
    int  *row, *col;
    int  *r, *c;
    IS   isrow, iscol;

    sprintf(msgBuf_dh, "using mat_ordering_type = %s ", orderingName);
    SET_INFO(msgBuf_dh);

    ierr = MatGetOrdering(A,orderingName,&isrow,&iscol);   CHKERRQ(ierr);
    ierr = ISGetIndices(iscol,&col); CHKERRQ(ierr);
    ierr = ISGetIndices(isrow,&row); CHKERRQ(ierr);
    r = ctx->n2o_row = (int*)MALLOC_DH(n*sizeof(int)); CHECK_V_ERROR;
    c = ctx->n2o_col = (int*)MALLOC_DH(n*sizeof(int)); CHECK_ERROR(errFlag_dh);
    memcpy(c, col, n*sizeof(int));
    memcpy(r, row, n*sizeof(int));
    ierr = ISRestoreIndices(isrow, &row); CHKERRQ(ierr); 
    ierr = ISRestoreIndices(iscol, &col); CHKERRQ(ierr); 
  }

#endif

#undef __FUNC__
#define __FUNC__ "buildEuclidFromPetscMat"
void buildEuclidFromPetscMat(Mat Ain, Mat_dh *Aout)
{
  START_FUNC_DH
  Mat_dh A;

  Mat_dhCreate(&A); CHECK_V_ERROR;
  *Aout = A;
  extractMat(Ain, &(A->n), &(A->m), &(A->beg_row),
        &(A->rp), &(A->cval), &(A->aval), NULL, NULL); CHECK_V_ERROR;
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "extractMat"
void extractMat(Mat Ain, int *globalRows, int *localRows, int *beg_row,
                int **rpOUT, int **cvalOUT, double **avalOUT,
                int *n2o_row, int *n2o_col)

{
  START_FUNC_DH
  int tmp, i, j, n, m, nz, nz2, ierr, *rp, *cval;
  int ncols, *cols;
  int firstRow, lastRow, gRows, gCols;
  Scalar *vals;
  double *aval;
  MatInfo info;
  bool permFlag = false;
  bool isSEQ;

  isSEQ = (np_dh == 1) ? true : false;

  if (n2o_row!= NULL) permFlag = true;

  ierr = MatGetOwnershipRange(Ain, &firstRow, &lastRow);
  ierr = MatGetSize(Ain, &gRows, &gCols);
  ierr = MatGetInfo(Ain, MAT_LOCAL, &info); CHECK_PV_ERROR(ierr, "called MatGetInfo");
  nz = (int)info.nz_used;

  *globalRows = gRows;
  *localRows = m = lastRow - firstRow;
  *beg_row = firstRow;

  /* hack, since MatGetRow won't work for factored matrices */
  if (isSEQ) {
    tmp = Ain->factor;  
    Ain->factor = 0;
  }

  rp   = *rpOUT   = (int*)MALLOC_DH((m+1)*sizeof(int)); CHECK_V_ERROR;
  cval = *cvalOUT = (int*)MALLOC_DH(nz*sizeof(int));    CHECK_V_ERROR;
  aval = *avalOUT = (double*)MALLOC_DH(nz*sizeof(double)); CHECK_V_ERROR;
  rp[0] = 0;
  nz2 = 0;

  /* permuted matrix */
  if (permFlag && n2o_row != NULL && n2o_col != NULL) {

    if (! isSEQ) {
      SET_V_ERROR("permutation not implemented for parallel");
    } else {

      /* first, must invert the column permutation */
      int *c = (int*)MALLOC_DH(m*sizeof(int)); CHECK_V_ERROR;
      for (i=0; i<m; ++i) c[n2o_col[i]] = i;
  
      for (i=0; i<m; ++i) {
        int row = n2o_row[i]; 
        ierr = MatGetRow(Ain, row, &ncols, &cols, &vals); CHECK_PV_ERROR(ierr, "called MatGetRow");
        memcpy(cval+nz2, cols, ncols*sizeof(int));
        for (j=0; j<ncols; ++j) *(aval+nz2+j) = vals[j];
        ierr = MatRestoreRow(Ain, i, &ncols, &cols, &vals); CHECK_PV_ERROR(ierr,"called MatRestoreRow");

        /* permute column indices */
        for (j=rp[i]; j<rp[i]+ncols; ++j) {
          cval[j] = c[cval[j]];
        }
        nz2 += ncols;
        rp[i+1] = nz2;
      }
      FREE_DH(c); CHECK_V_ERROR(errFlag_dh);
    }
  }

  /* unpermuted matrix */
  else {
    for (i=0; i<m; ++i) {
      int row = i +  firstRow;
      ierr = MatGetRow(Ain, row, &ncols, &cols, &vals); CHECK_PV_ERROR(ierr, "called MatGetRow");
      memcpy(cval+nz2, cols, ncols*sizeof(int));
      for (j=0; j<ncols; ++j) *(aval+nz2+j) = vals[j];
      ierr = MatRestoreRow(Ain, i, &ncols, &cols, &vals); CHECK_PV_ERROR(ierr, "called MatRestoreRow");
      nz2 += ncols;
      rp[i+1] = nz2;
    }
  }

  /* end of hack */
  if (isSEQ) Ain->factor = tmp; 

  END_FUNC_DH
}



#undef __FUNC__
#define __FUNC__ "buildPetscMat"
void buildPetscMat(int m, int n, int beg_row, int* rp, int* cval, 
                                                double* aval, Mat *Aout)
{
  START_FUNC_DH

  /* uni-processor case: build a MATSEQAIJ object */
  if (np_dh == 1) {
    int i, count, ierr;
    int *nnz = (int*)MALLOC_DH(n*sizeof(int)); CHECK_V_ERROR;
    for (i=0; i<m; ++i) nnz[i] = rp[i+1]-rp[i];

    ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD, m, n, PETSC_DEFAULT, nnz, Aout);  CHECK_PV_ERROR(ierr, "called MatCreateSeqAIJ");

    for (i=0; i<n; ++i) {
      count = rp[i+1]-rp[i];
      if (count) {
        ierr = MatSetValues(*Aout, 1, &i, rp[i+1]-rp[i], cval+rp[i], 
                                      aval+rp[i], INSERT_VALUES); CHECK_PV_ERROR(ierr, "called MatSetValues");
      }
    }


    ierr = MatAssemblyBegin(*Aout, MAT_FINAL_ASSEMBLY); CHECK_PV_ERROR(ierr, "called MatAssemblyBegin");
    ierr = MatAssemblyEnd(*Aout, MAT_FINAL_ASSEMBLY);   CHECK_PV_ERROR(ierr, "called MatAssemblyBegin");
    FREE_DH(nnz);  CHECK_V_ERROR;
  } 

  /* multi-processor case: build a MATMPIAIJ object */
  else {
    int ierr, d_nz = 0, o_nz = 0, i, j;
    int end_row = beg_row + m;

    /* determine number of nonzeros in diagonal and off-diagonal portions
       of local submatrix
     */
    for (i=0; i<m; ++i) {
      for (j=rp[i]; j<rp[i+1]; ++i) {
        int col = cval[j];
        if (col < beg_row || col >= end_row) { ++o_nz; }
        else                                 { ++d_nz; }
      }
    }
    o_nz = (o_nz % m) ? (o_nz/m)+1 : o_nz/m;
    d_nz = (d_nz % m) ? (d_nz/m)+1 : d_nz/m;

    ierr = MatCreateMPIAIJ(PETSC_COMM_WORLD, m, m, n, n, 
              d_nz, PETSC_NULL, o_nz, PETSC_NULL, Aout); CHECK_PV_ERROR(ierr, "called MatCreateMPIAIJ");

    for (i=0; i<m; ++i) {
      int row = i+beg_row;
      int count = rp[i+1]-rp[i];
      if (count) {
        ierr = MatSetValues(*Aout, 1, &row, rp[i+1]-rp[i], cval+rp[i], 
                                    aval+rp[i], INSERT_VALUES); CHECK_PV_ERROR(ierr, "called MatSetValues");
      }
    }
    ierr = MatAssemblyBegin(*Aout, MAT_FINAL_ASSEMBLY); CHECK_PV_ERROR(ierr, "called MatAssemblyBegin");
    ierr = MatAssemblyEnd(*Aout, MAT_FINAL_ASSEMBLY);   CHECK_PV_ERROR(ierr, "called MatAssemblyEnd");
  }
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "buildRandVec"
int buildRandVec(int n, int m, int beg_row, Vec *Xout)
{
  START_FUNC_DH 
  int i, ierr;
  double *guess, max=0.0;
  int *idx;

  guess = (double*)MALLOC_DH(m*sizeof(double)); CHECK_ERROR(errFlag_dh);
  idx = (int*)MALLOC_DH(m*sizeof(int)); CHECK_ERROR(errFlag_dh);

  if (np_dh == 1) {
    ierr = VecCreateMPI(PETSC_COMM_WORLD,m,n,Xout); CHECK_P_ERROR(ierr, "called VecCreateMPI");
  } else {
    ierr = VecCreateSeq(PETSC_COMM_WORLD,m,Xout); CHECK_P_ERROR(ierr, "called VecCreateSeq");
  }

  for (i=0; i<m; ++i) {
    idx[i] = i+beg_row;; 
    guess[i] = random(); 
  }

  /* find largest value in vector, and scale vector, 
   * so all values are in [0.0,1.0]
  */
  for (i=0; i<m; ++i) max = (guess[i] > max) ? guess[i] : max; 
  for (i=0; i<m; ++i) guess[i] = guess[i]/max; 

  ierr = VecSetValues(*Xout,m,idx,guess,INSERT_VALUES); CHECK_P_ERROR(ierr, "called VecSetValues");
  FREE_DH(idx); CHECK_ERROR(errFlag_dh);
  FREE_DH(guess); CHECK_ERROR(errFlag_dh);
  END_FUNC_VAL(0)
}

#endif /* #ifdef PETSC_MODE */
