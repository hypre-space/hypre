#include "getRow_dh.h"
#include "Mat_dh.h"
#include "Euclid_dh.h"

/*-------------------------------------------------------------------
 *  HYPRE
 *-------------------------------------------------------------------*/
#if defined(HYPRE_GET_ROW)

#undef __FUNC__
#define __FUNC__ "EuclidGetRow (HYPRE_GET_ROW)"
void EuclidGetRow(void *A, int row, int *len, int **ind, double **val) 
{
  START_FUNC_DH
  int ierr;
  HYPRE_ParCSRMatrix mat = (HYPRE_ParCSRMatrix) A;
  ierr = HYPRE_ParCSRMatrixGetRow(mat, row, len, ind, val); 
  if (ierr) {
    sprintf(msgBuf_dh, "HYPRE_ParCSRMatrixRestoreRow(row= %i) returned %i", row+1, ierr);
    SET_V_ERROR(msgBuf_dh);
  }
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "EuclidRestoreRow (HYPRE_GET_ROW)"
void EuclidRestoreRow(void *A, int row, int *len, int **ind, double **val) 
{
  START_FUNC_DH
  int ierr;
  HYPRE_ParCSRMatrix mat = (HYPRE_ParCSRMatrix) A;
  ierr = HYPRE_ParCSRMatrixRestoreRow(mat, row, len, ind, val); 
  if (ierr) {
    sprintf(msgBuf_dh, "HYPRE_ParCSRMatrixRestoreRow(row= %i) returned %i", row+1, ierr);
    SET_V_ERROR(msgBuf_dh);
  }
  END_FUNC_DH
}

/*-------------------------------------------------------------------
 *  PETSc
 *-------------------------------------------------------------------*/
#elif defined(PETSC_GET_ROW)

#undef __FUNC__
#define __FUNC__ "EuclidGetRow (PETSC_GET_ROW)"
void EuclidGetRow(void *A, int row, int *len, int **ind, double **val) 
{
  START_FUNC_DH
  SET_V_ERROR("not implemented");
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "EuclidRestoreRow (PETSC_GET_ROW)"
void EuclidRestoreRow(void *A, int row, int *len, int **ind, double **val) 
{
  START_FUNC_DH
  END_FUNC_DH
}


/*-------------------------------------------------------------------
 *  Euclid  
 *-------------------------------------------------------------------*/
#elif defined(EUCLID_GET_ROW)


#undef __FUNC__
#define __FUNC__ "EuclidGetRow (EUCLID_GET_ROW)"
void EuclidGetRow(void *A, int globalRow, int *len, int **ind, double **val) 
{
  START_FUNC_DH
  Mat_dh B = (Mat_dh)A;  
  int row = globalRow - B->beg_row;
  *len = B->rp[row+1] - B->rp[row];
  if (ind != NULL) *ind = B->cval + B->rp[row]; 
  if (val != NULL) *val = B->aval + B->rp[row]; 
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "EuclidRestoreRow (EUCLID_GET_ROW)"
void EuclidRestoreRow(void *A, int row, int *len, int **ind, double **val) 
{
  START_FUNC_DH
  END_FUNC_DH
}


/*-------------------------------------------------------------------
 *  Default
 *-------------------------------------------------------------------*/
#else

#undef __FUNC__
#define __FUNC__ "EuclidGetRow (ERROR)"
void EuclidGetRow(void *A, int row, int *len, int **ind, double **val) 
{
  START_FUNC_DH
  SET_ERROR(EUCLID_ERROR, "Oops; missing XXX_GET_ROW definition!");
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "EuclidRestoreRow (ERROR)"
void EuclidRestoreRow(void *A, int row, int *len, int **ind, double **val) 
{
  START_FUNC_DH
  SET_ERROR(EUCLID_ERROR, "Oops; missing XXX_GET_ROW definition!");
  END_FUNC_DH
}

#endif

/*-------------------------------------------------------------------
 *  end of GET_ROW definitions
 *-------------------------------------------------------------------*/

#undef __FUNC__
#define __FUNC__ "PrintMatUsingGetRow"
void PrintMatUsingGetRow(void* A, int beg_row, int m,
                          int *n2o_row, int *n2o_col, char *filename)
{
  START_FUNC_DH
  FILE *fp;
  int *o2n_col, pe, i, j, *cval, len;
  int newCol, newRow;
  double *aval;

  /* form inverse column permutation */
  if (n2o_col != NULL) {
    o2n_col = (int*)MALLOC_DH(m*sizeof(int)); CHECK_V_ERROR;
    for (i=0; i<m; ++i) o2n_col[n2o_col[i]] = i;
  }

  for (pe=0; pe<np_dh; ++pe) {

    MPI_Barrier(comm_dh);

    if (myid_dh == pe) {
      if (pe == 0) {
        fp=fopen(filename, "w");
      } else {
        fp=fopen(filename, "a");
      }
      if (fp == NULL) {
        sprintf(msgBuf_dh, "can't open %s for writing\n", filename);
        SET_V_ERROR(msgBuf_dh);
      }

      for (i=0; i<m; ++i) {

        if (n2o_row == NULL) {
          EuclidGetRow(A, i+beg_row, &len, &cval, &aval); CHECK_V_ERROR;
          for (j=0; j<len; ++j) {
            fprintf(fp, "%i %i %g\n", i+1, cval[j], aval[j]);
          }
          EuclidRestoreRow(A, i, &len, &cval, &aval); CHECK_V_ERROR;
        } else {
          newRow = n2o_row[i] + beg_row;
          EuclidGetRow(A, newRow, &len, &cval, &aval); CHECK_V_ERROR;
          for (j=0; j<len; ++j) {
            newCol = o2n_col[cval[j]-beg_row] + beg_row; 
            fprintf(fp, "%i %i %g\n", i+1, newCol, aval[j]);
          }
          EuclidRestoreRow(A, i, &len, &cval, &aval); CHECK_V_ERROR;
        }
      }
    }
  }
  fclose(fp);

  if (n2o_col != NULL) {
    FREE_DH(o2n_col); CHECK_V_ERROR;
  }
  END_FUNC_DH
}

/*------------------------------------------------------------------------
 *  functions for setting matrices
 *------------------------------------------------------------------------*/

#ifdef EUCLID_GET_ROW
#undef __FUNC__
#define __FUNC__ "Euclid_dhInputCSRMat"
void Euclid_dhInputCSRMat(Euclid_dh ctx, int globalRows,
                         int localRows, int beg_row,
                         int *rp, int *cval, double *aval)
{
  START_FUNC_DH
  Mat_dh A;

  Mat_dhCreate(&A); CHECK_V_ERROR;
  ctx->A = (void*) A;
  ctx->ownsAstruct = true;

  A->m = ctx->m = localRows;
  A->n = ctx->n = globalRows;
  A->beg_row = ctx->beg_row = beg_row;
  A->rp = rp;
  A->cval = cval;
  A->aval = aval;
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "Euclid_dhInputEuclidMat"
void Euclid_dhInputEuclidMat(Euclid_dh ctx, Mat_dh A)
{
  START_FUNC_DH
  ctx->m = A->m;
  ctx->n = A->n;
  ctx->beg_row = A->beg_row;
  ctx->A = (void*)A;
  END_FUNC_DH
}
#endif

#ifdef PETSC_GET_ROW
#undef __FUNC__
#define __FUNC__ "Euclid_dhInputPetscMat"
void Euclid_dhInputPetscMat(Euclid_dh ctx, Mat A)
{
  START_FUNC_DH
  int ierr, beg_row, end_row, m, n;

  ierr = MatGetLocalSize(A, &m, &n); CHECK_V_ERROR;
  ierr = MatGetOwnershipRange(A, &beg_row, &end_row); CHECK_V_ERROR;

  ctx->m = m;
  ctx->n = n;
  ctx->beg_row = beg_row;
  ctx->A = (void*)A;
  END_FUNC_DH
}
#endif


#ifdef HYPRE_MODE
#undef __FUNC__
#define __FUNC__ "Euclid_dhInputHypreMat"
void Euclid_dhInputHypreMat(Euclid_dh ctx, HYPRE_ParCSRMatrix A)
{
  START_FUNC_DH
  int M, N;
  int beg_row, end_row, junk;

  /* get dimension and ownership information */
  HYPRE_ParCSRMatrixGetDims(A, &M , &N);
  if (M != N) {
    sprintf(msgBuf_dh, "Global matrix is not square: M= %i, N= %i", M, N);
    SET_V_ERROR(msgBuf_dh);
  }
  HYPRE_ParCSRMatrixGetLocalRange(A, &beg_row, &end_row, &junk, &junk);

  ctx->m = end_row - beg_row + 1;
  ctx->n = M;
  ctx->beg_row = beg_row;
  ctx->A = (void*)A;

  END_FUNC_DH
}
#endif
