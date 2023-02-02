/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include <string.h>
#include <stdio.h>
#include "_hypre_utilities.h"
#include "HYPRE.h"
#include "_hypre_parcsr_mv.h"
#include "HYPRE_IJ_mv.h"
#include "mli_matrix.h"
#include "mli_utils.h"

/***************************************************************************
 * constructor function for the MLI_Matrix
 *--------------------------------------------------------------------------*/

MLI_Matrix::MLI_Matrix(void *inMatrix,char *inName, MLI_Function *func)
{
#ifdef MLI_DEBUG_DETAILED
   printf("MLI_Matrix::MLI_Matrix : %s\n", inName);
#endif
   matrix_ = inMatrix;
   if (func != NULL) destroyFunc_ = (int (*)(void *)) func->func_;
   else              destroyFunc_ = NULL;
   strncpy(name_, inName, 100);
   gNRows_  = -1;
   maxNNZ_  = -1;
   minNNZ_  = -1;
   totNNZ_  = -1;
   dtotNNZ_ = 0.0;
   maxVal_  = 0.0;
   minVal_  = 0.0;
   subMatrixLength_ = 0;
   subMatrixEqnList_ = NULL;
}

/***************************************************************************
 * destructor function for the MLI_Matrix
 *--------------------------------------------------------------------------*/

MLI_Matrix::~MLI_Matrix()
{
#ifdef MLI_DEBUG
   printf("MLI_Matrix::~MLI_Matrix : %s\n", name_);
#endif
   if (matrix_ != NULL && destroyFunc_ != NULL) destroyFunc_(matrix_);
   matrix_      = NULL;
   destroyFunc_ = NULL;
   if (subMatrixEqnList_ != NULL) delete [] subMatrixEqnList_;
   subMatrixEqnList_ = NULL;
}

/***************************************************************************
 * apply function ( vec3 = alpha * Matrix * vec1 + beta * vec2)
 *--------------------------------------------------------------------------*/

int MLI_Matrix::apply(double alpha, MLI_Vector *vec1, double beta,
                      MLI_Vector *vec2, MLI_Vector *vec3)
{
   int                irow, status, ncolsA, nrowsV, mypid, index;
   int                startRow, endRow, *partitioning, ierr;
   double             *V1_data, *V2_data, *V3_data;
   double             *V1S_data, *V2S_data, *V3S_data;
   char               *vname;
   hypre_ParVector    *hypreV1, *hypreV2, *hypreV3;
   hypre_ParVector    *hypreV1S, *hypreV2S, *hypreV3S;
   hypre_ParCSRMatrix *hypreA = (hypre_ParCSRMatrix *) matrix_;
   HYPRE_IJVector     IJV1, IJV2, IJV3;
   MPI_Comm           comm;

#ifdef MLI_DEBUG_DETAILED
   printf("MLI_Matrix::apply : %s\n", name_);
#endif

   /* -----------------------------------------------------------------------
    * error checking
    * ----------------------------------------------------------------------*/

   if (!strcmp(name_, "HYPRE_ParCSR") && !strcmp(name_, "HYPRE_ParCSRT"))
   {
      printf("MLI_Matrix::apply ERROR : matrix not HYPRE_ParCSR.\n");
      exit(1);
   }
   vname = vec1->getName();
   if (strcmp(vname, "HYPRE_ParVector"))
   {
      printf("MLI_Matrix::apply ERROR : vec1 not HYPRE_ParVector.\n");
      printf("MLI_Matrix::vec1 of type = %s\n", vname);
      exit(1);
   }
   if (vec2 != NULL)
   {
      vname = vec2->getName();
      if (strcmp(vname, "HYPRE_ParVector"))
      {
         printf("MLI_Matrix::apply ERROR : vec2 not HYPRE_ParVector.\n");
         exit(1);
      }
   }
   vname = vec3->getName();
   if (strcmp(vname, "HYPRE_ParVector"))
   {
      printf("MLI_Matrix::apply ERROR : vec3 not HYPRE_ParVector.\n");
      exit(1);
   }

   /* -----------------------------------------------------------------------
    * fetch matrix and vectors; and then operate
    * ----------------------------------------------------------------------*/

   hypreA  = (hypre_ParCSRMatrix *) matrix_;
   hypreV1 = (hypre_ParVector *) vec1->getVector();
   nrowsV = hypre_VectorSize(hypre_ParVectorLocalVector(hypreV1));
   if (!strcmp(name_, "HYPRE_ParCSR"))
      ncolsA = hypre_ParCSRMatrixNumCols(hypreA);
   else
      ncolsA = hypre_ParCSRMatrixNumRows(hypreA);
   if (subMatrixLength_ == 0 || ncolsA == nrowsV)
   {
      hypreV1 = (hypre_ParVector *) vec1->getVector();
      hypreV3 = (hypre_ParVector *) vec3->getVector();
      if (vec2 != NULL)
      {
         hypreV2 = (hypre_ParVector *) vec2->getVector();
         status  = hypre_ParVectorCopy( hypreV2, hypreV3 );
      }
      else status = hypre_ParVectorSetConstantValues( hypreV3, 0.0e0 );

      if (!strcmp(name_, "HYPRE_ParCSR"))
      {
         status += hypre_ParCSRMatrixMatvec(alpha,hypreA,hypreV1,beta,hypreV3);
      }
      else
      {
         status += hypre_ParCSRMatrixMatvecT(alpha,hypreA,hypreV1,beta,hypreV3);
      }
      return status;
   }
   else if (subMatrixLength_ != 0 && ncolsA != nrowsV)
   {
      comm = hypre_ParCSRMatrixComm(hypreA);
      MPI_Comm_rank(comm, &mypid);
      if (!strcmp(name_, "HYPRE_ParCSR"))
         HYPRE_ParCSRMatrixGetColPartitioning((HYPRE_ParCSRMatrix)hypreA,
                                           &partitioning);
      else
         HYPRE_ParCSRMatrixGetColPartitioning((HYPRE_ParCSRMatrix)hypreA,
                                              &partitioning);
      startRow = partitioning[mypid];
      endRow   = partitioning[mypid+1];
      free(partitioning);
      ierr  = HYPRE_IJVectorCreate(comm, startRow, endRow-1, &IJV1);
      ierr += HYPRE_IJVectorSetObjectType(IJV1, HYPRE_PARCSR);
      ierr += HYPRE_IJVectorInitialize(IJV1);
      ierr += HYPRE_IJVectorAssemble(IJV1);
      ierr += HYPRE_IJVectorGetObject(IJV1, (void **) &hypreV1S);
      ierr  = HYPRE_IJVectorCreate(comm, startRow, endRow-1, &IJV3);
      ierr += HYPRE_IJVectorSetObjectType(IJV3, HYPRE_PARCSR);
      ierr += HYPRE_IJVectorInitialize(IJV3);
      ierr += HYPRE_IJVectorAssemble(IJV3);
      ierr += HYPRE_IJVectorGetObject(IJV3, (void **) &hypreV3S);
      V1S_data = hypre_VectorData(hypre_ParVectorLocalVector(hypreV1S));
      V3S_data = hypre_VectorData(hypre_ParVectorLocalVector(hypreV3S));
      hypreV1 = (hypre_ParVector *) vec1->getVector();
      hypreV3 = (hypre_ParVector *) vec3->getVector();
      V1_data = hypre_VectorData(hypre_ParVectorLocalVector(hypreV1));
      V3_data = hypre_VectorData(hypre_ParVectorLocalVector(hypreV3));
      if (vec2 != NULL)
      {
         ierr  = HYPRE_IJVectorCreate(comm, startRow, endRow-1, &IJV2);
         ierr += HYPRE_IJVectorSetObjectType(IJV2, HYPRE_PARCSR);
         ierr += HYPRE_IJVectorInitialize(IJV2);
         ierr += HYPRE_IJVectorAssemble(IJV2);
         ierr += HYPRE_IJVectorGetObject(IJV2, (void **) &hypreV2S);
         hypreV2 = (hypre_ParVector *) vec2->getVector();
         V2_data = hypre_VectorData(hypre_ParVectorLocalVector(hypreV2));
         V2S_data = hypre_VectorData(hypre_ParVectorLocalVector(hypreV2S));
      }
      for (irow = 0; irow < subMatrixLength_; irow++)
      {
         index = subMatrixEqnList_[irow];
         V1S_data[irow] = V1_data[index];
         V3S_data[irow] = V3_data[index];
         if (vec2 != NULL) V2S_data[irow] = V2_data[index];
      }
      if (!strcmp(name_, "HYPRE_ParCSR"))
      {
         status = hypre_ParCSRMatrixMatvec(alpha,hypreA,hypreV1S,
                                            beta,hypreV3S);
      }
      else
      {
         status = hypre_ParCSRMatrixMatvecT(alpha,hypreA,hypreV1S,
                                             beta,hypreV3S);
      }
      for (irow = 0; irow < subMatrixLength_; irow++)
      {
         index = subMatrixEqnList_[irow];
         V3_data[index] = V3S_data[irow];
      }
      ierr += HYPRE_IJVectorDestroy(IJV1);
      ierr += HYPRE_IJVectorDestroy(IJV2);
      ierr += HYPRE_IJVectorDestroy(IJV3);
      return status;
   }
   return 0;
}

/******************************************************************************
 * create a vector from information of this matrix
 *---------------------------------------------------------------------------*/

MLI_Vector *MLI_Matrix::createVector()
{
   int                mypid, nprocs, startRow, endRow;
   int                ierr, *partitioning;
   char               paramString[100];
   MPI_Comm           comm;
   HYPRE_ParVector    newVec;
   hypre_ParCSRMatrix *hypreA;
   HYPRE_IJVector     IJvec;
   MLI_Vector         *mli_vec;
   MLI_Function       *funcPtr;

   if (strcmp(name_, "HYPRE_ParCSR"))
   {
      printf("MLI_Matrix::createVector ERROR - matrix has invalid type.\n");
      exit(1);
   }
   hypreA = (hypre_ParCSRMatrix *) matrix_;
   comm = hypre_ParCSRMatrixComm(hypreA);
   MPI_Comm_rank(comm, &mypid);
   MPI_Comm_size(comm, &nprocs);
   if (!strcmp(name_, "HYPRE_ParCSR"))
      HYPRE_ParCSRMatrixGetColPartitioning((HYPRE_ParCSRMatrix)hypreA,
                                            &partitioning);
   else
      HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix)hypreA,
                                            &partitioning);
   startRow = partitioning[mypid];
   endRow   = partitioning[mypid+1];
   free( partitioning );
   ierr  = HYPRE_IJVectorCreate(comm, startRow, endRow-1, &IJvec);
   ierr += HYPRE_IJVectorSetObjectType(IJvec, HYPRE_PARCSR);
   ierr += HYPRE_IJVectorInitialize(IJvec);
   ierr += HYPRE_IJVectorAssemble(IJvec);
   ierr += HYPRE_IJVectorGetObject(IJvec, (void **) &newVec);
   ierr += HYPRE_IJVectorSetObjectType(IJvec, -1);
   ierr += HYPRE_IJVectorDestroy(IJvec);
   hypre_assert( !ierr );
   HYPRE_ParVectorSetConstantValues(newVec, 0.0);
   sprintf(paramString, "HYPRE_ParVector");
   funcPtr = new MLI_Function();
   MLI_Utils_HypreParVectorGetDestroyFunc(funcPtr);
   mli_vec = new MLI_Vector((void*) newVec, paramString, funcPtr);
   delete funcPtr;
   return mli_vec;
}

/******************************************************************************
 * create a vector from information of this matrix
 *---------------------------------------------------------------------------*/

int MLI_Matrix::getMatrixInfo(char *paramString, int &intParams,
                              double &dbleParams)
{
   int      matInfo[4];
   double   valInfo[3];

   if (!strcmp(name_, "HYPRE_ParCSR") && !strcmp(name_, "HYPRE_ParCSRT"))
   {
      printf("MLI_Matrix::getInfo ERROR : matrix not HYPRE_ParCSR.\n");
      intParams  = -1;
      dbleParams = 0.0;
      return 1;
   }
   if (gNRows_ < 0)
   {
      MLI_Utils_HypreMatrixGetInfo(matrix_, matInfo, valInfo);
      gNRows_  = matInfo[0];
      maxNNZ_  = matInfo[1];
      minNNZ_  = matInfo[2];
      totNNZ_  = matInfo[3];
      maxVal_  = valInfo[0];
      minVal_  = valInfo[1];
      dtotNNZ_ = valInfo[2];
   }
   intParams  = 0;
   dbleParams = 0.0;
   if      (!strcmp(paramString, "nrows" ))  intParams  = gNRows_;
   else if (!strcmp(paramString, "maxnnz"))  intParams  = maxNNZ_;
   else if (!strcmp(paramString, "minnnz"))  intParams  = minNNZ_;
   else if (!strcmp(paramString, "totnnz"))  intParams  = totNNZ_;
   else if (!strcmp(paramString, "maxval"))  dbleParams = maxVal_;
   else if (!strcmp(paramString, "minval"))  dbleParams = minVal_;
   else if (!strcmp(paramString, "dtotnnz")) dbleParams = dtotNNZ_;
   return 0;
}

/******************************************************************************
 * load submatrix equation list
 *---------------------------------------------------------------------------*/

void MLI_Matrix::setSubMatrixEqnList(int length, int *list)
{
   if (length <= 0) return;
   if (subMatrixEqnList_ != NULL) delete [] subMatrixEqnList_;
   subMatrixLength_ = length;
   subMatrixEqnList_ = new int[length];
   for (int i = 0; i < subMatrixLength_; i++)
      subMatrixEqnList_[i] = list[i];
}

/******************************************************************************
 * get matrix
 *---------------------------------------------------------------------------*/

void *MLI_Matrix::getMatrix()
{
   return matrix_;
}

/******************************************************************************
 * take matrix (will be destroyed by the taker)
 *---------------------------------------------------------------------------*/

void *MLI_Matrix::takeMatrix()
{
   destroyFunc_ = NULL;
   return matrix_;
}

/******************************************************************************
 * get the name of this matrix
 *---------------------------------------------------------------------------*/

char *MLI_Matrix::getName()
{
   return name_;
}

/******************************************************************************
 * print a matrix
 *---------------------------------------------------------------------------*/

int MLI_Matrix::print(char *filename)
{
   if (!strcmp(name_, "HYPRE_ParCSR") && !strcmp(name_, "HYPRE_ParCSRT"))
   {
      printf("MLI_Matrix::print ERROR : matrix not HYPRE_ParCSR.\n");
      return 1;
   }
   MLI_Utils_HypreMatrixPrint((void *) matrix_, filename);
   return 0;
}

