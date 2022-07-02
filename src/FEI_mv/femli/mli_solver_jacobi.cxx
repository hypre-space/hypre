/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "mli_utils.h"
#include "mli_solver_jacobi.h"

/******************************************************************************
 * constructor
 *---------------------------------------------------------------------------*/

MLI_Solver_Jacobi::MLI_Solver_Jacobi(char *name) : MLI_Solver(name)
{
   Amat_             = NULL;
   nSweeps_          = 1;
   relaxWeights_     = new double[1];
   relaxWeights_[0]  = 0.0;
   zeroInitialGuess_ = 0;
   diagonal_         = NULL;
   auxVec_           = NULL;
   auxVec2_          = NULL;
   auxVec3_          = NULL;
   maxEigen_         = 0.0;
   numFpts_          = 0;
   FptList_          = NULL;
   ownAmat_          = 0;
   modifiedD_        = 0;
}

/******************************************************************************
 * destructor
 *---------------------------------------------------------------------------*/

MLI_Solver_Jacobi::~MLI_Solver_Jacobi()
{
   if ( relaxWeights_ != NULL ) delete [] relaxWeights_;
   if ( diagonal_     != NULL ) delete [] diagonal_;
   if ( auxVec_       != NULL ) delete auxVec_;
   if ( auxVec2_      != NULL ) delete auxVec2_;
   if ( auxVec3_      != NULL ) delete auxVec3_;
   if ( FptList_      != NULL ) delete FptList_;
   if ( ownAmat_ == 1 ) delete Amat_;
}

/******************************************************************************
 * set up the smoother
 *---------------------------------------------------------------------------*/

int MLI_Solver_Jacobi::setup(MLI_Matrix *Amat)
{
   int                i, globalNRows, *partition, *ADiagI, *ADiagJ;
   int                j, localNRows, status;
   double             *ADiagA, *ritzValues;
   char               *paramString;
   MPI_Comm           comm;
   hypre_ParCSRMatrix *A;
   hypre_CSRMatrix    *ADiag;
   hypre_ParVector    *hypreVec;
   MLI_Function       *funcPtr;

   /*-----------------------------------------------------------------
    * fetch machine and matrix parameters
    *-----------------------------------------------------------------*/

   Amat_       = Amat;
   A           = (hypre_ParCSRMatrix *) Amat_->getMatrix();
   comm        = hypre_ParCSRMatrixComm(A);
   ADiag       = hypre_ParCSRMatrixDiag(A);
   ADiagI      = hypre_CSRMatrixI(ADiag);
   ADiagJ      = hypre_CSRMatrixJ(ADiag);
   ADiagA      = hypre_CSRMatrixData(ADiag);
   localNRows  = hypre_CSRMatrixNumRows(ADiag);
   globalNRows = hypre_ParCSRMatrixGlobalNumRows( A );

   /*-----------------------------------------------------------------
    * extract and store matrix diagonal
    *-----------------------------------------------------------------*/

   if ( localNRows > 0 ) diagonal_ = new double[localNRows];
   for ( i = 0; i < localNRows; i++ )
   {
      diagonal_[i] = 0.0;
      for ( j = ADiagI[i]; j < ADiagI[i+1]; j++ )
      {
         if ( ADiagJ[j] == i && ADiagA[j] != 0.0 ) 
         {
            diagonal_[i] = ADiagA[j];
            break;
         }
      }
      if ((modifiedD_ & 1) == 1)
      {
         if (diagonal_[i] > 0)
         {
            for ( j = ADiagI[i]; j < ADiagI[i+1]; j++ )
            {
               if ( ADiagJ[j] != i && ADiagA[j] > 0.0 ) 
                  diagonal_[i] += ADiagA[j];
            }
         }
         else
         {
            for ( j = ADiagI[i]; j < ADiagI[i+1]; j++ )
            {
               if ( ADiagJ[j] != i && ADiagA[j] < 0.0 ) 
                  diagonal_[i] += ADiagA[j];
            }
         }
      }
      diagonal_[i] = 1.0 / diagonal_[i];
   }

   /*-----------------------------------------------------------------
    * create temporary vector
    *-----------------------------------------------------------------*/

   funcPtr = hypre_TAlloc( MLI_Function , 1, HYPRE_MEMORY_HOST);
   MLI_Utils_HypreParVectorGetDestroyFunc(funcPtr);
   paramString = new char[20];
   strcpy( paramString, "HYPRE_ParVector" );

   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) A, &partition);
   hypreVec = hypre_ParVectorCreate(comm, globalNRows, partition);
   hypre_ParVectorInitialize(hypreVec);
   auxVec_ = new MLI_Vector(hypreVec, paramString, funcPtr);

   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) A, &partition);
   hypreVec = hypre_ParVectorCreate(comm, globalNRows, partition);
   hypre_ParVectorInitialize(hypreVec);
   auxVec2_ = new MLI_Vector(hypreVec, paramString, funcPtr);

   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) A, &partition);
   hypreVec = hypre_ParVectorCreate(comm, globalNRows, partition);
   hypre_ParVectorInitialize(hypreVec);
   auxVec3_ = new MLI_Vector(hypreVec, paramString, funcPtr);

   delete [] paramString;

   free( funcPtr );

   /*-----------------------------------------------------------------
    * compute spectral radius of A
    *-----------------------------------------------------------------*/

   if (maxEigen_ == 0.0 && (relaxWeights_ == NULL || relaxWeights_[0] == 0.0))
   {
      ritzValues = new double[2];
      status = MLI_Utils_ComputeExtremeRitzValues(A, ritzValues, 1);
      if ( status != 0 ) MLI_Utils_ComputeMatrixMaxNorm(A, ritzValues, 1);
      maxEigen_ = ritzValues[0]; 
      delete [] ritzValues;
   }
   if ( relaxWeights_ == NULL ) relaxWeights_ = new double[nSweeps_];
   if (maxEigen_ != 0.0)
   {
      for (i = 0; i < nSweeps_; i++) relaxWeights_[i] = 1.0 / maxEigen_;
   }
   return 0;
}

/******************************************************************************
 * apply function
 *---------------------------------------------------------------------------*/

int MLI_Solver_Jacobi::solve(MLI_Vector *fIn, MLI_Vector *uIn)
{
   int                i, j, is, localNRows, *ADiagI, *ADiagJ, index;
   double             *rData, *uData, weight, *f2Data, *u2Data, *fData;
   double             *ADiagA, dtemp, coeff;
   hypre_ParCSRMatrix *A;
   hypre_CSRMatrix    *ADiag;
   hypre_ParVector    *f, *u, *r, *f2, *u2;

   /*-----------------------------------------------------------------
    * fetch machine and matrix parameters
    *-----------------------------------------------------------------*/

   A          = (hypre_ParCSRMatrix *) Amat_->getMatrix();
   ADiag      = hypre_ParCSRMatrixDiag(A);
   localNRows = hypre_CSRMatrixNumRows(ADiag);
   f          = (hypre_ParVector *) fIn->getVector();
   u          = (hypre_ParVector *) uIn->getVector();
   r          = (hypre_ParVector *) auxVec_->getVector();
   rData      = hypre_VectorData(hypre_ParVectorLocalVector(r));
   uData      = hypre_VectorData(hypre_ParVectorLocalVector(u));
   ADiagI     = hypre_CSRMatrixI(ADiag);
   ADiagJ     = hypre_CSRMatrixJ(ADiag);
   ADiagA     = hypre_CSRMatrixData(ADiag);
   
   /*-----------------------------------------------------------------
    * loop 
    *-----------------------------------------------------------------*/

   if (numFpts_ == 0)
   {
      for ( is = 0; is < nSweeps_; is++ )
      {
         weight = relaxWeights_[is];
         hypre_ParVectorCopy(f, r); 
         if ( zeroInitialGuess_ == 0 )
         {
            if ((modifiedD_ & 2) == 0)
               hypre_ParCSRMatrixMatvec(-1.0, A, u, 1.0, r);
            else
            {
               for ( i = 0; i < localNRows; i++ ) 
               {
                  dtemp = rData[i];
                  for ( j = ADiagI[i]; j < ADiagI[i+1]; j++ ) 
                  {
                     index = ADiagJ[j];
                     coeff = ADiagA[j];
                     if (coeff*diagonal_[i] < 0.0)
                        dtemp -= coeff * uData[index];
                     else
                        dtemp -= coeff * uData[i];
                  }
                  rData[i] = dtemp;
               }
            }
         }

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
         for ( i = 0; i < localNRows; i++ ) 
            uData[i] += weight * rData[i] * diagonal_[i];

         zeroInitialGuess_ = 0;
      }
   }
   else
   {
      if (numFpts_ != localNRows)
      {
         printf("MLI_Solver_Jacobi::solve ERROR : length mismatch.\n");
         exit(1);
      }
      f2 = (hypre_ParVector *) auxVec2_->getVector();
      u2 = (hypre_ParVector *) auxVec3_->getVector();
      fData  = hypre_VectorData(hypre_ParVectorLocalVector(f));
      f2Data = hypre_VectorData(hypre_ParVectorLocalVector(f2));
      u2Data = hypre_VectorData(hypre_ParVectorLocalVector(u2));
      for (i = 0; i < numFpts_; i++) f2Data[i] = fData[FptList_[i]]; 
      for (i = 0; i < numFpts_; i++) u2Data[i] = uData[FptList_[i]]; 

      for ( is = 0; is < nSweeps_; is++ )
      {
         weight = relaxWeights_[is];
         hypre_ParVectorCopy(f2, r); 
         if ( zeroInitialGuess_ == 0 )
            hypre_ParCSRMatrixMatvec(-1.0, A, u2, 1.0, r);
 
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
         for ( i = 0; i < localNRows; i++ ) 
            u2Data[i] += weight * rData[i] * diagonal_[i];

         zeroInitialGuess_ = 0;
      }
      for (i = 0; i < numFpts_; i++) uData[FptList_[i]] = u2Data[i]; 
   }
   return 0;
}

/******************************************************************************
 * set Jacobi parameters
 *---------------------------------------------------------------------------*/

int MLI_Solver_Jacobi::setParams( char *paramString, int argc, char **argv )
{
   int    i, *fList;
   double *weights=NULL;

   if ( !strcmp(paramString, "numSweeps") )
   {
      if ( argc != 1 ) 
      {
         printf("MLI_Solver_Jacobi::setParams ERROR : needs 1 arg.\n");
         return 1;
      }
      nSweeps_ = *(int*) argv[0];
      if ( nSweeps_ < 1 ) nSweeps_ = 1;
      if ( relaxWeights_ != NULL ) delete [] relaxWeights_;
      relaxWeights_ = NULL;
      return 0;
   }
   else if ( !strcmp(paramString, "setMaxEigen") )
   {
      if ( argc != 1 ) 
      {
         printf("MLI_Solver_Jacobi::setParams ERROR : needs 1 arg.\n");
         return 1;
      }
      maxEigen_ = *(double*) argv[0];
      return 0;
   }
   else if ( !strcmp(paramString, "relaxWeight") )
   {
      if ( argc != 2 && argc != 1 ) 
      {
         printf("MLI_Solver_Jacobi::setParams ERROR : needs 1 or 2 args.\n");
         return 1;
      }
      if ( argc >= 1 ) nSweeps_ = *(int*)  argv[0];
      if ( argc == 2 ) weights = (double*) argv[1];
      if ( nSweeps_ < 1 ) nSweeps_ = 1;
      if ( relaxWeights_ != NULL ) delete [] relaxWeights_;
      relaxWeights_ = NULL;
      if ( weights != NULL )
      {
         relaxWeights_ = new double[nSweeps_];
         for ( i = 0; i < nSweeps_; i++ ) relaxWeights_[i] = weights[i];
      }
   }
   else if ( !strcmp(paramString, "zeroInitialGuess") )
   {
      zeroInitialGuess_ = 1;
      return 0;
   }
   else if ( !strcmp(paramString, "setModifiedDiag") )
   {
      modifiedD_ |= 1;
      return 0;
   }
   else if ( !strcmp(paramString, "useModifiedDiag") )
   {
      modifiedD_ |= 2;
      return 0;
   }
   else if ( !strcmp(paramString, "setFptList") )
   {
      if ( argc != 2 ) 
      {
         printf("MLI_Solver_Jacobi::setParams ERROR : needs 2 args.\n");
         return 1;
      }
      numFpts_ = *(int*)  argv[0];
      fList = (int*) argv[1];
      if ( FptList_ != NULL ) delete [] FptList_;
      FptList_ = NULL;
      if (numFpts_ <= 0) return 0;
      FptList_ = new int[numFpts_];;
      for ( i = 0; i < numFpts_; i++ ) FptList_[i] = fList[i];
      return 0;
   }
   else if ( !strcmp(paramString, "ownAmat") )
   {
      ownAmat_ = 1;
      return 0;
   }
#if 0
   else
   {   
      printf("MLI_Solver_Jacobi::setParams - parameter not recognized.\n");
      printf("                Params = %s\n", paramString);
      return 1;
   }
#endif
   return 0;
}

/******************************************************************************
 * set Jacobi parameters
 *---------------------------------------------------------------------------*/

int MLI_Solver_Jacobi::setParams( int ntimes, double *weights )
{
   int i, nsweeps;

   nsweeps = ntimes;
   if ( ntimes <= 0 )
   {
      printf("MLI_Solver_Jacobi::setParams WARNING : nSweeps set to 1.\n");
      nsweeps = 1;
   }
   nSweeps_ = nsweeps;
   if ( relaxWeights_ != NULL ) delete [] relaxWeights_;
   relaxWeights_ = new double[nsweeps];
   if ( weights == NULL )
   {
      printf("MLI_Solver_Jacobi::setParams - relaxWeights set to 0.0.\n");
      for ( i = 0; i < nsweeps; i++ ) relaxWeights_[i] = 0.0;
   }
   else
   {
      for ( i = 0; i < nsweeps; i++ ) 
      {
         if (weights[i] >= 0. && weights[i] <= 2.) 
            relaxWeights_[i] = weights[i];
         else 
         {
            printf("MLI_Solver_Jacobi::setParams - weights set to 0.0.\n");
            relaxWeights_[i] = 0.0;
         }
      }
   }
   return 0;
}

/******************************************************************************
 * get Jacobi parameters
 *---------------------------------------------------------------------------*/

int MLI_Solver_Jacobi::getParams( char *paramString, int *argc, char **argv )
{
   double *ddata, *ritzValues;

   if ( !strcmp(paramString, "getMaxEigen") )
   {
      if ( maxEigen_ == 0.0 )
      {
         ritzValues = new double[2];
         MLI_Utils_ComputeExtremeRitzValues((hypre_ParCSRMatrix *) 
                                 Amat_->getMatrix(), ritzValues, 1);
         maxEigen_ = ritzValues[0]; 
         delete [] ritzValues;
      }
      ddata = (double *) argv[0];
      ddata[0] = maxEigen_;
      *argc = 1;
      return 0;
   }
   else return -1;
}

