/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <stdio.h>

#include "base/mli_defs.h"
#include "util/mli_utils.h"
#include "solver/mli_solver_jacobi.h"

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
   maxEigen_         = 0.0;
}

/******************************************************************************
 * destructor
 *---------------------------------------------------------------------------*/

MLI_Solver_Jacobi::~MLI_Solver_Jacobi()
{
   if ( relaxWeights_ != NULL ) delete [] relaxWeights_;
   if ( diagonal_     != NULL ) delete [] diagonal_;
   if ( auxVec_       != NULL ) delete auxVec_;
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
            diagonal_[i] = 1.0 / ADiagA[j];
            break;
         }
      }
   }

   /*-----------------------------------------------------------------
    * create temporary vector
    *-----------------------------------------------------------------*/

   funcPtr = (MLI_Function *) malloc( sizeof( MLI_Function ) );
   MLI_Utils_HypreParVectorGetDestroyFunc(funcPtr);
   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) A, &partition);
   hypreVec = hypre_ParVectorCreate(comm, globalNRows, partition);
   hypre_ParVectorInitialize(hypreVec);
   paramString = new char[20];
   strcpy( paramString, "HYPRE_ParVector" );
   auxVec_ = new MLI_Vector(hypreVec, paramString, funcPtr);
   delete [] paramString;
   free( funcPtr );

   /*-----------------------------------------------------------------
    * compute spectral radius of A
    *-----------------------------------------------------------------*/

   if ( maxEigen_ == 0.0 && relaxWeights_[0] == 0.0 )
   {
      ritzValues = new double[2];
      status = MLI_Utils_ComputeExtremeRitzValues(A, ritzValues, 1);
      if ( status != 0 ) MLI_Utils_ComputeMatrixMaxNorm(A, ritzValues, 1);
      maxEigen_ = ritzValues[0]; 
      delete [] ritzValues;
   }
   for (i = 0; i < nSweeps_; i++) relaxWeights_[i] = 4.0 / (3.0*maxEigen_);
   return 0;
}

/******************************************************************************
 * apply function
 *---------------------------------------------------------------------------*/

int MLI_Solver_Jacobi::solve(MLI_Vector *fIn, MLI_Vector *uIn)
{
   int                i, is, localNRows;
   double             *rData, *uData, weight;
   hypre_ParCSRMatrix *A;
   hypre_CSRMatrix    *ADiag;
   hypre_ParVector    *f, *u, *r;

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
   
   /*-----------------------------------------------------------------
    * loop 
    *-----------------------------------------------------------------*/

   for ( is = 0; is < nSweeps_; is++ )
   {
      weight = relaxWeights_[is];

      hypre_ParVectorCopy(f, r); 
      if ( zeroInitialGuess_ == 0 )
         hypre_ParCSRMatrixMatvec(-1.0, A, u, 1.0, r);

#define HYPRE_SMP_PRIVATE i
#include "utilities/hypre_smp_forloop.h"
      for ( i = 0; i < localNRows; i++ ) 
         uData[i] += weight * rData[i] * diagonal_[i];

      zeroInitialGuess_ = 0;
   }
   return 0;
}

/******************************************************************************
 * set Jacobi parameters
 *---------------------------------------------------------------------------*/

int MLI_Solver_Jacobi::setParams( char *paramString, int argc, char **argv )
{
   int    i;
   double *weights;

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
   else
   {   
      printf("MLI_Solver_Jacobi::setParams - parameter not recognized.\n");
      printf("                Params = %s\n", paramString);
      return 1;
   }
   return 0;
}

/******************************************************************************
 * set Jacobi parameters
 *---------------------------------------------------------------------------*/

int MLI_Solver_Jacobi::setParams( int ntimes, double *weights )
{
   int i, nsweeps;

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

