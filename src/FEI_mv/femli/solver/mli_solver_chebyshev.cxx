/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.8 $
 ***********************************************************************EHEADER*/





#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#ifdef WIN32
#define strcmp _stricmp
#endif

#include "solver/mli_solver_jacobi.h"
#include "solver/mli_solver_chebyshev.h"
#include "parcsr_mv/_hypre_parcsr_mv.h"

/******************************************************************************
 * constructor
 *---------------------------------------------------------------------------*/

MLI_Solver_Chebyshev::MLI_Solver_Chebyshev(char *name) : MLI_Solver(name)
{
   Amat_             = NULL;
   degree_           = 2;
   rVec_             = NULL;
   zVec_             = NULL;
   pVec_             = NULL;
   diagonal_         = NULL;
   maxEigen_         = 0.0;
   minEigen_         = 0.0;
   zeroInitialGuess_ = 0;
}

/******************************************************************************
 * destructor
 *---------------------------------------------------------------------------*/

MLI_Solver_Chebyshev::~MLI_Solver_Chebyshev()
{
   Amat_ = NULL;
   if ( rVec_     != NULL ) delete rVec_;
   if ( zVec_     != NULL ) delete zVec_;
   if ( pVec_     != NULL ) delete pVec_;
   if ( diagonal_ != NULL ) delete [] diagonal_;
}

/******************************************************************************
 * set up the smoother
 *---------------------------------------------------------------------------*/

int MLI_Solver_Chebyshev::setup(MLI_Matrix *mat)
{
   int                i, j, localNRows, *ADiagI, *ADiagJ;
   double             *ADiagA, *ritzValues, omega=3.0/3.0, scale;
   hypre_ParCSRMatrix *A;
   hypre_CSRMatrix    *ADiag;

   /*-----------------------------------------------------------------
    * fetch parameters
    *-----------------------------------------------------------------*/

   Amat_      = mat;
   A          = (hypre_ParCSRMatrix *) Amat_->getMatrix();
   ADiag      = hypre_ParCSRMatrixDiag(A);
   ADiagI     = hypre_CSRMatrixI(ADiag);
   ADiagJ     = hypre_CSRMatrixJ(ADiag);
   ADiagA     = hypre_CSRMatrixData(ADiag);
   localNRows = hypre_CSRMatrixNumRows(ADiag);

   /*-----------------------------------------------------------------
    * compute spectral radius of scaled Amat
    *-----------------------------------------------------------------*/

   if ( maxEigen_ == 0.0 )
   {
      ritzValues = new double[2];
      MLI_Utils_ComputeExtremeRitzValues( A, ritzValues, 1 );
      maxEigen_ = ritzValues[0];
      minEigen_ = ritzValues[1];
      delete [] ritzValues;
   }

   /*-----------------------------------------------------------------
    * extract and store matrix diagonal
    *-----------------------------------------------------------------*/

   scale = omega / maxEigen_;
   if ( localNRows > 0 ) diagonal_ = new double[localNRows];
   for ( i = 0; i < localNRows; i++ )
   {
      diagonal_[i] = 1.0;
      for ( j = ADiagI[i]; j < ADiagI[i+1]; j++ )
      {
         if ( ADiagJ[j] == i && ADiagA[j] != 0.0 ) 
         {
            diagonal_[i] = scale / ADiagA[j] ;
            break;
         }
      }
   }

   /*-----------------------------------------------------------------
    * allocate temporary vectors
    *-----------------------------------------------------------------*/

   if ( rVec_ != NULL ) delete rVec_;
   if ( zVec_ != NULL ) delete zVec_;
   if ( pVec_ != NULL ) delete pVec_;
   rVec_ = mat->createVector();
   zVec_ = mat->createVector();
   pVec_ = mat->createVector();
   return 0;
}

/******************************************************************************
 * apply function
 *---------------------------------------------------------------------------*/

int MLI_Solver_Chebyshev::solve(MLI_Vector *f_in, MLI_Vector *u_in)
{
   int                i, j, localNRows;
   double             *pData, *zData, alpha, beta, cValue, dValue;
   double             *rData, lambdaMax, lambdaMin, omega=2.0/3.0;
   hypre_ParCSRMatrix *A;
   hypre_CSRMatrix    *ADiag;
   hypre_ParVector    *r, *z, *p, *u, *f;

   /*-----------------------------------------------------------------
    * fetch machine and smoother parameters
    *-----------------------------------------------------------------*/

   A          = (hypre_ParCSRMatrix *) Amat_->getMatrix();
   ADiag      = hypre_ParCSRMatrixDiag(A);
   localNRows = hypre_CSRMatrixNumRows(ADiag);
   u          = (hypre_ParVector *) u_in->getVector();
   f          = (hypre_ParVector *) f_in->getVector();
   r          = (hypre_ParVector *) rVec_->getVector();
   z          = (hypre_ParVector *) zVec_->getVector();
   p          = (hypre_ParVector *) pVec_->getVector();
   rData      = hypre_VectorData(hypre_ParVectorLocalVector(r));
   zData      = hypre_VectorData(hypre_ParVectorLocalVector(z));
   pData      = hypre_VectorData(hypre_ParVectorLocalVector(p));
   lambdaMin  = omega * minEigen_ / maxEigen_;
   lambdaMax  = omega;
   dValue     = 0.5 * (lambdaMax + lambdaMin);
   cValue     = 0.5 * (lambdaMax - lambdaMin);
   
   /*-----------------------------------------------------------------
    * Perform Chebyshev iterations
    *-----------------------------------------------------------------*/
 
   hypre_ParVectorCopy( f, r );
   if ( zeroInitialGuess_ == 0 )
      hypre_ParCSRMatrixMatvec( -1.0, A, u, 1.0, r ); 
   zeroInitialGuess_ = 0;
   for ( i = 1; i <= degree_; i++ )
   {
      for ( j = 0 ; j < localNRows; j++ )
         zData[j] = diagonal_[j] * rData[j];
      if ( i == 1 ) 
      {
         hypre_ParVectorCopy( z, p );
         alpha = 2.0 / dValue;
      }
      else
      {
         beta = 0.5 * alpha * cValue;
         beta = beta * beta;
         alpha = 1.0 / ( dValue - beta );
         for ( j = 0 ; j < localNRows; j++ ) 
            pData[j] = zData[j] + beta * pData[j];
      }
      hypre_ParVectorAxpy( alpha, p, u );
      hypre_ParCSRMatrixMatvec( -alpha, A, p, 1.0, r ); 
   }
   return(0); 
}

/******************************************************************************
 * set Chebyshev parameters
 *---------------------------------------------------------------------------*/

int MLI_Solver_Chebyshev::setParams(char *paramString, int argc, char **argv)
{
   char param1[200];

   sscanf( paramString, "%s", param1 ); 
   if ( !strcmp(param1, "relaxWeight") )
   {
      if ( argc >= 1 ) degree_ = *(int*)  argv[0];
      if ( degree_ < 3 ) degree_ = 3;
   }
   else if ( !strcmp(param1, "degree") )
   {
      if ( argc != 1 ) 
      {
         printf("MLI_Solver_Chebyshev::setParams ERROR : needs 1 arg.\n");
         return 1;
      }
      degree_ = *(int*) argv[0];
      if ( degree_ < 3 ) degree_ = 3;
   }
   else if ( !strcmp(param1, "zeroInitialGuess") )
   {
      zeroInitialGuess_ = 1;
   }
   return 0;
}

