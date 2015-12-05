/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.13 $
 ***********************************************************************EHEADER*/





#include <stdio.h>
#include <string.h>
#include "HYPRE.h"
#include "solver/mli_solver_cg.h"
#include "solver/mli_solver_jacobi.h"
#include "solver/mli_solver_bjacobi.h"
#include "solver/mli_solver_sgs.h"
#include "solver/mli_solver_bsgs.h"
#include "solver/mli_solver_hsgs.h"
#include "solver/mli_solver_mli.h"
#include "solver/mli_solver_amg.h"

/******************************************************************************
 * constructor
 *---------------------------------------------------------------------------*/

MLI_Solver_CG::MLI_Solver_CG(char *name) : MLI_Solver(name)
{
   Amat_             = NULL;
   rVec_             = NULL;
   zVec_             = NULL;
   pVec_             = NULL;
   apVec_            = NULL;
   maxIterations_    = 3;
   tolerance_        = 0.0;
   baseSolver_       = NULL;
   baseMethod_       = MLI_SOLVER_BSGS_ID;
   zeroInitialGuess_ = 0;
   // for domain decomposition with coarse overlaps
   nSends_    = 0;
   nRecvs_    = 0;
   sendProcs_ = NULL;
   recvProcs_ = NULL;
   sendLengs_ = NULL;
   recvLengs_ = NULL;
   PSmat_     = NULL;
   AComm_     = 0;
   PSvec_     = NULL;
   iluI_      = NULL;
   iluJ_      = NULL;
   iluA_      = NULL;
   iluD_      = NULL;
}

/******************************************************************************
 * destructor
 *---------------------------------------------------------------------------*/

MLI_Solver_CG::~MLI_Solver_CG()
{
   if ( rVec_  != NULL ) delete rVec_;
   if ( zVec_  != NULL ) delete zVec_;
   if ( pVec_  != NULL ) delete pVec_;
   if ( apVec_ != NULL ) delete apVec_;
   if ( PSmat_ != NULL ) delete PSmat_;
   if ( PSvec_ != NULL ) delete PSvec_;
   if ( sendProcs_  != NULL ) delete [] sendProcs_;
   if ( recvProcs_  != NULL ) delete [] recvProcs_;
   if ( sendLengs_  != NULL ) delete [] sendLengs_;
   if ( recvLengs_  != NULL ) delete [] recvLengs_;
   if ( baseSolver_ != NULL ) delete baseSolver_;
   if ( iluI_ != NULL ) delete iluI_;
   if ( iluJ_ != NULL ) delete iluJ_;
   if ( iluA_ != NULL ) delete iluA_;
   if ( iluD_ != NULL ) delete iluD_;
}

/******************************************************************************
 * set up the smoother
 *---------------------------------------------------------------------------*/

int MLI_Solver_CG::setup(MLI_Matrix *Amat_in)
{
   int    numSweeps;
   double value=4.0/3.0;
   char   paramString[100], *argv[1];;

   /*-----------------------------------------------------------------
    * set local matrix
    *-----------------------------------------------------------------*/

   Amat_ = Amat_in;

   /*-----------------------------------------------------------------
    * set up preconditioner
    *-----------------------------------------------------------------*/

   switch( baseMethod_ )
   {
      case MLI_SOLVER_JACOBI_ID : sprintf(paramString, "Jacobi");
                                  baseSolver_ = 
                                     new MLI_Solver_Jacobi(paramString);
                                  sprintf(paramString, "numSweeps");
                                  numSweeps = 1;
                                  argv[0] = (char *) &numSweeps;
                                  baseSolver_->setParams(paramString,1,argv);
                                  sprintf(paramString, "setMaxEigen");
                                  argv[0] = (char *) &value;
                                  baseSolver_->setParams(paramString,1,argv);
                                  break;
      case MLI_SOLVER_BJACOBI_ID: sprintf(paramString, "BJacobi");
                                  baseSolver_ = 
                                     new MLI_Solver_BJacobi(paramString);
                                  sprintf(paramString, "numSweeps");
                                  numSweeps = 1;
                                  argv[0] = (char *) &numSweeps;
                                  baseSolver_->setParams(paramString,1,argv);
                                  break;
      case MLI_SOLVER_SGS_ID :    sprintf(paramString, "HSGS");
                                  baseSolver_ = 
                                     new MLI_Solver_HSGS(paramString);
                                  sprintf(paramString, "numSweeps");
                                  numSweeps = 1;
                                  argv[0] = (char *) &numSweeps;
                                  baseSolver_->setParams(paramString,1,argv);
                                  break;
      case MLI_SOLVER_BSGS_ID :   sprintf(paramString, "BSGS");
                                  baseSolver_ = 
                                     new MLI_Solver_BSGS(paramString);
                                  sprintf(paramString, "numSweeps");
                                  numSweeps = 1;
                                  argv[0] = (char *) &numSweeps;
                                  baseSolver_->setParams(paramString,1,argv);
                                  break;
      case MLI_SOLVER_AMG_ID :    sprintf(paramString, "AMG");
                                  baseSolver_ = 
                                     new MLI_Solver_AMG(paramString);
                                  break;
      case MLI_SOLVER_MLI_ID :    sprintf(paramString, "MLI");
                                  baseSolver_ = 
                                     new MLI_Solver_MLI(paramString);
                                  break;
      case MLI_SOLVER_ILU_ID:     iluDecomposition();
                                  break;
      default : printf("MLI_Solver_CG ERROR : no base method.\n");
                exit(1);
   }
   if (baseMethod_ != MLI_SOLVER_ILU_ID) baseSolver_->setup(Amat_);
 
   /*-----------------------------------------------------------------
    * build auxiliary vectors
    *-----------------------------------------------------------------*/

   rVec_  = Amat_->createVector();
   zVec_  = Amat_->createVector();
   pVec_  = Amat_->createVector();
   apVec_ = Amat_->createVector();

   return 0;
}

/******************************************************************************
 * apply function
 *---------------------------------------------------------------------------*/

int MLI_Solver_CG::solve(MLI_Vector *f_in, MLI_Vector *u_in)
{
   int                i, iter, localNRows, iP, rlength, shortNRows;
   double             *pData, *zData, *rData, *u2Data, *f2Data, dZero=0.0;
   double             rho, rhom1, alpha, beta, sigma, rnorm, dOne=1.0;
   double             *fData, *uData;
   char               paramString[30];
   hypre_ParCSRMatrix *A, *P;
   hypre_CSRMatrix    *ADiag;
   hypre_ParVector    *f, *u, *p, *z, *r, *ap, *f2;
   MPI_Request        *mpiRequests;
   MPI_Status         mpiStatus;
   MLI_Vector         *zVecLocal, *rVecLocal;

   /*-----------------------------------------------------------------
    * fetch machine and smoother parameters
    *-----------------------------------------------------------------*/

   A          = (hypre_ParCSRMatrix *) Amat_->getMatrix();
   ADiag      = hypre_ParCSRMatrixDiag(A);
   localNRows = hypre_CSRMatrixNumRows(ADiag);
   
   /*-----------------------------------------------------------------
    * fetch local vectors
    *-----------------------------------------------------------------*/

   r  = (hypre_ParVector *) rVec_->getVector();  /* -- r  -- */
   z  = (hypre_ParVector *) zVec_->getVector();  /* -- z  -- */
   p  = (hypre_ParVector *) pVec_->getVector();  /* -- p  -- */
   ap = (hypre_ParVector *) apVec_->getVector(); /* -- ap -- */

   /*-----------------------------------------------------------------
    * for domain decomposition, set up for extended vector f
    *-----------------------------------------------------------------*/

   f = (hypre_ParVector *) f_in->getVector();
   u = (hypre_ParVector *) u_in->getVector();
   rData = hypre_VectorData(hypre_ParVectorLocalVector(r));
   if ( PSmat_ != NULL )
   {
      P  = (hypre_ParCSRMatrix *) PSmat_->getMatrix();
      f2 = (hypre_ParVector *) PSvec_->getVector();
      hypre_ParCSRMatrixMatvecT(dOne, P, f, dZero, f2);
      rlength = 0;
      for ( iP = 0; iP < nRecvs_; iP++ ) rlength += recvLengs_[iP];
      shortNRows = localNRows - rlength;
      f2Data = hypre_VectorData(hypre_ParVectorLocalVector(f2));
      if ( nRecvs_ > 0 ) mpiRequests = new MPI_Request[nRecvs_];
      for ( iP = 0; iP < nRecvs_; iP++ )
      {
         MPI_Irecv(&rData[shortNRows],recvLengs_[iP],MPI_DOUBLE,
                   recvProcs_[iP], 45716, AComm_, &(mpiRequests[iP]));
         shortNRows += recvLengs_[iP];
      }
      for ( iP = 0; iP < nSends_; iP++ )
         MPI_Send(f2Data,sendLengs_[iP],MPI_DOUBLE,sendProcs_[iP],45716,
                  AComm_);
      for ( iP = 0; iP < nRecvs_; iP++ )
         MPI_Wait( &(mpiRequests[iP]), &mpiStatus );
      if ( nRecvs_ > 0 ) delete [] mpiRequests;
      shortNRows = localNRows - rlength;
      fData = hypre_VectorData(hypre_ParVectorLocalVector(f));
      for ( i = 0; i < shortNRows; i++ ) rData[i] = fData[i];
      zeroInitialGuess_ = 0;
      u2Data = new double[localNRows];
      for ( i = 0; i < localNRows; i++ ) u2Data[i] = 0.0;
   }
   else
   {
      hypre_ParVectorCopy(f, r);
      if (zeroInitialGuess_==0) hypre_ParCSRMatrixMatvec(-1.0,A,u,1.0,r);
      zeroInitialGuess_ = 0;
   }

   /*-----------------------------------------------------------------
    * set up for outer iterations 
    *-----------------------------------------------------------------*/

   if ( tolerance_ != 0.0 ) rnorm = sqrt(hypre_ParVectorInnerProd(r, r));
   else                     rnorm = 1.0;

   /*-----------------------------------------------------------------
    * fetch auxiliary vectors
    *-----------------------------------------------------------------*/

   pData  = hypre_VectorData(hypre_ParVectorLocalVector(p));
   zData  = hypre_VectorData(hypre_ParVectorLocalVector(z));

   /*-----------------------------------------------------------------
    * Perform iterations
    *-----------------------------------------------------------------*/
 
   iter = 0;
   rho  = 0.0;
   while ( iter < maxIterations_ && rnorm > tolerance_ )
   {
      iter++;
      hypre_ParVectorSetConstantValues(z, 0.0);
      strcpy(paramString, "zeroInitialGuess");
      if (baseMethod_ != MLI_SOLVER_ILU_ID) 
         baseSolver_->setParams( paramString, 0, NULL );
      strcpy( paramString, "HYPRE_ParVector" );
      zVecLocal = new MLI_Vector( (void *) z, paramString, NULL);
      rVecLocal = new MLI_Vector( (void *) r, paramString, NULL);
      if (baseMethod_ == MLI_SOLVER_ILU_ID) iluSolve(rData, zData);
      else               baseSolver_->solve( rVecLocal, zVecLocal );
      rhom1 = rho;
      rho   = hypre_ParVectorInnerProd(r, z);
      if ( iter == 1 ) 
      {
         beta = 0.0;
         hypre_ParVectorCopy(z, p);
      }
      else
      {
         beta = rho / rhom1;

#define HYPRE_SMP_PRIVATE i
#include "utilities/hypre_smp_forloop.h"
         for ( i = 0; i < localNRows; i++ ) 
            pData[i] = beta * pData[i] + zData[i];

      }
      hypre_ParCSRMatrixMatvec(1.0, A, p, 0.0, ap);
      sigma = hypre_ParVectorInnerProd(p, ap);
      alpha = rho /sigma;
      if ( PSmat_ == NULL )
         hypre_ParVectorAxpy(alpha, p, u);  /* u = u + alpha p */
      else
         for ( i = 0; i < localNRows; i++ ) u2Data[i] += alpha * pData[i];

      hypre_ParVectorAxpy(-alpha, ap, r); /* r = r - alpha ap */
      if (tolerance_ != 0.0 && maxIterations_ > 1) 
         rnorm = sqrt(hypre_ParVectorInnerProd(r, r));
   }

   /*-----------------------------------------------------------------
    * for domain decomposition, recover the solution vector
    *-----------------------------------------------------------------*/

   if ( PSmat_ != NULL )
   {
      uData  = hypre_VectorData(hypre_ParVectorLocalVector(u));
      for ( i = 0; i < shortNRows; i++ ) uData[i] = u2Data[i];
      delete [] u2Data;
   }
   return(0); 
}

/******************************************************************************
 * set CG parameters
 *---------------------------------------------------------------------------*/

int MLI_Solver_CG::setParams( char *paramString, int argc, char **argv )
{
   int    i, *iArray;
   char   param1[100], param2[100];

   sscanf(paramString, "%s", param1);
   if ( !strcmp(param1, "maxIterations") )
   {
      sscanf(paramString, "%s %d", param1, &maxIterations_);
      return 0;
   }
   else if ( !strcmp(param1, "tolerance") )
   {
      sscanf(paramString, "%s %lg", param1, &tolerance_);
      return 0;
   }
   else if ( !strcmp(param1, "zeroInitialGuess") )
   {
      zeroInitialGuess_ = 1;
      return 0;
   }
   else if ( !strcmp(param1, "numSweeps") )
   {
      sscanf(paramString, "%s %d", param1, &maxIterations_);
      return 0;
   }
   else if ( !strcmp(param1, "relaxWeight") )
   {
      if ( argc != 2 && argc != 1 ) 
      {
         printf("MLI_Solver_CG::setParams ERROR : needs 1 or 2 args.\n");
         return 1;
      }
      if ( argc >= 1 ) maxIterations_ = *(int*) argv[0];
      return 0;
   }
   else if ( !strcmp(param1, "baseMethod") )
   {
      sscanf(paramString, "%s %s", param1, param2);
      if ( !strcmp(param2, "Jacobi") ) 
         baseMethod_ = MLI_SOLVER_JACOBI_ID;
      else if ( !strcmp(param2, "BJacobi") )
         baseMethod_ = MLI_SOLVER_BJACOBI_ID;
      else if ( !strcmp(param2, "SGS") )
         baseMethod_ = MLI_SOLVER_SGS_ID;
      else if ( !strcmp(param2, "BSGS") )
         baseMethod_ = MLI_SOLVER_BSGS_ID;
      else if ( !strcmp(param2, "AMG") )
         baseMethod_ = MLI_SOLVER_AMG_ID;
      else if ( !strcmp(param2, "MLI") )
         baseMethod_ = MLI_SOLVER_MLI_ID;
      else if ( !strcmp(param2, "ILU") )
         baseMethod_ = MLI_SOLVER_ILU_ID;
      else
         baseMethod_ = MLI_SOLVER_BJACOBI_ID;
      return 0;
   }
   else if ( !strcmp(param1, "setPmat") )
   {
      if ( argc != 1 )
      {
       	 printf("MLI_Solver_CG::setParams ERROR : needs 1 arg.\n");
	 return 1;
      }
      HYPRE_IJVector auxVec;
      PSmat_ = (MLI_Matrix *) argv[0];
      hypre_ParCSRMatrix *hypreAux;
      hypre_ParCSRMatrix *ps = (hypre_ParCSRMatrix *) PSmat_->getMatrix();
      int nCols = hypre_ParCSRMatrixNumCols(ps);
      int start = hypre_ParCSRMatrixFirstColDiag(ps);
      MPI_Comm vComm = hypre_ParCSRMatrixComm(ps);
      HYPRE_IJVectorCreate(vComm, start, start+nCols-1, &auxVec);
      HYPRE_IJVectorSetObjectType(auxVec, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(auxVec);
      HYPRE_IJVectorAssemble(auxVec);
      HYPRE_IJVectorGetObject(auxVec, (void **) &hypreAux);
      HYPRE_IJVectorSetObjectType(auxVec, -1);
      HYPRE_IJVectorDestroy(auxVec);
      strcpy( paramString, "HYPRE_ParVector" );
      MLI_Function *funcPtr = new MLI_Function();
      MLI_Utils_HypreParVectorGetDestroyFunc(funcPtr);
      PSvec_ = new MLI_Vector( (void*) hypreAux, paramString, funcPtr );
      delete funcPtr;
   }
   else if ( !strcmp(param1, "setCommData") )
   {
      if ( argc != 7 )
      {
         printf("MLI_Solver_CG::setParams ERROR : needs 7 arg.\n");
         return 1;
      }
      nRecvs_ = *(int *) argv[0];
      if ( nRecvs_ > 0 )
      {
         recvProcs_ = new int[nRecvs_];
         recvLengs_ = new int[nRecvs_];
         iArray =  (int *) argv[1];
         for ( i = 0; i < nRecvs_; i++ ) recvProcs_[i] = iArray[i];
         iArray =  (int *) argv[2];
         for ( i = 0; i < nRecvs_; i++ ) recvLengs_[i] = iArray[i];
      }
      nSends_ = *(int *) argv[3];
      if ( nSends_ > 0 )
      {
         sendProcs_ = new int[nSends_];
         sendLengs_ = new int[nSends_];
         iArray =  (int *) argv[4];
         for ( i = 0; i < nSends_; i++ ) sendProcs_[i] = iArray[i];
         iArray =  (int *) argv[5];
         for ( i = 0; i < nSends_; i++ ) sendLengs_[i] = iArray[i];
      }
      AComm_ = *(MPI_Comm *) argv[6];
   }
   else
   {   
      printf("MLI_Solver_CG::setParams - parameter not recognized.\n");
      printf("                Params = %s\n", paramString);
      return 1;
   }
   return 0;
}

/******************************************************************************
 * ilu decomposition 
 *---------------------------------------------------------------------------*/

int MLI_Solver_CG::iluDecomposition()
{
   int                 nrows, i, j, jj, jjj, k, rownum, colnum;
   int                 *ADiagI, *ADiagJ;
   double              *ADiagA, *darray, dt;
   hypre_ParCSRMatrix *A;
   hypre_CSRMatrix    *ADiag;

   A       = (hypre_ParCSRMatrix *) Amat_->getMatrix();
   ADiag   = hypre_ParCSRMatrixDiag(A);
   nrows   = hypre_CSRMatrixNumRows(ADiag);
   ADiagI  = hypre_CSRMatrixI(ADiag);
   ADiagJ  = hypre_CSRMatrixJ(ADiag);
   ADiagA  = hypre_CSRMatrixData(ADiag);
   iluI_   = new int[nrows+2];
   iluJ_   = new int[ADiagI[nrows]];
   iluA_   = new double[ADiagI[nrows]];
   iluD_   = new int[nrows+1];

   // -----------------------------------------------------------------
   // then put the elements (submatrix) of A into lu array
   // -----------------------------------------------------------------

   for (i = 0; i <= nrows; i++) iluI_[i+1] = ADiagI[i];
   for (i = 1; i <= nrows; i++)
   {
      rownum = i;
      for (j = iluI_[i]; j < iluI_[i+1]; j++)
      {
         colnum = ADiagJ[j] + 1;
         if (colnum == rownum) iluD_[i] = j;
         iluJ_[j] = colnum;
         iluA_[j] = ADiagA[j];
      }
   }
   // -----------------------------------------------------------------
   // loop on all the rows
   // -----------------------------------------------------------------
   darray = new double[nrows+1];
   for ( i = 1; i <= nrows; i++)
   {
      if (iluI_[i] != iluI_[i+1])
      {
         for (j = 1; j <= nrows; j++) darray[j] = 0.0e0;
         for (j = iluI_[i]; j < iluI_[i+1]; j++)
         {
            jj = iluJ_[j];
            if (iluI_[jj] != iluI_[jj+1]) darray[jj] = iluA_[j];
         }
         // ----------------------------------------------------------
         // eliminate L part of row i
         // ----------------------------------------------------------
         for (j = iluI_[i]; j < iluI_[i+1]; j++)
         {
            jj = iluJ_[j];
            if (jj < i && iluI_[jj] != iluI_[jj+1])
            {
               dt = darray[jj];
                if (dt != 0.0) {
                  dt = dt * iluA_[iluD_[jj]];
                  darray[jj] = dt;
                  for (k = iluI_[jj]; k < iluI_[jj+1]; k++) {
                     jjj = iluJ_[k];
                     if (jjj > jj) darray[jjj] -= dt * iluA_[k];
                  }
               }
            }
         }
         // ----------------------------------------------------------
         // put modified row part to lu array
         // ----------------------------------------------------------
         for (j = iluI_[i]; j < iluI_[i+1]; j++) {
            jj = iluJ_[j];
            if (iluI_[jj] != iluI_[jj+1]) iluA_[j] = darray[jj];
            else                          iluA_[j] = 0.0;
         }
         iluA_[iluD_[i]] = 1.0e0 / iluA_[iluD_[i]];
      }
   }
   delete [] darray;
   return 0;
}

/******************************************************************************
 * ilu solve 
 *---------------------------------------------------------------------------*/

int MLI_Solver_CG::iluSolve(double *inData, double *outData)
{
   int                i, j, k, nrows;
   double             dtmp;
   hypre_ParCSRMatrix *A;
   hypre_CSRMatrix    *ADiag;

   A     = (hypre_ParCSRMatrix *) Amat_->getMatrix();
   ADiag = hypre_ParCSRMatrixDiag(A);
   nrows = hypre_CSRMatrixNumRows(ADiag);

   for (i = 0; i < nrows; i++) outData[i] = inData[i];
   for (i = 1; i <= nrows; i++) {
      if (iluI_[i] != iluI_[i+1]) {
         dtmp = 0.0e0;
         for (k = iluI_[i]; k < iluD_[i]; k++) {
            j = iluJ_[k];
            dtmp += (iluA_[k] * outData[j-1]);
         }
         outData[i-1] = outData[i-1] - dtmp;
      }
   }
   for (i = nrows; i >= 1; i--) {
      if (iluI_[i] != iluI_[i+1]) {
         dtmp = 0.0e0;
         for (k = iluD_[i]+1; k < iluI_[i+1]; k++) {
            j = iluJ_[k];
            dtmp += (iluA_[k] * outData[j-1]);
         }
         outData[i-1] = (outData[i-1] - dtmp) * iluA_[iluD_[i]];
      }
   }
   return 0;
}

