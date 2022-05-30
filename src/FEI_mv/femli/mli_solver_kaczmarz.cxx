/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include <string.h>
#include "mli_solver_kaczmarz.h"
#include "_hypre_parcsr_mv.h"

/******************************************************************************
 * Kaczmarz relaxation scheme
 *****************************************************************************/

/******************************************************************************
 * constructor
 *---------------------------------------------------------------------------*/

MLI_Solver_Kaczmarz::MLI_Solver_Kaczmarz(char *name) : MLI_Solver(name)
{
   Amat_             = NULL;
   nSweeps_          = 1;
   AsqDiag_          = NULL;
   zeroInitialGuess_ = 0;
}

/******************************************************************************
 * destructor
 *---------------------------------------------------------------------------*/

MLI_Solver_Kaczmarz::~MLI_Solver_Kaczmarz()
{
   if ( AsqDiag_ != NULL ) delete [] AsqDiag_;
   AsqDiag_ = NULL;
}

/******************************************************************************
 * set up the smoother
 *---------------------------------------------------------------------------*/

int MLI_Solver_Kaczmarz::setup(MLI_Matrix *mat)
{
   int                irow, jcol, localNRows, *ADiagI, *AOffdI;
   double             *ADiagA, *AOffdA, rowNorm;
   hypre_ParCSRMatrix *A;
   hypre_CSRMatrix    *ADiag, *AOffd;

   Amat_ = mat;

   A          = (hypre_ParCSRMatrix *) Amat_->getMatrix();
   ADiag      = hypre_ParCSRMatrixDiag(A);
   AOffd      = hypre_ParCSRMatrixOffd(A);
   localNRows = hypre_CSRMatrixNumRows(ADiag);
   ADiagI     = hypre_CSRMatrixI(ADiag);
   ADiagA     = hypre_CSRMatrixData(ADiag);
   AOffdI     = hypre_CSRMatrixI(AOffd);
   AOffdA     = hypre_CSRMatrixData(AOffd);

   if ( AsqDiag_ != NULL ) delete [] AsqDiag_;
   AsqDiag_ = new double[localNRows];
   for ( irow = 0; irow < localNRows; irow++ )
   {
      rowNorm = 0.0;
      for ( jcol = ADiagI[irow]; jcol < ADiagI[irow+1]; jcol++ )
         rowNorm += (ADiagA[jcol] * ADiagA[jcol]);
      for ( jcol = AOffdI[irow]; jcol < AOffdI[irow+1]; jcol++ )
         rowNorm += (AOffdA[jcol] * AOffdA[jcol]);
      if ( rowNorm != 0.0 ) AsqDiag_[irow] = 1.0 / rowNorm;
      else                  AsqDiag_[irow] = 1.0;
   }
   return 0;
}

/******************************************************************************
 * apply function
 *---------------------------------------------------------------------------*/

int MLI_Solver_Kaczmarz::solve(MLI_Vector *fIn, MLI_Vector *uIn)
{
   hypre_ParCSRMatrix  *A;
   hypre_CSRMatrix     *ADiag, *AOffd;
   int                 *ADiagI, *ADiagJ, *AOffdI, *AOffdJ;
   double              *ADiagA, *AOffdA, *uData, *fData;
   int                 irow, jcol, is, localNRows, retFlag=0, nprocs, start;
   int                 nSends, extNRows, index, endp1;
   double              *vBufData, *vExtData, res;
   MPI_Comm            comm;
   hypre_ParCSRCommPkg    *commPkg;
   hypre_ParVector        *f, *u;
   hypre_ParCSRCommHandle *commHandle;

   /*-----------------------------------------------------------------
    * fetch machine and smoother parameters
    *-----------------------------------------------------------------*/

   A          = (hypre_ParCSRMatrix *) Amat_->getMatrix();
   comm       = hypre_ParCSRMatrixComm(A);
   commPkg    = hypre_ParCSRMatrixCommPkg(A);
   ADiag      = hypre_ParCSRMatrixDiag(A);
   localNRows = hypre_CSRMatrixNumRows(ADiag);
   ADiagI     = hypre_CSRMatrixI(ADiag);
   ADiagJ     = hypre_CSRMatrixJ(ADiag);
   ADiagA     = hypre_CSRMatrixData(ADiag);
   AOffd      = hypre_ParCSRMatrixOffd(A);
   extNRows   = hypre_CSRMatrixNumCols(AOffd);
   AOffdI     = hypre_CSRMatrixI(AOffd);
   AOffdJ     = hypre_CSRMatrixJ(AOffd);
   AOffdA     = hypre_CSRMatrixData(AOffd);
   u          = (hypre_ParVector *) uIn->getVector();
   f          = (hypre_ParVector *) fIn->getVector();
   uData      = hypre_VectorData(hypre_ParVectorLocalVector(u));
   fData      = hypre_VectorData(hypre_ParVectorLocalVector(f));
   MPI_Comm_size(comm,&nprocs);  

   /*-----------------------------------------------------------------
    * setting up for interprocessor communication
    *-----------------------------------------------------------------*/

   if (nprocs > 1)
   {
      nSends = hypre_ParCSRCommPkgNumSends(commPkg);
      vBufData = new double[hypre_ParCSRCommPkgSendMapStart(commPkg,nSends)];
      vExtData = new double[extNRows];
      for ( irow = 0; irow < extNRows; irow++ ) vExtData[irow] = 0.0;
   }

   /*-----------------------------------------------------------------
    * perform Kaczmarz sweeps
    *-----------------------------------------------------------------*/
 
   for( is = 0; is < nSweeps_; is++ )
   {
      if (nprocs > 1 && zeroInitialGuess_ != 1 )
      {
         index = 0;
         for (irow = 0; irow < nSends; irow++)
         {
            start = hypre_ParCSRCommPkgSendMapStart(commPkg, irow);
            endp1 = hypre_ParCSRCommPkgSendMapStart(commPkg,irow+1);
            for ( jcol = start; jcol < endp1; jcol++ )
               vBufData[index++]
                      = uData[hypre_ParCSRCommPkgSendMapElmt(commPkg,jcol)];
         }
         commHandle = hypre_ParCSRCommHandleCreate(1,commPkg,vBufData,
                                                   vExtData);
         hypre_ParCSRCommHandleDestroy(commHandle);
         commHandle = NULL;
      }

      for ( irow = 0; irow < localNRows; irow++ )
      {
         res = fData[irow];
         for ( jcol = ADiagI[irow];  jcol < ADiagI[irow+1]; jcol++ )
         {
            index = ADiagJ[jcol];
            res -= ADiagA[jcol] * uData[index];
         }
         if (nprocs > 1 && zeroInitialGuess_ != 1 )
         {
            for ( jcol = AOffdI[irow]; jcol < AOffdI[irow+1]; jcol++ )
            {
               index = AOffdJ[jcol];
               res -= AOffdA[jcol] * vExtData[index];
            }
         }
         res *= AsqDiag_[irow];
         for ( jcol = ADiagI[irow];  jcol < ADiagI[irow+1]; jcol++ )
         {
            index = ADiagJ[jcol];
            uData[index] += res * ADiagA[jcol];
         }
      }
      for ( irow = localNRows-1; irow >= 0; irow-- )
      {
         res = fData[irow];
         for ( jcol = ADiagI[irow];  jcol < ADiagI[irow+1]; jcol++ )
         {
            index = ADiagJ[jcol];
            res -= ADiagA[jcol] * uData[index];
         }
         if (nprocs > 1 && zeroInitialGuess_ != 1 )
         {
            for ( jcol = AOffdI[irow]; jcol < AOffdI[irow+1]; jcol++ )
            {
               index = AOffdJ[jcol];
               res -= AOffdA[jcol] * vExtData[index];
            }
         }
         res *= AsqDiag_[irow];
         for ( jcol = ADiagI[irow]; jcol < ADiagI[irow+1]; jcol++ )
         {
            index = ADiagJ[jcol];
            uData[index] += res * ADiagA[jcol];
         }
         for ( jcol = AOffdI[irow]; jcol < AOffdI[irow+1]; jcol++ )
         {
            index = AOffdJ[jcol];
            vExtData[index] += res * AOffdA[jcol];
         }
      }
      zeroInitialGuess_ = 0;
   }

   /*-----------------------------------------------------------------
    * clean up and return
    *-----------------------------------------------------------------*/

   if (nprocs > 1)
   {
      delete [] vExtData;
      delete [] vBufData;
   }
   return (retFlag); 
}

/******************************************************************************
 * set SGS parameters
 *---------------------------------------------------------------------------*/

int MLI_Solver_Kaczmarz::setParams(char *paramString, int argc, char **argv)
{
   if (!strcmp(paramString,"numSweeps") || !strcmp(paramString,"relaxWeight"))
   {
      if ( argc >= 1 ) nSweeps_ = *(int*)  argv[0];
      if ( nSweeps_ < 1 ) nSweeps_ = 1;
   }
   else if ( !strcmp(paramString, "zeroInitialGuess") )
   {
      zeroInitialGuess_ = 1;
   }
   return 0;
}

