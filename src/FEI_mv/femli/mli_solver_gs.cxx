/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include <string.h>
#include "mli_solver_gs.h"
#include "_hypre_parcsr_mv.h"

/******************************************************************************
 * Gauss-Seidel relaxation scheme
 *****************************************************************************/

/******************************************************************************
 * constructor
 *---------------------------------------------------------------------------*/

MLI_Solver_GS::MLI_Solver_GS(char *name) : MLI_Solver(name)
{
   Amat_             = NULL;
   nSweeps_          = 1;
   relaxWeights_     = new double[1];
   relaxWeights_[0]  = 1.0;
   zeroInitialGuess_ = 0;
}

/******************************************************************************
 * destructor
 *---------------------------------------------------------------------------*/

MLI_Solver_GS::~MLI_Solver_GS()
{
   if ( relaxWeights_ != NULL ) delete [] relaxWeights_;
   relaxWeights_ = NULL;
}

/******************************************************************************
 * set up the smoother
 *---------------------------------------------------------------------------*/

int MLI_Solver_GS::setup(MLI_Matrix *mat)
{
   Amat_ = mat;
   return 0;
}

/******************************************************************************
 * apply function
 *---------------------------------------------------------------------------*/

int MLI_Solver_GS::solve(MLI_Vector *fIn, MLI_Vector *uIn)
{
   hypre_ParCSRMatrix  *A;
   hypre_CSRMatrix     *ADiag, *AOffd;
   int                 *ADiagI, *ADiagJ, *AOffdI, *AOffdJ;
   double              *ADiagA, *AOffdA, *uData, *fData;
   int                 i, j, is, localNRows, relaxError=0;
   int                 ii, jj, nprocs, nthreads, start, length;
   int                 nSends, extNRows, index, size, ns, ne, rest;
   double              zero = 0.0, relaxWeight, res;
   double              *vBufData;
   double              *vExtData, *tmpData;
   MPI_Comm            comm;
   hypre_ParCSRCommPkg     *commPkg;
   hypre_ParVector         *f, *u;
   hypre_ParCSRCommHandle *commHandle;

   /*-----------------------------------------------------------------
    * fetch machine and smoother parameters
    *-----------------------------------------------------------------*/

   nthreads   = hypre_NumThreads();
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

   vBufData = vExtData = tmpData  = NULL;
   if (nprocs > 1)
   {
      nSends = hypre_ParCSRCommPkgNumSends(commPkg);
      length = hypre_ParCSRCommPkgSendMapStart(commPkg,nSends);
      if ( length > 0 ) vBufData = new double[length];
      if ( extNRows > 0 ) vExtData = new double[extNRows];
   }
   if (nthreads > 1 && localNRows > 0) tmpData = new double[localNRows];

   /*-----------------------------------------------------------------
    * perform GS sweeps
    *-----------------------------------------------------------------*/
 
   for( is = 0; is < nSweeps_; is++ )
   {
      if ( relaxWeights_ != NULL ) relaxWeight = relaxWeights_[is];
      else                         relaxWeight = 1.0;

      if (nprocs > 1 && zeroInitialGuess_ != 1 )
      {
         index = 0;
         for (i = 0; i < nSends; i++)
         {
            start = hypre_ParCSRCommPkgSendMapStart(commPkg, i);
            for (j=start;j<hypre_ParCSRCommPkgSendMapStart(commPkg,i+1);j++)
               vBufData[index++]
                      = uData[hypre_ParCSRCommPkgSendMapElmt(commPkg,j)];
         }
         commHandle = hypre_ParCSRCommHandleCreate(1,commPkg,vBufData,
                                                   vExtData);
         hypre_ParCSRCommHandleDestroy(commHandle);
         commHandle = NULL;
      }

      if (nthreads > 1)
      {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < localNRows; i++) tmpData[i] = uData[i];

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,ns,ne,res,rest,size) HYPRE_SMP_SCHEDULE
#endif
         for (j = 0; j < nthreads; j++)
         {
            size = localNRows/nthreads;
            rest = localNRows - size*nthreads;
            if (j < rest)
            {
               ns = j*size+j;
               ne = (j+1)*size+j+1;
            }
            else
            {
               ns = j*size+rest;
               ne = (j+1)*size+rest;
            }
            for (i = ns; i < ne; i++)   /* interior points first */
            {
               /*-----------------------------------------------------------
                * If diagonal is nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/

               if ( ADiagA[ADiagI[i]] != zero)
               {
                  res = fData[i];
                  for (jj = ADiagI[i]; jj < ADiagI[i+1]; jj++)
                  {
                     ii = ADiagJ[jj];
                     if (ii >= ns && ii < ne)
                        res -= ADiagA[jj] * uData[ii];
                     else
                        res -= ADiagA[jj] * tmpData[ii];
                  }
                  for (jj = AOffdI[i]; jj < AOffdI[i+1]; jj++)
                  {
                     ii = AOffdJ[jj];
                     res -= AOffdA[jj] * vExtData[ii];
                  }
                  uData[i] += relaxWeight * (res / ADiagA[ADiagI[i]]);
               }
            }
         }
      }
      else
      {
         for (i = 0; i < localNRows; i++)     /* interior points first */
         {
            /*-----------------------------------------------------------
             * If diagonal is nonzero, relax point i; otherwise, skip it.
             *-----------------------------------------------------------*/

            if ( ADiagA[ADiagI[i]] != zero)
            {
               res = fData[i];
               for (jj = ADiagI[i]; jj < ADiagI[i+1]; jj++)
               {
                  ii = ADiagJ[jj];
                  res -= ADiagA[jj] * uData[ii];
               }
               for (jj = AOffdI[i]; jj < AOffdI[i+1]; jj++)
               {
                  ii = AOffdJ[jj];
                  res -= AOffdA[jj] * vExtData[ii];
               }
               uData[i] += relaxWeight * (res / ADiagA[ADiagI[i]]);
            }
         }
      }
      zeroInitialGuess_ = 0;
   }

   /*-----------------------------------------------------------------
    * clean up and return
    *-----------------------------------------------------------------*/

   if ( vExtData != NULL ) delete [] vExtData;
   if ( vBufData != NULL ) delete [] vBufData;
   if ( tmpData  != NULL ) delete [] tmpData;
   return(relaxError); 
}

/******************************************************************************
 * set parameters
 *---------------------------------------------------------------------------*/

int MLI_Solver_GS::setParams(char *paramString, int argc, char **argv)
{
   int    i;
   double *weights=NULL;

   if ( !strcmp(paramString, "numSweeps") )
   {
      if ( argc == 1 ) nSweeps_ = *(int*) argv[0];
      if ( nSweeps_ < 1 ) nSweeps_ = 1;
      return 0;
   }
   else if ( !strcmp(paramString, "relaxWeight") )
   {
      if ( argc != 2 && argc != 1 ) 
      {
         printf("MLI_Solver_GS::setParams ERROR : needs 1 or 2 args.\n");
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
         for ( i = 0; i < nSweeps_; i++ ) 
         {
            if ( weights[i] > 0.0 ) relaxWeights_[i] = weights[i];
            else                    relaxWeights_[i] = 1.0;
         }
      }
   }
   else if ( strcmp(paramString, "zeroInitialGuess") )
   {   
      printf("MLI_Solver_GS::setParams - parameter not recognized.\n");
      printf("              Params = %s\n", paramString);
      return 1;
   }
   return 0;
}

/******************************************************************************
 * set parameters
 *---------------------------------------------------------------------------*/

int MLI_Solver_GS::setParams( int ntimes, double *weights )
{
   int i, nsweeps=0;

   if ( ntimes <= 0 )
   {
      printf("MLI_Solver_GS::setParams WARNING : nsweeps set to 1.\n");
      nsweeps = 1;
   }
   nSweeps_ = nsweeps;
   if ( relaxWeights_ != NULL ) delete [] relaxWeights_;
   relaxWeights_ = new double[ntimes];
   if ( weights == NULL )
   {
      printf("MLI_Solver_GS::setParams - relaxWeights set to 0.5.\n");
      for ( i = 0; i < nsweeps; i++ ) relaxWeights_[i] = 0.5;
   }
   else
   {
      for ( i = 0; i < nsweeps; i++ ) 
      {
         if (weights[i] >= 0. && weights[i] <= 2.) 
            relaxWeights_[i] = weights[i];
         else 
         {
            printf("MLI_Solver_GS::setParams - some weights set to 1.0.\n");
            relaxWeights_[i] = 1.0;
         }
      }
   }
   return 0;
}

