/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

#include <string.h>
#include <strings.h>
#include "parcsr_mv/parcsr_mv.h"
#include "solver/mli_solver_sgs.h"
#include "base/mli_defs.h"

/******************************************************************************
 * symmetric Gauss-Seidel relaxation scheme
 *****************************************************************************/

/******************************************************************************
 * constructor
 *---------------------------------------------------------------------------*/

MLI_Solver_SGS::MLI_Solver_SGS() : MLI_Solver(MLI_SOLVER_SGS_ID)
{
   Amat_             = NULL;
   zeroInitialGuess_ = 0;
   nSweeps_          = 1;
   relaxWeights_     = new double[1];
   relaxWeights_[0]  = 1.0;
}

/******************************************************************************
 * destructor
 *---------------------------------------------------------------------------*/

MLI_Solver_SGS::~MLI_Solver_SGS()
{
   if ( relaxWeights_ != NULL ) delete [] relaxWeights_;
   relaxWeights_ = NULL;
}

/******************************************************************************
 * set up the smoother
 *---------------------------------------------------------------------------*/

int MLI_Solver_SGS::setup(MLI_Matrix *mat)
{
   Amat_ = mat;
   return 0;
}

/******************************************************************************
 * apply function
 *---------------------------------------------------------------------------*/

int MLI_Solver_SGS::solve(MLI_Vector *fIn, MLI_Vector *uIn)
{
   int                 *ADiagI, *ADiagJ, *AOffdI, *AOffdJ;
   double              *ADiagA, *AOffdA, *uData, *fData;
   register int        iStart, iEnd, jj;
   int                 i, j, is, localNRows, extNRows, *tmpJ, relaxError=0;
   int                 index, nprocs, nSends, start;
   register double     res;
   double              zero = 0.0, relaxWeight;
   double              *vBufData, *tmpData, *vExtData;
   MPI_Comm            comm;
   hypre_ParCSRMatrix     *A;
   hypre_CSRMatrix        *ADiag, *AOffd;
   hypre_ParVector        *f, *u;
   hypre_ParCSRCommPkg    *commPkg;
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
   uData      = hypre_VectorData(hypre_ParVectorLocalVector(u));
   f          = (hypre_ParVector *) fIn->getVector();
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

      if (extNRows)
      {
         AOffdJ = hypre_CSRMatrixJ(AOffd);
         AOffdA = hypre_CSRMatrixData(AOffd);
      }
   }

   /*-----------------------------------------------------------------
    * perform GS sweeps
    *-----------------------------------------------------------------*/
 
   for( is = 0; is < nSweeps_; is++ )
   {
      relaxWeight = relaxWeights_[is];

      /*-----------------------------------------------------------------
       * communicate data on processor boundaries
       *-----------------------------------------------------------------*/

      if (nprocs > 1)
      {
         if ( ! zeroInitialGuess_ )
         {
            index = 0;
            for (i = 0; i < nSends; i++)
            {
               start = hypre_ParCSRCommPkgSendMapStart(commPkg, i);
               for (j=start;j<hypre_ParCSRCommPkgSendMapStart(commPkg,i+1);
                    j++)
                  vBufData[index++]
                      = uData[hypre_ParCSRCommPkgSendMapElmt(commPkg,j)];
            }
            commHandle = hypre_ParCSRCommHandleCreate(1,commPkg,vBufData,
                                                      vExtData);
            hypre_ParCSRCommHandleDestroy(commHandle);
            commHandle = NULL;
         }
      }

      /*-----------------------------------------------------------------
       * forward sweep
       *-----------------------------------------------------------------*/

      for (i = 0; i < localNRows; i++)     /* interior points first */
      {
         /*-----------------------------------------------------------
          * If diagonal is nonzero, relax point i; otherwise, skip it.
          *-----------------------------------------------------------*/

         if ( ADiagA[ADiagI[i]] != zero)
         {
            res      = fData[i];
            iStart   = ADiagI[i];
            iEnd     = ADiagI[i+1];
            tmpJ    = &(ADiagJ[iStart]);
            tmpData = &(ADiagA[iStart]);
            for (jj = iStart; jj < iEnd; jj++)
               res -= (*tmpData++) * uData[*tmpJ++];
            if ( ! zeroInitialGuess_ && nprocs > 1 )
            {
               iStart  = AOffdI[i];
               iEnd    = AOffdI[i+1];
               tmpJ    = &(AOffdJ[iStart]);
               tmpData = &(AOffdA[iStart]);
               for (jj = iStart; jj < iEnd; jj++)
                  res -= (*tmpData++) * vExtData[*tmpJ++];
            }
            uData[i] += relaxWeight * res / ADiagA[ADiagI[i]];
         }
      }

      /*-----------------------------------------------------------------
       * backward sweep
       *-----------------------------------------------------------------*/

      for (i = localNRows-1; i > -1; i--)  /* interior points first */
      {
         /*-----------------------------------------------------------
          * If diagonal is nonzero, relax point i; otherwise, skip it.
          *-----------------------------------------------------------*/

         if ( ADiagA[ADiagI[i]] != zero)
         {
            res     = fData[i];
            iStart  = ADiagI[i];
            iEnd    = ADiagI[i+1];
            tmpJ    = &(ADiagJ[iStart]);
            tmpData = &(ADiagA[iStart]);
            for (jj = iStart; jj < iEnd; jj++)
               res -= (*tmpData++) * uData[*tmpJ++];
            if ( ! zeroInitialGuess_ && nprocs > 1 )
            {
               iStart  = AOffdI[i];
               iEnd    = AOffdI[i+1];
               tmpJ    = &(AOffdJ[iStart]);
               tmpData = &(AOffdA[iStart]);
               for (jj = iStart; jj < iEnd; jj++)
                  res -= (*tmpData++) * vExtData[*tmpJ++];
            }
            uData[i] += relaxWeight * res / ADiagA[ADiagI[i]];
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
   return(relaxError); 
}

/******************************************************************************
 * set SGS parameters
 *---------------------------------------------------------------------------*/

int MLI_Solver_SGS::setParams( char *paramString, int argc, char **argv )
{
   int    i;
   double *weights;
   char   param1[200];

   if ( !strcasecmp(paramString, "numSweeps") )
   {
      if ( argc != 1 ) 
      {
         printf("MLI_Solver_SGS::setParams ERROR : needs 1 arg.\n");
         return 1;
      }
      nSweeps_ = *(int*) argv[0];
      if ( nSweeps_ < 1 ) nSweeps_ = 1;
      if ( relaxWeights_ != NULL ) delete [] relaxWeights_;
      relaxWeights_ = new double[nSweeps_];
      for ( i = 0; i < nSweeps_; i++ ) relaxWeights_[i] = 1.0;
      return 0;
   }
   else if ( !strcasecmp(paramString, "relaxWeight") )
   {
      if ( argc != 2 && argc != 1 ) 
      {
         printf("MLI_Solver_SGS::setParams ERROR : needs 1 or 2 args.\n");
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
   else if ( !strcasecmp(paramString, "zeroInitialGuess") )
   {
      zeroInitialGuess_ = 1;
      return 0;
   }
   else
   {   
      printf("MLI_Solver_SGS::setParams - parameter not recognized.\n");
      printf("                 Params = %s\n", paramString);
      return 1;
   }
   return 0;
}

/******************************************************************************
 * set SGS parameters
 *---------------------------------------------------------------------------*/

int MLI_Solver_SGS::setParams( int ntimes, double *weights )
{
   int i, nsweeps;

   if ( ntimes <= 0 )
   {
      printf("MLI_Solver_SGS::setParams WARNING : nsweeps set to 1.\n");
      nsweeps = 1;
   }
   nSweeps_ = nsweeps;
   if ( relaxWeights_ != NULL ) delete [] relaxWeights_;
   relaxWeights_ = new double[nsweeps];
   if ( weights == NULL )
   {
      printf("MLI_Solver_SGS::setParams - relax_weights set to 0.5.\n");
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
            printf("MLI_Solver_SGS::setParams - some weights set to 0.5.\n");
            relaxWeights_[i] = 0.5;
         }
      }
   }
   return 0;
}

