/*BHEADER**********************************************************************
 * (c) 2003   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

#include <math.h>
#include <string.h>
#include "parcsr_mv/parcsr_mv.h"
#include "parcsr_ls/parcsr_ls.h"
#include "parcsr_ls/par_amg.h"
#include "solver/mli_solver_hsgs.h"

/******************************************************************************
 * symmetric Gauss-Seidel relaxation scheme in BoomerAMG
 *****************************************************************************/

/******************************************************************************
 * constructor
 *---------------------------------------------------------------------------*/

MLI_Solver_HSGS::MLI_Solver_HSGS(char *name) : MLI_Solver(name)
{
   Amat_            = NULL;
   nSweeps_         = 1;
   relaxWeights_    = new double[1];
   relaxWeights_[0] = 1.0;
   mliVec_          = NULL;
   calcOmega_       = 1;
   printRNorm_      = 0;
}

/******************************************************************************
 * destructor
 *---------------------------------------------------------------------------*/

MLI_Solver_HSGS::~MLI_Solver_HSGS()
{
   if ( relaxWeights_ != NULL ) delete [] relaxWeights_;
   relaxWeights_ = NULL;
   if ( mliVec_ != NULL ) delete mliVec_;
}

/******************************************************************************
 * set up the smoother
 *---------------------------------------------------------------------------*/

int MLI_Solver_HSGS::setup(MLI_Matrix *mat)
{
   Amat_   = mat;
   mliVec_ = Amat_->createVector();
   if ( calcOmega_ == 1 ) calcOmega();
   return 0;
}

/******************************************************************************
 * apply function
 *---------------------------------------------------------------------------*/

int MLI_Solver_HSGS::solve(MLI_Vector *fIn, MLI_Vector *uIn)
{
   int                 *ADiagI, *ADiagJ, *AOffdI, *AOffdJ;
   double              *ADiagA, *AOffdA, *uData, *fData;
   register int        iStart, iEnd, jj;
   int                 i, j, is, localNRows, extNRows, *tmpJ, relaxError=0;
   int                 index, nprocs, mypid, nSends, start;
   register double     res;
   double              zero = 0.0, relaxWeight, rnorm;
   double              *vBufData=NULL, *tmpData, *vExtData=NULL;
   MPI_Comm            comm;
   hypre_ParCSRMatrix     *A;
   hypre_CSRMatrix        *ADiag, *AOffd;
   hypre_ParVector        *f, *u, *hypreR;
   hypre_ParCSRCommPkg    *commPkg;
   hypre_ParCSRCommHandle *commHandle;
   MLI_Vector             *mliRvec;

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
   MPI_Comm_rank(comm,&mypid);  

   /*-----------------------------------------------------------------
    * setting up for interprocessor communication
    *-----------------------------------------------------------------*/

   if (nprocs > 1)
   {
      nSends = hypre_ParCSRCommPkgNumSends(commPkg);
      if ( nSends > 0 )
         vBufData = new double[hypre_ParCSRCommPkgSendMapStart(commPkg,nSends)];
      else vBufData = NULL;
      if ( extNRows > 0 ) vExtData = new double[extNRows];
      else                vExtData = NULL;
   }
   if ( printRNorm_ == 1 )
   {
      mliRvec = Amat_->createVector();
      hypreR  = (hypre_ParVector *) mliRvec->getVector();
   }

   /*-----------------------------------------------------------------
    * perform SGS sweeps
    *-----------------------------------------------------------------*/
 
   relaxWeight = 1.0;
   for( is = 0; is < nSweeps_; is++ )
   {
      if ( relaxWeights_ != NULL ) relaxWeight = relaxWeights_[is];
      if ( relaxWeight <= 0.0 ) relaxWeight = 1.0;

      /*-----------------------------------------------------------------
       * forward sweep
       *-----------------------------------------------------------------*/

      if (nprocs > 1)
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

      for (i = 0; i < localNRows; i++)
      {
         if ( ADiagA[ADiagI[i]] != zero)
         {
            res      = fData[i];
            iStart   = ADiagI[i];
            iEnd     = ADiagI[i+1];
            tmpJ    = &(ADiagJ[iStart]);
            tmpData = &(ADiagA[iStart]);
            for (jj = iStart; jj < iEnd; jj++)
               res -= (*tmpData++) * uData[*tmpJ++];
            if ( nprocs > 1 )
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
         else printf("MLI_Solver_HSGS error : diag = 0.\n");
      }

      /*-----------------------------------------------------------------
       * backward sweep
       *-----------------------------------------------------------------*/

      for (i = localNRows-1; i > -1; i--)
      {
         if ( ADiagA[ADiagI[i]] != zero)
         {
            res     = fData[i];
            iStart  = ADiagI[i];
            iEnd    = ADiagI[i+1];
            tmpJ    = &(ADiagJ[iStart]);
            tmpData = &(ADiagA[iStart]);
            for (jj = iStart; jj < iEnd; jj++)
               res -= (*tmpData++) * uData[*tmpJ++];
            if ( nprocs > 1 )
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
      if ( printRNorm_ == 1 )
      {
         hypre_ParVectorCopy( f, hypreR );
         hypre_ParCSRMatrixMatvec( -1.0, A, u, 1.0, hypreR );
         rnorm = sqrt(hypre_ParVectorInnerProd( hypreR, hypreR ));
         if ( mypid == 0 )
            printf("\tMLI_Solver_HSGS iter = %4d, rnorm = %e (omega=%e)\n", 
                   is, rnorm, relaxWeight);
      }
   }
   if ( printRNorm_ == 1 ) delete mliRvec;

   /*-----------------------------------------------------------------
    * clean up and return
    *-----------------------------------------------------------------*/

   if ( vExtData != NULL ) delete [] vExtData;
   if ( vBufData != NULL ) delete [] vBufData;
   return(relaxError); 
}

/******************************************************************************
 * set SGS parameters
 *---------------------------------------------------------------------------*/

int MLI_Solver_HSGS::setParams( char *paramString, int argc, char **argv )
{
   int    i;
   double *weights;
   char   param1[100];

   sscanf(paramString, "%s", param1);
   if ( !strcmp(param1, "numSweeps") )
   {
      if ( argc != 1 ) 
      {
         printf("MLI_Solver_HSGS::setParams ERROR : needs 1 arg.\n");
         return 1;
      }
      nSweeps_ = *(int*) argv[0];
      if ( nSweeps_ < 1 ) nSweeps_ = 1;
      if ( relaxWeights_ != NULL ) delete [] relaxWeights_;
      relaxWeights_ = new double[nSweeps_];
      for ( i = 0; i < nSweeps_; i++ ) relaxWeights_[i] = 1.0;
      return 0;
   }
   else if ( !strcmp(param1, "relaxWeight") )
   {
      if ( argc != 2 && argc != 1 ) 
      {
         printf("MLI_Solver_HSGS::setParams ERROR : needs 1 or 2 args.\n");
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
   else if ( !strcmp(param1, "printRNorm") )
   {
      printRNorm_ = 1;
   }
   else if ( !strcmp(param1, "calcOmega") )
   {
      calcOmega_ = 1;
   }
   else
   {   
      printf("MLI_Solver_HSGS::setParams - parameter not recognized.\n");
      printf("                 Params = %s\n", paramString);
      return 1;
   }
   return 0;
}

/******************************************************************************
 * calculate relax weight
 *---------------------------------------------------------------------------*/

int MLI_Solver_HSGS::calcOmega()
{
   int                i, relaxType=6, relaxTypes[2], level=0, numCGSweeps=10;
   double             relaxWt;
   hypre_ParCSRMatrix *A;
   hypre_ParVector    *vTemp;
   hypre_ParAMGData   *amgData;
   MPI_Comm           comm;

   A = (hypre_ParCSRMatrix *) Amat_->getMatrix();
   comm       = hypre_ParCSRMatrixComm(A);
   amgData = (hypre_ParAMGData *) hypre_BoomerAMGCreate();
   amgData->CF_marker_array = new int*[1];
   amgData->CF_marker_array[0] = NULL;
   amgData->A_array = new hypre_ParCSRMatrix*[1];
   amgData->A_array[0] = A;
   vTemp = (hypre_ParVector *) mliVec_->getVector();
   amgData->Vtemp = vTemp;
   relaxTypes[0] = 0;
   relaxTypes[1] = relaxType;
   amgData->grid_relax_type = relaxTypes;
   amgData->smooth_num_levels = 0;
   amgData->smooth_type = 0;
   hypre_BoomerAMGCGRelaxWt((void *) amgData, level, numCGSweeps, &relaxWt);
   for ( i = 0; i < nSweeps_; i++ ) relaxWeights_[i] = relaxWt;
   printf("HSGS : relaxWt = %e\n", relaxWt);
   delete [] amgData->A_array;
   delete [] amgData->CF_marker_array;
   hypre_TFree(amgData);
   return 0;
}

