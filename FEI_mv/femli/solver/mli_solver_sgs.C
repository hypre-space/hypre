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

MLI_Solver_SGS::MLI_Solver_SGS(char *name) : MLI_Solver(name)
{
   Amat_             = NULL;
   zeroInitialGuess_ = 0;
   nSweeps_          = 1;
   relaxWeights_     = new double[1];
   relaxWeights_[0]  = 1.0;
   myColor_          = 0;
   numColors_        = 1;
   scheme_           = 1;
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
   hypre_ParCSRMatrix *A;
   MPI_Comm           comm;

   if      ( scheme_ == 0 ) doProcColoring();
   else if ( scheme_ == 1 ) 
   {
      myColor_   = 0;
      numColors_ = 1;
   }
   else
   {
      A    = (hypre_ParCSRMatrix *) Amat_->getMatrix();
      comm = hypre_ParCSRMatrixComm(A);
      MPI_Comm_size(comm, &numColors_);
      MPI_Comm_rank(comm, &myColor_);
   }
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
   int                 iC, index, nprocs, mypid, nSends, start;
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
   MPI_Comm_rank(comm,&mypid);  

   /*-----------------------------------------------------------------
    * setting up for interprocessor communication
    *-----------------------------------------------------------------*/

   if (nprocs > 1)
   {
      nSends = hypre_ParCSRCommPkgNumSends(commPkg);
      vBufData = new double[hypre_ParCSRCommPkgSendMapStart(commPkg,nSends)];
      vExtData = new double[extNRows];
   }

   /*-----------------------------------------------------------------
    * perform SGS sweeps
    *-----------------------------------------------------------------*/
 
#if 1
   if ( relaxWeights_ != NULL && relaxWeights_[0] != 0.0 && 
        relaxWeights_[0] != 1.0 ) 
      printf("\t SGS smoother : relax weight != 1.0 (%e).\n",relaxWeights_[0]);
#endif

   relaxWeight = 1.0;
   for( is = 0; is < nSweeps_; is++ )
   {
      if ( relaxWeights_ != NULL ) relaxWeight = relaxWeights_[is];
      if ( relaxWeight <= 0.0 ) relaxWeight = 1.0;

      /*-----------------------------------------------------------------
       * forward sweep
       *-----------------------------------------------------------------*/

      for ( iC = 0; iC < numColors_; iC++ )
      {
         if (nprocs > 1)
         {
            if ( zeroInitialGuess_ == 0 )
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

         if ( iC == myColor_ )
         {
            for (i = 0; i < localNRows; i++)
            {
               if ( ADiagA[ADiagI[i]] != zero)
               {
                  res      = fData[i];
                  iStart   = ADiagI[i] + 1;
                  iEnd     = ADiagI[i+1];
                  tmpJ    = &(ADiagJ[iStart]);
                  tmpData = &(ADiagA[iStart]);
                  for (jj = iStart; jj < iEnd; jj++)
                     res -= (*tmpData++) * uData[*tmpJ++];
                  if ( (zeroInitialGuess_ == 0) && (nprocs > 1) )
                  {
                     iStart  = AOffdI[i];
                     iEnd    = AOffdI[i+1];
                     tmpJ    = &(AOffdJ[iStart]);
                     tmpData = &(AOffdA[iStart]);
                     for (jj = iStart; jj < iEnd; jj++)
                        res -= (*tmpData++) * vExtData[*tmpJ++];
                  }
                  uData[i] = relaxWeight * res / ADiagA[ADiagI[i]];
               }
               else printf("MLI_Solver_SGS error : diag = 0.\n");
            }
         }
         zeroInitialGuess_ = 0;
      }

      /*-----------------------------------------------------------------
       * backward sweep
       *-----------------------------------------------------------------*/

      for ( iC = numColors_-1; iC >= 0; iC-- )
      {
         if ( (numColors_ > 1) && (nprocs > 1) )
         {
            if ( zeroInitialGuess_ == 0 )
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

         if ( iC == myColor_ )
         {
            for (i = localNRows-1; i > -1; i--)
            {
               if ( ADiagA[ADiagI[i]] != zero)
               {
                  res     = fData[i];
                  iStart  = ADiagI[i] + 1;
                  iEnd    = ADiagI[i+1];
                  tmpJ    = &(ADiagJ[iStart]);
                  tmpData = &(ADiagA[iStart]);
                  for (jj = iStart; jj < iEnd; jj++)
                     res -= (*tmpData++) * uData[*tmpJ++];
                  if ( (zeroInitialGuess_ == 0 ) && (nprocs > 1) )
                  {
                     iStart  = AOffdI[i];
                     iEnd    = AOffdI[i+1];
                     tmpJ    = &(AOffdJ[iStart]);
                     tmpData = &(AOffdA[iStart]);
                     for (jj = iStart; jj < iEnd; jj++)
                        res -= (*tmpData++) * vExtData[*tmpJ++];
                  }
                  uData[i] = relaxWeight * res / ADiagA[ADiagI[i]];
               }
            }
         }
      }
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
   char   param1[100], param2[100];

   sscanf(paramString, "%s", param1);
   if ( !strcmp(param1, "numSweeps") )
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
   else if ( !strcmp(param1, "relaxWeight") )
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
   else if ( !strcmp(param1, "zeroInitialGuess") )
   {
      zeroInitialGuess_ = 1;
      return 0;
   }
   else if ( !strcmp(param1, "setScheme") )
   {
      sscanf(paramString, "%s %s", param1, param2);
      if      ( !strcmp(param2, "multicolor") ) scheme_ = 0;
      else if ( !strcmp(param2, "parallel") )   scheme_ = 1;
      else if ( !strcmp(param2, "sequential") ) scheme_ = 2;
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
      printf("MLI_Solver_SGS::setParams - relax_weights set to 1.0.\n");
      for ( i = 0; i < nsweeps; i++ ) relaxWeights_[i] = 1.0;
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
            relaxWeights_[i] = 1.0;
         }
      }
   }
   return 0;
}

/******************************************************************************
 * color processors
 *---------------------------------------------------------------------------*/

int MLI_Solver_SGS::doProcColoring()
{
   int                 nSends, *sendProcs, mypid, nprocs, *commGraphI;
   int                 *commGraphJ, *recvCounts, i, j, *colors, *colorsAux;
   int                 pIndex, pColor;
   MPI_Comm            comm;
   hypre_ParCSRMatrix  *A;
   hypre_ParCSRCommPkg *commPkg;

   A       = (hypre_ParCSRMatrix *) Amat_->getMatrix();
   comm    = hypre_ParCSRMatrixComm(A);
   commPkg = hypre_ParCSRMatrixCommPkg(A);
   if ( commPkg == NULL )
   {
      hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) A);
      commPkg = hypre_ParCSRMatrixCommPkg(A);
   }
   nSends    = hypre_ParCSRCommPkgNumSends(commPkg);
   sendProcs = hypre_ParCSRCommPkgSendProcs(commPkg);

   MPI_Comm_rank(comm, &mypid);
   MPI_Comm_size(comm, &nprocs);

   commGraphI = new int[nprocs+1]; 
   recvCounts = new int[nprocs];
   MPI_Allgather(&nSends, 1, MPI_INT, recvCounts, 1, MPI_INT, comm);
   commGraphI[0] = 0;
   for ( i = 1; i <= nprocs; i++ )
      commGraphI[i] = commGraphI[i-1] + recvCounts[i-1];
   commGraphJ = new int[commGraphI[nprocs]]; 
   MPI_Allgatherv(sendProcs, nSends, MPI_INT, commGraphJ,
                  recvCounts, commGraphI, MPI_INT, comm);
   delete [] recvCounts;

#if 0
   if ( mypid == 0 )
   {
      for ( i = 0; i < nprocs; i++ )
         for ( j = commGraphI[i]; j < commGraphI[i+1]; j++ )
            printf("Graph(%d,%d)\n", i, commGraphJ[j]);
   }
#endif

   colors = new int[nprocs];
   colorsAux = new int[nprocs];
   for ( i = 0; i < nprocs; i++ ) colors[i] = colorsAux[i] = -1;
   for ( i = 0; i < nprocs; i++ )
   {
      for ( j = commGraphI[i]; j < commGraphI[i+1]; j++ )
      {
         pIndex = commGraphJ[j];
         pColor = colors[pIndex];
         if ( pColor >= 0 ) colorsAux[pColor] = 1;
      }
      for ( j = 0; j < nprocs; j++ ) 
         if ( colorsAux[j] < 0 ) break;
      colors[i] = j;
      for ( j = commGraphI[i]; j < commGraphI[i+1]; j++ )
      {
         pIndex = commGraphJ[j];
         pColor = colors[pIndex];
         if ( pColor >= 0 ) colorsAux[pColor] = -1;
      }
   }
   delete [] colorsAux;
   myColor_ = colors[mypid];
   numColors_ = 0;
   for ( j = 0; j < nprocs; j++ ) 
      if ( colors[j]+1 > numColors_ ) numColors_ = colors[j]+1;
   delete [] colors;
printf("MLI_Solver_SGS : number of colors = %d\n", numColors_);
   return 0;
}

