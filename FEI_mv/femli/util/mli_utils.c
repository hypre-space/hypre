/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * Utilities functions 
 *
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * include files 
 *--------------------------------------------------------------------------*/

#include <assert.h>
#include <math.h>
#include "HYPRE.h"
#include "util/mli_utils.h"
#include "IJ_mv/HYPRE_IJ_mv.h"
/*
#include <mpi.h>
*/

/*--------------------------------------------------------------------------
 * external function 
 *--------------------------------------------------------------------------*/

#ifdef __cplusplus
extern "C" {
#else 
extern
#endif
int hypre_BoomerAMGBuildCoarseOperator(hypre_ParCSRMatrix*,hypre_ParCSRMatrix*,
                                    hypre_ParCSRMatrix *,hypre_ParCSRMatrix **);
void qsort1(int *, double *, int, int);
int  MLI_Utils_IntTreeUpdate(int treeLeng, int *tree,int *treeInd);

#ifdef __cplusplus
}
#endif

#define habs(x) (((x) > 0) ? x : -(x))

/*****************************************************************************
 * destructor for hypre_ParCSRMatrix conforming to MLI requirements 
 *--------------------------------------------------------------------------*/

int MLI_Utils_HypreParCSRMatrixGetDestroyFunc( MLI_Function *funcPtr )
{
   funcPtr->func_ = (int (*)(void *)) hypre_ParCSRMatrixDestroy;
   return 0;
}

/*****************************************************************************
 * destructor for hypre_CSRMatrix conforming to MLI requirements 
 *--------------------------------------------------------------------------*/

int MLI_Utils_HypreCSRMatrixGetDestroyFunc( MLI_Function *funcPtr )
{
   funcPtr->func_ = (int (*)(void *)) hypre_CSRMatrixDestroy;
   return 0;
}

/*****************************************************************************
 * destructor for hypre_ParVector conforming to MLI requirements 
 *--------------------------------------------------------------------------*/

int MLI_Utils_HypreParVectorGetDestroyFunc( MLI_Function *funcPtr )
{
   funcPtr->func_ = (int (*)(void *)) hypre_ParVectorDestroy;
   return 0;
}

/*****************************************************************************
 * destructor for hypre_Vector conforming to MLI requirements 
 *--------------------------------------------------------------------------*/

int MLI_Utils_HypreVectorGetDestroyFunc( MLI_Function *funcPtr )
{
   funcPtr->func_ = (int (*)(void *)) hypre_SeqVectorDestroy;
   return 0;
}

/***************************************************************************
 * FormJacobi ( Jmat = I - alpha * Amat )
 *--------------------------------------------------------------------------*/

int MLI_Utils_HypreMatrixFormJacobi(void *A, double alpha, void **J)
{
   int                *rowPart, mypid, nprocs;
   int                localNRows, startRow, ierr, irow, *rowLengths;
   int                rownum, rowSize, *colInd, *newColInd, newRowSize;
   int                icol, maxnnz;
   double             *colVal, *newColVal;
   MPI_Comm           comm;
   HYPRE_IJMatrix     IJmat;
   hypre_ParCSRMatrix *Amat, *Jmat;

   /* -----------------------------------------------------------------------
    * get matrix parameters
    * ----------------------------------------------------------------------*/

   Amat = (hypre_ParCSRMatrix *) A;
   comm = hypre_ParCSRMatrixComm(Amat);
   MPI_Comm_rank(comm, &mypid);
   MPI_Comm_size(comm, &nprocs);
   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix)Amat,&rowPart);
   localNRows = rowPart[mypid+1] - rowPart[mypid];
   startRow   = rowPart[mypid];

   /* -----------------------------------------------------------------------
    * initialize new matrix
    * ----------------------------------------------------------------------*/

   ierr =  HYPRE_IJMatrixCreate(comm, startRow, startRow+localNRows-1, 
                                startRow, startRow+localNRows-1, &IJmat);
   ierr += HYPRE_IJMatrixSetObjectType(IJmat, HYPRE_PARCSR);
   assert( !ierr );
   maxnnz = 0;
   rowLengths = (int *) calloc( localNRows, sizeof(int) );
   if ( rowLengths == NULL ) 
   {
      printf("FormJacobi ERROR : memory allocation.\n");
      exit(1);
   }
   for ( irow = 0; irow < localNRows; irow++ )
   {
      rownum = startRow + irow; 
      hypre_ParCSRMatrixGetRow(Amat, rownum, &rowSize, &colInd, NULL);
      rowLengths[irow] = rowSize;
      if ( rowSize <= 0 )
      {
         printf("FormJacobi ERROR : Amat has rowSize <= 0 (%d)\n", rownum);
         exit(1);
      }
      for ( icol = 0; icol < rowSize; icol++ )
         if ( colInd[icol] == rownum ) break;
      if ( icol == rowSize ) rowLengths[irow]++;
      hypre_ParCSRMatrixRestoreRow(Amat, rownum, &rowSize, &colInd, NULL);
      maxnnz = ( rowLengths[irow] > maxnnz ) ? rowLengths[irow] : maxnnz;
   }
   ierr = HYPRE_IJMatrixSetRowSizes(IJmat, rowLengths);
   assert( !ierr );
   HYPRE_IJMatrixInitialize(IJmat);

   /* -----------------------------------------------------------------------
    * load the new matrix 
    * ----------------------------------------------------------------------*/

   newColInd = (int *) calloc( maxnnz, sizeof(int) );
   newColVal = (double *) calloc( maxnnz, sizeof(double) );

   for ( irow = 0; irow < localNRows; irow++ )
   {
      rownum = startRow + irow; 
      hypre_ParCSRMatrixGetRow(Amat, rownum, &rowSize, &colInd, &colVal);
      for ( icol = 0; icol < rowSize; icol++ )
      {
         newColInd[icol] = colInd[icol];
#if 0
// Note : no diagonal scaling
newColVal[icol] = - alpha * colVal[icol] / colVal[0];
if ( colInd[icol] == rownum ) newColVal[icol] = 1.0 - alpha;
#endif
         newColVal[icol] = - alpha * colVal[icol];
         if ( colInd[icol] == rownum ) newColVal[icol] += 1.0;
      } 
      newRowSize = rowSize;
      if ( rowLengths[irow] == rowSize+1 ) 
      {
         newColInd[newRowSize] = rownum;
         newColVal[newRowSize++] = 1.0;
      }
      hypre_ParCSRMatrixRestoreRow(Amat, rownum, &rowSize, &colInd, &colVal);
      HYPRE_IJMatrixSetValues(IJmat, 1, &newRowSize,(const int *) &rownum,
                (const int *) newColInd, (const double *) newColVal);
   }
   HYPRE_IJMatrixAssemble(IJmat);

   /* -----------------------------------------------------------------------
    * create new MLI_matrix and then clean up
    * ----------------------------------------------------------------------*/

   HYPRE_IJMatrixGetObject(IJmat, (void **) &Jmat);
   HYPRE_IJMatrixSetObjectType(IJmat, -1);
   HYPRE_IJMatrixDestroy(IJmat);
   hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) Jmat);
   (*J) = (void *) Jmat;

   free( newColInd );
   free( newColVal );
   free( rowLengths );
   free( rowPart );
   return 0;
}

/***************************************************************************
 * Given a local degree of freedom, construct an array for that for all
 *--------------------------------------------------------------------------*/
 
int MLI_Utils_GenPartition(MPI_Comm comm, int nlocal, int **rowPart)
{
   int i, nprocs, mypid, *garray, count=0, count2;
 
   MPI_Comm_rank(comm, &mypid);
   MPI_Comm_size(comm, &nprocs);
   garray = (int *) calloc( nprocs+1, sizeof(int) );
   garray[mypid] = nlocal;
   MPI_Allgather(&nlocal, 1, MPI_INT, garray, 1, MPI_INT, comm);
   count = 0;
   for ( i = 0; i < nprocs; i++ )
   {
      count2 = garray[i];
      garray[i] = count;
      count += count2;
   }
   garray[nprocs] = count;
   (*rowPart) = garray;
   return 0;
}

/***************************************************************************
 * Given a matrix, find its maximum eigenvalue
 *--------------------------------------------------------------------------*/

int MLI_Utils_ComputeSpectralRadius(hypre_ParCSRMatrix *Amat, double *maxEigen)
{
   int             mypid, nprocs, *partition, startRow, endRow;
   int             it, maxits=20, ierr;
   double          norm2, lambda;
   MPI_Comm        comm;
   HYPRE_IJVector  IJvec1, IJvec2;
   HYPRE_ParVector vec1, vec2;

   /* -----------------------------------------------------------------
    * fetch matrix paramters
    * ----------------------------------------------------------------*/

   comm = hypre_ParCSRMatrixComm(Amat);
   MPI_Comm_rank( comm, &mypid );
   MPI_Comm_size( comm, &nprocs );
   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix)Amat,&partition);
   startRow    = partition[mypid];
   endRow      = partition[mypid+1];
   free( partition );

   /* -----------------------------------------------------------------
    * create two temporary vectors
    * ----------------------------------------------------------------*/

   ierr =  HYPRE_IJVectorCreate(comm, startRow, endRow-1, &IJvec1);
   ierr += HYPRE_IJVectorSetObjectType(IJvec1, HYPRE_PARCSR);
   ierr += HYPRE_IJVectorInitialize(IJvec1);
   ierr += HYPRE_IJVectorAssemble(IJvec1);
   ierr += HYPRE_IJVectorCreate(comm, startRow, endRow-1, &IJvec2);
   ierr += HYPRE_IJVectorSetObjectType(IJvec2, HYPRE_PARCSR);
   ierr += HYPRE_IJVectorInitialize(IJvec2);
   ierr += HYPRE_IJVectorAssemble(IJvec2);

   /* -----------------------------------------------------------------
    * perform the power iterations
    * ----------------------------------------------------------------*/
 
   ierr += HYPRE_IJVectorGetObject(IJvec1, (void **) &vec1);
   ierr += HYPRE_IJVectorGetObject(IJvec2, (void **) &vec2);
   assert(!ierr);
   HYPRE_ParVectorSetRandomValues( vec1, 2934731 );
   HYPRE_ParCSRMatrixMatvec(1.0,(HYPRE_ParCSRMatrix) Amat,vec1,0.0,vec2 );
   HYPRE_ParVectorInnerProd( vec2, vec2, &norm2);
   for ( it = 0; it < maxits; it++ )
   {
      HYPRE_ParVectorInnerProd( vec2, vec2, &norm2);
      HYPRE_ParVectorCopy( vec2, vec1);
      norm2 = 1.0 / sqrt(norm2);
      HYPRE_ParVectorScale( norm2, vec1 );
      HYPRE_ParCSRMatrixMatvec(1.0,(HYPRE_ParCSRMatrix) Amat,vec1,0.0,vec2 );
      HYPRE_ParVectorInnerProd( vec1, vec2, &lambda);
   }
   (*maxEigen) = lambda*1.05;
   HYPRE_IJVectorDestroy(IJvec1);
   HYPRE_IJVectorDestroy(IJvec2);
   return 0;
} 

/******************************************************************************
 * compute Ritz Values that approximates extreme eigenvalues
 *--------------------------------------------------------------------------*/

int MLI_Utils_ComputeExtremeRitzValues(hypre_ParCSRMatrix *A, double *ritz, 
                                       int scaleFlag)
{
   int      i, j, k, its, maxIter, nprocs, mypid, localNRows, globalNRows;
   int      startRow, endRow, *partition, *ADiagI;
   double   alpha, beta, rho, rhom1, sigma, offdiagNorm, *zData;
   double   rnorm, *alphaArray, *rnormArray, **Tmat, initOffdiagNorm;
   double   app, aqq, arr, ass, apq, sign, tau, t, c, s;
   double   *ADiagA, one=1.0, *rData;
   MPI_Comm comm;
   hypre_CSRMatrix *ADiag;
   hypre_ParVector *rVec, *zVec, *pVec, *apVec;

   /*-----------------------------------------------------------------
    * fetch matrix information 
    *-----------------------------------------------------------------*/

   comm = hypre_ParCSRMatrixComm(A);
   MPI_Comm_rank(comm,&mypid);  
   MPI_Comm_size(comm,&nprocs);  

   ADiag      = hypre_ParCSRMatrixDiag(A);
   ADiagA     = hypre_CSRMatrixData(ADiag);
   ADiagI     = hypre_CSRMatrixI(ADiag);
   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) A, &partition);
   startRow    = partition[mypid];
   endRow      = partition[mypid+1] - 1;
   globalNRows = partition[nprocs];
   localNRows  = endRow - startRow + 1;
   hypre_TFree( partition );
   maxIter     = 5;
   if ( globalNRows < maxIter ) maxIter = globalNRows;
   ritz[0] = ritz[1] = 0.0;

   /*-----------------------------------------------------------------
    * allocate space
    *-----------------------------------------------------------------*/

   if ( localNRows > 0 )
   {
      HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix)A,&partition);
      rVec = hypre_ParVectorCreate(comm, globalNRows, partition);
      hypre_ParVectorInitialize(rVec);
      HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix)A,&partition);
      zVec = hypre_ParVectorCreate(comm, globalNRows, partition);
      hypre_ParVectorInitialize(zVec);
      HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix)A,&partition);
      pVec = hypre_ParVectorCreate(comm, globalNRows, partition);
      hypre_ParVectorInitialize(pVec);
      HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix)A,&partition);
      apVec = hypre_ParVectorCreate(comm, globalNRows, partition);
      hypre_ParVectorInitialize(apVec);
      zData  = hypre_VectorData( hypre_ParVectorLocalVector(zVec) );
      rData  = hypre_VectorData( hypre_ParVectorLocalVector(rVec) );
   }
   HYPRE_ParVectorSetRandomValues((HYPRE_ParVector) rVec, 1209873 );
   alphaArray = (double  *) malloc( (maxIter+1) * sizeof(double) );
   rnormArray = (double  *) malloc( (maxIter+1) * sizeof(double) );
   Tmat       = (double **) malloc( (maxIter+1) * sizeof(double*) );
   for ( i = 0; i <= maxIter; i++ )
   {
      Tmat[i] = (double *) malloc( (maxIter+1) * sizeof(double) );
      for ( j = 0; j <= maxIter; j++ ) Tmat[i][j] = 0.0;
      Tmat[i][i] = 1.0;
   }

   /*-----------------------------------------------------------------
    * compute initial residual vector norm 
    *-----------------------------------------------------------------*/

   hypre_ParVectorSetRandomValues(rVec, 1209837);
   hypre_ParVectorSetConstantValues(pVec, 0.0);
   hypre_ParVectorSetConstantValues(zVec, 0.0);
   rho = hypre_ParVectorInnerProd(rVec, rVec); 
   rnorm = sqrt(rho);
   rnormArray[0] = rnorm;
   if ( rnorm == 0.0 )
   {
      printf("MLI_Utils_ComputeExtremeRitzValues : fail for res=0.\n");
      hypre_ParVectorDestroy( rVec );
      hypre_ParVectorDestroy( pVec );
      hypre_ParVectorDestroy( zVec );
      hypre_ParVectorDestroy( apVec );
      return 1;
   }

   /*-----------------------------------------------------------------
    * main loop 
    *-----------------------------------------------------------------*/

   for ( its = 0; its < maxIter; its++ )
   {
      if ( scaleFlag )
      {
         for ( i = 0; i < localNRows; i++ )
            if (ADiagA[ADiagI[i]] != 0.0) 
               zData[i] = rData[i]/sqrt(ADiagA[ADiagI[i]]);
      }
      else 
         for ( i = 0; i < localNRows; i++ ) zData[i] = rData[i];

      rhom1 = rho;
      rho = hypre_ParVectorInnerProd(rVec, zVec); 
      if (its == 0) beta = 0.0;
      else 
      {
         beta = rho / rhom1;
         Tmat[its-1][its] = -beta;
      }
      HYPRE_ParVectorScale( beta, (HYPRE_ParVector) pVec );
      hypre_ParVectorAxpy( one, zVec, pVec );
      hypre_ParCSRMatrixMatvec(one, A, pVec, 0.0, apVec);
      sigma = hypre_ParVectorInnerProd(pVec, apVec); 
      alpha  = rho / sigma;
      alphaArray[its] = sigma;
      hypre_ParVectorAxpy( -alpha, apVec, rVec );
      rnorm = sqrt(hypre_ParVectorInnerProd(rVec, rVec)); 
      rnormArray[its+1] = rnorm;
      if ( rnorm < 1.0E-8 * rnormArray[0] )
      {
         maxIter = its + 1;
         break;
      }
   }

   /*-----------------------------------------------------------------
    * construct T 
    *-----------------------------------------------------------------*/

   Tmat[0][0] = alphaArray[0];
   for ( i = 1; i < maxIter; i++ )
      Tmat[i][i]=alphaArray[i]+alphaArray[i-1]*Tmat[i-1][i]*Tmat[i-1][i];

   for ( i = 0; i < maxIter; i++ )
   {
      Tmat[i][i+1] *= alphaArray[i];
      Tmat[i+1][i] = Tmat[i][i+1];
      rnormArray[i] = 1.0 / rnormArray[i];
   }
   for ( i = 0; i < maxIter; i++ )
      for ( j = 0; j < maxIter; j++ )
         Tmat[i][j] = Tmat[i][j] * rnormArray[i] * rnormArray[j];

   /* ----------------------------------------------------------------*/
   /* diagonalize T using Jacobi iteration                            */
   /* ----------------------------------------------------------------*/

   offdiagNorm = 0.0;
   for ( i = 0; i < maxIter; i++ )
      for ( j = 0; j < i; j++ ) offdiagNorm += (Tmat[i][j] * Tmat[i][j]);
   offdiagNorm *= 2.0;
   initOffdiagNorm = offdiagNorm;

   while ( offdiagNorm > initOffdiagNorm * 1.0E-8 )
   {
      for ( i = 1; i < maxIter; i++ )
      {
         for ( j = 0; j < i; j++ )
         {
            apq = Tmat[i][j];
            if ( apq != 0.0 )
            {
               app = Tmat[j][j];
               aqq = Tmat[i][i];
               tau = ( aqq - app ) / (2.0 * apq);
               sign = (tau >= 0.0) ? 1.0 : -1.0;
               t  = sign / (tau * sign + sqrt(1.0 + tau * tau));
               c  = 1.0 / sqrt( 1.0 + t * t );
               s  = t * c;
               for ( k = 0; k < maxIter; k++ )
               {
                  arr = Tmat[j][k];
                  ass = Tmat[i][k];
                  Tmat[j][k] = c * arr - s * ass;
                  Tmat[i][k] = s * arr + c * ass;
               }
               for ( k = 0; k < maxIter; k++ )
               {
                  arr = Tmat[k][j];
                  ass = Tmat[k][i];
                  Tmat[k][j] = c * arr - s * ass;
                  Tmat[k][i] = s * arr + c * ass;
               }
            }
         }
      }
      offdiagNorm = 0.0;
      for ( i = 0; i < maxIter; i++ )
         for ( j = 0; j < i; j++ ) offdiagNorm += (Tmat[i][j] * Tmat[i][j]);
      offdiagNorm *= 2.0;
   }

   /* ----------------------------------------------------------------
    * search for max and min eigenvalues
    * ----------------------------------------------------------------*/

   t = Tmat[0][0];
   for (i = 1; i < maxIter; i++) t = (Tmat[i][i] > t) ? Tmat[i][i] : t;
   ritz[0] = t * 1.1;
   t = Tmat[0][0];
   for (i = 1; i < maxIter; i++) t = (Tmat[i][i] < t) ? Tmat[i][i] : t;
   ritz[1] = t / 1.1;

   /* ----------------------------------------------------------------*
    * de-allocate storage for temporary vectors
    * ----------------------------------------------------------------*/

   if ( localNRows > 0 )
   {
      hypre_ParVectorDestroy( rVec );
      hypre_ParVectorDestroy( zVec );
      hypre_ParVectorDestroy( pVec );
      hypre_ParVectorDestroy( apVec );
   }
   free(alphaArray);
   free(rnormArray);
   for (i = 0; i <= maxIter; i++) if ( Tmat[i] != NULL ) free( Tmat[i] );
   free(Tmat);
   return 0;
}

/******************************************************************************
 * compute matrix max norm
 *--------------------------------------------------------------------------*/

int MLI_Utils_ComputeMatrixMaxNorm(hypre_ParCSRMatrix *A, double *norm,
                                   int scaleFlag)
{
   int             i, j, iStart, iEnd, localNRows, *ADiagI, *AOffdI;
   int             mypid;
   double          *ADiagA, *AOffdA, maxVal, rowSum, dtemp;
   hypre_CSRMatrix *ADiag, *AOffd;
   MPI_Comm        comm;

   /*-----------------------------------------------------------------
    * fetch machine and smoother parameters
    *-----------------------------------------------------------------*/

   ADiag      = hypre_ParCSRMatrixDiag(A);
   ADiagA     = hypre_CSRMatrixData(ADiag);
   ADiagI     = hypre_CSRMatrixI(ADiag);
   AOffd      = hypre_ParCSRMatrixDiag(A);
   AOffdA     = hypre_CSRMatrixData(AOffd);
   AOffdI     = hypre_CSRMatrixI(AOffd);
   localNRows = hypre_CSRMatrixNumRows(ADiag);
   comm       = hypre_ParCSRMatrixComm(A);
   MPI_Comm_rank(comm,&mypid);  
   
   maxVal = 0.0;
   for (i = 0; i < localNRows; i++)
   {
      rowSum = 0.0;
      iStart = ADiagI[i];
      iEnd   = ADiagI[i+1];
      for (j = iStart; j < iEnd; j++) rowSum += habs(ADiagA[j]);
      iStart = AOffdI[i];
      iEnd   = AOffdI[i+1];
      for (j = iStart; j < iEnd; j++) rowSum += habs(AOffdA[j]);
      if ( scaleFlag == 1 )
      {
         if ( ADiagA[ADiagI[i]] == 0.0)
            printf("MLI_Utils_ComputeMatrixMaxNorm - zero diagonal.\n");
         else rowSum /= ADiagA[ADiagI[i]];
      }
      if ( rowSum > maxVal ) maxVal = rowSum;
   }
   MPI_Allreduce(&maxVal, &dtemp, 1, MPI_DOUBLE, MPI_MAX, comm);
   (*norm) = dtemp;
   return 0;
}

/***************************************************************************
 * Given a local degree of freedom, construct an array for that for all
 *--------------------------------------------------------------------------*/
 
double MLI_Utils_WTime()
{
   clock_t ticks;
   double  seconds;
   ticks   = clock() ;
   seconds = (double) ticks / (double) CLOCKS_PER_SEC;
   return seconds;
}

/***************************************************************************
 * Given a Hypre ParCSR matrix, output the matrix to a file
 *--------------------------------------------------------------------------*/
 
int MLI_Utils_HypreMatrixPrint(void *in_mat, char *name)
{
   MPI_Comm comm;
   int      i, mypid, localNRows, startRow, *rowPart, rowSize;
   int      j, *colInd, nnz;
   double   *colVal;
   char     fname[200];
   FILE     *fp;
   hypre_ParCSRMatrix *mat;
   HYPRE_ParCSRMatrix hypre_mat;

   mat       = (hypre_ParCSRMatrix *) in_mat;
   hypre_mat = (HYPRE_ParCSRMatrix) mat;
   comm = hypre_ParCSRMatrixComm(mat);  
   MPI_Comm_rank( comm, &mypid );
   HYPRE_ParCSRMatrixGetRowPartitioning( hypre_mat, &rowPart);
   localNRows  = rowPart[mypid+1] - rowPart[mypid];
   startRow    = rowPart[mypid];
   free( rowPart );

   sprintf(fname, "%s.%d", name, mypid);
   fp = fopen( fname, "w");
   nnz = 0;
   for ( i = startRow; i < startRow+localNRows; i++ )
   {
      HYPRE_ParCSRMatrixGetRow(hypre_mat, i, &rowSize, &colInd, NULL);
      nnz += rowSize;
      HYPRE_ParCSRMatrixRestoreRow(hypre_mat, i, &rowSize, &colInd, NULL);
   }
   fprintf(fp, "%6d  %7d \n", localNRows, nnz);
   for ( i = startRow; i < startRow+localNRows; i++ )
   {
      HYPRE_ParCSRMatrixGetRow(hypre_mat, i, &rowSize, &colInd, &colVal);
      for ( j = 0; j < rowSize; j++ )
         fprintf(fp, "%6d  %6d  %25.16e \n", i+1, colInd[j]+1, colVal[j]);
      HYPRE_ParCSRMatrixRestoreRow(hypre_mat, i, &rowSize, &colInd, &colVal);
   }
   fclose(fp);
   return 0;
}

/***************************************************************************
 * Given 2 Hypre ParCSR matrix A and P, create trans(P) * A * P 
 *--------------------------------------------------------------------------*/
 
int MLI_Utils_HypreMatrixComputeRAP(void *Pmat, void *Amat, void **RAPmat) 
{
   hypre_ParCSRMatrix *hypreP, *hypreA, *hypreRAP;
   hypreP = (hypre_ParCSRMatrix *) Pmat;
   hypreA = (hypre_ParCSRMatrix *) Amat;
   hypre_BoomerAMGBuildCoarseOperator(hypreP, hypreA, hypreP, &hypreRAP);
   (*RAPmat) = (void *) hypreRAP;
   return 0;
}

/***************************************************************************
 * Get matrix information of a Hypre ParCSR matrix 
 *--------------------------------------------------------------------------*/
 
int MLI_Utils_HypreMatrixGetInfo(void *Amat, int *matInfo, double *valInfo)
{
   int      mypid, nprocs, icol, isum[4], ibuf[4], *partition, thisNnz;
   int      localNRows, irow, rownum, rowsize, *colind, startrow;
   int      globalNRows, maxNnz, minNnz, totalNnz;
   double   *colval, dsum[2], dbuf[2], maxVal, minVal;
   MPI_Comm mpiComm;
   hypre_ParCSRMatrix *hypreA;

   hypreA = (hypre_ParCSRMatrix *) Amat;
   mpiComm = hypre_ParCSRMatrixComm(hypreA);
   MPI_Comm_rank( mpiComm, &mypid);
   MPI_Comm_size( mpiComm, &nprocs);
   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) hypreA,&partition);
   localNRows  = partition[mypid+1] - partition[mypid];
   startrow    = partition[mypid];
   globalNRows = partition[nprocs];
   free( partition );
   maxVal  = -1.0E-30;
   minVal  = +1.0E30;
   maxNnz  = 0;
   minNnz  = 1000000;
   thisNnz = 0;
   for ( irow = 0; irow < localNRows; irow++ )
   {
      rownum = startrow + irow;
      hypre_ParCSRMatrixGetRow(hypreA,rownum,&rowsize,&colind,&colval);
      for ( icol = 0; icol < rowsize; icol++ )
      {
         if ( colval[icol] > maxVal ) maxVal = colval[icol];
         if ( colval[icol] < minVal ) minVal = colval[icol];
      }
      if ( rowsize > maxNnz ) maxNnz = rowsize;
      if ( rowsize < minNnz ) minNnz = rowsize;
      thisNnz += rowsize;
      hypre_ParCSRMatrixRestoreRow(hypreA,rownum,&rowsize,&colind,&colval);
   }
   dsum[0] = maxVal;
   dsum[1] = - minVal;
   MPI_Allreduce( dsum, dbuf, 2, MPI_DOUBLE, MPI_MAX, mpiComm );
   maxVal  = dbuf[0];
   minVal  = - dbuf[1];
   isum[0] = maxNnz;
   isum[1] = - minNnz;
   MPI_Allreduce( isum, ibuf, 2, MPI_INT, MPI_MAX, mpiComm );
   maxNnz  = ibuf[0];
   minNnz  = - ibuf[1];
   MPI_Allreduce( &thisNnz, &totalNnz, 1, MPI_INT, MPI_SUM, mpiComm );
   matInfo[0] = globalNRows;
   matInfo[1] = maxNnz;
   matInfo[2] = minNnz;
   matInfo[3] = totalNnz;
   valInfo[0] = maxVal;
   valInfo[1] = minVal;
   return 0;
}

/***************************************************************************
 * Given a Hypre ParCSR matrix, compress it
 *--------------------------------------------------------------------------*/
 
int MLI_Utils_HypreMatrixCompress(void *Amat, int blksize, void **Amat2) 
{
   int                mypid, *partition, startRow, localNRows;
   int                newLNRows, newStartRow;
   int                ierr, *rowLengths, irow, rowNum, rowSize, *colInd;
   int                *newInd, newSize, j, k, nprocs;
   double             *colVal, *newVal;
   MPI_Comm           mpiComm;
   hypre_ParCSRMatrix *hypreA, *hypreA2;
   HYPRE_IJMatrix     IJAmat2;

   /* ----------------------------------------------------------------
    * fetch information about incoming matrix
    * ----------------------------------------------------------------*/
   
   hypreA  = (hypre_ParCSRMatrix *) Amat;
   mpiComm = hypre_ParCSRMatrixComm(hypreA);
   MPI_Comm_rank(mpiComm, &mypid);
   MPI_Comm_size(mpiComm, &nprocs);
   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) hypreA,&partition);
   startRow    = partition[mypid];
   localNRows  = partition[mypid+1] - startRow;
   free( partition );
   if ( localNRows % blksize != 0 )
   {
      printf("MLI_CompressMatrix ERROR : nrows not divisible by blksize.\n");
      printf("                nrows, blksize = %d %d\n",localNRows,blksize);
      exit(1);
   }

   /* ----------------------------------------------------------------
    * compute size of new matrix and create the new matrix
    * ----------------------------------------------------------------*/

   newLNRows   = localNRows / blksize;
   newStartRow = startRow / blksize;
   ierr =  HYPRE_IJMatrixCreate(mpiComm, newStartRow, 
                  newStartRow+newLNRows-1, newStartRow,
                  newStartRow+newLNRows-1, &IJAmat2);
   ierr += HYPRE_IJMatrixSetObjectType(IJAmat2, HYPRE_PARCSR);
   assert(!ierr);

   /* ----------------------------------------------------------------
    * compute the row lengths of the new matrix
    * ----------------------------------------------------------------*/

   if (newLNRows > 0) rowLengths = (int *) malloc(newLNRows*sizeof(int));
   else               rowLengths = NULL;

   for ( irow = 0; irow < newLNRows; irow++ )
   {
      rowLengths[irow] = 0;
      for ( j = 0; j < blksize; j++)
      {
         rowNum = startRow + irow * blksize + j;
         hypre_ParCSRMatrixGetRow(hypreA,rowNum,&rowSize,&colInd,NULL);
         rowLengths[irow] += rowSize;
         hypre_ParCSRMatrixRestoreRow(hypreA,rowNum,&rowSize,&colInd,NULL);
      }
   }
   ierr =  HYPRE_IJMatrixSetRowSizes(IJAmat2, rowLengths);
   ierr += HYPRE_IJMatrixInitialize(IJAmat2);
   assert(!ierr);

   /* ----------------------------------------------------------------
    * load the compressed matrix
    * ----------------------------------------------------------------*/

   for ( irow = 0; irow < newLNRows; irow++ )
   {
      newInd  = (int *)    malloc( rowLengths[irow] * sizeof(int) );
      newVal  = (double *) malloc( rowLengths[irow] * sizeof(double) );
      newSize = 0;
      for ( j = 0; j < blksize; j++)
      {
         rowNum = startRow + irow * blksize + j;
         hypre_ParCSRMatrixGetRow(hypreA,rowNum,&rowSize,&colInd,&colVal);
         for ( k = 0; k < rowSize; k++ )
         {
            newInd[newSize] = colInd[k] / blksize;
            newVal[newSize++] = colVal[k];
         }
         hypre_ParCSRMatrixRestoreRow(hypreA,rowNum,&rowSize,
                                      &colInd,&colVal);
      }
      if ( newSize > 0 )
      {
         qsort1(newInd, newVal, 0, newSize-1);
         k = 0;
         newVal[k] = newVal[k] * newVal[k];
         for ( j = 1; j < newSize; j++ )
         {
            if (newInd[j] == newInd[k]) 
               newVal[k] += (newVal[j] * newVal[j]);
            else
            {
               newInd[++k] = newInd[j];
               newVal[k]   = newVal[j] * newVal[j];
            }
         }
         newSize = k + 1;
      }
      rowNum = newStartRow + irow;
      HYPRE_IJMatrixSetValues(IJAmat2, 1, &newSize,(const int *) &rowNum,
                (const int *) newInd, (const double *) newVal);
      free( newInd );
      free( newVal );
   }
   ierr = HYPRE_IJMatrixAssemble(IJAmat2);
   assert( !ierr );
   HYPRE_IJMatrixGetObject(IJAmat2, (void **) &hypreA2);
   /*hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) hypreA2);*/
   HYPRE_IJMatrixSetObjectType( IJAmat2, -1 );
   HYPRE_IJMatrixDestroy( IJAmat2 );
   if ( rowLengths != NULL ) free( rowLengths );
   (*Amat2) = (void *) hypreA2;
   return 0;
}

/***************************************************************************
 * perform QR factorization
 *--------------------------------------------------------------------------*/
 
int MLI_Utils_QR(double *qArray, double *rArray, int nrows, int ncols)
{
   int    icol, irow, pcol;
   double innerProd, *currQ, *currR, *prevQ, alpha;

#ifdef MLI_DEBUG_DETAILED
   printf("(before) QR %6d %6d : \n", nrows, ncols);
   for ( irow = 0; irow < nrows; irow++ )
   {
      for ( icol = 0; icol < ncols; icol++ )
         printf(" %13.5e ", qArray[icol*nrows+irow]);
      printf("\n");
   }
#endif
   for ( icol = 0; icol < ncols; icol++ )
   {
      currQ = &qArray[icol*nrows];
      currR = &rArray[icol*ncols];
      for ( pcol = 0; pcol < icol; pcol++ )
      {
         prevQ = &qArray[pcol*nrows];
         alpha = 0.0;
         for ( irow = 0; irow < nrows; irow++ )
            alpha += (currQ[irow] * prevQ[irow]); 
         currR[pcol] = alpha;
         for ( irow = 0; irow < nrows; irow++ )
            currQ[irow] -= ( alpha * prevQ[irow] ); 
      }
      for ( pcol = icol; pcol < ncols; pcol++ ) currR[pcol] = 0.0;
      innerProd = 0.0;
      for ( irow = 0; irow < nrows; irow++ )
         innerProd += (currQ[irow] * currQ[irow]); 
      innerProd = sqrt( innerProd );
      if ( innerProd < 1.0e-10 ) return (icol+1);
      currR[icol] = innerProd;
      alpha = 1.0 / innerProd;
      for ( irow = 0; irow < nrows; irow++ )
         currQ[irow] = alpha * currQ[irow]; 
   }
#ifdef MLI_DEBUG_DETAILED
   printf("(after ) Q %6d %6d : \n", nrows, ncols);
   for ( irow = 0; irow < nrows; irow++ )
   {
      for ( icol = 0; icol < ncols; icol++ )
         printf(" %13.5e ", qArray[icol*nrows+irow]);
      printf("\n");
   }
   printf("(after ) R %6d %6d : \n", nrows, ncols);
   for ( irow = 0; irow < ncols; irow++ )
   {
      for ( icol = 0; icol < ncols; icol++ )
         printf(" %13.5e ", rArray[icol*ncols+irow]);
      printf("\n");
   }
#endif
   return 0;
}

/***************************************************************************
 * read a matrix file and create a hypre_ParCSRMatrix from it
 *--------------------------------------------------------------------------*/
 
int MLI_Utils_HypreMatrixReadTuminFormat(char *filename, MPI_Comm mpiComm, 
                 int blksize, void **Amat, int scaleFlag, double **scaleVec)
{
   int    mypid, nprocs, currProc, globalNRows, localNRows, startRow;
   int    irow, colNum, *inds, *matIA, *matJA, *tempJA, length, rowNum;
   int    j, nnz, currBufSize, *rowLengths, ierr;
   double colVal, *vals, *matAA, *tempAA, *diag=NULL, *diag2=NULL, scale;
   FILE   *fp;
   hypre_ParCSRMatrix *hypreA;
   HYPRE_IJMatrix     IJmat;

   MPI_Comm_rank( mpiComm, &mypid );
   MPI_Comm_size( mpiComm, &nprocs );
   currProc = 0;
   while ( currProc < nprocs )
   {
      if ( mypid == currProc )
      {
         fp = fopen( filename, "r" );
         if ( fp == NULL )
         {
            printf("MLI_Utils_HypreMatrixReadTuminFormat ERROR : ");
            printf("file %s not found.\n", filename);
            exit(1);
         }
         fscanf( fp, "%d", &globalNRows );
         if ( globalNRows < 0 || globalNRows > 1000000000 )
         {
            printf("MLI_Utils_HypreMatrixRead ERROR : invalid nrows %d.\n",
                   globalNRows);
            exit(1);
         }
         if ( globalNRows % blksize != 0 )
         {
            printf("MLI_Utils_HypreMatrixReadTuminFormat ERROR : ");
            printf("nrows,blksize (%d,%d) mismatch.\n", globalNRows,blksize);
            exit(1);
         }
         localNRows = globalNRows / blksize / nprocs * blksize;
         startRow   = localNRows * mypid;
         if ( mypid == nprocs - 1 ) localNRows = globalNRows - startRow;

         if (scaleFlag) diag = (double *) malloc(sizeof(double)*globalNRows);
         for ( irow = 0; irow < startRow; irow++ )
         {
            fscanf( fp, "%d", &colNum );
            while ( colNum != -1 )
            {
               fscanf( fp, "%lg", &colVal );
               fscanf( fp, "%d", &colNum );
               if ( scaleFlag && colNum == irow ) diag[irow] = colVal;
            } 
         } 

         currBufSize = localNRows * 27;
         matIA = (int *)    malloc((localNRows+1) * sizeof(int));
         matJA = (int *)    malloc(currBufSize * sizeof(int));
         matAA = (double *) malloc(currBufSize * sizeof(double));
         nnz    = 0;
         matIA[0] = nnz;
         for ( irow = startRow; irow < startRow+localNRows; irow++ )
         {
            fscanf( fp, "%d", &colNum );
            while ( colNum != -1 )
            {
               fscanf( fp, "%lg", &colVal );
               matJA[nnz] = colNum;
               matAA[nnz++] = colVal;
               if ( scaleFlag && colNum == irow ) diag[irow] = colVal;
               if ( nnz >= currBufSize )
               {
                  tempJA = matJA;
                  tempAA = matAA;
                  currBufSize += ( 27 * localNRows );
                  matJA = (int *)    malloc(currBufSize * sizeof(int));
                  matAA = (double *) malloc(currBufSize * sizeof(double));
                  for ( j = 0; j < nnz; j++ )
                  {
                     matJA[j] = tempJA[j];
                     matAA[j] = tempAA[j];
                  }
                  free( tempJA );
                  free( tempAA );
               }
               fscanf( fp, "%d", &colNum );
            } 
            matIA[irow-startRow+1] = nnz;
         }
         for ( irow = startRow+localNRows; irow < globalNRows; irow++ )
         {
            fscanf( fp, "%d", &colNum );
            while ( colNum != -1 )
            {
               fscanf( fp, "%lg", &colVal );
               fscanf( fp, "%d", &colNum );
               if ( scaleFlag && colNum == irow ) diag[irow] = colVal;
            } 
         } 
         fclose( fp );
      }
      MPI_Barrier( mpiComm );
      currProc++;
   }
   printf("%5d : MLI_Utils_HypreMatrixReadTuminFormat : nlocal, nnz = %d %d\n", 
          mypid, localNRows, nnz);
   rowLengths = (int *) malloc(localNRows * sizeof(int));
   for ( irow = 0; irow < localNRows; irow++ )
      rowLengths[irow] = matIA[irow+1] - matIA[irow];

   ierr = HYPRE_IJMatrixCreate(mpiComm, startRow, startRow+localNRows-1,
                               startRow, startRow+localNRows-1, &IJmat);
   ierr = HYPRE_IJMatrixSetObjectType(IJmat, HYPRE_PARCSR);
   assert(!ierr);
   ierr = HYPRE_IJMatrixSetRowSizes(IJmat, rowLengths);
   ierr = HYPRE_IJMatrixInitialize(IJmat);
   assert(!ierr);
   for ( irow = 0; irow < localNRows; irow++ )
   {
      length = rowLengths[irow];
      rowNum = irow + startRow;
      inds = &(matJA[matIA[irow]]);
      vals = &(matAA[matIA[irow]]);
      if ( scaleFlag ) 
      {
         scale = 1.0 / sqrt( diag[irow] );
         for ( j = 0; j < length; j++ )
            vals[j] = vals[j] * scale / ( sqrt(diag[inds[j]]) );
      }
      ierr = HYPRE_IJMatrixSetValues(IJmat, 1, &length,(const int *) &rowNum,
                (const int *) inds, (const double *) vals);
      assert( !ierr );
   }
   free( rowLengths );
   free( matIA );
   free( matJA );
   free( matAA );

   ierr = HYPRE_IJMatrixAssemble(IJmat);
   assert( !ierr );
   HYPRE_IJMatrixGetObject(IJmat, (void**) &hypreA);
   HYPRE_IJMatrixSetObjectType(IJmat, -1);
   HYPRE_IJMatrixDestroy(IJmat);
   (*Amat) = (void *) hypreA;
   if ( scaleFlag )
   {
      diag2 = (double *) malloc( sizeof(double) * localNRows);
      for ( irow = 0; irow < localNRows; irow++ )
         diag2[irow] = diag[startRow+irow];
      free( diag );
   }
   (*scaleVec) = diag2;
   return ierr;
}

/***************************************************************************
 * read a matrix file and create a hypre_ParCSRMatrix from it
 *--------------------------------------------------------------------------*/
 
int MLI_Utils_HypreMatrixReadIJAFormat(char *filename, MPI_Comm mpiComm, 
              int blksize, void **Amat, int scaleFlag, double **scaleVec)
{
   int    mypid, nprocs, currProc, globalNRows, localNRows, startRow;
   int    irow, colNum, *inds, *matIA, *matJA, length, rowNum;
   int    j, nnz, currBufSize, *rowLengths, ierr, globalNnz, currRow;
   double colVal, *vals, *matAA, *diag=NULL, *diag2=NULL, scale;
   char   fname[20];
   FILE   *fp;
   hypre_ParCSRMatrix *hypreA;
   HYPRE_IJMatrix     IJmat;

   MPI_Comm_rank( mpiComm, &mypid );
   MPI_Comm_size( mpiComm, &nprocs );
   currProc = 0;
   while ( currProc < nprocs )
   {
      if ( mypid == currProc )
      {
         printf("Processor %d reading matrix file %s.\n", mypid, filename);
         fp = fopen( filename, "r" );
         if ( fp == NULL )
         {
            printf("MLI_Utils_HypreMatrixReadIJAFormat ERROR : ");
            printf("file %s not found.\n", filename);
            exit(1);
         }
         fscanf( fp, "%d %d", &globalNRows, &globalNnz );
         if ( globalNRows < 0 || globalNRows > 1000000000 )
         {
            printf("MLI_Utils_HypreMatrixReadIJAFormat ERROR : ");
            printf("invalid nrows %d.\n", globalNRows);
            exit(1);
         }
         if ( globalNRows % blksize != 0 )
         {
            printf("MLI_Utils_HypreMatrixReadIJAFormat ERROR : nrows,blksize");
            printf("(%d,%d) mismatch.\n", globalNRows, blksize);
            exit(1);
         }
         localNRows = globalNRows / blksize / nprocs * blksize;
         startRow   = localNRows * mypid;
         if ( mypid == nprocs - 1 ) localNRows = globalNRows - startRow;
         currBufSize = globalNnz / nprocs * 3;
         matIA = (int *)    malloc((localNRows+1) * sizeof(int));
         matJA = (int *)    malloc(currBufSize * sizeof(int));
         matAA = (double *) malloc(currBufSize * sizeof(double));

         if (scaleFlag == 1) 
            diag = (double *) malloc(sizeof(double)*globalNRows);
         for ( irow = 0; irow < globalNnz; irow++ )
         {
            fscanf( fp, "%d %d %lg", &rowNum, &colNum, &colVal );
            rowNum--;
            if ( scaleFlag == 1 && rowNum == colNum-1 ) 
               diag[rowNum] = colVal;
            if ( rowNum >= startRow ) break;
         }
         nnz = 0;
         matIA[0] = nnz;
         matJA[nnz] = colNum - 1;
         matAA[nnz++] = colVal;
         currRow = rowNum;

         for ( j = irow+1; j < globalNnz; j++ )
         {
            fscanf( fp, "%d %d %lg", &rowNum, &colNum, &colVal );
            rowNum--;
            if ( scaleFlag == 1 && rowNum == colNum-1 ) 
               diag[rowNum] = colVal;
            if ( rowNum >= startRow+localNRows ) break;
            if ( rowNum != currRow )
            {
               currRow = rowNum;
               matIA[currRow-startRow] = nnz;
            }
            matJA[nnz] = colNum - 1;
            matAA[nnz++] = colVal;
         } 
         if ( j == globalNnz ) matIA[rowNum+1-startRow] = nnz;
         else                   matIA[rowNum-startRow] = nnz;

         for ( irow = j+1; irow < globalNnz; irow++ )
         {
            fscanf( fp, "%d %d %lg", &rowNum, &colNum, &colVal );
            rowNum--;
            if ( scaleFlag == 1 && rowNum == colNum-1 ) 
               diag[rowNum] = colVal;
         } 
         fclose( fp );
         printf("Processor %d finished reading matrix file.\n", mypid);
      }
      MPI_Barrier( mpiComm );
      currProc++;
   }
   printf("%5d : MLI_Utils_HypreMatrixRead : nlocal, nnz = %d %d\n", 
          mypid, localNRows, nnz);
   rowLengths = (int *) malloc(localNRows * sizeof(int));
   for ( irow = 0; irow < localNRows; irow++ )
      rowLengths[irow] = matIA[irow+1] - matIA[irow];

   ierr = HYPRE_IJMatrixCreate(mpiComm, startRow, startRow+localNRows-1,
                               startRow, startRow+localNRows-1, &IJmat);
   ierr = HYPRE_IJMatrixSetObjectType(IJmat, HYPRE_PARCSR);
   assert(!ierr);
   ierr = HYPRE_IJMatrixSetRowSizes(IJmat, rowLengths);
   ierr = HYPRE_IJMatrixInitialize(IJmat);
   assert(!ierr);
   for ( irow = 0; irow < localNRows; irow++ )
   {
      length = rowLengths[irow];
      rowNum = irow + startRow;
      inds = &(matJA[matIA[irow]]);
      vals = &(matAA[matIA[irow]]);
      if ( scaleFlag == 1 ) 
      {
         scale = 1.0 / sqrt( diag[rowNum] );
         for ( j = 0; j < length; j++ )
         {
            vals[j] = vals[j] * scale / ( sqrt(diag[inds[j]]) );
            if ( rowNum == inds[j] && habs(vals[j]-1.0) > 1.0e-6  )
            {
               printf("Proc %d : diag %d = %e != 1.\n",mypid,rowNum,vals[j]);
               exit(1);
            }
         }
      }
      ierr = HYPRE_IJMatrixSetValues(IJmat, 1, &length,(const int *) &rowNum,
                (const int *) inds, (const double *) vals);
      assert( !ierr );
   }
   free( rowLengths );
   free( matIA );
   free( matJA );
   free( matAA );

   ierr = HYPRE_IJMatrixAssemble(IJmat);
   assert( !ierr );
   HYPRE_IJMatrixGetObject(IJmat, (void**) &hypreA);
   HYPRE_IJMatrixSetObjectType(IJmat, -1);
   HYPRE_IJMatrixDestroy(IJmat);
   (*Amat) = (void *) hypreA;
   if ( scaleFlag )
   {
      diag2 = (double *) malloc( sizeof(double) * localNRows);
      for ( irow = 0; irow < localNRows; irow++ )
         diag2[irow] = diag[startRow+irow];
      free( diag );
   }
   (*scaleVec) = diag2;
#if 0
   sprintf(fname, "mat.%d", mypid);
   fp = fopen(fname, "w");
   for ( irow = 0; irow < localNRows; irow++ )
   {
      rowNum = startRow + irow; 
      hypre_ParCSRMatrixGetRow(hypreA, rowNum, &length, &inds, &vals);
      for ( colNum = 0; colNum < length; colNum++ )
         fprintf(fp, "%d %d %e\n", rowNum, inds[colNum], vals[colNum]);
      hypre_ParCSRMatrixRestoreRow(hypreA, rowNum, &length, &inds, &vals);
   }
   fclose(fp);
#endif

   return ierr;
}

/***************************************************************************
 * read matrix files and create a hypre_ParCSRMatrix from them
 *--------------------------------------------------------------------------*/
 
int MLI_Utils_HypreParMatrixReadIJAFormat(char *filename, MPI_Comm mpiComm, 
              void **Amat, int scaleFlag, double **scaleVec)
{
   int    mypid, nprocs, globalNRows, localNRows, localNnz, startRow;
   int    irow, colNum, *inds, *matIA, *matJA, length, rowNum, index;
   int    j, *rowLengths, ierr, globalNnz, currRow, *rowsArray;
   double colVal, *vals, *matAA, *diag=NULL, *diag2=NULL, scale;
   char   fname[20];
   FILE   *fp;
   hypre_ParCSRMatrix *hypreA;
   HYPRE_IJMatrix     IJmat;

   MPI_Comm_rank( mpiComm, &mypid );
   MPI_Comm_size( mpiComm, &nprocs );
   sprintf( fname, "%s.%d", filename, mypid);
   printf("Processor %d reading matrix file %s.\n", mypid, fname);
   fp = fopen( fname, "r" );
   if ( fp == NULL )
   {
      printf("MLI_Utils_HypreParMatrixReadIJAFormat ERROR : ");
      printf("file %s not found.\n", filename);
      exit(1);
   }
   fscanf( fp, "%d %d", &localNRows, &localNnz );
   printf("%5d : MLI_Utils_HypreParMatrixRead : nlocal, nnz = %d %d\n", 
          mypid, localNRows, localNnz);
   fflush(stdout);
   if ( localNRows < 0 || localNnz > 1000000000 )
   {
      printf("MLI_Utils_HypreMatrixReadIJAFormat ERROR : ");
      printf("invalid nrows %d.\n", localNRows);
      exit(1);
   }
   rowsArray = (int *) malloc( nprocs * sizeof(int) );
   MPI_Allgather(&localNRows, 1, MPI_INT, rowsArray, 1, MPI_INT, mpiComm);
   globalNRows = 0;
   for ( j = 0; j < nprocs; j++ )
   {
      if ( j == mypid ) startRow = globalNRows;
      globalNRows += rowsArray[j];
   }
   free( rowsArray );
   matIA = (int *)    malloc((localNRows+1) * sizeof(int));
   matJA = (int *)    malloc(localNnz * sizeof(int));
   matAA = (double *) malloc(localNnz * sizeof(double));

   if (scaleFlag == 1) 
   {
      diag  = (double *) malloc(sizeof(double)*globalNRows);
      diag2 = (double *) malloc(sizeof(double)*globalNRows);
      for (irow = 0; irow < globalNRows; irow++) diag[irow] = diag2[irow] = 0.0;
   }
   index = 0;
   matIA[0] = index;
   currRow = startRow;
   for ( j = 0; j < localNnz; j++ )
   {
      fscanf( fp, "%d %d %lg", &rowNum, &colNum, &colVal );
      rowNum--;
      if ( scaleFlag == 1 && rowNum == colNum-1 ) diag[rowNum] = colVal;
      if ( rowNum != currRow )
      {
         currRow = rowNum;
         matIA[currRow-startRow] = index;
      }
      matJA[index] = colNum - 1;
      matAA[index++] = colVal;
   } 
   matIA[localNRows] = index;
   fclose(fp);

   printf("Processor %d finished reading matrix file.\n", mypid);
   fflush(stdout);

   if ( scaleFlag == 1 )
      MPI_Allreduce(diag, diag2, globalNRows, MPI_DOUBLE, MPI_SUM, mpiComm);

   rowLengths = (int *) malloc(localNRows * sizeof(int));
   for ( irow = 0; irow < localNRows; irow++ )
      rowLengths[irow] = matIA[irow+1] - matIA[irow];

   ierr = HYPRE_IJMatrixCreate(mpiComm, startRow, startRow+localNRows-1,
                               startRow, startRow+localNRows-1, &IJmat);
   ierr = HYPRE_IJMatrixSetObjectType(IJmat, HYPRE_PARCSR);
   assert(!ierr);
   ierr = HYPRE_IJMatrixSetRowSizes(IJmat, rowLengths);
   ierr = HYPRE_IJMatrixInitialize(IJmat);
   assert(!ierr);
   for ( irow = 0; irow < localNRows; irow++ )
   {
      length = rowLengths[irow];
      rowNum = irow + startRow;
      inds = &(matJA[matIA[irow]]);
      vals = &(matAA[matIA[irow]]);
      if ( scaleFlag == 1 ) 
      {
         scale = 1.0 / sqrt( diag2[rowNum] );
         for ( j = 0; j < length; j++ )
         {
            vals[j] = vals[j] * scale / ( sqrt(diag2[inds[j]]) );
            if ( rowNum == inds[j] && habs(vals[j]-1.0) > 1.0e-6  )
            {
               printf("Proc %d : diag %d = %e != 1.\n",mypid,rowNum,vals[j]);
               exit(1);
            }
         }
      }
      ierr = HYPRE_IJMatrixSetValues(IJmat, 1, &length,(const int *) &rowNum,
                (const int *) inds, (const double *) vals);
      assert( !ierr );
   }
   free( rowLengths );
   free( matIA );
   free( matJA );
   free( matAA );

   ierr = HYPRE_IJMatrixAssemble(IJmat);
   assert( !ierr );
   HYPRE_IJMatrixGetObject(IJmat, (void**) &hypreA);
   HYPRE_IJMatrixSetObjectType(IJmat, -1);
   HYPRE_IJMatrixDestroy(IJmat);
   (*Amat) = (void *) hypreA;
   if ( scaleFlag == 1 )
   {
      free( diag );
      diag = (double *) malloc( sizeof(double) * localNRows);
      for ( irow = 0; irow < localNRows; irow++ )
         diag[irow] = diag2[startRow+irow];
      free( diag2 );
   }
   (*scaleVec) = diag;

   return ierr;
}

/***************************************************************************
 * read a vector from a file 
 *--------------------------------------------------------------------------*/
 
int MLI_Utils_DoubleVectorRead(char *filename, MPI_Comm mpiComm, 
                               int length, int start, double *vec)
{
   int    mypid, nprocs, currProc, globalNRows;
   int    irow, k, k2, base, numparams=2;
   double value;
   FILE   *fp;

   MPI_Comm_rank( mpiComm, &mypid );
   MPI_Comm_size( mpiComm, &nprocs );
   currProc = 0;
   while ( currProc < nprocs )
   {
      if ( mypid == currProc )
      {
         fp = fopen( filename, "r" );
         if ( fp == NULL )
         {
            printf("MLI_Utils_DbleVectorRead ERROR : file not found.\n");
            return -1;
         }
         fscanf( fp, "%d", &globalNRows );
         if ( globalNRows < 0 || globalNRows > 1000000000 )
         {
            printf("MLI_Utils_DoubleVectorRead ERROR : invalid nrows %d.\n",
                   globalNRows);
            exit(1);
         }
         if ( start+length > globalNRows )
         {
            printf("MLI_Utils_DoubleVectorRead ERROR : invalid start %d %d.\n",
                   start, length);
            exit(1);
         }
         fscanf( fp, "%d %lg %d", &k, &value, &k2 );
         if ( k == 0 ) base = 0; else base = 1;
         if ( k2 != 1 && k2 != 2 ) numparams = 3;
         fclose( fp );
         fp = fopen( filename, "r" );
         fscanf( fp, "%d", &globalNRows );
         for ( irow = 0; irow < start; irow++ )
         {
            fscanf( fp, "%d", &k );
            fscanf( fp, "%lg", &value );
            if ( numparams == 3 ) fscanf( fp, "%d", &k2 );
         } 
         for ( irow = start; irow < start+length; irow++ )
         {
            fscanf( fp, "%d", &k );
            if ( irow+base != k )
               printf("Utils::VectorRead Warning : index mismatch (%d,%d).\n",
                      irow+base,k);
            fscanf( fp, "%lg", &value );
            if ( numparams == 3 ) fscanf( fp, "%d", &k2 );
            vec[irow-start] = value;
         }
         fclose( fp );
      }
      MPI_Barrier( mpiComm );
      currProc++;
   }
   printf("%5d : MLI_Utils_DoubleVectorRead : nlocal, start = %d %d\n", 
          mypid, length, start);
   return 0;
}

/***************************************************************************
 * read a vector from a file 
 *--------------------------------------------------------------------------*/
 
int MLI_Utils_DoubleParVectorRead(char *filename, MPI_Comm mpiComm, 
                                  int length, int start, double *vec)
{
   int    mypid, nprocs, localNRows;
   int    irow, k, k2, base=0;
   double value;
   char   fname[20];
   FILE   *fp;

   MPI_Comm_rank( mpiComm, &mypid );
   MPI_Comm_size( mpiComm, &nprocs );
   sprintf( fname, "%s.%d", filename, mypid);
   fp = fopen( fname, "r" );
   if ( fp == NULL )
   {
      printf("MLI_Utils_DoubleParVectorRead ERROR : file %s not found.\n", 
              fname);
      return -1;
   }
   fscanf( fp, "%d", &localNRows );
   if ( length != localNRows )
   {
      printf("MLI_Utils_DoubleParVectorRead ERROR : invalid nrows %d (%d).\n",
             localNRows, length);
      exit(1);
   }
   for ( irow = start; irow < start+length; irow++ )
   {
      fscanf( fp, "%d %lg", &k, &value );
      vec[irow-start] = value;
   }
   fclose( fp );
   return 0;
}

/***************************************************************************
 * conform to the preconditioner set up from HYPRE
 *--------------------------------------------------------------------------*/
 
int MLI_Utils_ParCSRMLISetup( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                              HYPRE_ParVector b, HYPRE_ParVector x )
{
   int  ierr=0;
   CMLI *cmli;
   (void) A;
   (void) b;
   (void) x;
   cmli = (CMLI *) solver;
   MLI_Setup( cmli );
   return ierr;
}

/***************************************************************************
 * conform to the preconditioner apply from HYPRE
 *--------------------------------------------------------------------------*/
 
int MLI_Utils_ParCSRMLISolve( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                              HYPRE_ParVector b, HYPRE_ParVector x )
{
   int          ierr;
   CMLI         *cmli;
   CMLI_Vector  *csol, *crhs;

   (void) A;
   cmli = (CMLI *) solver;
   csol = MLI_VectorCreate((void*) x, "HYPRE_ParVector", NULL);
   crhs = MLI_VectorCreate((void*) b, "HYPRE_ParVector", NULL);
   ierr = MLI_Solve( cmli, csol, crhs );
   MLI_VectorDestroy(csol);
   MLI_VectorDestroy(crhs);
   return ierr;
}

/***************************************************************************
 * solve the system using HYPRE pcg
 *--------------------------------------------------------------------------*/
 
int MLI_Utils_HyprePCGSolve( CMLI *cmli, HYPRE_Matrix A,
                             HYPRE_Vector b, HYPRE_Vector x )
{
   int          numIterations, maxIter=500, mypid;
   double       tol=1.0e-6, norm, setupTime, solveTime;
   MPI_Comm     mpiComm;
   HYPRE_Solver pcgSolver, pcgPrecond;
   HYPRE_ParCSRMatrix hypreA;

   hypreA = (HYPRE_ParCSRMatrix) A;
   MLI_SetMaxIterations( cmli, 1 );
   HYPRE_ParCSRMatrixGetComm( hypreA , &mpiComm );
   HYPRE_ParCSRPCGCreate(mpiComm, &pcgSolver);
   HYPRE_PCGSetMaxIter(pcgSolver, maxIter );
   HYPRE_PCGSetTol(pcgSolver, tol);
   HYPRE_PCGSetTwoNorm(pcgSolver, 1);
   HYPRE_PCGSetRelChange(pcgSolver, 1);
   HYPRE_PCGSetLogging(pcgSolver, 2);
   pcgPrecond = (HYPRE_Solver) cmli;
   HYPRE_PCGSetPrecond(pcgSolver,
                       (HYPRE_PtrToSolverFcn) MLI_Utils_ParCSRMLISolve,
                       (HYPRE_PtrToSolverFcn) MLI_Utils_ParCSRMLISetup,
                       pcgPrecond);
   setupTime = MLI_Utils_WTime();
   HYPRE_PCGSetup(pcgSolver, A, b, x);
   solveTime = MLI_Utils_WTime();
   setupTime = solveTime - setupTime;
   HYPRE_PCGSolve(pcgSolver, A, b, x);
   solveTime = MLI_Utils_WTime() - solveTime;
   HYPRE_PCGGetNumIterations(pcgSolver, &numIterations);
   HYPRE_PCGGetFinalRelativeResidualNorm(pcgSolver, &norm);
   HYPRE_ParCSRPCGDestroy(pcgSolver);
   MPI_Comm_rank(mpiComm, &mypid);
   if ( mypid == 0 )
   {
      printf("\tPCG maximum iterations           = %d\n", maxIter);
      printf("\tPCG convergence tolerance        = %e\n", tol);
      printf("\tPCG number of iterations         = %d\n", numIterations);
      printf("\tPCG final relative residual norm = %e\n", norm);
      printf("\tPCG setup time                   = %e seconds\n",setupTime);
      printf("\tPCG solve time                   = %e seconds\n",solveTime);
   }
   return 0;
}

/***************************************************************************
 * solve the system using HYPRE gmres
 *--------------------------------------------------------------------------*/
 
int MLI_Utils_HypreGMRESSolve( CMLI *cmli, HYPRE_Matrix A,
                             HYPRE_Vector b, HYPRE_Vector x )
{
   int          numIterations, maxIter=500, mypid;
   double       tol=1.0e-6, norm, setupTime, solveTime;
   MPI_Comm     mpiComm;
   HYPRE_Solver gmresSolver, gmresPrecond;
   HYPRE_ParCSRMatrix hypreA;

   hypreA = (HYPRE_ParCSRMatrix) A;
   MLI_SetMaxIterations( cmli, 1 );
   HYPRE_ParCSRMatrixGetComm( hypreA , &mpiComm );
   HYPRE_ParCSRGMRESCreate(mpiComm, &gmresSolver);
   HYPRE_GMRESSetMaxIter(gmresSolver, maxIter );
   HYPRE_GMRESSetTol(gmresSolver, tol);
   HYPRE_GMRESSetRelChange(gmresSolver, 0);
   HYPRE_GMRESSetLogging(gmresSolver, 2);
   HYPRE_ParCSRGMRESSetKDim(gmresSolver, 200);
   gmresPrecond = (HYPRE_Solver) cmli;
   HYPRE_GMRESSetPrecond(gmresSolver,
                       (HYPRE_PtrToSolverFcn) MLI_Utils_ParCSRMLISolve,
                       (HYPRE_PtrToSolverFcn) MLI_Utils_ParCSRMLISetup,
                       gmresPrecond);
   setupTime = MLI_Utils_WTime();
   HYPRE_GMRESSetup(gmresSolver, A, b, x);
   solveTime = MLI_Utils_WTime();
   setupTime = solveTime - setupTime;
   HYPRE_GMRESSolve(gmresSolver, A, b, x);
   solveTime = MLI_Utils_WTime() - solveTime;
   HYPRE_GMRESGetNumIterations(gmresSolver, &numIterations);
   HYPRE_GMRESGetFinalRelativeResidualNorm(gmresSolver, &norm);
   HYPRE_ParCSRGMRESDestroy(gmresSolver);
   MPI_Comm_rank(mpiComm, &mypid);
   if ( mypid == 0 )
   {
      printf("\tGMRES Krylov dimension             = 200\n");
      printf("\tGMRES maximum iterations           = %d\n", maxIter);
      printf("\tGMRES convergence tolerance        = %e\n", tol);
      printf("\tGMRES number of iterations         = %d\n", numIterations);
      printf("\tGMRES final relative residual norm = %e\n", norm);
      printf("\tGMRES setup time                   = %e seconds\n",setupTime);
      printf("\tGMRES solve time                   = %e seconds\n",solveTime);
   }
   return 0;
}

/***************************************************************************
 * solve the system using HYPRE bicgstab
 *--------------------------------------------------------------------------*/
 
int MLI_Utils_HypreBiCGSTABSolve( CMLI *cmli, HYPRE_Matrix A,
                                  HYPRE_Vector b, HYPRE_Vector x )
{
   int          numIterations, maxIter=500;
   double       tol=1.0e-6, norm, setupTime, solveTime;
   MPI_Comm     mpiComm;
   HYPRE_Solver cgstabSolver, cgstabPrecond;
   HYPRE_ParCSRMatrix hypreA;

   hypreA = (HYPRE_ParCSRMatrix) A;
   MLI_SetMaxIterations( cmli, 1 );
   HYPRE_ParCSRMatrixGetComm( hypreA , &mpiComm );
   HYPRE_ParCSRBiCGSTABCreate(mpiComm, &cgstabSolver);
   HYPRE_BiCGSTABSetMaxIter(cgstabSolver, maxIter );
   HYPRE_BiCGSTABSetTol(cgstabSolver, tol);
   HYPRE_BiCGSTABSetStopCrit(cgstabSolver, 0);
   HYPRE_BiCGSTABSetLogging(cgstabSolver, 2);
   cgstabPrecond = (HYPRE_Solver) cmli;
   HYPRE_BiCGSTABSetPrecond(cgstabSolver,
                       (HYPRE_PtrToSolverFcn) MLI_Utils_ParCSRMLISolve,
                       (HYPRE_PtrToSolverFcn) MLI_Utils_ParCSRMLISetup,
                       cgstabPrecond);
   setupTime = MLI_Utils_WTime();
   HYPRE_BiCGSTABSetup(cgstabSolver, A, b, x);
   solveTime = MLI_Utils_WTime();
   setupTime = solveTime - setupTime;
   HYPRE_BiCGSTABSolve(cgstabSolver, A, b, x);
   solveTime = MLI_Utils_WTime() - solveTime;
   HYPRE_BiCGSTABGetNumIterations(cgstabSolver, &numIterations);
   HYPRE_BiCGSTABGetFinalRelativeResidualNorm(cgstabSolver, &norm);
   HYPRE_BiCGSTABDestroy(cgstabSolver);
   printf("\tBiCGSTAB maximum iterations           = %d\n", maxIter);
   printf("\tBiCGSTAB convergence tolerance        = %e\n", tol);
   printf("\tBiCGSTAB number of iterations         = %d\n", numIterations);
   printf("\tBiCGSTAB final relative residual norm = %e\n", norm);
   printf("\tBiCGSTAB setup time                   = %e seconds\n",setupTime);
   printf("\tBiCGSTAB solve time                   = %e seconds\n",solveTime);
   return 0;
}

/***************************************************************************
 * binary search
 *--------------------------------------------------------------------------*/

int MLI_Utils_BinarySearch(int key, int *list, int size)
{
   int  nfirst, nlast, nmid, found, index;

   if (size <= 0) return -1;
   nfirst = 0;
   nlast  = size - 1;
   if (key > list[nlast])  return -(nlast+1);
   if (key < list[nfirst]) return -(nfirst+1);
   found = 0;
   while ((found == 0) && ((nlast-nfirst)>1))
   {
      nmid = (nfirst + nlast) / 2;
      if      (key == list[nmid]) {index  = nmid; found = 1;}
      else if (key > list[nmid])  nfirst = nmid;
      else                        nlast  = nmid;
   }
   if (found == 1)                    return index;
   else if (key == list[nfirst]) return nfirst;
   else if (key == list[nlast])  return nlast;
   else                          return -(nfirst+1);
}

/***************************************************************************
 * quicksort on integers
 *--------------------------------------------------------------------------*/

int MLI_Utils_IntQSort2(int *ilist, int *ilist2, int left, int right)
{
   int i, last, mid, itemp;

   if (left >= right) return 0;
   mid          = (left + right) / 2;
   itemp        = ilist[left];
   ilist[left]  = ilist[mid];
   ilist[mid]   = itemp;
   if ( ilist2 != NULL )
   {
      itemp        = ilist2[left];
      ilist2[left] = ilist2[mid];
      ilist2[mid]  = itemp;
   }
   last         = left;
   for (i = left+1; i <= right; i++)
   {
      if (ilist[i] < ilist[left])
      {
         last++;
         itemp        = ilist[last];
         ilist[last]  = ilist[i];
         ilist[i]     = itemp;
         if ( ilist2 != NULL )
         {
            itemp        = ilist2[last];
            ilist2[last] = ilist2[i];
            ilist2[i]    = itemp;
         }
      }
   }
   itemp        = ilist[left];
   ilist[left]  = ilist[last];
   ilist[last]  = itemp;
   if ( ilist2 != NULL )
   {
      itemp        = ilist2[left];
      ilist2[left] = ilist2[last];
      ilist2[last] = itemp;
   }
   MLI_Utils_IntQSort2(ilist, ilist2, left, last-1);
   MLI_Utils_IntQSort2(ilist, ilist2, last+1, right);
   return 0;
}

/***************************************************************************
 * quicksort on integers and permute doubles
 *--------------------------------------------------------------------------*/

int MLI_Utils_IntQSort2a(int *ilist, double *dlist, int left, int right)
{
   int    i, last, mid, itemp;
   double dtemp;

   if (left >= right) return 0;
   mid          = (left + right) / 2;
   itemp        = ilist[left];
   ilist[left]  = ilist[mid];
   ilist[mid]   = itemp;
   if ( dlist != NULL )
   {
      dtemp       = dlist[left];
      dlist[left] = dlist[mid];
      dlist[mid]  = dtemp;
   }
   last         = left;
   for (i = left+1; i <= right; i++)
   {
      if (ilist[i] < ilist[left])
      {
         last++;
         itemp        = ilist[last];
         ilist[last]  = ilist[i];
         ilist[i]     = itemp;
         if ( dlist != NULL )
         {
            dtemp       = dlist[last];
            dlist[last] = dlist[i];
            dlist[i]    = dtemp;
         }
      }
   }
   itemp        = ilist[left];
   ilist[left]  = ilist[last];
   ilist[last]  = itemp;
   if ( dlist != NULL )
   {
      dtemp       = dlist[left];
      dlist[left] = dlist[last];
      dlist[last] = dtemp;
   }
   MLI_Utils_IntQSort2a(ilist, dlist, left, last-1);
   MLI_Utils_IntQSort2a(ilist, dlist, last+1, right);
   return 0;
}

/***************************************************************************
 * merge sort on integers 
 *--------------------------------------------------------------------------*/

int MLI_Utils_IntMergeSort(int nList, int *listLengs, int **lists,
                           int **lists2, int *newNListOut, int **newListOut)
{
   int i, totalLeng, *indices, *newList, parseCnt, newListCnt, minInd;
   int minVal, *tree, *treeInd;
#if 0
   int sortFlag;
#endif

   totalLeng = 0;
   for ( i = 0; i < nList; i++ ) totalLeng += listLengs[i];
   if ( totalLeng <= 0 ) return 1;

#if 0
   for ( i = 0; i < nList; i++ ) 
   {
      sortFlag = 0;
      for ( j = 1; j < listLengs[i]; j++ ) 
         if ( lists[i][j] < lists[i][j-1] )
         {
            sortFlag = 1;
            break;
         } 
      if ( sortFlag == 1 )
         MLI_Utils_IntQSort2(lists[i], lists2[i], 0, listLengs[i]-1);
   }
#endif

   newList  = (int *) malloc( totalLeng * sizeof(int) ); 
   indices  = (int *) malloc( nList * sizeof(int) );
   tree     = (int *) malloc( nList * sizeof(int) );
   treeInd  = (int *) malloc( nList * sizeof(int) );
   for ( i = 0; i < nList; i++ ) indices[i] = 0;
   for ( i = 0; i < nList; i++ ) 
   {
      if ( listLengs[i] > 0 ) 
      {
         tree[i] = lists[i][0];
         treeInd[i] = i;
      }
      else
      {
         tree[i] = 1 << 31 - 1;
         treeInd[i] = -1;
      }
   }
   MLI_Utils_IntQSort2(tree, treeInd, 0, nList-1);

   parseCnt = newListCnt = 0;
   while ( parseCnt < totalLeng )
   {
      minInd = treeInd[0];
      minVal = tree[0];
      if ( newListCnt == 0 || minVal != newList[newListCnt-1] )
      {
         newList[newListCnt] = minVal;
         lists2[minInd][indices[minInd]++] = newListCnt++;
      }
      else if ( minVal == newList[newListCnt-1] )
      {
         lists2[minInd][indices[minInd]++] = newListCnt - 1;
      }
      if ( indices[minInd] < listLengs[minInd] )
      {
         tree[0] = lists[minInd][indices[minInd]];
         treeInd[0] = minInd;
      }
      else
      {
         tree[0] = 1 << 31 - 1;
         treeInd[0] = - 1;
      }
      MLI_Utils_IntTreeUpdate(nList, tree, treeInd);
      parseCnt++;
   }
   (*newListOut) = newList;   
   (*newNListOut) = newListCnt;   
   free( indices );
   free( tree );
   free( treeInd );
   return 0;
}

/***************************************************************************
 * tree sort on integers 
 *--------------------------------------------------------------------------*/

int MLI_Utils_IntTreeUpdate(int treeLeng, int *tree, int *treeInd)
{
   int i, itemp, seed, next, nextp1, ndigits, minInd, minVal;

   ndigits = 0;
   if ( treeLeng > 0 ) ndigits++;
   itemp = treeLeng;
   while ( (itemp >>= 1) > 0 ) ndigits++;

   if ( tree[1] < tree[0] )
   {
      itemp = tree[0];
      tree[0] = tree[1];
      tree[1] = itemp;
      itemp = treeInd[0];
      treeInd[0] = treeInd[1];
      treeInd[1] = itemp;
   }
   else return 0;

   seed = 1;
   for ( i = 0; i < ndigits-1; i++ )
   {
      next   = seed * 2;
      nextp1 = next + 1;
      minInd = seed;
      minVal = tree[seed];
      if ( next < treeLeng && tree[next] < minVal ) 
      {
         minInd = next;
         minVal = tree[next];
      }
      if ( nextp1 < treeLeng && tree[nextp1] < minVal ) 
      {
         minInd = next + 1;
         minVal = tree[nextp1];
      }
      if ( minInd == seed ) return 0;
      itemp = tree[minInd];
      tree[minInd] = tree[seed];
      tree[seed] = itemp;
      itemp = treeInd[minInd];
      treeInd[minInd] = treeInd[seed];
      treeInd[seed] = itemp;
      seed = minInd;
   }
   return 0;
}

/* ******************************************************************** */
/* inverse of a dense matrix                                             */
/* -------------------------------------------------------------------- */

int MLI_Utils_DenseMatrixInverse( double **Amat, int ndim, double ***Bmat )
{
   int    i, j, k;
   double denom, **Cmat, dmax;

   (*Bmat) = NULL;
   if ( ndim == 1 )
   {
      if ( habs(Amat[0][0]) <= 1.0e-16 ) return -1;
      Cmat = (double **) malloc( ndim * sizeof(double*) );
      for ( i = 0; i < ndim; i++ )
         Cmat[i] = (double *) malloc( ndim * sizeof(double) );
      Cmat[0][0] = 1.0 / Amat[0][0];
      (*Bmat) = Cmat;
      return 0;
   }
   else if ( ndim == 2 )
   {
      denom = Amat[0][0] * Amat[1][1] - Amat[0][1] * Amat[1][0];
      if ( habs( denom ) <= 1.0e-16 ) return -1;
      Cmat = (double **) malloc( ndim * sizeof(double*) );
      for ( i = 0; i < ndim; i++ )
         Cmat[i] = (double *) malloc( ndim * sizeof(double) );
      Cmat[0][0] = Amat[1][1] / denom;
      Cmat[1][1] = Amat[0][0] / denom;
      Cmat[0][1] = - ( Amat[0][1] / denom );
      Cmat[1][0] = - ( Amat[1][0] / denom );
      (*Bmat) = Cmat;
      return 0;
   }
   else
   {
      Cmat = (double **) malloc( ndim * sizeof(double*) );
      for ( i = 0; i < ndim; i++ )
      {
         Cmat[i] = (double *) malloc( ndim * sizeof(double) );
         for ( j = 0; j < ndim; j++ ) Cmat[i][j] = 0.0;
         Cmat[i][i] = 1.0;
      }
      for ( i = 1; i < ndim; i++ )
      {
         for ( j = 0; j < i; j++ )
         {
            if ( habs(Amat[j][j]) < 1.0e-16 ) return -1;
            denom = Amat[i][j] / Amat[j][j];
            for ( k = 0; k < ndim; k++ )
            {
               Amat[i][k] -= denom * Amat[j][k];
               Cmat[i][k] -= denom * Cmat[j][k];
            }
         }
      }
      for ( i = ndim-2; i >= 0; i-- )
      {
         for ( j = ndim-1; j >= i+1; j-- )
         {
            if ( habs(Amat[j][j]) < 1.0e-16 ) return -1;
            denom = Amat[i][j] / Amat[j][j];
            for ( k = 0; k < ndim; k++ )
            {
               Amat[i][k] -= denom * Amat[j][k];
               Cmat[i][k] -= denom * Cmat[j][k];
            }
         }
      }
      for ( i = 0; i < ndim; i++ )
      {
         denom = Amat[i][i];
         if ( habs(denom) < 1.0e-16 ) return -1;
         for ( j = 0; j < ndim; j++ ) Cmat[i][j] /= denom;
      }

      for ( i = 0; i < ndim; i++ )
         for ( j = 0; j < ndim; j++ )
            if ( habs(Cmat[i][j]) < 1.0e-17 ) Cmat[i][j] = 0.0;
      dmax = 0.0;
      for ( i = 0; i < ndim; i++ )
      {
         for ( j = 0; j < ndim; j++ )
            if ( habs(Cmat[i][j]) > dmax ) dmax = habs(Cmat[i][j]);
      }
      (*Bmat) = Cmat;
      if ( dmax > 1.0e6 ) return 1;
      else                return 0;
   }
}

/* ******************************************************************** */
/* matvec given a dense matrix (Amat 2 D array)                         */
/* -------------------------------------------------------------------- */

int MLI_Utils_DenseMatvec( double **Amat, int ndim, double *x, double *Ax )
{
   int    i, j;
   double ddata, *matLocal;

   for ( i = 0; i < ndim; i++ )
   {
      matLocal = Amat[i];
      ddata = 0.0;
      for ( j = 0; j < ndim; j++ ) ddata += *matLocal++ * x[j];
      Ax[i] = ddata;
   }
   return 0;
}

