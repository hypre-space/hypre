/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Utilities functions
 *
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * include files
 *--------------------------------------------------------------------------*/

#include <stdlib.h>
#include <math.h>
#include "HYPRE.h"
#include "mli_utils.h"
#include "HYPRE_IJ_mv.h"
#include "../fei-hypre/HYPRE_parcsr_fgmres.h"
#include "_hypre_lapack.h"

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
void hypre_qsort0(int *, int, int);
void hypre_qsort1(int *, double *, int, int);
int  MLI_Utils_IntTreeUpdate(int treeLeng, int *tree,int *treeInd);

#ifdef __cplusplus
}
#endif

#define habs(x) (((x) > 0) ? x : -(x))

/*****************************************************************************
 * destructor for hypre_ParCSRMatrix conforming to MLI requirements
 *--------------------------------------------------------------------------*/

int MLI_Utils_HypreParCSRMatrixGetDestroyFunc(MLI_Function *funcPtr)
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
   double             *colVal, *newColVal, dtemp;
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
   hypre_assert( !ierr );
   maxnnz = 0;
   rowLengths = hypre_CTAlloc(int,  localNRows, HYPRE_MEMORY_HOST);
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
   hypre_assert( !ierr );
   HYPRE_IJMatrixInitialize(IJmat);

   /* -----------------------------------------------------------------------
    * load the new matrix
    * ----------------------------------------------------------------------*/

   newColInd = hypre_CTAlloc(int,  maxnnz, HYPRE_MEMORY_HOST);
   newColVal = hypre_CTAlloc(double,  maxnnz, HYPRE_MEMORY_HOST);

   for ( irow = 0; irow < localNRows; irow++ )
   {
      rownum = startRow + irow;
      hypre_ParCSRMatrixGetRow(Amat, rownum, &rowSize, &colInd, &colVal);
      dtemp = 1.0;
      for ( icol = 0; icol < rowSize; icol++ )
         if ( colInd[icol] == rownum ) {dtemp = colVal[icol]; break;}
      if ( habs(dtemp) > 1.0e-16 ) dtemp = 1.0 / dtemp;
      else                         dtemp = 1.0;
      for ( icol = 0; icol < rowSize; icol++ )
      {
         newColInd[icol] = colInd[icol];
         newColVal[icol] = - alpha * colVal[icol] * dtemp;
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

   hypre_TFree(newColInd , HYPRE_MEMORY_HOST);
   hypre_TFree(newColVal , HYPRE_MEMORY_HOST);
   hypre_TFree(rowLengths , HYPRE_MEMORY_HOST);
   hypre_TFree(rowPart , HYPRE_MEMORY_HOST);
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
   garray = hypre_CTAlloc(int,  nprocs+1, HYPRE_MEMORY_HOST);
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
 * Given matrix A and vector v, scale the vector by (v'*v)/(v'*A*v).
 *--------------------------------------------------------------------------*/

int MLI_Utils_ScaleVec(hypre_ParCSRMatrix *Amat, hypre_ParVector *vec)
{
   MPI_Comm        comm;
   int             mypid, nprocs, *partition;
   hypre_ParVector *temp;
   double          norm1, norm2;

   /* -----------------------------------------------------------------
    * fetch matrix parameters
    * ----------------------------------------------------------------*/

   comm = hypre_ParCSRMatrixComm(Amat);
   MPI_Comm_rank(comm, &mypid);
   MPI_Comm_size(comm, &nprocs);
   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix)Amat,&partition);

   /* -----------------------------------------------------------------
    * create temporary vector
    * ----------------------------------------------------------------*/

   temp = hypre_ParVectorCreate(comm, partition[nprocs], partition);
   hypre_ParVectorInitialize(temp);

   /* -----------------------------------------------------------------
    * normalize vector
    * ----------------------------------------------------------------*/
   norm2 = hypre_ParVectorInnerProd(vec, vec);
   hypre_ParVectorScale(1./sqrt(norm2), vec);

   /* -----------------------------------------------------------------
    * multiply by matrix, perform inner product, and scale
    * ----------------------------------------------------------------*/

   norm1 = hypre_ParVectorInnerProd(vec, vec);
   hypre_ParCSRMatrixMatvec(1.0, Amat, vec, 0.0, temp);
   norm2 = hypre_ParVectorInnerProd(vec, temp);
   hypre_ParVectorScale(norm1/norm2, vec);
   /* printf("Rayleigh quotient: %f\n", norm2/norm1); */

   hypre_ParVectorDestroy(temp);
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
   hypre_TFree(partition, HYPRE_MEMORY_HOST);

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
   hypre_assert(!ierr);
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
   int      startRow, endRow, *partition, *ADiagI, *ADiagJ;
   double   alpha, beta, rho, rhom1, sigma, offdiagNorm, *zData;
   double   rnorm, *alphaArray, *rnormArray, **Tmat, initOffdiagNorm;
   double   app, aqq, arr, ass, apq, sign, tau, t, c, s;
   double   *ADiagA, one=1.0, *srdiag;
   MPI_Comm comm;
   hypre_CSRMatrix *ADiag;
   hypre_ParVector *rVec=NULL, *zVec, *pVec, *apVec;

   double   *pData, *apData;

   /*-----------------------------------------------------------------
    * fetch matrix information
    *-----------------------------------------------------------------*/

   comm = hypre_ParCSRMatrixComm(A);
   MPI_Comm_rank(comm,&mypid);
   MPI_Comm_size(comm,&nprocs);

   ADiag      = hypre_ParCSRMatrixDiag(A);
   ADiagA     = hypre_CSRMatrixData(ADiag);
   ADiagI     = hypre_CSRMatrixI(ADiag);
   ADiagJ     = hypre_CSRMatrixJ(ADiag);
   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) A, &partition);
   startRow    = partition[mypid];
   endRow      = partition[mypid+1] - 1;
   globalNRows = partition[nprocs];
   localNRows  = endRow - startRow + 1;
   hypre_TFree( partition , HYPRE_MEMORY_HOST);
   maxIter     = 5;
   if ( globalNRows < maxIter ) maxIter = globalNRows;
   ritz[0] = ritz[1] = 0.0;
   srdiag = hypre_TAlloc(double, localNRows , HYPRE_MEMORY_HOST);
   for ( i = 0; i < localNRows; i++ )
   {
      srdiag[i] = 1.0;
      for ( j = ADiagI[i]; j < ADiagI[i+1]; j++ )
         if (ADiagJ[j] == i) {srdiag[i] = ADiagA[j]; break;}
      if ( srdiag[i] > 0.0 ) srdiag[i] = 1.0 / sqrt(srdiag[i]);
      else                   srdiag[i] = 1.0 / sqrt(-srdiag[i]);
   }

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

      pData  = hypre_VectorData( hypre_ParVectorLocalVector(pVec) );
      apData  = hypre_VectorData( hypre_ParVectorLocalVector(apVec) );
   }
   HYPRE_ParVectorSetRandomValues((HYPRE_ParVector) rVec, 1209873 );
   alphaArray = hypre_TAlloc(double,  (maxIter+1) , HYPRE_MEMORY_HOST);
   rnormArray = hypre_TAlloc(double,  (maxIter+1) , HYPRE_MEMORY_HOST);
   Tmat       = hypre_TAlloc(double*,  (maxIter+1) , HYPRE_MEMORY_HOST);
   for ( i = 0; i <= maxIter; i++ )
   {
      Tmat[i] = hypre_TAlloc(double,  (maxIter+1) , HYPRE_MEMORY_HOST);
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
      rhom1 = rho;
      rho   = hypre_ParVectorInnerProd(rVec, rVec);
      if (its == 0) beta = 0.0;
      else
      {
         beta = rho / rhom1;
         Tmat[its-1][its] = -beta;
      }
      HYPRE_ParVectorScale( beta, (HYPRE_ParVector) pVec );
      hypre_ParVectorAxpy( one, rVec, pVec );

      if (scaleFlag)
         for ( i = 0; i < localNRows; i++ ) apData[i] = pData[i]*srdiag[i];
      else
         for ( i = 0; i < localNRows; i++ ) apData[i] = pData[i];

      hypre_ParCSRMatrixMatvec(one, A, apVec, 0.0, zVec);

      if (scaleFlag)
         for ( i = 0; i < localNRows; i++ ) apData[i] = zData[i]*srdiag[i];
      else
         for ( i = 0; i < localNRows; i++ ) apData[i] = zData[i];

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
   hypre_TFree(alphaArray, HYPRE_MEMORY_HOST);
   hypre_TFree(rnormArray, HYPRE_MEMORY_HOST);
   for (i = 0; i <= maxIter; i++) 
      hypre_TFree(Tmat[i], HYPRE_MEMORY_HOST);
   hypre_TFree(Tmat, HYPRE_MEMORY_HOST);
   hypre_TFree(srdiag, HYPRE_MEMORY_HOST);
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
   hypre_TFree(rowPart, HYPRE_MEMORY_HOST);

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
   hypre_TFree(partition, HYPRE_MEMORY_HOST);
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
   isum[0] = thisNnz % 16;
   isum[1] = thisNnz >> 4;
   MPI_Allreduce( isum, ibuf, 2, MPI_INT, MPI_SUM, mpiComm );
   totalNnz = ibuf[1] * 16 + ibuf[0];
   matInfo[0] = globalNRows;
   matInfo[1] = maxNnz;
   matInfo[2] = minNnz;
   matInfo[3] = totalNnz;
   valInfo[0] = maxVal;
   valInfo[1] = minVal;
   valInfo[2] = 16.0 * ((double) ibuf[1]) + ((double) ibuf[0]);
   return 0;
}

/***************************************************************************
 * Given a Hypre ParCSR matrix, compress it
 *--------------------------------------------------------------------------*/

int MLI_Utils_HypreMatrixCompress(void *Amat, int blksize, void **Amat2)
{
   int                mypid, *partition, startRow, localNRows;
   int                newLNRows, newStartRow, blksize2;
   int                ierr, *rowLengths, irow, rowNum, rowSize, *colInd;
   int                *newInd, newSize, j, k, nprocs;
   double             *colVal, *newVal, *newVal2;
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
   hypre_TFree(partition, HYPRE_MEMORY_HOST);
   if ( blksize < 0 ) blksize2 = - blksize;
   else               blksize2 = blksize;
   if ( localNRows % blksize2 != 0 )
   {
      printf("MLI_CompressMatrix ERROR : nrows not divisible by blksize.\n");
      printf("                nrows, blksize = %d %d\n",localNRows,blksize2);
      exit(1);
   }

   /* ----------------------------------------------------------------
    * compute size of new matrix and create the new matrix
    * ----------------------------------------------------------------*/

   newLNRows   = localNRows / blksize2;
   newStartRow = startRow / blksize2;
   ierr =  HYPRE_IJMatrixCreate(mpiComm, newStartRow,
                  newStartRow+newLNRows-1, newStartRow,
                  newStartRow+newLNRows-1, &IJAmat2);
   ierr += HYPRE_IJMatrixSetObjectType(IJAmat2, HYPRE_PARCSR);
   hypre_assert(!ierr);

   /* ----------------------------------------------------------------
    * compute the row lengths of the new matrix
    * ----------------------------------------------------------------*/

   if (newLNRows > 0) rowLengths = hypre_TAlloc(int, newLNRows, HYPRE_MEMORY_HOST);
   else               rowLengths = NULL;

   for ( irow = 0; irow < newLNRows; irow++ )
   {
      rowLengths[irow] = 0;
      for ( j = 0; j < blksize2; j++)
      {
         rowNum = startRow + irow * blksize2 + j;
         hypre_ParCSRMatrixGetRow(hypreA,rowNum,&rowSize,&colInd,NULL);
         rowLengths[irow] += rowSize;
         hypre_ParCSRMatrixRestoreRow(hypreA,rowNum,&rowSize,&colInd,NULL);
      }
   }
   ierr =  HYPRE_IJMatrixSetRowSizes(IJAmat2, rowLengths);
   ierr += HYPRE_IJMatrixInitialize(IJAmat2);
   hypre_assert(!ierr);

   /* ----------------------------------------------------------------
    * load the compressed matrix
    * ----------------------------------------------------------------*/

   for ( irow = 0; irow < newLNRows; irow++ )
   {
      newInd  = hypre_TAlloc(int,  rowLengths[irow] , HYPRE_MEMORY_HOST);
      newVal  = hypre_TAlloc(double,  rowLengths[irow] , HYPRE_MEMORY_HOST);
      newVal2 = hypre_TAlloc(double,  rowLengths[irow] , HYPRE_MEMORY_HOST);
      newSize = 0;
      for ( j = 0; j < blksize2; j++)
      {
         rowNum = startRow + irow * blksize2 + j;
         hypre_ParCSRMatrixGetRow(hypreA,rowNum,&rowSize,&colInd,&colVal);
         for ( k = 0; k < rowSize; k++ )
         {
            newInd[newSize] = colInd[k] / blksize2;
            newVal[newSize++] = colVal[k];
         }
         hypre_ParCSRMatrixRestoreRow(hypreA,rowNum,&rowSize,
                                      &colInd,&colVal);
      }
      if ( newSize > 0 )
      {
         hypre_qsort1(newInd, newVal, 0, newSize-1);
         if ( blksize > 0 )
         {
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
            for ( j = 0; j < newSize; j++ ) newVal[j] = sqrt(newVal[j]);
         }
         else
         {
            k = 0;
            newVal[k] = newVal[k];
            newVal2[k] = newVal[k];
            for ( j = 1; j < newSize; j++ )
            {
               if (newInd[j] == newInd[k])
               {
                  newVal2[k] += newVal[j];
                  if ( habs(newVal[j]) > habs(newVal[k]) )
                     newVal[k] = newVal[j];
               }
               else
               {
                  newInd[++k] = newInd[j];
                  newVal2[k]  = newVal[j];
                  newVal[k]   = newVal[j];
               }
            }
            newSize = k + 1;
            for ( j = 0; j < newSize; j++ )
            {
               if ( newInd[j] == newStartRow+irow )
                    newVal[j] = (newVal[j])/((double) blksize2);
               else
                  newVal[j] = (newVal[j])/((double) blksize2);
/*
               else if ( newVal2[j] >= 0.0 )
                  newVal[j] = (newVal[j])/((double) blksize2);
               else
                  newVal[j] = -(newVal[j])/((double) blksize2);
*/
            }
         }
      }
      rowNum = newStartRow + irow;
      HYPRE_IJMatrixSetValues(IJAmat2, 1, &newSize,(const int *) &rowNum,
                (const int *) newInd, (const double *) newVal);
      hypre_TFree(newInd, HYPRE_MEMORY_HOST);
      hypre_TFree(newVal, HYPRE_MEMORY_HOST);
      hypre_TFree(newVal2, HYPRE_MEMORY_HOST);
   }
   ierr = HYPRE_IJMatrixAssemble(IJAmat2);
   hypre_assert( !ierr );
   HYPRE_IJMatrixGetObject(IJAmat2, (void **) &hypreA2);
   /*hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) hypreA2);*/
   HYPRE_IJMatrixSetObjectType( IJAmat2, -1 );
   HYPRE_IJMatrixDestroy( IJAmat2 );
   hypre_TFree(rowLengths, HYPRE_MEMORY_HOST);
   (*Amat2) = (void *) hypreA2;
   return 0;
}

/***************************************************************************
 * Given a Hypre ParCSR matrix, compress it
 *--------------------------------------------------------------------------*/

int MLI_Utils_HypreBoolMatrixDecompress(void *Smat, int blkSize,
                                        void **Smat2, void *Amat)
{
   int                mypid, *partition, startRow, localNRows, newLNRows;
   int                newStartRow, maxRowLeng, index, ierr, irow, sRowNum;
   int                *rowLengths=NULL, rowNum, rowSize, *colInd, *sInd=NULL;
   int                *newInd=NULL, newSize, j, k, nprocs, searchInd;
   int                sRowSize;
   double             *newVal=NULL;
   MPI_Comm           mpiComm;
   hypre_ParCSRMatrix *hypreA, *hypreS, *hypreS2;
   HYPRE_IJMatrix     IJSmat2;

   /* ----------------------------------------------------------------
    * fetch information about incoming matrix
    * ----------------------------------------------------------------*/

   hypreS  = (hypre_ParCSRMatrix *) Smat;
   hypreA  = (hypre_ParCSRMatrix *) Amat;
   mpiComm = hypre_ParCSRMatrixComm(hypreA);
   MPI_Comm_rank(mpiComm, &mypid);
   MPI_Comm_size(mpiComm, &nprocs);
   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) hypreA,&partition);
   startRow    = partition[mypid];
   localNRows  = partition[mypid+1] - startRow;
   hypre_TFree(partition, HYPRE_MEMORY_HOST);
   if ( localNRows % blkSize != 0 )
   {
      printf("MLI_DecompressMatrix ERROR : nrows not divisible by blksize.\n");
      printf("                nrows, blksize = %d %d\n",localNRows,blkSize);
      exit(1);
   }

   /* ----------------------------------------------------------------
    * compute size of new matrix and create the new matrix
    * ----------------------------------------------------------------*/

   newLNRows   = localNRows / blkSize;
   newStartRow = startRow / blkSize;
   ierr =  HYPRE_IJMatrixCreate(mpiComm, startRow,
                  startRow+localNRows-1, startRow,
                  startRow+localNRows-1, &IJSmat2);
   ierr += HYPRE_IJMatrixSetObjectType(IJSmat2, HYPRE_PARCSR);
   hypre_assert(!ierr);

   /* ----------------------------------------------------------------
    * compute the row lengths of the new matrix
    * ----------------------------------------------------------------*/

   if (localNRows > 0) rowLengths = hypre_TAlloc(int, localNRows, HYPRE_MEMORY_HOST);

   maxRowLeng = 0;
   for ( irow = 0; irow < localNRows; irow++ )
   {
      rowNum = startRow + irow;
      hypre_ParCSRMatrixGetRow(hypreA,rowNum,&rowSize,&colInd,NULL);
      rowLengths[irow] = rowSize;
      if ( rowSize > maxRowLeng ) maxRowLeng = rowSize;
      hypre_ParCSRMatrixRestoreRow(hypreA,rowNum,&rowSize,&colInd,NULL);
   }
   ierr =  HYPRE_IJMatrixSetRowSizes(IJSmat2, rowLengths);
   ierr += HYPRE_IJMatrixInitialize(IJSmat2);
   hypre_assert(!ierr);
   hypre_TFree(rowLengths, HYPRE_MEMORY_HOST);

   /* ----------------------------------------------------------------
    * load the decompressed matrix
    * ----------------------------------------------------------------*/

   if ( maxRowLeng > 0 )
   {
      newInd  = hypre_TAlloc(int,  maxRowLeng , HYPRE_MEMORY_HOST);
      newVal  = hypre_TAlloc(double,  maxRowLeng , HYPRE_MEMORY_HOST);
      sInd    = hypre_TAlloc(int,  maxRowLeng , HYPRE_MEMORY_HOST);
      for ( irow = 0; irow < maxRowLeng; irow++ ) newVal[irow] = 1.0;
   }
   for ( irow = 0; irow < newLNRows; irow++ )
   {
      sRowNum = newStartRow + irow;
      hypre_ParCSRMatrixGetRow(hypreS,sRowNum,&sRowSize,&colInd,NULL);
      for ( k = 0; k < sRowSize; k++ ) sInd[k] = colInd[k];
      hypre_ParCSRMatrixRestoreRow(hypreS,sRowNum,&sRowSize,&colInd,NULL);
      hypre_qsort0(sInd, 0, sRowSize-1);
      for ( j = 0; j < blkSize; j++)
      {
         rowNum = startRow + irow * blkSize + j;
         hypre_ParCSRMatrixGetRow(hypreA,rowNum,&rowSize,&colInd,NULL);
         for ( k = 0; k < rowSize; k++ )
         {
            index = colInd[k] / blkSize;
            searchInd = MLI_Utils_BinarySearch(index, sInd, sRowSize);
            if ( searchInd >= 0 && colInd[k] == index*blkSize+j )
                 newInd[k] = colInd[k];
            else newInd[k] = -1;
         }
         newSize = 0;
         for ( k = 0; k < rowSize; k++ )
            if ( newInd[k] >= 0 ) newInd[newSize++] = newInd[k];
         hypre_ParCSRMatrixRestoreRow(hypreA,rowNum,&rowSize,&colInd,NULL);
         HYPRE_IJMatrixSetValues(IJSmat2, 1, &newSize,(const int *) &rowNum,
                (const int *) newInd, (const double *) newVal);
      }
   }
   hypre_TFree(newInd, HYPRE_MEMORY_HOST);
   hypre_TFree(newVal, HYPRE_MEMORY_HOST);
   hypre_TFree(sInd, HYPRE_MEMORY_HOST);
   ierr = HYPRE_IJMatrixAssemble(IJSmat2);
   hypre_assert( !ierr );
   HYPRE_IJMatrixGetObject(IJSmat2, (void **) &hypreS2);
   HYPRE_IJMatrixSetObjectType( IJSmat2, -1 );
   HYPRE_IJMatrixDestroy( IJSmat2 );
   (*Smat2) = (void *) hypreS2;
   return 0;
}

/***************************************************************************
 * perform QR factorization
 *--------------------------------------------------------------------------*/

int MLI_Utils_QR(double *qArray, double *rArray, int nrows, int ncols)
{
   int    icol, irow, pcol, retFlag=0;
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
      if ( innerProd < 1.0e-18 )
      {
         return icol + 1;
      }
      else
      {
         currR[icol] = innerProd;
         alpha = 1.0 / innerProd;
         for ( irow = 0; irow < nrows; irow++ )
            currQ[irow] = alpha * currQ[irow];
      }
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
   return retFlag;
}

/***************************************************************************
 * perform SVD factorization
 *
 * Inputs:
 *    uArray = input matrix (array of length m*n)
 *    m = number of rows of input matrix
 *    n = number of cols of input matrix
 *
 * Outputs:
 *    uArray = min(m,n) by m; left singular vectors
 *    sArray = min(m,n) singular values (decreasing order)
 *    vtArray = min(m,n) rows of transpose of
 *
 * Work space:
 *    workArray = array of length workLen
 *    workLen   = suggest 5*(m+n)
 *--------------------------------------------------------------------------*/

#include "fortran.h"

int MLI_Utils_SVD(double *uArray, double *sArray, double *vtArray,
    double *workArray, int m, int n, int workLen)
{
#ifndef MIN
#define MIN(a,b) ((a)<(b)?(a):(b))
#endif

#ifdef HYPRE_USING_ESSL
    /* undone */
    int info;
    info = -1;
#else
    char jobu  = 'O'; /* overwrite input with U */
    char jobvt = 'S'; /* return rows of V in vtArray */
    int  dim = MIN(m,n);
    int  info;

    hypre_dgesvd(&jobu, &jobvt, &m, &n, uArray,
        &m, sArray, (double *) NULL, &m, vtArray, &dim, workArray,
        &workLen, &info);
#endif

    return info;
}

/******************************************************************************
 * Return the left singular vectors of a square matrix
 *--------------------------------------------------------------------------*/

int MLI_Utils_singular_vectors(int n, double *uArray)
{
    int info;

#ifdef HYPRE_USING_ESSL
    info = -1;
#else
    char jobu  = 'O'; /* overwrite input with U */
    char jobvt = 'N';
    double *sArray = hypre_TAlloc(double, n, HYPRE_MEMORY_HOST);
    int workLen = 5*n;
    double *workArray = hypre_TAlloc(double, workLen, HYPRE_MEMORY_HOST);

    hypre_dgesvd(&jobu, &jobvt, &n, &n, uArray,
        &n, sArray, NULL, &n, NULL, &n, workArray, &workLen, &info);

    hypre_TFree(workArray, HYPRE_MEMORY_HOST);
    hypre_TFree(sArray, HYPRE_MEMORY_HOST);
#endif

    return info;
}

/******************************************************************************
 * MLI_Utils_ComputeLowEnergyLanczos
 * inputs:
 * A = matrix
 * maxIter = number of Lanczos steps
 * num_vecs_to_return = number of low energy vectors to return
 * le_vectors = pointer to storage space where vectors will be returned
 *--------------------------------------------------------------------------*/

int MLI_Utils_ComputeLowEnergyLanczos(hypre_ParCSRMatrix *A,
    int maxIter, int num_vecs_to_return, double *le_vectors)
{
   int      i, j, k, its, nprocs, mypid, localNRows, globalNRows;
   int      startRow, endRow, *partition;
   double   alpha, beta, rho, rhom1, sigma, *zData;
   double   rnorm, *alphaArray, *rnormArray, **Tmat;
   double   one=1.0, *rData;
   MPI_Comm comm;
   hypre_ParVector *rVec=NULL, *zVec, *pVec, *apVec;
   double *lanczos, *lanczos_p, *Umat, *ptr, *Uptr, *curr_le_vector;
   double rVecNorm;

   /*-----------------------------------------------------------------
    * fetch matrix information
    *-----------------------------------------------------------------*/

   comm = hypre_ParCSRMatrixComm(A);
   MPI_Comm_rank(comm,&mypid);
   MPI_Comm_size(comm,&nprocs);

   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) A, &partition);
   startRow    = partition[mypid];
   endRow      = partition[mypid+1] - 1;
   globalNRows = partition[nprocs];
   localNRows  = endRow - startRow + 1;
   hypre_TFree( partition , HYPRE_MEMORY_HOST);

   if ( globalNRows < maxIter )
   {
       fprintf(stderr, "Computing Low energy vectors: "
          "more steps than dim of matrix.\n");
       exit(-1);
   }

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
   alphaArray = hypre_TAlloc(double,  (maxIter+1) , HYPRE_MEMORY_HOST);
   rnormArray = hypre_TAlloc(double,  (maxIter+1) , HYPRE_MEMORY_HOST);
   Tmat       = hypre_TAlloc(double*,  (maxIter+1) , HYPRE_MEMORY_HOST);
   for ( i = 0; i <= maxIter; i++ )
   {
      Tmat[i] = hypre_TAlloc(double,  (maxIter+1) , HYPRE_MEMORY_HOST);
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
      printf("MLI_Utils_ComputeLowEnergyLanczos : fail for res=0.\n");
      hypre_ParVectorDestroy( rVec );
      hypre_ParVectorDestroy( pVec );
      hypre_ParVectorDestroy( zVec );
      hypre_ParVectorDestroy( apVec );
      return 1;
   }

   /* allocate storage for lanzcos vectors */

   lanczos = hypre_TAlloc(double, maxIter*localNRows, HYPRE_MEMORY_HOST);
   lanczos_p = lanczos;

   /*-----------------------------------------------------------------
    * main loop
    *-----------------------------------------------------------------*/

   for ( its = 0; its < maxIter; its++ )
   {
      for ( i = 0; i < localNRows; i++ )
          zData[i] = rData[i];

      /* scale copy lanczos vector r for use later */
      rVecNorm = sqrt(hypre_ParVectorInnerProd(rVec, rVec));
      for ( i = 0; i < localNRows; i++ )
          *lanczos_p++ = rData[i] / rVecNorm;

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
         fprintf(stderr, "Computing Low energy vectors: "
          "too many Lanczos steps for this problem.\n");
         exit(-1);
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
   /* Compute eigenvectors and eigenvalues of T.                      */
   /* Since we need the smallest eigenvalue eigenvectors, use an SVD  */
   /* and return all the singular vectors.                            */
   /* ----------------------------------------------------------------*/

   Umat = hypre_TAlloc(double, maxIter*maxIter, HYPRE_MEMORY_HOST);
   ptr = Umat;
   /* copy Tmat into Umat */
   for ( i = 0; i < maxIter; i++ )
      for ( j = 0; j < maxIter; j++ )
         *ptr++ = Tmat[i][j];

   MLI_Utils_singular_vectors(maxIter, Umat);

   /* ----------------------------------------------------------------
    * compute low-energy vectors
    * ----------------------------------------------------------------*/

   if (num_vecs_to_return > maxIter)
   {
       fprintf(stderr, "Computing Low energy vectors: "
          "requested more vectors than number of Lanczos steps.\n");
       exit(-1);
   }

   for (i=0; i<num_vecs_to_return; i++)
   {
       Uptr = Umat + maxIter * (maxIter - num_vecs_to_return + i);

       lanczos_p = lanczos;

       curr_le_vector = le_vectors + i*localNRows;

       for (j=0; j<localNRows; j++)
           curr_le_vector[j] = 0.;

       for (j=0; j<maxIter; j++)
       {
           for (k=0; k<localNRows; k++)
               curr_le_vector[k] += *Uptr * *lanczos_p++;

           Uptr++;
       }
   }

   hypre_TFree(Umat, HYPRE_MEMORY_HOST);
   hypre_TFree(lanczos, HYPRE_MEMORY_HOST);

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
   hypre_TFree(alphaArray, HYPRE_MEMORY_HOST);
   hypre_TFree(rnormArray, HYPRE_MEMORY_HOST);
   for (i = 0; i <= maxIter; i++) 
      hypre_TFree(Tmat[i], HYPRE_MEMORY_HOST);
   hypre_TFree(Tmat, HYPRE_MEMORY_HOST);
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

         if (scaleFlag) diag = hypre_TAlloc(double, globalNRows, HYPRE_MEMORY_HOST);
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
         matIA = hypre_TAlloc(int, (localNRows+1) , HYPRE_MEMORY_HOST);
         matJA = hypre_TAlloc(int, currBufSize , HYPRE_MEMORY_HOST);
         matAA = hypre_TAlloc(double, currBufSize , HYPRE_MEMORY_HOST);
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
                  matJA = hypre_TAlloc(int, currBufSize , HYPRE_MEMORY_HOST);
                  matAA = hypre_TAlloc(double, currBufSize , HYPRE_MEMORY_HOST);
                  for ( j = 0; j < nnz; j++ )
                  {
                     matJA[j] = tempJA[j];
                     matAA[j] = tempAA[j];
                  }
                  hypre_TFree(tempJA , HYPRE_MEMORY_HOST);
                  hypre_TFree(tempAA , HYPRE_MEMORY_HOST);
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
   rowLengths = hypre_TAlloc(int, localNRows , HYPRE_MEMORY_HOST);
   for ( irow = 0; irow < localNRows; irow++ )
      rowLengths[irow] = matIA[irow+1] - matIA[irow];

   ierr = HYPRE_IJMatrixCreate(mpiComm, startRow, startRow+localNRows-1,
                               startRow, startRow+localNRows-1, &IJmat);
   ierr = HYPRE_IJMatrixSetObjectType(IJmat, HYPRE_PARCSR);
   hypre_assert(!ierr);
   ierr = HYPRE_IJMatrixSetRowSizes(IJmat, rowLengths);
   ierr = HYPRE_IJMatrixInitialize(IJmat);
   hypre_assert(!ierr);
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
      hypre_assert( !ierr );
   }
   hypre_TFree(rowLengths , HYPRE_MEMORY_HOST);
   hypre_TFree(matIA , HYPRE_MEMORY_HOST);
   hypre_TFree(matJA , HYPRE_MEMORY_HOST);
   hypre_TFree(matAA , HYPRE_MEMORY_HOST);

   ierr = HYPRE_IJMatrixAssemble(IJmat);
   hypre_assert( !ierr );
   HYPRE_IJMatrixGetObject(IJmat, (void**) &hypreA);
   HYPRE_IJMatrixSetObjectType(IJmat, -1);
   HYPRE_IJMatrixDestroy(IJmat);
   (*Amat) = (void *) hypreA;
   if ( scaleFlag )
   {
      diag2 = hypre_TAlloc(double,  localNRows, HYPRE_MEMORY_HOST);
      for ( irow = 0; irow < localNRows; irow++ )
         diag2[irow] = diag[startRow+irow];
      hypre_TFree(diag, HYPRE_MEMORY_HOST);
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
#if 0
   char   fname[20];
#endif
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
            system("ls");
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
         matIA = hypre_TAlloc(int, (localNRows+1) , HYPRE_MEMORY_HOST);
         matJA = hypre_TAlloc(int, currBufSize , HYPRE_MEMORY_HOST);
         matAA = hypre_TAlloc(double, currBufSize , HYPRE_MEMORY_HOST);

         if (scaleFlag == 1)
            diag = hypre_TAlloc(double, globalNRows, HYPRE_MEMORY_HOST);
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
   rowLengths = hypre_TAlloc(int, localNRows , HYPRE_MEMORY_HOST);
   for ( irow = 0; irow < localNRows; irow++ )
      rowLengths[irow] = matIA[irow+1] - matIA[irow];

   ierr = HYPRE_IJMatrixCreate(mpiComm, startRow, startRow+localNRows-1,
                               startRow, startRow+localNRows-1, &IJmat);
   ierr = HYPRE_IJMatrixSetObjectType(IJmat, HYPRE_PARCSR);
   hypre_assert(!ierr);
   ierr = HYPRE_IJMatrixSetRowSizes(IJmat, rowLengths);
   ierr = HYPRE_IJMatrixInitialize(IJmat);
   hypre_assert(!ierr);
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
      hypre_assert( !ierr );
   }
   hypre_TFree(rowLengths , HYPRE_MEMORY_HOST);
   hypre_TFree(matIA , HYPRE_MEMORY_HOST);
   hypre_TFree(matJA , HYPRE_MEMORY_HOST);
   hypre_TFree(matAA , HYPRE_MEMORY_HOST);

   ierr = HYPRE_IJMatrixAssemble(IJmat);
   hypre_assert( !ierr );
   HYPRE_IJMatrixGetObject(IJmat, (void**) &hypreA);
   HYPRE_IJMatrixSetObjectType(IJmat, -1);
   HYPRE_IJMatrixDestroy(IJmat);
   (*Amat) = (void *) hypreA;
   if ( scaleFlag )
   {
      diag2 = hypre_TAlloc(double,  localNRows, HYPRE_MEMORY_HOST);
      for ( irow = 0; irow < localNRows; irow++ )
         diag2[irow] = diag[startRow+irow];
      hypre_TFree(diag, HYPRE_MEMORY_HOST);
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
   int    j, *rowLengths, ierr, currRow, *rowsArray;
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
   rowsArray = hypre_TAlloc(int,  nprocs , HYPRE_MEMORY_HOST);
   MPI_Allgather(&localNRows, 1, MPI_INT, rowsArray, 1, MPI_INT, mpiComm);
   globalNRows = 0;
   for ( j = 0; j < nprocs; j++ )
   {
      if ( j == mypid ) startRow = globalNRows;
      globalNRows += rowsArray[j];
   }
   hypre_TFree(rowsArray, HYPRE_MEMORY_HOST);
   matIA = hypre_TAlloc(int, (localNRows+1) , HYPRE_MEMORY_HOST);
   matJA = hypre_TAlloc(int, localNnz , HYPRE_MEMORY_HOST);
   matAA = hypre_TAlloc(double, localNnz , HYPRE_MEMORY_HOST);

   if (scaleFlag == 1)
   {
      diag  = hypre_TAlloc(double, globalNRows, HYPRE_MEMORY_HOST);
      diag2 = hypre_TAlloc(double, globalNRows, HYPRE_MEMORY_HOST);
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

   rowLengths = hypre_TAlloc(int, localNRows , HYPRE_MEMORY_HOST);
   for ( irow = 0; irow < localNRows; irow++ )
      rowLengths[irow] = matIA[irow+1] - matIA[irow];

   ierr = HYPRE_IJMatrixCreate(mpiComm, startRow, startRow+localNRows-1,
                               startRow, startRow+localNRows-1, &IJmat);
   ierr = HYPRE_IJMatrixSetObjectType(IJmat, HYPRE_PARCSR);
   hypre_assert(!ierr);
   ierr = HYPRE_IJMatrixSetRowSizes(IJmat, rowLengths);
   ierr = HYPRE_IJMatrixInitialize(IJmat);
   hypre_assert(!ierr);
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
      hypre_assert( !ierr );
   }
   hypre_TFree(rowLengths, HYPRE_MEMORY_HOST);
   hypre_TFree(matIA, HYPRE_MEMORY_HOST);
   hypre_TFree(matJA, HYPRE_MEMORY_HOST);
   hypre_TFree(matAA, HYPRE_MEMORY_HOST);

   ierr = HYPRE_IJMatrixAssemble(IJmat);
   hypre_assert( !ierr );
   HYPRE_IJMatrixGetObject(IJmat, (void**) &hypreA);
   HYPRE_IJMatrixSetObjectType(IJmat, -1);
   HYPRE_IJMatrixDestroy(IJmat);
   (*Amat) = (void *) hypreA;
   if ( scaleFlag == 1 )
   {
      hypre_TFree(diag, HYPRE_MEMORY_HOST);
      diag = hypre_TAlloc(double,  localNRows, HYPRE_MEMORY_HOST);
      for ( irow = 0; irow < localNRows; irow++ )
         diag[irow] = diag2[startRow+irow];
      hypre_TFree(diag2, HYPRE_MEMORY_HOST);
   }
   (*scaleVec) = diag;

   return ierr;
}

/***************************************************************************
 * read a matrix file in HB format (sequential)
 *--------------------------------------------------------------------------*/

int MLI_Utils_HypreMatrixReadHBFormat(char *filename, MPI_Comm mpiComm,
                                      void **Amat)
{
   int    *matIA, *matJA, *rowLengths, length, rowNum, startRow,*inds;
   int    irow, lineLeng=200, localNRows, localNCols, localNnz, ierr;
   int    rhsl;
   double *matAA, *vals;
   char   line[200], junk[100];
   FILE   *fp;
   hypre_ParCSRMatrix *hypreA;
   HYPRE_IJMatrix     IJmat;

   fp = fopen(filename, "r");
   if (fp == NULL)
   {
      printf("file not found.\n");
      exit(1);
   }
   fgets(line, lineLeng, fp);
   fgets(line, lineLeng, fp);
   sscanf(line, "%s %s %s %s %d", junk, junk, junk, junk, &rhsl );
   fgets(line, lineLeng, fp);
   sscanf(line, "%s %d %d %d", junk, &localNRows, &localNCols, &localNnz );
   printf("matrix info = %d %d %d\n", localNRows, localNCols, localNnz);
   fgets(line, lineLeng, fp);
   if (rhsl)
      fgets(line, lineLeng, fp);

   matIA = hypre_TAlloc(int, (localNRows+1) , HYPRE_MEMORY_HOST);
   matJA = hypre_TAlloc(int, localNnz , HYPRE_MEMORY_HOST);
   matAA = hypre_TAlloc(double, localNnz , HYPRE_MEMORY_HOST);
   for (irow = 0; irow <= localNRows; irow++) fscanf(fp, "%d", &matIA[irow]);
   for (irow = 0; irow < localNnz; irow++) fscanf(fp, "%d", &matJA[irow]);
   for (irow = 0; irow < localNnz; irow++) fscanf(fp, "%lg", &matAA[irow]);
   for (irow = 0; irow <= localNRows; irow++) matIA[irow]--;
   for (irow = 0; irow < localNnz; irow++) matJA[irow]--;
   if (matAA[0] < 0.0)
      for (irow = 0; irow < localNnz; irow++) matAA[irow] = -matAA[irow];

   fclose(fp);

   startRow = 0;
   rowLengths = hypre_TAlloc(int, localNRows , HYPRE_MEMORY_HOST);
   for ( irow = 0; irow < localNRows; irow++ )
      rowLengths[irow] = matIA[irow+1] - matIA[irow];

   ierr = HYPRE_IJMatrixCreate(mpiComm, startRow, startRow+localNRows-1,
                               startRow, startRow+localNRows-1, &IJmat);
   ierr = HYPRE_IJMatrixSetObjectType(IJmat, HYPRE_PARCSR);
   hypre_assert(!ierr);
   ierr = HYPRE_IJMatrixSetRowSizes(IJmat, rowLengths);
   ierr = HYPRE_IJMatrixInitialize(IJmat);
   hypre_assert(!ierr);
   for (irow = 0; irow < localNRows; irow++)
   {
      length = rowLengths[irow];
      rowNum = irow + startRow;
      inds = &(matJA[matIA[irow]]);
      vals = &(matAA[matIA[irow]]);
      ierr = HYPRE_IJMatrixSetValues(IJmat, 1, &length,(const int *) &rowNum,
                (const int *) inds, (const double *) vals);
      hypre_assert( !ierr );
   }
   hypre_TFree(rowLengths, HYPRE_MEMORY_HOST);
   hypre_TFree(matIA, HYPRE_MEMORY_HOST);
   hypre_TFree(matJA, HYPRE_MEMORY_HOST);
   hypre_TFree(matAA, HYPRE_MEMORY_HOST);

   ierr = HYPRE_IJMatrixAssemble(IJmat);
   hypre_assert( !ierr );
   HYPRE_IJMatrixGetObject(IJmat, (void**) &hypreA);
   HYPRE_IJMatrixSetObjectType(IJmat, -1);
   HYPRE_IJMatrixDestroy(IJmat);
   (*Amat) = (void *) hypreA;
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
   int    irow, k;
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
 * constructor for m-Jacobi preconditioner
 *--------------------------------------------------------------------------*/

int MLI_Utils_mJacobiCreate(MPI_Comm comm, HYPRE_Solver *solver)
{
   HYPRE_MLI_mJacobi *jacobiPtr;

   jacobiPtr = hypre_TAlloc(HYPRE_MLI_mJacobi, 1, HYPRE_MEMORY_HOST);

   if (jacobiPtr == NULL) return 1;

   jacobiPtr->comm_     = comm;
   jacobiPtr->diagonal_ = NULL;
   jacobiPtr->degree_   = 1;
   jacobiPtr->hypreRes_ = NULL;

   *solver = (HYPRE_Solver) jacobiPtr;
   return 0;
}

/***************************************************************************
 * destructor for m-Jacobi preconditioner
 *--------------------------------------------------------------------------*/

int MLI_Utils_mJacobiDestroy(HYPRE_Solver solver)
{
   HYPRE_MLI_mJacobi *jacobiPtr = (HYPRE_MLI_mJacobi *) solver;
   if (jacobiPtr == NULL) return 1;
   hypre_TFree(jacobiPtr->diagonal_, HYPRE_MEMORY_HOST);
   if (jacobiPtr->hypreRes_ != NULL)
      HYPRE_ParVectorDestroy(jacobiPtr->hypreRes_);
   jacobiPtr->diagonal_ = NULL;
   jacobiPtr->hypreRes_ = NULL;
   return 0;
}

/***************************************************************************
 * set polynomial degree
 *--------------------------------------------------------------------------*/

int MLI_Utils_mJacobiSetParams(HYPRE_Solver solver, int degree)
{
   HYPRE_MLI_mJacobi *jacobiPtr = (HYPRE_MLI_mJacobi *) solver;
   if (jacobiPtr == NULL) return 1;
   if (degree > 0) jacobiPtr->degree_ = degree;
   return 0;
}

/***************************************************************************
 * conform to the preconditioner set up from HYPRE
 *--------------------------------------------------------------------------*/

int MLI_Utils_mJacobiSetup(HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                           HYPRE_ParVector b, HYPRE_ParVector x)
{
   int    i, j, nrows, *AI, *AJ, gnrows, *partition, *newPartition, nprocs;
   double *AData;
   hypre_ParCSRMatrix *hypreA;
   hypre_ParVector    *hypreX;
   HYPRE_MLI_mJacobi  *jacobiPtr;

   jacobiPtr = (HYPRE_MLI_mJacobi *) solver;
   if (jacobiPtr == NULL) return 1;
   hypre_TFree(jacobiPtr->diagonal_, HYPRE_MEMORY_HOST);
   hypreX = (hypre_ParVector *) x;
   nrows = hypre_VectorSize(hypre_ParVectorLocalVector(hypreX));
   jacobiPtr->diagonal_ = hypre_TAlloc(double, nrows , HYPRE_MEMORY_HOST);
   hypreA = (hypre_ParCSRMatrix *) A;
   AI = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(hypreA));
   AJ = hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(hypreA));
   AData = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(hypreA));
   for (i = 0; i < nrows; i++)
   {
      jacobiPtr->diagonal_[i] = 1.0;
      for (j = AI[i]; j < AI[i+1]; j++)
      {
         if (AJ[j] == i && AData[j] != 0.0)
         {
            jacobiPtr->diagonal_[i] = AData[j];
            break;
         }
      }
      if (jacobiPtr->diagonal_[i] >= 0.0)
      {
         for (j = AI[i]; j < AI[i+1]; j++)
            if (AJ[j] != i && AData[j] > 0.0)
               jacobiPtr->diagonal_[i] += AData[j];
      }
      else
      {
         for (j = AI[i]; j < AI[i+1]; j++)
            if (AJ[j] != i && AData[j] < 0.0)
               jacobiPtr->diagonal_[i] += AData[j];
      }
      jacobiPtr->diagonal_[i] = 1.0 / jacobiPtr->diagonal_[i];
   }
   if (jacobiPtr->hypreRes_ != NULL)
      HYPRE_ParVectorDestroy(jacobiPtr->hypreRes_);
   gnrows = hypre_ParVectorGlobalSize(hypreX);
   partition = hypre_ParVectorPartitioning(hypreX);
   MPI_Comm_size(jacobiPtr->comm_, &nprocs);
   newPartition = hypre_TAlloc(int, (nprocs+1) , HYPRE_MEMORY_HOST);
   for (i = 0; i <= nprocs; i++) newPartition[i] = partition[i];
   HYPRE_ParVectorCreate(jacobiPtr->comm_, gnrows, newPartition,
                         &(jacobiPtr->hypreRes_));
   HYPRE_ParVectorInitialize(jacobiPtr->hypreRes_);
   return 0;
}

/***************************************************************************
 * conform to the preconditioner apply from HYPRE
 *--------------------------------------------------------------------------*/

int MLI_Utils_mJacobiSolve(HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                           HYPRE_ParVector b, HYPRE_ParVector x)
{
   int                i, j, nrows;
   double             *xData, *rData, omega=1;
   HYPRE_ParVector    res;
   hypre_ParVector    *hypreX, *hypreR;
   HYPRE_MLI_mJacobi  *jacobiPtr;

   jacobiPtr = (HYPRE_MLI_mJacobi *) solver;
   if (jacobiPtr == NULL) return 1;
   res = (HYPRE_ParVector) jacobiPtr->hypreRes_;
   hypreX = (hypre_ParVector *) x;
   hypreR = (hypre_ParVector *) res;
   xData = hypre_VectorData(hypre_ParVectorLocalVector(hypreX));
   rData = hypre_VectorData(hypre_ParVectorLocalVector(hypreR));
   nrows = hypre_VectorSize(hypre_ParVectorLocalVector(hypreX));
   HYPRE_ParVectorCopy(b, res);
   for (j = 0; j < nrows; j++)
      xData[j] = (rData[j] * jacobiPtr->diagonal_[j]);
   for (i = 1; i < jacobiPtr->degree_; i++)
   {
      HYPRE_ParVectorCopy(b, res);
      HYPRE_ParCSRMatrixMatvec(-1.0e0, A, x, 1.0, res);
      for (j = 0; j < nrows; j++)
         xData[j] += omega * (rData[j] * jacobiPtr->diagonal_[j]);
   }
   return 0;
}

/***************************************************************************
 * solve the system using HYPRE pcg
 *--------------------------------------------------------------------------*/

int MLI_Utils_HyprePCGSolve( CMLI *cmli, HYPRE_Matrix A,
                             HYPRE_Vector b, HYPRE_Vector x )
{
   int          numIterations, maxIter=500, mypid;
   double       tol=1.0e-8, norm, setupTime, solveTime;
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

#if 0
      printf("& %3d & %7.2f & %7.2f & %7.2f \\\\\n",numIterations,
        setupTime,solveTime,setupTime+solveTime);
#endif
   }
   return 0;
}

/***************************************************************************
 * solve the system using HYPRE gmres
 *--------------------------------------------------------------------------*/

int MLI_Utils_HypreGMRESSolve(void *precon, HYPRE_Matrix A,
                              HYPRE_Vector b, HYPRE_Vector x, char *pname)
{
   int          numIterations, maxIter=1000, mypid, i, *nSweeps, *rTypes;
   double       tol=1.0e-8, norm, setupTime, solveTime;
   double       *relaxWt, *relaxOmega;
   MPI_Comm     mpiComm;
   HYPRE_Solver gmresSolver, gmresPrecond;
   HYPRE_ParCSRMatrix hypreA;
   CMLI         *cmli;

   hypreA = (HYPRE_ParCSRMatrix) A;
   HYPRE_ParCSRMatrixGetComm(hypreA , &mpiComm);
   HYPRE_ParCSRGMRESCreate(mpiComm, &gmresSolver);
   HYPRE_ParCSRGMRESSetMaxIter(gmresSolver, maxIter);
   HYPRE_ParCSRGMRESSetTol(gmresSolver, tol);
   HYPRE_GMRESSetRelChange(gmresSolver, 0);
   HYPRE_ParCSRGMRESSetPrintLevel(gmresSolver, 2);
   HYPRE_ParCSRGMRESSetKDim(gmresSolver, 100);
   if (!strcmp(pname, "boomeramg"))
   {
      HYPRE_BoomerAMGCreate(&gmresPrecond);
      HYPRE_BoomerAMGSetMaxIter(gmresPrecond, 1);
      HYPRE_BoomerAMGSetCycleType(gmresPrecond, 1);
      HYPRE_BoomerAMGSetMaxLevels(gmresPrecond, 25);
      HYPRE_BoomerAMGSetMeasureType(gmresPrecond, 0);
      HYPRE_BoomerAMGSetDebugFlag(gmresPrecond, 0);
      HYPRE_BoomerAMGSetPrintLevel(gmresPrecond, 0);
      HYPRE_BoomerAMGSetCoarsenType(gmresPrecond, 0);
      HYPRE_BoomerAMGSetStrongThreshold(gmresPrecond, 0.9);
      nSweeps = hypre_TAlloc(int, 4 , HYPRE_MEMORY_HOST);
      for (i = 0; i < 4; i++) nSweeps[i] = 1;
      HYPRE_BoomerAMGSetNumGridSweeps(gmresPrecond, nSweeps);
      rTypes = hypre_TAlloc(int, 4 , HYPRE_MEMORY_HOST);
      for (i = 0; i < 4; i++) rTypes[i] = 6;
      relaxWt = hypre_TAlloc(double, 25 , HYPRE_MEMORY_HOST);
      for (i = 0; i < 25; i++) relaxWt[i] = 1.0;
      HYPRE_BoomerAMGSetRelaxWeight(gmresPrecond, relaxWt);
      relaxOmega = hypre_TAlloc(double, 25 , HYPRE_MEMORY_HOST);
      for (i = 0; i < 25; i++) relaxOmega[i] = 1.0;
      HYPRE_BoomerAMGSetOmega(gmresPrecond, relaxOmega);
      HYPRE_GMRESSetPrecond(gmresSolver,
                       (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                       (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
                       gmresPrecond);
   }
   else if (!strcmp(pname, "mli"))
   {
      cmli = (CMLI *) precon;
      MLI_SetMaxIterations(cmli, 1);
      gmresPrecond = (HYPRE_Solver) cmli;
      HYPRE_GMRESSetPrecond(gmresSolver,
                       (HYPRE_PtrToSolverFcn) MLI_Utils_ParCSRMLISolve,
                       (HYPRE_PtrToSolverFcn) MLI_Utils_ParCSRMLISetup,
                       gmresPrecond);
   }
   else if (!strcmp(pname, "pJacobi"))
   {
      gmresPrecond = (HYPRE_Solver) precon;
      HYPRE_ParCSRGMRESSetMaxIter(gmresSolver, 10);
      HYPRE_ParCSRGMRESSetPrintLevel(gmresSolver, 0);
      HYPRE_GMRESSetPrecond(gmresSolver,
                       (HYPRE_PtrToSolverFcn) MLI_Utils_mJacobiSolve,
                       (HYPRE_PtrToSolverFcn) MLI_Utils_mJacobiSetup,
                       gmresPrecond);
   }
   else if (!strcmp(pname, "mJacobi"))
   {
      gmresPrecond = (HYPRE_Solver) precon;
      HYPRE_ParCSRGMRESSetMaxIter(gmresSolver, 5); /* change this in amgcr too */
      HYPRE_ParCSRGMRESSetPrintLevel(gmresSolver, 0);
      HYPRE_GMRESSetPrecond(gmresSolver,
                       (HYPRE_PtrToSolverFcn) MLI_Utils_mJacobiSolve,
                       (HYPRE_PtrToSolverFcn) MLI_Utils_mJacobiSetup,
                       gmresPrecond);
   }
   setupTime = MLI_Utils_WTime();
   HYPRE_GMRESSetup(gmresSolver, A, b, x);
   solveTime = MLI_Utils_WTime();
   setupTime = solveTime - setupTime;
   HYPRE_GMRESSolve(gmresSolver, A, b, x);
   solveTime = MLI_Utils_WTime() - solveTime;
   HYPRE_ParCSRGMRESGetNumIterations(gmresSolver, &numIterations);
   HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm(gmresSolver, &norm);
   HYPRE_ParCSRGMRESDestroy(gmresSolver);
   MPI_Comm_rank(mpiComm, &mypid);
   if (mypid == 0 && ((!strcmp(pname, "mli")) || (!strcmp(pname, "boomeramg"))))
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
 * solve the system using HYPRE fgmres
 *--------------------------------------------------------------------------*/

int MLI_Utils_HypreFGMRESSolve(void *precon, HYPRE_Matrix A,
                               HYPRE_Vector b, HYPRE_Vector x, char *pname)
{
   int          numIterations, maxIter=1000, mypid, i, *nSweeps, *rTypes;
   double       tol=1.0e-8, norm, setupTime, solveTime;
   double       *relaxWt, *relaxOmega;
   MPI_Comm     mpiComm;
   HYPRE_Solver gmresSolver, gmresPrecond;
   HYPRE_ParCSRMatrix hypreA;
   CMLI         *cmli;

   hypreA = (HYPRE_ParCSRMatrix) A;
   HYPRE_ParCSRMatrixGetComm(hypreA , &mpiComm);
   HYPRE_ParCSRFGMRESCreate(mpiComm, &gmresSolver);
   HYPRE_ParCSRFGMRESSetMaxIter(gmresSolver, maxIter);
   HYPRE_ParCSRFGMRESSetTol(gmresSolver, tol);
   HYPRE_ParCSRFGMRESSetLogging(gmresSolver, 2);
   HYPRE_ParCSRFGMRESSetKDim(gmresSolver, 100);
   if (!strcmp(pname, "boomeramg"))
   {
      HYPRE_BoomerAMGCreate(&gmresPrecond);
      HYPRE_BoomerAMGSetMaxIter(gmresPrecond, 1);
      HYPRE_BoomerAMGSetCycleType(gmresPrecond, 1);
      HYPRE_BoomerAMGSetMaxLevels(gmresPrecond, 25);
      HYPRE_BoomerAMGSetMeasureType(gmresPrecond, 0);
      HYPRE_BoomerAMGSetDebugFlag(gmresPrecond, 0);
      HYPRE_BoomerAMGSetPrintLevel(gmresPrecond, 0);
      HYPRE_BoomerAMGSetCoarsenType(gmresPrecond, 0);
      HYPRE_BoomerAMGSetStrongThreshold(gmresPrecond, 0.9);
      nSweeps = hypre_TAlloc(int, 4 , HYPRE_MEMORY_HOST);
      for (i = 0; i < 4; i++) nSweeps[i] = 1;
      HYPRE_BoomerAMGSetNumGridSweeps(gmresPrecond, nSweeps);
      rTypes = hypre_TAlloc(int, 4 , HYPRE_MEMORY_HOST);
      for (i = 0; i < 4; i++) rTypes[i] = 6;
      relaxWt = hypre_TAlloc(double, 25 , HYPRE_MEMORY_HOST);
      for (i = 0; i < 25; i++) relaxWt[i] = 1.0;
      HYPRE_BoomerAMGSetRelaxWeight(gmresPrecond, relaxWt);
      relaxOmega = hypre_TAlloc(double, 25 , HYPRE_MEMORY_HOST);
      for (i = 0; i < 25; i++) relaxOmega[i] = 1.0;
      HYPRE_BoomerAMGSetOmega(gmresPrecond, relaxOmega);
      HYPRE_ParCSRFGMRESSetMaxIter(gmresSolver, maxIter);
      HYPRE_ParCSRFGMRESSetPrecond(gmresSolver, HYPRE_BoomerAMGSolve,
                       HYPRE_BoomerAMGSetup, gmresPrecond);
   }
   else if (!strcmp(pname, "mli"))
   {
      cmli = (CMLI *) precon;
      MLI_SetMaxIterations(cmli, 1);
      gmresPrecond = (HYPRE_Solver) cmli;
      HYPRE_ParCSRFGMRESSetPrecond(gmresSolver, MLI_Utils_ParCSRMLISolve,
                       MLI_Utils_ParCSRMLISetup, gmresPrecond);
   }
   else if (!strcmp(pname, "pJacobi"))
   {
      gmresPrecond = (HYPRE_Solver) precon;
      HYPRE_ParCSRFGMRESSetMaxIter(gmresSolver, 10);
      HYPRE_ParCSRFGMRESSetLogging(gmresSolver, 0);
      HYPRE_ParCSRFGMRESSetPrecond(gmresSolver, MLI_Utils_mJacobiSolve,
                       MLI_Utils_mJacobiSetup, gmresPrecond);
   }
   else if (!strcmp(pname, "mJacobi"))
   {
      gmresPrecond = (HYPRE_Solver) precon;
      HYPRE_ParCSRFGMRESSetMaxIter(gmresSolver, 5); /* change this in amgcr too */
      HYPRE_ParCSRFGMRESSetLogging(gmresSolver, 0);
      HYPRE_ParCSRFGMRESSetPrecond(gmresSolver, MLI_Utils_mJacobiSolve,
                       MLI_Utils_mJacobiSetup, gmresPrecond);
   }
   setupTime = MLI_Utils_WTime();
   HYPRE_ParCSRFGMRESSetup(gmresSolver, hypreA, (HYPRE_ParVector) b,
                           (HYPRE_ParVector) x);
   solveTime = MLI_Utils_WTime();
   setupTime = solveTime - setupTime;
   HYPRE_ParCSRFGMRESSolve(gmresSolver, hypreA, (HYPRE_ParVector) b,
                           (HYPRE_ParVector) x);
   solveTime = MLI_Utils_WTime() - solveTime;
   HYPRE_ParCSRFGMRESGetNumIterations(gmresSolver, &numIterations);
   HYPRE_ParCSRFGMRESGetFinalRelativeResidualNorm(gmresSolver, &norm);
   HYPRE_ParCSRFGMRESDestroy(gmresSolver);
   MPI_Comm_rank(mpiComm, &mypid);
   if (mypid == 0 && ((!strcmp(pname, "mli")) || (!strcmp(pname, "boomeramg"))))
   {
      printf("\tFGMRES Krylov dimension             = 200\n");
      printf("\tFGMRES maximum iterations           = %d\n", maxIter);
      printf("\tFGMRES convergence tolerance        = %e\n", tol);
      printf("\tFGMRES number of iterations         = %d\n", numIterations);
      printf("\tFGMRES final relative residual norm = %e\n", norm);
      printf("\tFGMRES setup time                   = %e seconds\n",setupTime);
      printf("\tFGMRES solve time                   = %e seconds\n",solveTime);
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
 * quicksort on double and permute integers
 *--------------------------------------------------------------------------*/

int MLI_Utils_DbleQSort2a(double *dlist, int *ilist, int left, int right)
{
   int    i, last, mid, itemp;
   double dtemp;

   if (left >= right) return 0;
   mid          = (left + right) / 2;
   dtemp        = dlist[left];
   dlist[left]  = dlist[mid];
   dlist[mid]   = dtemp;
   if ( ilist != NULL )
   {
      itemp       = ilist[left];
      ilist[left] = ilist[mid];
      ilist[mid]  = itemp;
   }
   last = left;
   for (i = left+1; i <= right; i++)
   {
      if (dlist[i] < dlist[left])
      {
         last++;
         dtemp        = dlist[last];
         dlist[last]  = dlist[i];
         dlist[i]     = dtemp;
         if ( ilist != NULL )
         {
            itemp       = ilist[last];
            ilist[last] = ilist[i];
            ilist[i]    = itemp;
         }
      }
   }
   dtemp        = dlist[left];
   dlist[left]  = dlist[last];
   dlist[last]  = dtemp;
   if ( ilist != NULL )
   {
      itemp       = ilist[left];
      ilist[left] = ilist[last];
      ilist[last] = itemp;
   }
   MLI_Utils_DbleQSort2a(dlist, ilist, left, last-1);
   MLI_Utils_DbleQSort2a(dlist, ilist, last+1, right);
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

   newList  = hypre_TAlloc(int,  totalLeng , HYPRE_MEMORY_HOST);
   indices  = hypre_TAlloc(int,  nList , HYPRE_MEMORY_HOST);
   tree     = hypre_TAlloc(int,  nList , HYPRE_MEMORY_HOST);
   treeInd  = hypre_TAlloc(int,  nList , HYPRE_MEMORY_HOST);
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
         tree[i] = (1 << 30) - 1;
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
         tree[0] = (1 << 30) - 1;
         treeInd[0] = - 1;
      }
      MLI_Utils_IntTreeUpdate(nList, tree, treeInd);
      parseCnt++;
   }
   (*newListOut) = newList;
   (*newNListOut) = newListCnt;
   hypre_TFree(indices, HYPRE_MEMORY_HOST);
   hypre_TFree(tree, HYPRE_MEMORY_HOST);
   hypre_TFree(treeInd, HYPRE_MEMORY_HOST);
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
      Cmat = hypre_TAlloc(double*,  ndim , HYPRE_MEMORY_HOST);
      for ( i = 0; i < ndim; i++ )
         Cmat[i] = hypre_TAlloc(double,  ndim , HYPRE_MEMORY_HOST);
      Cmat[0][0] = 1.0 / Amat[0][0];
      (*Bmat) = Cmat;
      return 0;
   }
   else if ( ndim == 2 )
   {
      denom = Amat[0][0] * Amat[1][1] - Amat[0][1] * Amat[1][0];
      if ( habs( denom ) <= 1.0e-16 ) return -1;
      Cmat = hypre_TAlloc(double*,  ndim , HYPRE_MEMORY_HOST);
      for ( i = 0; i < ndim; i++ )
         Cmat[i] = hypre_TAlloc(double,  ndim , HYPRE_MEMORY_HOST);
      Cmat[0][0] = Amat[1][1] / denom;
      Cmat[1][1] = Amat[0][0] / denom;
      Cmat[0][1] = - ( Amat[0][1] / denom );
      Cmat[1][0] = - ( Amat[1][0] / denom );
      (*Bmat) = Cmat;
      return 0;
   }
   else
   {
      Cmat = hypre_TAlloc(double*,  ndim , HYPRE_MEMORY_HOST);
      for ( i = 0; i < ndim; i++ )
      {
         Cmat[i] = hypre_TAlloc(double,  ndim , HYPRE_MEMORY_HOST);
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

