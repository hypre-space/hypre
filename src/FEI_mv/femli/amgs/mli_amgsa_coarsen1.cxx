/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.36 $
 ***********************************************************************EHEADER*/





// *********************************************************************
// This file is customized to use HYPRE matrix format
// *********************************************************************

// *********************************************************************
// local includes
// ---------------------------------------------------------------------

#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include "HYPRE.h"
#include "utilities/_hypre_utilities.h"
#include "IJ_mv/HYPRE_IJ_mv.h"
#include "seq_mv/seq_mv.h"
#include "parcsr_mv/_hypre_parcsr_mv.h"

#include "vector/mli_vector.h"
#include "amgs/mli_method_amgsa.h"
#include "util/mli_utils.h"
#include "solver/mli_solver.h"
#include "solver/mli_solver_sgs.h"
 
// *********************************************************************
// local defines
// ---------------------------------------------------------------------

#define MLI_METHOD_AMGSA_READY       -1
#define MLI_METHOD_AMGSA_SELECTED    -2
#define MLI_METHOD_AMGSA_PENDING     -3
#define MLI_METHOD_AMGSA_NOTSELECTED -4

#define habs(x) ((x > 0 ) ? x : -(x))

// ********************************************************************* 
// Purpose   : Given Amat and aggregation information, create the 
//             corresponding Pmat using the local aggregation scheme 
// ---------------------------------------------------------------------

double MLI_Method_AMGSA::genP(MLI_Matrix *mli_Amat,
                              MLI_Matrix **Pmat_out,
                              int initCount, int *initAggr)
{
   HYPRE_IJMatrix         IJPmat;
   hypre_ParCSRMatrix     *Amat, *A2mat, *Pmat, *Gmat=NULL, *Jmat, *Pmat2;
   hypre_ParCSRCommPkg    *comm_pkg;
   MLI_Matrix             *mli_Pmat, *mli_Jmat, *mli_A2mat=NULL;
   MLI_Function           *funcPtr;
   MPI_Comm  comm;
   int       i, j, k, index, irow, mypid, numProcs, AStartRow, AEndRow;
   int       ALocalNRows, *partition, naggr, *node2aggr, *eqn2aggr, ierr;
   int       PLocalNCols, PStartCol, PGlobalNCols, *colInd, *P_cols;
   int       PLocalNRows, PStartRow, *rowLengths, rowNum, GGlobalNRows;
   int       blkSize, maxAggSize, *aggCntArray, **aggIndArray;
   int       aggSize, info, nzcnt, *localLabels=NULL, AGlobalNRows;
   double    *colVal, **P_vecs, maxEigen=0, alpha, dtemp;
   double    *qArray, *newNull, *rArray, ritzValues[2];
   char      paramString[200];

   /*-----------------------------------------------------------------
    * fetch matrix and machine information
    *-----------------------------------------------------------------*/

   Amat = (hypre_ParCSRMatrix *) mli_Amat->getMatrix();

#if 0
   sprintf(paramString, "matrix%d", currLevel_);
   hypre_ParCSRMatrixPrintIJ(Amat, 1, 1, paramString);
   printf("matrix %s printed\n", paramString);
#endif

   comm = hypre_ParCSRMatrixComm(Amat);
   MPI_Comm_rank(comm,&mypid);
   MPI_Comm_size(comm,&numProcs);

   /*-----------------------------------------------------------------
    * fetch other matrix information
    *-----------------------------------------------------------------*/

   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) Amat,&partition);
   AStartRow = partition[mypid];
   AEndRow   = partition[mypid+1] - 1;
   AGlobalNRows = partition[numProcs];
   ALocalNRows  = AEndRow - AStartRow + 1;
   free( partition );
   if ( AGlobalNRows < minCoarseSize_ ) 
   {
      if ( mypid == 0 && outputLevel_ > 2 )
      {
         printf("\tMETHOD_AMGSA::genP - stop coarsening (less than min %d)\n",
                minCoarseSize_);
      }
      (*Pmat_out) = NULL;
      return 0.0;
   }

   /*-----------------------------------------------------------------
    * read aggregate information if desired
    *-----------------------------------------------------------------*/

#if 0 /* read aggregate info from stdin */
    if (currLevel_ == 0)
    {
          FILE *file;

          printf("reading aggregates from marker file, n=%d\n",ALocalNRows);
          file = stdin;

          initCount = 0;
          initAggr = new int[ALocalNRows];
          for (i=0; i<ALocalNRows; i++)
          {
              fscanf(file, "%d", &initAggr[i]);
              if (initAggr[i] > initCount)
                  initCount = initAggr[i];
          }
          initCount++;
          printf("number of aggregates = %d\n", initCount);
    }
#endif

   /*-----------------------------------------------------------------
    * reduce Amat based on the block size information (if nodeDofs_ > 1)
    *-----------------------------------------------------------------*/

   if ( initAggr == NULL )
   {
      blkSize = currNodeDofs_;
      if (blkSize > 1 && scalar_ == 0) 
      {
         MLI_Matrix_Compress(mli_Amat, blkSize, &mli_A2mat);
         if ( saLabels_ != NULL && saLabels_[currLevel_] != NULL )
         {
            localLabels = new int[ALocalNRows/blkSize];
            for ( i = 0; i < ALocalNRows; i+=blkSize )
               localLabels[i/blkSize] = saLabels_[currLevel_][i];
         }
         else localLabels = NULL;
      }
      else 
      {
         mli_A2mat = mli_Amat;
         if ( saLabels_ != NULL && saLabels_[currLevel_] != NULL )
            localLabels = saLabels_[currLevel_];
         else
            localLabels = NULL;
      }
      A2mat = (hypre_ParCSRMatrix *) mli_A2mat->getMatrix();
   }

   /*-----------------------------------------------------------------
    * modify minimum aggregate size, if needed
    *-----------------------------------------------------------------*/

   if ( scalar_ == 0 )
   {
      minAggrSize_ = nullspaceDim_ / currNodeDofs_;
      if ( minAggrSize_ <= 1 ) minAggrSize_ = 2;
      if ( currLevel_ == (numLevels_-1) ) minAggrSize_ = 2;
   }
   else minAggrSize_ = nullspaceDim_ * 2;
   if (currLevel_ == 0) minAggrSize_ = minAggrSize_ * 3 / 2;

   /*-----------------------------------------------------------------
    * perform coarsening (If aggregate information is not given, then
    * if dimension of A is small enough and hybrid is on, switch to
    * global coarsening.  Otherwise if the scheme is local, do local
    * coarsening). 
    *-----------------------------------------------------------------*/
  
   if ( initAggr == NULL ) 
   {
      GGlobalNRows = hypre_ParCSRMatrixGlobalNumRows(A2mat);
      if ( GGlobalNRows <= minAggrSize_*numProcs ) 
      {
         formGlobalGraph(A2mat, &Gmat);
         coarsenGlobal(Gmat, &naggr, &node2aggr);
         hypre_ParCSRMatrixDestroy(Gmat);
      }
      else if ( GGlobalNRows > minAggrSize_*numProcs ) 
      {
         formLocalGraph(A2mat, &Gmat, localLabels);
         coarsenLocal(Gmat, &naggr, &node2aggr);
         hypre_ParCSRMatrixDestroy(Gmat);
      }
      if ( blkSize > 1 && scalar_ == 0 ) 
      {
         if ( saLabels_ != NULL && saLabels_[currLevel_] != NULL )
            if (localLabels != NULL) delete [] localLabels;
         if (mli_A2mat != NULL) delete mli_A2mat;
      }
   }
   else 
   {
      blkSize = currNodeDofs_;
      naggr = initCount;
      node2aggr = new int[ALocalNRows];
      for ( i = 0; i < ALocalNRows; i++ ) node2aggr[i] = initAggr[i];
   }

   /*-----------------------------------------------------------------
    * create global P 
    *-----------------------------------------------------------------*/

   if ( initAggr == NULL & numSmoothVec_ == 0 ) 
   {
      if ( GGlobalNRows <= minAggrSize_*numProcs ) 
      {
         genPGlobal(Amat, Pmat_out, naggr, node2aggr);
         if (node2aggr != NULL) delete [] node2aggr;
         return 1.0e39;
      }
   }

   /*-----------------------------------------------------------------
    * fetch the coarse grid information and instantiate P
    * If coarse grid size is below a given threshold, stop
    *-----------------------------------------------------------------*/

   PLocalNCols  = naggr * nullspaceDim_;
   MLI_Utils_GenPartition(comm, PLocalNCols, &partition);
   PStartCol    = partition[mypid];
   PGlobalNCols = partition[numProcs];
   free( partition );
   if ( PGlobalNCols > AGlobalNRows*3/4 )
   {
      (*Pmat_out) = NULL;
      delete [] node2aggr;
      if ( mypid == 0 && outputLevel_ > 2 )
      {
         printf("METHOD_AMGSA::genP - cannot coarsen any further.\n");
      }
      return 0.0;
   }
   PLocalNRows = ALocalNRows;
   PStartRow   = AStartRow;
   ierr = HYPRE_IJMatrixCreate(comm,PStartRow,PStartRow+PLocalNRows-1,
                          PStartCol,PStartCol+PLocalNCols-1,&IJPmat);
   ierr = HYPRE_IJMatrixSetObjectType(IJPmat, HYPRE_PARCSR);
   assert(!ierr);

   /*-----------------------------------------------------------------
    * expand the aggregation information if block size > 1 ==> eqn2aggr
    *-----------------------------------------------------------------*/

   if ( blkSize > 1 && initAggr == NULL && scalar_ == 0 )
   {
      eqn2aggr = new int[ALocalNRows];
      for ( i = 0; i < ALocalNRows; i++ )
         eqn2aggr[i] = node2aggr[i/blkSize];
      delete [] node2aggr;
   }
   else eqn2aggr = node2aggr;
 
   /*-----------------------------------------------------------------
    * construct the next set of labels for the next level
    *-----------------------------------------------------------------*/

   if ( saLabels_ != NULL && saLabels_[currLevel_] != NULL )
   {
      if ( (currLevel_+1) < maxLevels_ )
      {
         if ( saLabels_[currLevel_+1] != NULL ) 
            delete [] saLabels_[currLevel_+1];
         saLabels_[currLevel_+1] = new int[PLocalNCols];
         for ( i = 0; i < PLocalNCols; i++ ) saLabels_[currLevel_+1][i] = -1;
         for ( i = 0; i < naggr; i++ )
         {
            for ( j = 0; j < ALocalNRows; j++ )
               if ( eqn2aggr[j] == i ) break;
            for ( k = 0; k < nullspaceDim_; k++ )
               saLabels_[currLevel_+1][i*nullspaceDim_+k] = 
                                              saLabels_[currLevel_][j];
         }
         for ( i = 0; i < PLocalNCols; i++ ) 
            if ( saLabels_[currLevel_+1][i] < 0 ||
                 saLabels_[currLevel_+1][i] >= naggr ) 
               printf("saLabels[%d][%d] = %d (%d)\n",currLevel_+1,i,
                      saLabels_[currLevel_+1][i], naggr);
      }
   }

   /*-----------------------------------------------------------------
    * compute smoothing factor for the prolongation smoother
    *-----------------------------------------------------------------*/

   if ( (currLevel_ >= SPLevel_ && Pweight_ != 0.0) || 
        !strcmp(preSmoother_, "MLS") ||
        !strcmp(postSmoother_, "MLS"))
   {
      MLI_Utils_ComputeExtremeRitzValues(Amat, ritzValues, 1);
      maxEigen = ritzValues[0];
      if ( mypid == 0 && outputLevel_ > 1 )
         printf("\tEstimated spectral radius of A = %e\n", maxEigen);
      assert ( maxEigen > 0.0 );
      alpha = Pweight_ / maxEigen;
   }

   /*-----------------------------------------------------------------
    * create smooth vectors if this option was chosen
    *-----------------------------------------------------------------*/

   if (currLevel_ == 0 && numSmoothVec_ != 0)
      formSmoothVec(mli_Amat);
      //formSmoothVecLanczos(mli_Amat);

   if (currLevel_ > 0 && numSmoothVec_ != 0)
      smoothTwice(mli_Amat);

   /*-----------------------------------------------------------------
    * create a compact form for the null space vectors 
    * (get ready to perform QR on them)
    *-----------------------------------------------------------------*/

   P_vecs = new double*[nullspaceDim_];
   P_cols = new int[PLocalNRows];
   for (i = 0; i < nullspaceDim_; i++) P_vecs[i] = new double[PLocalNRows];
   for ( irow = 0; irow < PLocalNRows; irow++ )
   {
      if ( eqn2aggr[irow] >= 0 )
      {
         P_cols[irow] = PStartCol + eqn2aggr[irow] * nullspaceDim_;
         if ( nullspaceVec_ != NULL )
         {
            for ( j = 0; j < nullspaceDim_; j++ )
               P_vecs[j][irow] = nullspaceVec_[j*PLocalNRows+irow];
         }
         else
         {
            for ( j = 0; j < nullspaceDim_; j++ )
            {
               if ( irow % nullspaceDim_ == j ) P_vecs[j][irow] = 1.0;
               else                             P_vecs[j][irow] = 0.0;
            }
         }
      }
      else
      {
         P_cols[irow] = -1;
         for ( j = 0; j < nullspaceDim_; j++ ) P_vecs[j][irow] = 0.0;
      }
   }

   /*-----------------------------------------------------------------
    * perform QR for null space
    *-----------------------------------------------------------------*/

   newNull = NULL;
   if ( PLocalNRows > 0 && numSmoothVec_ == 0)
   {
      /* ------ count the size of each aggregate ------ */

      aggCntArray = new int[naggr];
      for ( i = 0; i < naggr; i++ ) aggCntArray[i] = 0;
      for ( irow = 0; irow < PLocalNRows; irow++ )
         if ( eqn2aggr[irow] >= 0 ) aggCntArray[eqn2aggr[irow]]++;
      maxAggSize = 0;
      for ( i = 0; i < naggr; i++ ) 
         if (aggCntArray[i] > maxAggSize) maxAggSize = aggCntArray[i];

      /* ------ register which equation is in which aggregate ------ */

      aggIndArray = new int*[naggr];
      for ( i = 0; i < naggr; i++ ) 
      {
         aggIndArray[i] = new int[aggCntArray[i]];
         aggCntArray[i] = 0;
      }
      for ( irow = 0; irow < PLocalNRows; irow++ )
      {
         index = eqn2aggr[irow];
         if ( index >= 0 )
            aggIndArray[index][aggCntArray[index]++] = irow;
      }

      /* ------ allocate storage for QR factorization ------ */

      qArray  = new double[maxAggSize * nullspaceDim_];
      rArray  = new double[nullspaceDim_ * nullspaceDim_];
      newNull = new double[naggr*nullspaceDim_*nullspaceDim_]; 

      /* ------ perform QR on each aggregate ------ */

      for ( i = 0; i < naggr; i++ ) 
      {
         aggSize = aggCntArray[i];

         if ( aggSize < nullspaceDim_ )
         {
            printf("Aggregation ERROR : underdetermined system in QR.\n");
            printf("            error on Proc %d\n", mypid);
            printf("            error on aggr %d (%d)\n", i, naggr);
            printf("            aggr size is %d\n", aggSize);
            exit(1);
         }
          
         /* ------ put data into the temporary array ------ */

         for ( j = 0; j < aggSize; j++ ) 
         {
            for ( k = 0; k < nullspaceDim_; k++ ) 
               qArray[aggSize*k+j] = P_vecs[k][aggIndArray[i][j]]; 
         }

         /* ------ call QR function ------ */

#if 0
         if ( mypid == 0 && i == 0)
         {
            for ( j = 0; j < aggSize; j++ ) 
            {
               printf("%5d : (size=%d)\n", aggIndArray[i][j], aggSize);
               for ( k = 0; k < nullspaceDim_; k++ ) 
                  printf("%10.3e ", qArray[aggSize*k+j]);
               printf("\n");
            }
         }
#endif
         if ( currLevel_ < (numLevels_-1) )
         {
            info = MLI_Utils_QR(qArray, rArray, aggSize, nullspaceDim_); 
            if (info != 0)
            {
               printf("%4d : Aggregation WARNING : QR returns non-zero for\n",
                      mypid);
               printf("  aggregate %d, size = %d, info = %d\n",i,aggSize,info);
#if 0
/*
               for ( j = 0; j < aggSize; j++ ) 
               {
                  for ( k = 0; k < nullspaceDim_; k++ ) 
                     qArray[aggSize*k+j] = P_vecs[k][aggIndArray[i][j]]; 
               }
*/
               printf("PArray : \n");
               for ( j = 0; j < aggSize; j++ ) 
               {
                  index = aggIndArray[i][j];;
                  printf("%5d : ", index);
                  for ( k = 0; k < nullspaceDim_; k++ ) 
                     printf("%16.8e ", P_vecs[k][index]);
                  printf("\n");
               }
               printf("RArray : \n");
               for ( j = 0; j < nullspaceDim_; j++ )
               {
                  for ( k = 0; k < nullspaceDim_; k++ )
                     printf("%16.8e ", rArray[j+nullspaceDim_*k]);
                  printf("\n");
               }
#endif
            }
         }
         else
         {
            for ( k = 0; k < nullspaceDim_; k++ ) 
            {
               dtemp = 0.0;
               for ( j = 0; j < aggSize; j++ ) 
                  dtemp += qArray[aggSize*k+j] * qArray[aggSize*k+j];
               dtemp = 1.0 / sqrt(dtemp);
               for ( j = 0; j < aggSize; j++ ) 
                  qArray[aggSize*k+j] *= dtemp;
            }
         }

         /* ------ after QR, put the R into the next null space ------ */

         for ( j = 0; j < nullspaceDim_; j++ )
            for ( k = 0; k < nullspaceDim_; k++ )
               newNull[i*nullspaceDim_+j+k*naggr*nullspaceDim_] = 
                         rArray[j+nullspaceDim_*k];

         /* ------ put the P to P_vecs ------ */

         for ( j = 0; j < aggSize; j++ )
         {
            for ( k = 0; k < nullspaceDim_; k++ )
            {
               index = aggIndArray[i][j];
               P_vecs[k][index] = qArray[ k*aggSize + j ];
            }
         } 
      }
      for ( i = 0; i < naggr; i++ ) delete [] aggIndArray[i];
      delete [] aggIndArray;
      delete [] aggCntArray;
      delete [] qArray;
      delete [] rArray;
   }
   else if ( PLocalNRows > 0 && numSmoothVec_ != 0)
   {
      double    *uArray, *sArray, *vtArray, *workArray;

      // printf("using SVD and numSmoothVec_ = %d\n", numSmoothVec_);

      /* ------ count the size of each aggregate ------ */

      aggCntArray = new int[naggr];
      for ( i = 0; i < naggr; i++ ) aggCntArray[i] = 0;
      for ( irow = 0; irow < PLocalNRows; irow++ )
         if ( eqn2aggr[irow] >= 0 ) aggCntArray[eqn2aggr[irow]]++;
      maxAggSize = 0;
      for ( i = 0; i < naggr; i++ ) 
         if (aggCntArray[i] > maxAggSize) maxAggSize = aggCntArray[i];

      /* ------ register which equation is in which aggregate ------ */

      aggIndArray = new int*[naggr];
      for ( i = 0; i < naggr; i++ ) 
      {
         aggIndArray[i] = new int[aggCntArray[i]];
         aggCntArray[i] = 0;
      }
      for ( irow = 0; irow < PLocalNRows; irow++ )
      {
         index = eqn2aggr[irow];
         if ( index >= 0 )
            aggIndArray[index][aggCntArray[index]++] = irow;
      }

      /* ------ allocate storage for SVD factorization ------ */
#ifndef MIN
#define MIN(a,b) ((a)<(b)?(a):(b))
#endif

      uArray  = new double[maxAggSize * numSmoothVec_ ];
      sArray  = new double[MIN(maxAggSize, numSmoothVec_)];
      vtArray = new double[MIN(maxAggSize, numSmoothVec_) * numSmoothVec_];
      workArray = new double[5*(maxAggSize + numSmoothVec_)];
      newNull = new double[naggr*nullspaceDim_*numSmoothVec_]; 

#if 0
      // print members of each aggregate
      for (i=0; i<naggr; i++)
      {
          printf("ien(%d)={[", i+1);
          aggSize = aggCntArray[i];
          for (k=0; k<aggSize; k++)
              printf("%d ", aggIndArray[i][k]+1);
          printf("]};\n");
      }
#endif

      /* ------ perform SVD on each aggregate ------ */

      for ( i = 0; i < naggr; i++ ) 
      {
         aggSize = aggCntArray[i];

         if ( aggSize < nullspaceDim_ )
         {
            printf("Aggregation ERROR : underdetermined system in SVD.\n");
            printf("            error on Proc %d\n", mypid);
            printf("            error on aggr %d (%d)\n", i, naggr);
            printf("            aggr size is %d\n", aggSize);
            exit(1);
         }
          
         /* ------ put data into the temporary array ------ */

         for ( k = 0; k < numSmoothVec_; k++ ) 
         {
            for ( j = 0; j < aggSize; j++ ) 
               uArray[aggSize*k+j] = 
                   nullspaceVec_[PLocalNRows*k+aggIndArray[i][j]];
         }

#if 0
         // look at uArray
         for (k=0; k<numSmoothVec_; k++)
             for (j=0; j<aggSize; j++ )
                 printf("a(%d,%d) = %e\n", j+1, k+1, uArray[aggSize*k+j]);
#endif

         /* ------ call SVD function ------ */

         info = MLI_Utils_SVD(uArray, sArray, vtArray, workArray, 
             aggSize, numSmoothVec_, 5*(maxAggSize + numSmoothVec_)); 

         if (info != 0)
         {
            printf("%4d : Aggregation WARNING : SVD returns non-zero for\n",
                   mypid);
            printf("  aggregate %d, size = %d, info = %d\n",i,aggSize,info);
         }

#if 0
         // look at uArray
         for (k=0; k<3; k++)
             for (j=0; j<aggSize; j++ )
                 printf("u(%d,%d) = %e\n", j+1, k+1, uArray[aggSize*k+j]);
#endif

         /* ------ after SVD, save the next null space ------ */

         for ( k = 0; k < numSmoothVec_; k++ )
            for ( j = 0; j < nullspaceDim_; j++ )
               newNull[i*nullspaceDim_ + j + k*naggr*nullspaceDim_] = 
                         sArray[j] * vtArray[j+MIN(aggSize, numSmoothVec_)*k];

         /* ------ store into P_vecs ------ */

         for ( j = 0; j < aggSize; j++ )
         {
            for ( k = 0; k < nullspaceDim_; k++ )
            {
               index = aggIndArray[i][j];
               P_vecs[k][index] = uArray[ k*aggSize + j ];
            }
         } 
      }
      for ( i = 0; i < naggr; i++ ) delete [] aggIndArray[i];
      delete [] aggIndArray;
      delete [] aggCntArray;
      delete [] uArray;
      delete [] sArray;
      delete [] vtArray;
      delete [] workArray;
   }

#if 0
   // print null space for next level
   for ( k = 0; k < numSmoothVec_; k++ )
       for (j=0; j<naggr*nullspaceDim_; j++)
           printf("r(%3d,%3d) = %15.10e\n", j+1, k+1, newNull[k*naggr*nullspaceDim_+j]);
#endif

   if ( nullspaceVec_ != NULL ) delete [] nullspaceVec_;
   nullspaceVec_ = newNull;
   currNodeDofs_ = nullspaceDim_;

   /*-----------------------------------------------------------------
    * if damping factor for prolongator smoother = 0
    *-----------------------------------------------------------------*/

   if ( currLevel_ < SPLevel_ || Pweight_ == 0.0 )
   {
      /*--------------------------------------------------------------
       * create and initialize Pmat 
       *--------------------------------------------------------------*/

      rowLengths = new int[PLocalNRows];
      for ( i = 0; i < PLocalNRows; i++ ) rowLengths[i] = nullspaceDim_;
      ierr = HYPRE_IJMatrixSetRowSizes(IJPmat, rowLengths);
      ierr = HYPRE_IJMatrixInitialize(IJPmat);
      assert(!ierr);
      delete [] rowLengths;

      /*-----------------------------------------------------------------
       * load and assemble Pmat 
       *-----------------------------------------------------------------*/

      colInd = new int[nullspaceDim_];
      colVal = new double[nullspaceDim_];
      for ( irow = 0; irow < PLocalNRows; irow++ )
      {
         if ( P_cols[irow] >= 0 )
         {
            nzcnt = 0;
            for ( j = 0; j < nullspaceDim_; j++ )
            {
               if ( P_vecs[j][irow] != 0.0 )
               {
                  colInd[nzcnt] = P_cols[irow] + j;
                  colVal[nzcnt++] = P_vecs[j][irow];
               }
            }
            rowNum = PStartRow + irow;
            HYPRE_IJMatrixSetValues(IJPmat, 1, &nzcnt, 
                             (const int *) &rowNum, (const int *) colInd, 
                             (const double *) colVal);
         }
      }
      ierr = HYPRE_IJMatrixAssemble(IJPmat);
      assert( !ierr );
      HYPRE_IJMatrixGetObject(IJPmat, (void **) &Pmat);
      hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) Pmat);
      comm_pkg = hypre_ParCSRMatrixCommPkg(Amat);
      if (!comm_pkg) hypre_MatvecCommPkgCreate(Amat);
      HYPRE_IJMatrixSetObjectType(IJPmat, -1);
      HYPRE_IJMatrixDestroy( IJPmat );
      delete [] colInd;
      delete [] colVal;
   }

   /*-----------------------------------------------------------------
    * form prolongator by P = (I - alpha A) tentP
    *-----------------------------------------------------------------*/

   else
   {
      MLI_Matrix_FormJacobi(mli_Amat, alpha, &mli_Jmat);
      Jmat = (hypre_ParCSRMatrix *) mli_Jmat->getMatrix();
      rowLengths = new int[PLocalNRows];
      for ( i = 0; i < PLocalNRows; i++ ) rowLengths[i] = nullspaceDim_;
      ierr = HYPRE_IJMatrixSetRowSizes(IJPmat, rowLengths);
      ierr = HYPRE_IJMatrixInitialize(IJPmat);
      assert(!ierr);
      delete [] rowLengths;
      colInd = new int[nullspaceDim_];
      colVal = new double[nullspaceDim_];
      for ( irow = 0; irow < PLocalNRows; irow++ )
      {
         if ( P_cols[irow] >= 0 )
         {
            nzcnt = 0;
            for ( j = 0; j < nullspaceDim_; j++ )
            {
               if ( P_vecs[j][irow] != 0.0 )
               {
                  colInd[nzcnt] = P_cols[irow] + j;
                  colVal[nzcnt++] = P_vecs[j][irow];
               }
            }
            rowNum = PStartRow + irow;
            HYPRE_IJMatrixSetValues(IJPmat, 1, &nzcnt, 
                             (const int *) &rowNum, (const int *) colInd, 
                             (const double *) colVal);
         }
      }
      ierr = HYPRE_IJMatrixAssemble(IJPmat);
      assert( !ierr );
      HYPRE_IJMatrixGetObject(IJPmat, (void **) &Pmat2);
      HYPRE_IJMatrixSetObjectType(IJPmat, -1);
      HYPRE_IJMatrixDestroy( IJPmat );
      delete [] colInd;
      delete [] colVal;
      Pmat = hypre_ParMatmul( Jmat, Pmat2);
      hypre_ParCSRMatrixOwnsRowStarts(Jmat) = 0; 
      hypre_ParCSRMatrixOwnsColStarts(Pmat2) = 0;
      hypre_ParCSRMatrixDestroy(Pmat2);
      delete mli_Jmat;
   }

#if 0
   hypre_ParCSRMatrixPrintIJ(Pmat, 1, 1, "pmat");
#endif

   /*-----------------------------------------------------------------
    * clean up
    *-----------------------------------------------------------------*/

   if ( P_cols != NULL ) delete [] P_cols;
   if ( P_vecs != NULL ) 
   {
      for (i = 0; i < nullspaceDim_; i++) 
         if ( P_vecs[i] != NULL ) delete [] P_vecs[i];
      delete [] P_vecs;
   }
   delete [] eqn2aggr;

   /*-----------------------------------------------------------------
    * set up and return the Pmat 
    *-----------------------------------------------------------------*/

   funcPtr = new MLI_Function();
   MLI_Utils_HypreParCSRMatrixGetDestroyFunc(funcPtr);
   sprintf(paramString, "HYPRE_ParCSR" ); 
   mli_Pmat = new MLI_Matrix( Pmat, paramString, funcPtr );
   (*Pmat_out) = mli_Pmat;
   delete funcPtr;
   return maxEigen;
}

// ********************************************************************* 
// Purpose   : Given Amat and aggregation information, create the 
//             corresponding Pmat using the global aggregation scheme 
// ---------------------------------------------------------------------

double MLI_Method_AMGSA::genPGlobal(hypre_ParCSRMatrix *Amat,
                                    MLI_Matrix **PmatOut,
                                    int nAggr, int *aggrMap)
{
   int      mypid, nprocs, *partition, *aggrCnt, PLocalNRows, PStartRow;
   int      irow, jcol, nzcnt, ierr;
   int      PLocalNCols, PStartCol, *rowLengths, rowInd, *colInd;
   double   *colVal, dtemp, *accum, *accum2;
   char     paramString[50];
   MPI_Comm comm;
   hypre_ParCSRMatrix  *Pmat;
   hypre_ParCSRCommPkg *commPkg;
   HYPRE_IJMatrix      IJPmat;
   MLI_Matrix          *mli_Pmat;
   MLI_Function        *funcPtr;

   /*-----------------------------------------------------------------
    * fetch and construct matrix parameters
    *-----------------------------------------------------------------*/

   comm = hypre_ParCSRMatrixComm(Amat);
   MPI_Comm_rank(comm,&mypid);
   MPI_Comm_size(comm,&nprocs);
   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) Amat, 
                                        &partition);
   PStartRow   = partition[mypid];
   PLocalNRows = partition[mypid+1] - PStartRow;
   free(partition);
   if ( nAggr > 0 ) aggrCnt = new int[nAggr];
   for ( irow = 0; irow < nAggr; irow++ ) aggrCnt[irow] = -1;
   for ( irow = 0; irow < nprocs; irow++ )
      if ( aggrCnt[aggrMap[irow]] == -1 ) aggrCnt[aggrMap[irow]] = irow; 
   PStartCol = 0;
   for ( irow = 0; irow < mypid; irow++ )
      if ( aggrCnt[aggrMap[irow]] == irow ) PStartCol += nullspaceDim_;
   if ( aggrCnt[aggrMap[mypid]] == mypid ) PLocalNCols = nullspaceDim_;
   else                                    PLocalNCols = 0;
   if ( nAggr > 0 ) delete [] aggrCnt;
      
   /*-----------------------------------------------------------------
    * initialize P matrix
    *-----------------------------------------------------------------*/

   ierr = HYPRE_IJMatrixCreate(comm,PStartRow,PStartRow+PLocalNRows-1,
                  PStartCol,PStartCol+PLocalNCols-1,&IJPmat);
   ierr = HYPRE_IJMatrixSetObjectType(IJPmat, HYPRE_PARCSR);
   assert(!ierr);

   rowLengths = new int[PLocalNRows];
   for (irow = 0; irow < PLocalNRows; irow++) rowLengths[irow] = nullspaceDim_;
   ierr = HYPRE_IJMatrixSetRowSizes(IJPmat, rowLengths);
   ierr = HYPRE_IJMatrixInitialize(IJPmat);
   assert(!ierr);
   delete [] rowLengths;

   /*-----------------------------------------------------------------
    * create scaling array
    *-----------------------------------------------------------------*/

   accum  = new double[nprocs*nullspaceDim_];
   accum2 = new double[nprocs*nullspaceDim_];
   for ( irow = 0; irow < nprocs*nullspaceDim_; irow++ ) accum[irow] = 0.0;
   for ( irow = 0; irow < nprocs*nullspaceDim_; irow++ ) accum2[irow] = 0.0;
   for ( irow = 0; irow < PLocalNRows; irow++ )
   {
      for ( jcol = 0; jcol < nullspaceDim_; jcol++ )
      {
         dtemp = nullspaceVec_[jcol*PLocalNRows+irow];
         accum[mypid*nullspaceDim_+jcol] += (dtemp * dtemp);
      }
   }
   MPI_Allreduce(accum,accum2,nullspaceDim_*nprocs,MPI_DOUBLE,MPI_SUM,comm);
   for ( irow = 0; irow < nullspaceDim_; irow++ ) accum[irow] = 0.0;
   for ( irow = 0; irow < nprocs; irow++ )
   {
      if ( aggrMap[irow] == aggrMap[mypid] )
      {
         for ( jcol = 0; jcol < nullspaceDim_; jcol++ )
            accum[jcol] += accum2[irow*nullspaceDim_+jcol];
      }
   }
   for ( irow = 0; irow < nullspaceDim_; irow++ )
      accum[irow] = 1.0 / sqrt(accum[irow]);

   /*-----------------------------------------------------------------
    * load P matrix
    *-----------------------------------------------------------------*/

   colInd = new int[nullspaceDim_];
   colVal = new double[nullspaceDim_];
   for ( irow = 0; irow < PLocalNRows; irow++ )
   {
      nzcnt = 0;
      for ( jcol = 0; jcol < nullspaceDim_; jcol++ )
      {
         dtemp = nullspaceVec_[jcol*PLocalNRows+irow];
	 if ( dtemp != 0.0 )
         {
            colInd[nzcnt] = aggrMap[mypid] * nullspaceDim_ + jcol;
            colVal[nzcnt++] = dtemp * accum[jcol];
         }
      }
      rowInd = PStartRow + irow;
      HYPRE_IJMatrixSetValues(IJPmat, 1, &nzcnt, (const int *) &rowInd, 
                              (const int *) colInd, (const double *) colVal);
   }
   delete [] colInd;
   delete [] colVal;
   delete [] accum;
   delete [] accum2;

   /*-----------------------------------------------------------------
    * assemble and create the MLI_Matrix format of the P matrix
    *-----------------------------------------------------------------*/

   ierr = HYPRE_IJMatrixAssemble(IJPmat);
   assert( !ierr );
   HYPRE_IJMatrixGetObject(IJPmat, (void **) &Pmat);
   hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) Pmat);
   commPkg = hypre_ParCSRMatrixCommPkg(Amat);
   if ( !commPkg ) hypre_MatvecCommPkgCreate(Amat);
   HYPRE_IJMatrixSetObjectType(IJPmat, -1);
   HYPRE_IJMatrixDestroy( IJPmat );

   funcPtr = new MLI_Function();
   MLI_Utils_HypreParCSRMatrixGetDestroyFunc(funcPtr);
   sprintf(paramString, "HYPRE_ParCSR" );
   mli_Pmat = new MLI_Matrix( Pmat, paramString, funcPtr );
   (*PmatOut) = mli_Pmat;
   delete funcPtr;

   return 0.0;
}

/* ********************************************************************* *
 * local coarsening scheme (Given a graph, aggregate on the local subgraph)
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::coarsenLocal(hypre_ParCSRMatrix *hypre_graph,
                                   int *mliAggrLeng, int **mliAggrArray)
{
   MPI_Comm  comm;
   int       mypid, numProcs, *partition, startRow, endRow, maxInd;
   int       localNRows, naggr=0, *node2aggr, *aggrSizes, nUndone;
   int       irow, icol, colNum, rowNum, rowLeng, *cols, global_nrows;
   int       *nodeStat, selectFlag, nSelected=0, nNotSelected=0, count;
   int       ibuf[2], itmp[2];
   double    maxVal, *vals;

   /*-----------------------------------------------------------------
    * fetch machine and matrix parameters
    *-----------------------------------------------------------------*/

   comm = hypre_ParCSRMatrixComm(hypre_graph);
   MPI_Comm_rank(comm,&mypid);
   MPI_Comm_size(comm,&numProcs);
   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) hypre_graph, 
                                        &partition);
   startRow = partition[mypid];
   endRow   = partition[mypid+1] - 1;
   free( partition );
   localNRows = endRow - startRow + 1;
   MPI_Allreduce(&localNRows, &global_nrows, 1, MPI_INT, MPI_SUM, comm);
   if ( mypid == 0 && outputLevel_ > 1 )
   {
      printf("\t*** Aggregation(U) : total nodes to aggregate = %d\n",
             global_nrows);
   }

   /*-----------------------------------------------------------------
    * this array is used to determine which row has been aggregated
    *-----------------------------------------------------------------*/

   if ( localNRows > 0 )
   {
      node2aggr = new int[localNRows];
      aggrSizes = new int[localNRows];
      nodeStat  = new int[localNRows];
      for ( irow = 0; irow < localNRows; irow++ ) 
      {
         aggrSizes[irow] = 0;
         node2aggr[irow] = -1;
         nodeStat[irow] = MLI_METHOD_AMGSA_READY;
         rowNum = startRow + irow;
         hypre_ParCSRMatrixGetRow(hypre_graph,rowNum,&rowLeng,NULL,NULL);
         if (rowLeng <= 0) 
         {
            nodeStat[irow] = MLI_METHOD_AMGSA_NOTSELECTED;
            nNotSelected++;
         }
         hypre_ParCSRMatrixRestoreRow(hypre_graph,rowNum,&rowLeng,NULL,NULL);
      }
   }
   else node2aggr = aggrSizes = nodeStat = NULL;

   /*-----------------------------------------------------------------
    * Phase 1 : form aggregates
    *-----------------------------------------------------------------*/

   for ( irow = 0; irow < localNRows; irow++ )
   {
      if ( nodeStat[irow] == MLI_METHOD_AMGSA_READY )
      {
         rowNum = startRow + irow;
         hypre_ParCSRMatrixGetRow(hypre_graph,rowNum,&rowLeng,&cols,NULL);
         selectFlag = 1;
         count      = 1;
         for ( icol = 0; icol < rowLeng; icol++ )
         {
            colNum = cols[icol] - startRow;
            if ( colNum >= 0 && colNum < localNRows )
            {
               if ( nodeStat[colNum] != MLI_METHOD_AMGSA_READY )
               {
                  selectFlag = 0;
                  break;
               }
               else count++;
            }
         }
         if ( selectFlag == 1 && count >= minAggrSize_ )
         {
            nSelected++;
            node2aggr[irow]  = naggr;
            aggrSizes[naggr] = 1;
            nodeStat[irow]  = MLI_METHOD_AMGSA_SELECTED;
            for ( icol = 0; icol < rowLeng; icol++ )
            {
               colNum = cols[icol] - startRow;
               if ( colNum >= 0 && colNum < localNRows )
               {
                  node2aggr[colNum] = naggr;
                  nodeStat[colNum] = MLI_METHOD_AMGSA_SELECTED;
                  aggrSizes[naggr]++;
                  nSelected++;
               }
            }
            naggr++;
         }
         hypre_ParCSRMatrixRestoreRow(hypre_graph,rowNum,&rowLeng,&cols,NULL);
      }
   }
   itmp[0] = naggr;
   itmp[1] = nSelected;
   if (outputLevel_ > 1) MPI_Allreduce(itmp, ibuf, 2, MPI_INT, MPI_SUM, comm);
   if ( mypid == 0 && outputLevel_ > 1 )
   {
      printf("\t*** Aggregation(U) P1 : no. of aggregates     = %d\n",ibuf[0]);
      printf("\t*** Aggregation(U) P1 : no. nodes aggregated  = %d\n",ibuf[1]);
   }

   /*-----------------------------------------------------------------
    * Phase 2 : put the rest into one of the existing aggregates
    *-----------------------------------------------------------------*/

   if ( (nSelected+nNotSelected) < localNRows )
   {
      for ( irow = 0; irow < localNRows; irow++ )
      {
         if ( nodeStat[irow] == MLI_METHOD_AMGSA_READY )
         {
            rowNum = startRow + irow;
            hypre_ParCSRMatrixGetRow(hypre_graph,rowNum,&rowLeng,&cols,&vals);
            maxInd = -1;
            maxVal = 0.0;
            for ( icol = 0; icol < rowLeng; icol++ )
            {
               colNum = cols[icol] - startRow;
               if ( colNum >= 0 && colNum < localNRows )
               {
                  if ( nodeStat[colNum] == MLI_METHOD_AMGSA_SELECTED )
                  {
                     if (vals[icol] > maxVal)
                     {
                        maxInd = colNum;
                        maxVal = vals[icol];
                     }
                  }
               }
            }
            if ( maxInd != -1 )
            {
               node2aggr[irow] = node2aggr[maxInd];
               nodeStat[irow] = MLI_METHOD_AMGSA_PENDING;
               aggrSizes[node2aggr[maxInd]]++;
            }
            hypre_ParCSRMatrixRestoreRow(hypre_graph,rowNum,&rowLeng,&cols,
                                         &vals);
         }
      }
      for ( irow = 0; irow < localNRows; irow++ )
      {
         if ( nodeStat[irow] == MLI_METHOD_AMGSA_PENDING )
         {
            nodeStat[irow] = MLI_METHOD_AMGSA_SELECTED;
            nSelected++;
         }
      } 
   }
   itmp[0] = naggr;
   itmp[1] = nSelected;
   if (outputLevel_ > 1) MPI_Allreduce(itmp,ibuf,2,MPI_INT,MPI_SUM,comm);
   if ( mypid == 0 && outputLevel_ > 1 )
   {
      printf("\t*** Aggregation(U) P2 : no. of aggregates     = %d\n",ibuf[0]);
      printf("\t*** Aggregation(U) P2 : no. nodes aggregated  = %d\n",ibuf[1]);
   }

   /*-----------------------------------------------------------------
    * Phase 3 : form aggregates for all other rows
    *-----------------------------------------------------------------*/

   if ( (nSelected+nNotSelected) < localNRows )
   {
      for ( irow = 0; irow < localNRows; irow++ )
      {
         if ( nodeStat[irow] == MLI_METHOD_AMGSA_READY )
         {
            rowNum = startRow + irow;
            hypre_ParCSRMatrixGetRow(hypre_graph,rowNum,&rowLeng,&cols,NULL);
            count = 1;
            for ( icol = 0; icol < rowLeng; icol++ )
            {
               colNum = cols[icol] - startRow;
               if ( colNum >= 0 && colNum < localNRows )
               {
                  if ( nodeStat[colNum] == MLI_METHOD_AMGSA_READY ) count++;
               }
            }
            if ( count > 1 && count >= minAggrSize_ ) 
            {
               node2aggr[irow]  = naggr;
               nodeStat[irow]  = MLI_METHOD_AMGSA_SELECTED;
               aggrSizes[naggr] = 1;
               nSelected++;
               for ( icol = 0; icol < rowLeng; icol++ )
               {
                  colNum = cols[icol] - startRow;
                  if ( colNum >= 0 && colNum < localNRows )
                  {
                     if ( nodeStat[colNum] == MLI_METHOD_AMGSA_READY )
                     {
                        nodeStat[colNum] = MLI_METHOD_AMGSA_SELECTED;
                        node2aggr[colNum] = naggr;
                        aggrSizes[naggr]++;
                        nSelected++;
                     }
                  }
               }
               naggr++;
            }
            hypre_ParCSRMatrixRestoreRow(hypre_graph,rowNum,&rowLeng,&cols,
                                         NULL);
         }
      }
   }
   itmp[0] = naggr;
   itmp[1] = nSelected;
   if (outputLevel_ > 1) MPI_Allreduce(itmp,ibuf,2,MPI_INT,MPI_SUM,comm);
   if ( mypid == 0 && outputLevel_ > 1 )
   {
      printf("\t*** Aggregation(U) P3 : no. of aggregates     = %d\n",ibuf[0]);
      printf("\t*** Aggregation(U) P3 : no. nodes aggregated  = %d\n",ibuf[1]);
   }

   /*-----------------------------------------------------------------
    * Phase 4 : finally put all lone rows into some neighbor aggregate
    *-----------------------------------------------------------------*/

   if ( (nSelected+nNotSelected) < localNRows )
   {
      for ( irow = 0; irow < localNRows; irow++ )
      {
         if ( nodeStat[irow] == MLI_METHOD_AMGSA_READY )
         {
            rowNum = startRow + irow;
            hypre_ParCSRMatrixGetRow(hypre_graph,rowNum,&rowLeng,&cols,NULL);
            for ( icol = 0; icol < rowLeng; icol++ )
            {
               colNum = cols[icol] - startRow;
               if ( colNum >= 0 && colNum < localNRows )
               {
                  if ( nodeStat[colNum] == MLI_METHOD_AMGSA_SELECTED )
                  {
                     node2aggr[irow] = node2aggr[colNum];
                     nodeStat[irow] = MLI_METHOD_AMGSA_SELECTED;
                     aggrSizes[node2aggr[colNum]]++;
                     nSelected++;
                     break;
                  }
               }
            }
            hypre_ParCSRMatrixRestoreRow(hypre_graph,rowNum,&rowLeng,&cols,
                                         NULL);
         }
      }
   }
   itmp[0] = naggr;
   itmp[1] = nSelected;
   if ( outputLevel_ > 1 ) MPI_Allreduce(itmp,ibuf,2,MPI_INT,MPI_SUM,comm);
   if ( mypid == 0 && outputLevel_ > 1 )
   {
      printf("\t*** Aggregation(U) P4 : no. of aggregates     = %d\n",ibuf[0]);
      printf("\t*** Aggregation(U) P4 : no. nodes aggregated  = %d\n",ibuf[1]);
   }
   nUndone = localNRows - nSelected - nNotSelected;
//if ( nUndone > 0 )
   if ( nUndone > localNRows )
   {
      count = nUndone / minAggrSize_;
      if ( count == 0 ) count = 1;
      count += naggr;
      irow = icol = 0;
      while ( nUndone > 0 )
      {
         if ( nodeStat[irow] == MLI_METHOD_AMGSA_READY )
         {
            node2aggr[irow] = naggr;
            nodeStat[irow] = MLI_METHOD_AMGSA_SELECTED;
            nUndone--;
            nSelected++;
            icol++;
            if ( icol >= minAggrSize_ && naggr < count-1 ) 
            {
               icol = 0;
               naggr++;
            }
         }
         irow++;
      }
      naggr = count;
   }
   itmp[0] = naggr;
   itmp[1] = nSelected;
   if ( outputLevel_ > 1 ) MPI_Allreduce(itmp,ibuf,2,MPI_INT,MPI_SUM,comm);
   if ( mypid == 0 && outputLevel_ > 1 )
   {
      printf("\t*** Aggregation(U) P5 : no. of aggregates     = %d\n",ibuf[0]);
      printf("\t*** Aggregation(U) P5 : no. nodes aggregated  = %d\n",ibuf[1]);
   }

   /*-----------------------------------------------------------------
    * diagnostics
    *-----------------------------------------------------------------*/

   if ( (nSelected+nNotSelected) < localNRows )
   {
#ifdef MLI_DEBUG_DETAILED
      for ( irow = 0; irow < localNRows; irow++ )
      {
         if ( nodeStat[irow] == MLI_METHOD_AMGSA_READY )
         {
            rowNum = startRow + irow;
            printf("%5d : unaggregated node = %8d\n", mypid, rowNum);
            hypre_ParCSRMatrixGetRow(hypre_graph,rowNum,&rowLeng,&cols,NULL);
            for ( icol = 0; icol < rowLeng; icol++ )
            {
               colNum = cols[icol];
               printf("ERROR : neighbor of unselected node %9d = %9d\n", 
                     rowNum, colNum);
            }
         }
      }
#endif
   }

   /*-----------------------------------------------------------------
    * clean up and initialize the output arrays 
    *-----------------------------------------------------------------*/

   if ( localNRows > 0 ) delete [] aggrSizes; 
   if ( localNRows > 0 ) delete [] nodeStat; 
   if ( localNRows == 1 && naggr == 0 )
   {
      node2aggr[0] = 0;
      naggr = 1;
   }
   (*mliAggrArray) = node2aggr;
   (*mliAggrLeng)  = naggr;
   return 0;
}

/* ********************************************************************* *
 * global coarsening scheme (for the coarsest grid only)
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::coarsenGlobal(hypre_ParCSRMatrix *Gmat,
                                    int *mliAggrLeng, int **mliAggrArray)
{
   int                 nRecvs, *recvProcs, mypid, nprocs, *commGraphI;
   int                 *commGraphJ, *recvCounts, i, j, *aggrInds, nAggr;
   int                 pIndex, *aggrCnts, *rowCounts, nRows;
   MPI_Comm            comm;
   hypre_ParCSRCommPkg *commPkg;

   comm    = hypre_ParCSRMatrixComm(Gmat);
   commPkg = hypre_ParCSRMatrixCommPkg(Gmat);
   if ( commPkg == NULL )
   {
      hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) Gmat);
      commPkg = hypre_ParCSRMatrixCommPkg(Gmat);
   }
   nRecvs    = hypre_ParCSRCommPkgNumRecvs(commPkg);
   recvProcs = hypre_ParCSRCommPkgRecvProcs(commPkg);

   MPI_Comm_rank(comm, &mypid);
   MPI_Comm_size(comm, &nprocs);

   commGraphI = new int[nprocs+1]; 
   recvCounts = new int[nprocs];
   MPI_Allgather(&nRecvs, 1, MPI_INT, recvCounts, 1, MPI_INT, comm);
   commGraphI[0] = 0;
   for ( i = 1; i <= nprocs; i++ )
      commGraphI[i] = commGraphI[i-1] + recvCounts[i-1];
   commGraphJ = new int[commGraphI[nprocs]]; 
   MPI_Allgatherv(recvProcs, nRecvs, MPI_INT, commGraphJ,
                  recvCounts, commGraphI, MPI_INT, comm);
   delete [] recvCounts;

   rowCounts = new int[nprocs];
   nRows = hypre_ParCSRMatrixNumRows(Gmat);
   MPI_Allgather(&nRows, 1, MPI_INT, rowCounts, 1, MPI_INT, comm);

   nAggr = 0;
   aggrInds = new int[nprocs];
   aggrCnts = new int[nprocs];
   for ( i = 0; i < nprocs; i++ ) aggrInds[i] = -1;
   for ( i = 0; i < nprocs; i++ ) aggrCnts[i] = 0;
   for ( i = 0; i < nprocs; i++ )
   {
      if (aggrInds[i] == -1)
      {
         aggrCnts[nAggr] = rowCounts[i];
         for ( j = commGraphI[i]; j < commGraphI[i+1]; j++ )
         {
            pIndex = commGraphJ[j];
            if (aggrInds[pIndex] == -1) aggrCnts[nAggr] += rowCounts[i];
         }
         if ( aggrCnts[nAggr] >= minAggrSize_ )
         {
            aggrInds[i] = nAggr; 
            for ( j = commGraphI[i]; j < commGraphI[i+1]; j++ )
            {
               pIndex = commGraphJ[j];
               if (aggrInds[pIndex] == -1) aggrInds[pIndex] = nAggr;
            }
            nAggr++;
         }
         else aggrCnts[nAggr] = 0;
      }
   }
   for ( i = 0; i < nprocs; i++ )
   {
      if (aggrInds[i] == -1)
      {
         aggrInds[i] = nAggr;
         aggrCnts[nAggr] += rowCounts[i];
         if (aggrCnts[nAggr] >= minAggrSize_) nAggr++;
      }
   }
   for ( i = 0; i < nprocs; i++ )
   {
      if (aggrInds[i] == nAggr) aggrInds[i] = nAggr - 1;
   }
   if ( outputLevel_ > 2 && mypid == 0 )
   {
      printf("\tMETHOD_AMGSA::coarsenGlobal - nAggr = %d\n", nAggr); 
   }
   if ( mypid == 0 && outputLevel_ > 1 )
   {
      printf("\t*** Aggregation(C) : no. of aggregates     = %d\n",nAggr);
      printf("\t*** Aggregation(C) : no. nodes aggregated  = %d\n",
             hypre_ParCSRMatrixGlobalNumRows(Gmat));
   }
   delete [] aggrCnts;
   delete [] rowCounts;
   (*mliAggrLeng)  = nAggr;
   (*mliAggrArray) = aggrInds;
   return 0;
}

/***********************************************************************
 * form graph from matrix (internal subroutine)
 * ------------------------------------------------------------------- */

int MLI_Method_AMGSA::formLocalGraph( hypre_ParCSRMatrix *Amat,
                                      hypre_ParCSRMatrix **graph_in,
                                      int *localLabels)
{
   HYPRE_IJMatrix     IJGraph;
   hypre_CSRMatrix    *AdiagBlock;
   hypre_ParCSRMatrix *graph;
   MPI_Comm           comm;
   int                i, j, jj, index, mypid, *partition;
   int                startRow, endRow, *rowLengths;
   int                *AdiagRPtr, *AdiagCols, AdiagNRows, length;
   int                irow, maxRowNnz, ierr, *colInd, labeli, labelj;
   double             *diagData=NULL, *colVal;
   double             *AdiagVals, dcomp1, dcomp2, epsilon;

   /*-----------------------------------------------------------------
    * fetch machine and matrix parameters
    *-----------------------------------------------------------------*/

   assert( Amat != NULL );
   comm = hypre_ParCSRMatrixComm(Amat);
   MPI_Comm_rank(comm,&mypid);

   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) Amat,&partition);
   startRow = partition[mypid];
   endRow   = partition[mypid+1] - 1;
   free( partition );
   AdiagBlock = hypre_ParCSRMatrixDiag(Amat);
   AdiagNRows = hypre_CSRMatrixNumRows(AdiagBlock);
   AdiagRPtr  = hypre_CSRMatrixI(AdiagBlock);
   AdiagCols  = hypre_CSRMatrixJ(AdiagBlock);
   AdiagVals  = hypre_CSRMatrixData(AdiagBlock);

#if 0
// seems to give some problems (12/31/03)
   if ( useSAMGeFlag_ )
   {
      if ( saLabels_ != NULL && saLabels_[currLevel_] != NULL )
      {
         partition = new int[AdiagNRows/currNodeDofs_];
         for ( i = 0; i < AdiagNRows; i+=currNodeDofs_ )
            partition[i/currNodeDofs_] = saLabels_[currLevel_][i];
         qsort0(partition, 0, AdiagNRows/currNodeDofs_-1);
         jj = 1;
         for ( i = 1; i < AdiagNRows/currNodeDofs_; i++ )
            if (partition[i] != partition[i-1]) jj++;
         if ( jj * currNodeDofs_ < AdiagNRows/2 ) 
         {
            for ( i = 0; i < AdiagNRows; i++ )
               saLabels_[currLevel_][i] = 0;
            for ( i = 0; i < AdiagNRows/currNodeDofs_; i++ )
               localLabels[i] = 0;
            if ( currNodeDofs_ > 6 ) nullspaceDim_ = currNodeDofs_ = 6;
         }
         delete [] partition;
      }
   }
#endif
   
   /*-----------------------------------------------------------------
    * construct the diagonal array (diagData) 
    *-----------------------------------------------------------------*/

   if ( threshold_ > 0.0 )
   {
      diagData = new double[AdiagNRows];

#define HYPRE_SMP_PRIVATE irow,j
#include "utilities/hypre_smp_forloop.h"
      for (irow = 0; irow < AdiagNRows; irow++)
      {
         for (j = AdiagRPtr[irow]; j < AdiagRPtr[irow+1]; j++)
         {
            if ( AdiagCols[j] == irow )
            {
               diagData[irow] = AdiagVals[j];
               break;
            }
         }
      }
   }

   /*-----------------------------------------------------------------
    * initialize the graph
    *-----------------------------------------------------------------*/

   ierr = HYPRE_IJMatrixCreate(comm, startRow, endRow, startRow,
                               endRow, &IJGraph);
   ierr = HYPRE_IJMatrixSetObjectType(IJGraph, HYPRE_PARCSR);
   assert(!ierr);

   /*-----------------------------------------------------------------
    * find and initialize the length of each row in the graph
    *-----------------------------------------------------------------*/

   epsilon = threshold_;
   for ( i = 0; i < currLevel_; i++ ) epsilon *= 0.5;
   if ( mypid == 0 && outputLevel_ > 1 )
   {
      printf("\t*** Aggregation(U) : strength threshold       = %8.2e\n",
             epsilon);
   }
   epsilon = epsilon * epsilon;
   rowLengths = new int[AdiagNRows];

#define HYPRE_SMP_PRIVATE irow,j,jj,index,dcomp1,dcomp2
#include "utilities/hypre_smp_forloop.h"
   for ( irow = 0; irow < AdiagNRows; irow++ )
   {
      rowLengths[irow] = 0;
      index = startRow + irow;
      if ( localLabels != NULL ) labeli = localLabels[irow];
      else                       labeli = 0;
      if ( epsilon > 0.0 )
      {
         for (j = AdiagRPtr[irow]; j < AdiagRPtr[irow+1]; j++)
         {
            jj = AdiagCols[j];
            if ( localLabels != NULL ) labelj = localLabels[jj];
            else                       labelj = 0;
            if ( jj != irow )
            {
               dcomp1 = AdiagVals[j] * AdiagVals[j];
               if (dcomp1 > 0.0 && labeli == labelj) rowLengths[irow]++;
            }
         }
      }
      else 
      {
         for (j = AdiagRPtr[irow]; j < AdiagRPtr[irow+1]; j++)
         {
            jj = AdiagCols[j];
            if ( localLabels != NULL ) labelj = localLabels[jj];
            else                       labelj = 0;
            if ( jj != irow && AdiagVals[j] != 0.0 && labeli == labelj )
               rowLengths[irow]++;
         }
      }
   }
   maxRowNnz = 0;
   for ( irow = 0; irow < AdiagNRows; irow++ )
   {
      if ( rowLengths[irow] > maxRowNnz ) maxRowNnz = rowLengths[irow];
   }
   ierr = HYPRE_IJMatrixSetRowSizes(IJGraph, rowLengths);
   ierr = HYPRE_IJMatrixInitialize(IJGraph);
   assert(!ierr);
   delete [] rowLengths;

   /*-----------------------------------------------------------------
    * load and assemble the graph
    *-----------------------------------------------------------------*/

   colInd = new int[maxRowNnz];
   colVal = new double[maxRowNnz];
   for ( irow = 0; irow < AdiagNRows; irow++ )
   {
      length = 0;
      index  = startRow + irow;
      if ( localLabels != NULL ) labeli = localLabels[irow];
      else                       labeli = 0;
      if ( epsilon > 0.0 )
      {
         for (j = AdiagRPtr[irow]; j < AdiagRPtr[irow+1]; j++)
         {
            jj = AdiagCols[j];
            if ( localLabels != NULL ) labelj = localLabels[jj];
            else                       labelj = 0;
            if ( jj != irow )
            {
               dcomp1 = AdiagVals[j] * AdiagVals[j];
               if ( dcomp1 > 0.0 )
               {
                  dcomp2 = habs(diagData[irow] * diagData[jj]);
                  if ( (dcomp1 >= epsilon * dcomp2) && (labeli == labelj) ) 
                  {
                     colVal[length] = dcomp1 / dcomp2;
                     colInd[length++] = jj + startRow;
                  }
               }
            }
         }
      }
      else 
      {
         for (j = AdiagRPtr[irow]; j < AdiagRPtr[irow+1]; j++)
         {
            jj = AdiagCols[j];
            if ( localLabels != NULL ) labelj = localLabels[jj];
            else                       labelj = 0;
            if ( jj != irow )
            {
               if (AdiagVals[j] != 0.0 && (labeli == labelj)) 
               {
                  colVal[length] = AdiagVals[j];
                  colInd[length++] = jj + startRow;
               }
            }
         }
      }
      HYPRE_IJMatrixSetValues(IJGraph, 1, &length, (const int *) &index, 
                              (const int *) colInd, (const double *) colVal);
   }
   ierr = HYPRE_IJMatrixAssemble(IJGraph);
   assert(!ierr);

   /*-----------------------------------------------------------------
    * return the graph and clean up
    *-----------------------------------------------------------------*/

   HYPRE_IJMatrixGetObject(IJGraph, (void **) &graph);
   HYPRE_IJMatrixSetObjectType(IJGraph, -1);
   HYPRE_IJMatrixDestroy(IJGraph);
   (*graph_in) = graph;
   delete [] colInd;
   delete [] colVal;
   if ( threshold_ > 0.0 ) delete [] diagData;
   return 0;
}

/***********************************************************************
 * form global graph from matrix (threshold assumed 0)
 * ------------------------------------------------------------------- */

int MLI_Method_AMGSA::formGlobalGraph( hypre_ParCSRMatrix *Amat,
                                       hypre_ParCSRMatrix **Gmat)
{
   HYPRE_IJMatrix     IJGraph;
   hypre_CSRMatrix    *ADiag, *AOffd;
   hypre_ParCSRMatrix *graph;
   MPI_Comm           comm;
   int                cInd, jcol, index, mypid, nprocs, *partition;
   int                startRow, endRow, *rowLengths, length;
   int                *ADiagI, *ADiagJ, *AOffdI, *AOffdJ, localNRows;
   int                irow, maxRowNnz, ierr, *colInd, *colMapOffd;
   double             *colVal, *ADiagA, *AOffdA;

   /*-----------------------------------------------------------------
    * fetch machine and matrix parameters
    *-----------------------------------------------------------------*/

   assert( Amat != NULL );
   comm = hypre_ParCSRMatrixComm(Amat);
   MPI_Comm_rank(comm,&mypid);
   MPI_Comm_size(comm,&nprocs);

   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) Amat,&partition);
   startRow = partition[mypid];
   endRow   = partition[mypid+1] - 1;
   free( partition );
   ADiag      = hypre_ParCSRMatrixDiag(Amat);
   AOffd      = hypre_ParCSRMatrixOffd(Amat);
   localNRows = hypre_CSRMatrixNumRows(ADiag);

   ADiagI = hypre_CSRMatrixI(ADiag);
   ADiagJ = hypre_CSRMatrixJ(ADiag);
   ADiagA = hypre_CSRMatrixData(ADiag);
   AOffdI = hypre_CSRMatrixI(AOffd);
   AOffdJ = hypre_CSRMatrixJ(AOffd);
   AOffdA = hypre_CSRMatrixData(AOffd);
   
   /*-----------------------------------------------------------------
    * initialize the graph
    *-----------------------------------------------------------------*/

   ierr = HYPRE_IJMatrixCreate(comm, startRow, endRow, startRow,
                               endRow, &IJGraph);
   ierr = HYPRE_IJMatrixSetObjectType(IJGraph, HYPRE_PARCSR);
   assert(!ierr);

   /*-----------------------------------------------------------------
    * find and initialize the length of each row in the graph
    *-----------------------------------------------------------------*/

   if (localNRows > 0) rowLengths = new int[localNRows];

   for ( irow = 0; irow < localNRows; irow++ )
   {
      rowLengths[irow] = 0;
      for (jcol = ADiagI[irow]; jcol < ADiagI[irow+1]; jcol++)
      {
         cInd = ADiagJ[jcol];
         if ( cInd != irow && ADiagA[jcol] != 0.0 ) rowLengths[irow]++;
      }
      if (nprocs > 1)
      {
         for (jcol = AOffdI[irow]; jcol < AOffdI[irow+1]; jcol++)
            if ( AOffdA[jcol] != 0.0 ) rowLengths[irow]++;
      }
   }
   maxRowNnz = 0;
   for ( irow = 0; irow < localNRows; irow++ )
   {
      if ( rowLengths[irow] > maxRowNnz ) maxRowNnz = rowLengths[irow];
   }
   ierr = HYPRE_IJMatrixSetRowSizes(IJGraph, rowLengths);
   ierr = HYPRE_IJMatrixInitialize(IJGraph);
   assert(!ierr);
   if (localNRows > 0) delete [] rowLengths;

   /*-----------------------------------------------------------------
    * load and assemble the graph
    *-----------------------------------------------------------------*/

   if (localNRows > 0) colInd = new int[maxRowNnz];
   if (localNRows > 0) colVal = new double[maxRowNnz];
   if (nprocs > 1) colMapOffd = hypre_ParCSRMatrixColMapOffd(Amat);
   for ( irow = 0; irow < localNRows; irow++ )
   {
      length = 0;
      index  = startRow + irow;
      for (jcol = ADiagI[irow]; jcol < ADiagI[irow+1]; jcol++)
      {
         cInd = ADiagJ[jcol];
         if ( cInd != irow && ADiagA[jcol] != 0.0) 
         {
            colVal[length] = ADiagA[jcol];
            colInd[length++] = cInd + startRow;
         }
      }
      if (nprocs > 1)
      {
         for (jcol = AOffdI[irow]; jcol < AOffdI[irow+1]; jcol++)
         {
            cInd = AOffdJ[jcol];
            if ( AOffdA[jcol] != 0.0) 
            {
               colVal[length] = AOffdA[jcol];
               colInd[length++] = colMapOffd[cInd];
            }
         }
      }
      HYPRE_IJMatrixSetValues(IJGraph, 1, &length, (const int *) &index, 
                              (const int *) colInd, (const double *) colVal);
   }
   ierr = HYPRE_IJMatrixAssemble(IJGraph);
   assert(!ierr);

   /*-----------------------------------------------------------------
    * return the graph and clean up
    *-----------------------------------------------------------------*/

   HYPRE_IJMatrixGetObject(IJGraph, (void **) &graph);
   HYPRE_IJMatrixSetObjectType(IJGraph, -1);
   HYPRE_IJMatrixDestroy(IJGraph);
   (*Gmat) = graph;
   if (localNRows > 0) delete [] colInd;
   if (localNRows > 0) delete [] colVal;
   return 0;
}

#undef MLI_METHOD_AMGSA_READY
#undef MLI_METHOD_AMGSA_SELECTED
#undef MLI_METHOD_AMGSA_PENDING
#undef MLI_METHOD_AMGSA_NOTSELECTED

/***********************************************************************
 * Construct the initial smooth vectors and put them in nullspaceVec_
 * ------------------------------------------------------------------- */

int MLI_Method_AMGSA::formSmoothVec(MLI_Matrix *mli_Amat)
{
   hypre_ParCSRMatrix     *Amat;
   MPI_Comm  comm;
   int mypid, nprocs;
   int *partition;
   hypre_ParVector    *trial_sol, *zero_rhs;
   MLI_Vector *mli_rhs, *mli_sol;
   hypre_Vector       *sol_local;
   double *sol_data;
   int local_nrows;
   char   paramString[200];

   MLI_Solver_SGS *smoother;
   double *nsptr;
   int i, j;

   /* warn if nullspaceVec_ is not NULL */
   if (nullspaceVec_ != NULL)
   {
       printf("Warning: formSmoothVec: zeroing nullspaceVec_\n");
       delete [] nullspaceVec_;
       nullspaceVec_ = NULL;
   }

   Amat = (hypre_ParCSRMatrix *) mli_Amat->getMatrix();
   comm = hypre_ParCSRMatrixComm(Amat);
   MPI_Comm_rank(comm,&mypid);
   MPI_Comm_size(comm,&nprocs);

   /*-----------------------------------------------------------------
    * create MLI_Vectors
    *-----------------------------------------------------------------*/

   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) Amat,
                                        &partition);
   zero_rhs = hypre_ParVectorCreate(comm, partition[nprocs], partition);
   hypre_ParVectorInitialize( zero_rhs );
   hypre_ParVectorSetConstantValues( zero_rhs, 0.0 );
   strcpy( paramString, "HYPRE_ParVector" );
   mli_rhs = new MLI_Vector( (void*) zero_rhs, paramString, NULL );

   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) Amat,
                                        &partition);
   trial_sol = hypre_ParVectorCreate(comm, partition[nprocs], partition);
   hypre_ParVectorInitialize( trial_sol );
   mli_sol = new MLI_Vector( (void*) trial_sol, paramString, NULL );

   local_nrows = partition[mypid+1] - partition[mypid];
   sol_local = hypre_ParVectorLocalVector(trial_sol);
   sol_data  = hypre_VectorData(sol_local);

   /*-----------------------------------------------------------------
    * allocate space for smooth vectors and set up smoother
    *-----------------------------------------------------------------*/

   nullspaceVec_ = new double[local_nrows*numSmoothVec_];
   nsptr = nullspaceVec_;

   strcpy( paramString, "SGS" );
   smoother = new MLI_Solver_SGS( paramString );
   smoother->setParams(numSmoothVecSteps_, NULL);
   smoother->setup(mli_Amat);

   /*-----------------------------------------------------------------
    * smooth the vectors
    *-----------------------------------------------------------------*/

   for (i=0; i<numSmoothVec_; i++)
   {
       for (j=0; j<local_nrows; j++)
           sol_data[j] = 2.*((double)rand() / (double)RAND_MAX)-1.;

       /* call smoother */
       smoother->solve(mli_rhs, mli_sol);
       MLI_Utils_ScaleVec(Amat, trial_sol);

       /* extract solution */
       for (j=0; j<local_nrows; j++)
           *nsptr++ = sol_data[j];
   }

   hypre_ParVectorDestroy(zero_rhs);
   hypre_ParVectorDestroy(trial_sol);
   delete smoother;

   return 0;
}

/***********************************************************************
 * Construct the initial smooth vectors and put them in nullspaceVec_
 * using Lanczos
 * ------------------------------------------------------------------- */

int MLI_Method_AMGSA::formSmoothVecLanczos(MLI_Matrix *mli_Amat)
{
    hypre_ParCSRMatrix *Amat;
    MPI_Comm comm;
    int mypid, nprocs;
    int *partition;
    hypre_ParVector    *trial_sol;
    int local_nrows;
    hypre_Vector       *sol_local;
    double *sol_data;
    double *nsptr;
    int i, j;

    Amat = (hypre_ParCSRMatrix *) mli_Amat->getMatrix();
    comm = hypre_ParCSRMatrixComm(Amat);
    MPI_Comm_rank(comm, &mypid);
    MPI_Comm_size(comm, &nprocs);

    HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) Amat, &partition);
    local_nrows = partition[mypid+1] - partition[mypid];

    trial_sol = hypre_ParVectorCreate(comm, partition[nprocs], partition);
    hypre_ParVectorInitialize( trial_sol );

    sol_local = hypre_ParVectorLocalVector(trial_sol);
    sol_data  = hypre_VectorData(sol_local);

    if (nullspaceVec_ != NULL)
    {
        printf("Warning: formSmoothVecLanczos: zeroing nullspaceVec_\n");
        delete [] nullspaceVec_;
        nullspaceVec_ = NULL;
    }

    nullspaceVec_ = new double[local_nrows*numSmoothVec_];
    nsptr = nullspaceVec_;

    MLI_Utils_ComputeLowEnergyLanczos(Amat, numSmoothVecSteps_,
        numSmoothVec_, nullspaceVec_);

    /* need to scale the individual vectors */
    for (i=0; i<numSmoothVec_; i++)
    {
        double *hold;

        /* copy part of vector into sol_data */
        hold = nsptr;
        for (j=0; j<local_nrows; j++)
        {
            sol_data[j] = *nsptr++;
            //printf("v%d %20.14f\n", i, sol_data[j]);
        }

        MLI_Utils_ScaleVec(Amat, trial_sol);

        /* extract scaled vector */
        nsptr = hold;
        for (j=0; j<local_nrows; j++)
        {
            *nsptr++ = sol_data[j];
            //printf("w%d %20.14f\n", i, sol_data[j]);
        }
    }

    return 0;
}

/***********************************************************************
 * Smooth the vectors in nullspaceVec_
 * There are numSmoothVec_ of them, and smooth them twice with SGS.
 * ------------------------------------------------------------------- */

int MLI_Method_AMGSA::smoothTwice(MLI_Matrix *mli_Amat)
{
   hypre_ParCSRMatrix     *Amat;
   MPI_Comm  comm;
   int mypid, nprocs;
   int *partition;
   hypre_ParVector    *trial_sol, *zero_rhs;
   MLI_Vector *mli_rhs, *mli_sol;
   hypre_Vector       *sol_local;
   double *sol_data;
   int local_nrows;
   char paramString[200];

   MLI_Solver_SGS *smoother;
   double *nsptr;
   int i, j;

   printf("Smoothing twice\n");

   Amat = (hypre_ParCSRMatrix *) mli_Amat->getMatrix();
   comm = hypre_ParCSRMatrixComm(Amat);
   MPI_Comm_rank(comm,&mypid);
   MPI_Comm_size(comm,&nprocs);

   /*-----------------------------------------------------------------
    * create MLI_Vectors
    *-----------------------------------------------------------------*/

   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) Amat,
                                        &partition);
   zero_rhs = hypre_ParVectorCreate(comm, partition[nprocs], partition);
   hypre_ParVectorInitialize( zero_rhs );
   hypre_ParVectorSetConstantValues( zero_rhs, 0.0 );
   strcpy( paramString, "HYPRE_ParVector" );
   mli_rhs = new MLI_Vector( (void*) zero_rhs,  paramString, NULL );

   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) Amat,
                                        &partition);
   trial_sol = hypre_ParVectorCreate(comm, partition[nprocs], partition);
   hypre_ParVectorInitialize( trial_sol );
   mli_sol = new MLI_Vector( (void*) trial_sol, paramString, NULL );

   local_nrows = partition[mypid+1] - partition[mypid];
   sol_local = hypre_ParVectorLocalVector(trial_sol);
   sol_data  = hypre_VectorData(sol_local);

   /*-----------------------------------------------------------------
    * set up smoother
    *-----------------------------------------------------------------*/

   strcpy( paramString, "SGS" );
   smoother = new MLI_Solver_SGS( paramString );
   smoother->setParams(2, NULL); // 2 smoothing steps
   smoother->setup(mli_Amat);

   /*-----------------------------------------------------------------
    * smooth the vectors
    *-----------------------------------------------------------------*/

   nsptr = nullspaceVec_;
   for (i=0; i<numSmoothVec_; i++)
   {
       double *hold;

       /* fill in current vector */
       hold = nsptr;
       for (j=0; j<local_nrows; j++)
           sol_data[j] = *nsptr++;

       /* call smoother */
       smoother->solve(mli_rhs, mli_sol);
       MLI_Utils_ScaleVec(Amat, trial_sol); // need this

       /* extract solution */
       nsptr = hold;
       for (j=0; j<local_nrows; j++)
           *nsptr++ = sol_data[j];
   }

   hypre_ParVectorDestroy(zero_rhs);
   hypre_ParVectorDestroy(trial_sol);
   delete smoother;

   return 0;
}
