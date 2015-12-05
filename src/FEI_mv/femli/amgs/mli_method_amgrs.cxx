/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.12 $
 ***********************************************************************EHEADER*/





#define PDEGREE 1
#define MU      0.5
#define pruneFactor 0.1

#ifdef WIN32
#define strcmp _stricmp
#endif

/* #define MLI_USE_HYPRE_MATMATMULT */

#include <string.h>
#include <assert.h>
#include "HYPRE.h"
#include "parcsr_ls/_hypre_parcsr_ls.h"
#include "util/mli_utils.h"
#include "matrix/mli_matrix.h"
#include "matrix/mli_matrix_misc.h"
#include "vector/mli_vector.h"
#include "solver/mli_solver.h"
#include "amgs/mli_method_amgrs.h"

#define habs(x)  ((x) > 0 ? x : -(x))

/* ********************************************************************* *
 * constructor
 * --------------------------------------------------------------------- */

MLI_Method_AMGRS::MLI_Method_AMGRS( MPI_Comm comm ) : MLI_Method( comm )
{
   char name[100];

   strcpy(name, "AMGRS");
   setName( name );
   setID( MLI_METHOD_AMGRS_ID );
   outputLevel_      = 0;
   maxLevels_        = 25;
   numLevels_        = 25;
   currLevel_        = 0;
   coarsenScheme_    = MLI_METHOD_AMGRS_FALGOUT;
   measureType_      = 0;              /* default : local measure */
   threshold_        = 0.5;
   nodeDOF_          = 1;
   minCoarseSize_    = 200;
   maxRowSum_        = 0.9;
   symmetric_        = 1;
   useInjectionForR_ = 0;
   truncFactor_      = 0.0;
   mxelmtsP_         = 0;
   strcpy(smoother_, "Jacobi");
   smootherNSweeps_ = 2;
   smootherWeights_  = new double[2];
   smootherWeights_[0] = smootherWeights_[1] = 0.667;
   smootherPrintRNorm_ = 0;
   smootherFindOmega_  = 0;
   strcpy(coarseSolver_, "SGS");
   coarseSolverNSweeps_ = 20;
   coarseSolverWeights_ = new double[20];
   for ( int j = 0; j < 20; j++ ) coarseSolverWeights_[j] = 1.0;
   RAPTime_            = 0.0;
   totalTime_          = 0.0;
}

/* ********************************************************************* *
 * destructor
 * --------------------------------------------------------------------- */

MLI_Method_AMGRS::~MLI_Method_AMGRS()
{
   if ( smootherWeights_     != NULL ) delete [] smootherWeights_;
   if ( coarseSolverWeights_ != NULL ) delete [] coarseSolverWeights_;
}

/* ********************************************************************* *
 * set parameters
 * --------------------------------------------------------------------- */

int MLI_Method_AMGRS::setParams(char *in_name, int argc, char *argv[])
{
   int        level, size, nSweeps=1;
   double     thresh, *weights=NULL;
   char       param1[256], param2[256];

   sscanf(in_name, "%s", param1);
   if ( !strcmp(param1, "setOutputLevel" ))
   {
      sscanf(in_name,"%s %d", param1, &level);
      return ( setOutputLevel( level ) );
   }
   else if ( !strcmp(param1, "setNumLevels" ))
   {
      sscanf(in_name,"%s %d", param1, &level);
      return ( setNumLevels( level ) );
   }
   else if ( !strcmp(param1, "setCoarsenScheme" ))
   {
      sscanf(in_name,"%s %s", param1, param2);
      if ( !strcmp(param2, "cljp" ) )
         return ( setCoarsenScheme( MLI_METHOD_AMGRS_CLJP ) );
      else if ( !strcmp(param2, "ruge" ) )
         return ( setCoarsenScheme( MLI_METHOD_AMGRS_RUGE ) );
      else if ( !strcmp(param2, "falgout" ) )
         return ( setCoarsenScheme( MLI_METHOD_AMGRS_FALGOUT ) );
      else
      {
         printf("MLI_Method_AMGRS::setParams ERROR : setCoarsenScheme not");
         printf(" valid.  Valid options are : cljp, ruge, and falgout \n");
         return 1;
      }
   }
   else if ( !strcmp(param1, "setMeasureType" ))
   {
      sscanf(in_name,"%s %s", param1, param2);
      if ( !strcmp(param2, "local" ) )
         return ( setMeasureType( 0 ) );
      else if ( !strcmp(param2, "global" ) )
         return ( setMeasureType( 1 ) );
      else
      {
         printf("MLI_Method_AMGRS::setParams ERROR : setMeasureType not");
         printf(" valid.  Valid options are : local or global\n");
         return 1;
      }
   }
   else if ( !strcmp(param1, "setStrengthThreshold" ))
   {
      sscanf(in_name,"%s %lg", param1, &thresh);
      return ( setStrengthThreshold( thresh ) );
   }
   else if ( !strcmp(param1, "setTruncationFactor" ))
   {
      sscanf(in_name,"%s %lg", param1, &truncFactor_);
      return 0;
   }
   else if ( !strcmp(param1, "setPMaxElmts" ))
   {
      sscanf(in_name,"%s %d", param1, &mxelmtsP_);
      return 0;
   }
   else if ( !strcmp(param1, "setNodeDOF" ))
   {
      sscanf(in_name,"%s %d", param1, &size);
      return ( setNodeDOF( size ) );
   }
   else if ( !strcmp(param1, "setNullSpace" ))
   {
      size = *(int *) argv[0];
      return ( setNodeDOF( size ) );
   }
   else if ( !strcmp(param1, "setMinCoarseSize" ))
   {
      sscanf(in_name,"%s %d", param1, &size);
      return ( setMinCoarseSize( size ) );
   }
   else if ( !strcmp(param1, "nonsymmetric" ))
   {
      symmetric_ = 0;
      return 0;
   }
   else if ( !strcmp(param1, "useInjectionForR" ))
   {
      useInjectionForR_ = 1;
      return 0;
   }
   else if ( !strcmp(param1, "setSmoother" ) || 
             !strcmp(param1, "setPreSmoother" ))
   {
      sscanf(in_name,"%s %s", param1, param2);
      if ( argc != 2 )
      {
         printf("MLI_Method_AMGRS::setParams ERROR - setSmoother needs");
         printf(" 2 arguments.\n");
         printf("     argument[0] : number of relaxation sweeps \n");
         printf("     argument[1] : relaxation weights\n");
         return 1;
      } 
      nSweeps = *(int *)   argv[0];
      weights = (double *) argv[1];
      return ( setSmoother(param2, nSweeps, weights) );
   }
   else if ( !strcmp(param1, "setSmootherPrintRNorm" ))
   {
      smootherPrintRNorm_ = 1;
      return 0;
   }
   else if ( !strcmp(param1, "setSmootherFindOmega" ))
   {
      smootherFindOmega_ = 1;
      return 0;
   }
   else if ( !strcmp(param1, "setCoarseSolver" ))
   {
      sscanf(in_name,"%s %s", param1, param2);
      if ( strcmp(param2, "SuperLU") && argc != 2 )
      {
         printf("MLI_Method_AMGRS::setParams ERROR - setCoarseSolver needs");
         printf(" 2 arguments.\n");
         printf("     argument[0] : number of relaxation sweeps \n");
         printf("     argument[1] : relaxation weights\n");
         return 1;
      } 
      else if ( strcmp(param2, "SuperLU") )
      {
         nSweeps   = *(int *)   argv[0];
         weights   = (double *) argv[1];
      }
      else if ( !strcmp(param2, "SuperLU") )
      {
         nSweeps = 1;
         weights = NULL;
      }
      return ( setCoarseSolver(param2, nSweeps, weights) );
   }
   else if ( !strcmp(param1, "print" ))
   {
      return ( print() );
   }
   return 1;
}

/***********************************************************************
 * generate multilevel structure
 * --------------------------------------------------------------------- */

int MLI_Method_AMGRS::setup( MLI *mli ) 
{
   int             k, level, irow, localNRows, mypid, nprocs, startRow;
   int             numNodes, one=1, globalNRows, *coarsePartition;
   int             *CFMarkers, coarseNRows, *dofArray, *cdofArray=NULL;
   int             *reduceArray1, *reduceArray2, *rowLengs, ierr, zeroNRows;
   int             startCol, localNCols, colInd, rowNum;
   int             globalCoarseNRows, numTrials;
   int             *mapStoA;
   double          startTime, elapsedTime, colVal=1.0;
   char            paramString[100], *targv[10];
   MLI_Matrix      *mli_Pmat, *mli_Rmat, *mli_APmat, *mli_Amat, *mli_cAmat;
   MLI_Matrix      *mli_ATmat, *mli_Affmat, *mli_Afcmat;
   MLI_Solver      *smootherPtr, *csolverPtr;
   MPI_Comm        comm;
   MLI_Function    *funcPtr;
   HYPRE_IJMatrix  IJRmat;
   hypre_ParCSRMatrix *hypreA, *hypreS, *hypreAT, *hypreST, *hypreP, *hypreR;
   hypre_ParCSRMatrix *hypreRT, *hypreS2=NULL;
#ifdef MLI_USE_HYPRE_MATMATMULT
   hypre_ParCSRMatrix *hypreAP, *hypreCA;
#endif

#ifdef MLI_DEBUG_DETAILED
   printf("MLI_Method_AMGRS::setup begins...\n");
#endif

   /* --------------------------------------------------------------- */
   /* fetch machine parameters                                        */
   /* --------------------------------------------------------------- */

   RAPTime_ = 0.0;
   comm     = getComm();
   MPI_Comm_rank( comm, &mypid );
   MPI_Comm_size( comm, &nprocs );
   totalTime_ = MLI_Utils_WTime();

   /* --------------------------------------------------------------- */
   /* traverse all levels                                             */
   /* --------------------------------------------------------------- */

   for ( level = 0; level < numLevels_; level++ )
   {
      if ( mypid == 0 && outputLevel_ > 0 )
      {
         printf("\t*****************************************************\n");
         printf("\t*** Ruge Stuben AMG : level = %d\n", level);
         printf("\t-----------------------------------------------------\n");
      }
      currLevel_ = level;
      if ( level == numLevels_-1 ) break;

      /* ------fetch fine grid matrix----------------------------------- */

      mli_Amat = mli->getSystemMatrix(level);
      assert ( mli_Amat != NULL );
      hypreA = (hypre_ParCSRMatrix *) mli_Amat->getMatrix();
      startRow    = hypre_ParCSRMatrixFirstRowIndex(hypreA);
      localNRows  = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(hypreA));
      globalNRows = hypre_ParCSRMatrixGlobalNumRows(hypreA);

      /* ------create strength matrix----------------------------------- */

      numNodes = localNRows / nodeDOF_;
      if ( level == 0 && (numNodes * nodeDOF_) != localNRows )
      {
         printf("\tMLI_Method_AMGRS::setup - nrows not divisible by dof.\n");
         printf("\tMLI_Method_AMGRS::setup - revert nodeDOF to 1.\n");
         nodeDOF_ = 1; 
         numNodes = localNRows / nodeDOF_;
      }
      if ( level == 0 )
      {
         if ( localNRows > 0 ) dofArray = new int[localNRows];
         else                  dofArray = NULL;
         for ( irow = 0; irow < localNRows; irow+=nodeDOF_ )
            for ( k = 0; k < nodeDOF_; k++ ) dofArray[irow+k] = k;
      }
      else
      {
         if ( level > 0 && dofArray != NULL ) delete [] dofArray;
         dofArray = cdofArray;
      }
      hypre_BoomerAMGCreateS(hypreA, threshold_, maxRowSum_, nodeDOF_,
                             dofArray, &hypreS);
      if ( threshold_ > 0 )
      {
         hypre_BoomerAMGCreateSCommPkg(hypreA, hypreS, &mapStoA);
      }
      else mapStoA = NULL;

      if ( coarsenScheme_ == MLI_METHOD_AMGRS_CR )
      {
         hypre_BoomerAMGCreateS(hypreA, 1.0e-16, maxRowSum_, nodeDOF_,
                                dofArray, &hypreS2);
      }
      else hypreS2 = NULL;

      /* ------perform coarsening--------------------------------------- */

      switch ( coarsenScheme_ )
      {
         case MLI_METHOD_AMGRS_CLJP :
              hypre_BoomerAMGCoarsen(hypreS, hypreA, 0, outputLevel_,
                            	     &CFMarkers);
              break;
         case MLI_METHOD_AMGRS_RUGE :
              hypre_BoomerAMGCoarsenRuge(hypreS, hypreA, measureType_,
                            	 coarsenScheme_, outputLevel_, &CFMarkers);
              break;
         case MLI_METHOD_AMGRS_FALGOUT :
              hypre_BoomerAMGCoarsenFalgout(hypreS, hypreA, measureType_, 
                                            outputLevel_, &CFMarkers);
              break;
         case MLI_METHOD_AMGRS_CR :
              hypre_BoomerAMGCoarsen(hypreS, hypreA, 0, outputLevel_,
                            	     &CFMarkers);
              k = 0;
              for (irow = 0; irow < localNRows; irow++) 
              {
                 if (CFMarkers[irow] > 0) {CFMarkers[irow] = 1; k++;}
                 else if (CFMarkers[irow] < 0) CFMarkers[irow] = 0;
              }
              printf("\tAMGRS_CR(1) nCoarse = %d\n", k);
              numTrials = 100;
              mli_Affmat = performCR(mli_Amat,CFMarkers,&mli_Afcmat,numTrials,
                                     hypreS2);
              k = 0;
              for (irow = 0; irow < localNRows; irow++) 
              {
                 if (CFMarkers[irow] > 0) {CFMarkers[irow] = 1; k++;}
                 else if (CFMarkers[irow] <= 0) CFMarkers[irow] = -1;
              }
              printf("\tAMGRS_CR(2) nCoarse = %d\n", k);
              break;
      }
      coarseNRows = 0;
      for ( irow = 0; irow < localNRows; irow++ )
         if ( CFMarkers[irow] == 1 ) coarseNRows++;

      /* ------if nonsymmetric, compute S for R------------------------- */

      if ( symmetric_ == 0 )
      {
         MLI_Matrix_Transpose( mli_Amat, &mli_ATmat );
         hypreAT = (hypre_ParCSRMatrix *) mli_ATmat->getMatrix();
         hypre_BoomerAMGCreateS(hypreAT, threshold_, maxRowSum_, nodeDOF_,
                                dofArray, &hypreST);
         hypre_BoomerAMGCoarsen(hypreST, hypreAT, 1, outputLevel_,
                            	&CFMarkers);
         coarseNRows = 0;
         for ( irow = 0; irow < localNRows; irow++ )
            if ( CFMarkers[irow] == 1 ) coarseNRows++;
      }

      /* ------construct processor maps for the coarse grid------------- */

      coarsePartition = (int *) hypre_CTAlloc(int, nprocs+1);
      coarsePartition[0] = 0;
      MPI_Allgather(&coarseNRows, 1, MPI_INT, &(coarsePartition[1]),
		    1, MPI_INT, comm);
      for ( irow = 2; irow < nprocs+1; irow++ )
         coarsePartition[irow] += coarsePartition[irow-1];
      globalCoarseNRows = coarsePartition[nprocs];

      if ( outputLevel_ > 1 && mypid == 0 )
         printf("\tMLI_Method_AMGRS::setup - # C dof = %d(%d)\n",
                globalCoarseNRows, globalNRows);

      /* ------if nonsymmetric, need to make sure localNRows > 0 ------ */
      /* ------ or the matrixTranspose function will give problems ----- */

      zeroNRows = 0;
      if ( symmetric_ == 0 )
      {
         for ( irow = 0; irow < nprocs; irow++ )
         {
            if ( (coarsePartition[irow+1]-coarsePartition[irow]) <= 0 )
            {
               zeroNRows = 1;
               break;
            }
         }
      }
          
      /* ------ wrap up creating the multigrid hierarchy --------------- */

      if ( coarsePartition[nprocs] < minCoarseSize_ ||
           coarsePartition[nprocs] == globalNRows || zeroNRows == 1 ) 
      {
         if ( symmetric_ == 0 )
         {
            delete mli_ATmat;
            hypre_ParCSRMatrixDestroy(hypreST);
         }
         hypre_TFree( coarsePartition );
         if ( CFMarkers != NULL ) hypre_TFree( CFMarkers );
         if ( hypreS  != NULL ) hypre_ParCSRMatrixDestroy(hypreS);
         if ( hypreS2 != NULL ) hypre_ParCSRMatrixDestroy(hypreS2);
         if ( coarsenScheme_ == MLI_METHOD_AMGRS_CR )
         {
            delete mli_Affmat;
            delete mli_Afcmat;
         }
         break;
      }
      k = (int) (globalNRows * 0.75);
      //if ( coarsenScheme_ > 0 && coarsePartition[nprocs] >= k )
      //   coarsenScheme_ = 0;

      /* ------create new dof array for coarse grid--------------------- */

      if ( coarseNRows > 0 ) cdofArray = new int[coarseNRows];
      else                   cdofArray = NULL;
      coarseNRows = 0;
      for ( irow = 0; irow < localNRows; irow++ )
      {
         if ( CFMarkers[irow] == 1 )
            cdofArray[coarseNRows++] = dofArray[irow];
      }

      /* ------build and set the interpolation operator----------------- */

#if 0
      //===============================================
      // This part is for future research on better Pmat
      //===============================================
      if ( coarsenScheme_ == MLI_METHOD_AMGRS_CR )
      {
         mli_Pmat = createPmat(CFMarkers, mli_Amat, mli_Affmat, mli_Afcmat);
         delete mli_Affmat;
         delete mli_Afcmat;
         mli->setProlongation(level+1, mli_Pmat);
      }
      else
      //===============================================
#endif
      {
         hypre_BoomerAMGBuildInterp(hypreA, CFMarkers, hypreS, 
                     coarsePartition, nodeDOF_, dofArray, outputLevel_, 
                     truncFactor_, mxelmtsP_, mapStoA, &hypreP);
         funcPtr = new MLI_Function();
         MLI_Utils_HypreParCSRMatrixGetDestroyFunc(funcPtr);
         sprintf(paramString, "HYPRE_ParCSR" ); 
         mli_Pmat = new MLI_Matrix( (void *) hypreP, paramString, funcPtr );
         mli->setProlongation(level+1, mli_Pmat);
         delete funcPtr;
      }
      if ( hypreS  != NULL ) hypre_ParCSRMatrixDestroy(hypreS);
      if ( hypreS2 != NULL ) hypre_ParCSRMatrixDestroy(hypreS2);

      /* ------build and set the restriction operator, if needed-------- */

      if ( useInjectionForR_ == 1 )
      {
         reduceArray1 = new int[nprocs+1];
         reduceArray2 = new int[nprocs+1];
         for ( k = 0; k < nprocs; k++ ) reduceArray1[k] = 0;
         reduceArray1[mypid] = coarseNRows;
         MPI_Allreduce(reduceArray1,reduceArray2,nprocs,MPI_INT,MPI_SUM,comm);
         for (k = nprocs-1; k >= 0; k--) reduceArray2[k+1] = reduceArray2[k];
         reduceArray2[0] = 0;
         for ( k = 2; k <= nprocs; k++ ) reduceArray2[k] += reduceArray2[k-1];
         startCol   = reduceArray2[mypid];
         localNCols = reduceArray2[mypid+1] - startCol;
         globalCoarseNRows = reduceArray2[nprocs];
         ierr = HYPRE_IJMatrixCreate(comm, startCol, startCol+localNCols-1,
                        startRow,startRow+localNRows-1,&IJRmat);
         ierr = HYPRE_IJMatrixSetObjectType(IJRmat, HYPRE_PARCSR);
         assert(!ierr);
         rowLengs = new int[localNCols];
         for ( k = 0; k < localNCols; k++ ) rowLengs[k] = 1;
         ierr = HYPRE_IJMatrixSetRowSizes(IJRmat, rowLengs);
         ierr = HYPRE_IJMatrixInitialize(IJRmat);
         assert(!ierr);
         delete [] rowLengs;
         delete [] reduceArray1;
         delete [] reduceArray2;
         k = 0;
         for ( irow = 0; irow < localNCols; irow++ ) 
         {
            while ( CFMarkers[k] != 1 ) k++; 
            rowNum = startCol + irow;
            colInd = k + startRow;
            HYPRE_IJMatrixSetValues(IJRmat, 1, &one, (const int *) &rowNum, 
                    (const int *) &colInd, (const double *) &colVal);
            k++;
         }
         ierr = HYPRE_IJMatrixAssemble(IJRmat);
         assert( !ierr );
         HYPRE_IJMatrixGetObject(IJRmat, (void **) &hypreR);
         hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) hypreR);
         funcPtr = new MLI_Function();
         MLI_Utils_HypreParCSRMatrixGetDestroyFunc(funcPtr);
         sprintf(paramString, "HYPRE_ParCSR" ); 
         mli_Rmat = new MLI_Matrix( (void *) hypreR, paramString, funcPtr );
         mli->setRestriction(level, mli_Rmat);
         delete funcPtr;
         delete mli_ATmat;
         hypre_ParCSRMatrixDestroy(hypreST);
      }
      else if ( symmetric_ == 0 )
      {
         hypre_BoomerAMGBuildInterp(hypreAT, CFMarkers, hypreST, 
                     coarsePartition, nodeDOF_, dofArray, outputLevel_, 
                     truncFactor_, mxelmtsP_, mapStoA, &hypreRT);
         hypreRT->owns_col_starts = 0;
         hypre_ParCSRMatrixTranspose( hypreRT, &hypreR, one );
         funcPtr = new MLI_Function();
         MLI_Utils_HypreParCSRMatrixGetDestroyFunc(funcPtr);
         sprintf(paramString, "HYPRE_ParCSR" ); 
         mli_Rmat = new MLI_Matrix( (void *) hypreR, paramString, funcPtr );
         mli->setRestriction(level, mli_Rmat);
         delete funcPtr;
         delete mli_ATmat;
         hypre_ParCSRMatrixDestroy(hypreST);
         hypre_ParCSRMatrixDestroy(hypreRT);
      }
      else
      {
         sprintf(paramString, "HYPRE_ParCSRT");
         mli_Rmat = new MLI_Matrix(mli_Pmat->getMatrix(), paramString, NULL);
         mli->setRestriction(level, mli_Rmat);
      }
      if ( CFMarkers != NULL ) hypre_TFree( CFMarkers );
      //if ( coarsePartition != NULL ) hypre_TFree( coarsePartition );

      startTime = MLI_Utils_WTime();

      /* ------construct and set the coarse grid matrix----------------- */

      if ( mypid == 0 && outputLevel_ > 0 ) printf("\tComputing RAP\n");
      if ( symmetric_ == 1 )
      {
         //hypreP = (hypre_ParCSRMatrix *) mli_Pmat->getMatrix();
         //hypre_ParCSRMatrixTranspose(hypreP, &hypreR, 1);
         //hypreAP = hypre_ParMatmul(hypreA, hypreP);
         //hypreCA = hypre_ParMatmul(hypreR, hypreAP);
         //if (hypre_ParCSRMatrixOwnsRowStarts(hypreCA) == 0)
         //{
         //   rowColStarts = hypre_ParCSRMatrixRowStarts(hypreR);
         //   newRowColStarts = (int *) malloc((nprocs+1) * sizeof(int));
         //   for (irow = 0; irow <= nprocs; irow++) 
         //      newRowColStarts[irow] = rowColStarts[irow];
         //   hypre_ParCSRMatrixRowStarts(hypreCA) = newRowColStarts;
         //   hypre_ParCSRMatrixOwnsRowStarts(hypreCA) = 1;
         //}
         //if (hypre_ParCSRMatrixOwnsColStarts(hypreCA) == 0)
         //{
         //   rowColStarts = hypre_ParCSRMatrixColStarts(hypreAP);
         //   newRowColStarts = (int *) malloc((nprocs+1) * sizeof(int));
         //   for (irow = 0; irow <= nprocs; irow++) 
         //      newRowColStarts[irow] = rowColStarts[irow];
         //   hypre_ParCSRMatrixColStarts(hypreCA) = newRowColStarts;
         //   hypre_ParCSRMatrixOwnsColStarts(hypreCA) = 1;
         //}
         //funcPtr = new MLI_Function();
         //MLI_Utils_HypreParCSRMatrixGetDestroyFunc(funcPtr);
         //sprintf(paramString, "HYPRE_ParCSR" ); 
         //mli_cAmat = new MLI_Matrix((void *) hypreCA, paramString, funcPtr);
         //delete funcPtr;
         //hypre_ParCSRMatrixDestroy( hypreR );
         //hypre_ParCSRMatrixDestroy( hypreAP );
         MLI_Matrix_ComputePtAP(mli_Pmat, mli_Amat, &mli_cAmat);
      }
      else
      {
#ifdef MLI_USE_HYPRE_MATMATMULT
         hypreP  = (hypre_ParCSRMatrix *) mli_Pmat->getMatrix();
         hypreR  = (hypre_ParCSRMatrix *) mli_Rmat->getMatrix();
         hypreAP = hypre_ParMatmul( hypreA, hypreP );
         hypreCA = hypre_ParMatmul( hypreR, hypreAP );
         hypre_ParCSRMatrixDestroy( hypreAP );
         funcPtr = new MLI_Function();
         MLI_Utils_HypreParCSRMatrixGetDestroyFunc(funcPtr);
         sprintf(paramString, "HYPRE_ParCSR" ); 
         mli_cAmat = new MLI_Matrix((void *) hypreCA, paramString, funcPtr);
         delete funcPtr;
#else
           MLI_Matrix_MatMatMult(mli_Amat, mli_Pmat, &mli_APmat);
           MLI_Matrix_MatMatMult(mli_Rmat, mli_APmat, &mli_cAmat);
           delete mli_APmat;
#endif
      }
      mli->setSystemMatrix(level+1, mli_cAmat);
      elapsedTime = (MLI_Utils_WTime() - startTime);
      RAPTime_ += elapsedTime;
      if ( mypid == 0 && outputLevel_ > 0 ) 
         printf("\tRAP computed, time = %e seconds.\n", elapsedTime);

      /* ------set the smoothers---------------------------------------- */

      smootherPtr = MLI_Solver_CreateFromName( smoother_ );
      targv[0] = (char *) &smootherNSweeps_;
      targv[1] = (char *) smootherWeights_;
      sprintf( paramString, "relaxWeight" );
      smootherPtr->setParams(paramString, 2, targv);
      if ( smootherPrintRNorm_ == 1 )
      {
         sprintf( paramString, "printRNorm" );
         smootherPtr->setParams(paramString, 0, NULL);
      }
      if ( smootherFindOmega_ == 1 )
      {
         sprintf( paramString, "findOmega" );
         smootherPtr->setParams(paramString, 0, NULL);
      }
      sprintf( paramString, "setModifiedDiag" );
      smootherPtr->setParams(paramString, 0, NULL);
      smootherPtr->setup(mli_Amat);
      mli->setSmoother( level, MLI_SMOOTHER_BOTH, smootherPtr );
   }
   if ( dofArray != NULL ) delete [] dofArray;

   /* ------set the coarse grid solver---------------------------------- */

   if (mypid == 0 && outputLevel_ > 0) printf("\tCoarse level = %d\n",level);
   csolverPtr = MLI_Solver_CreateFromName( coarseSolver_ );
   if ( strcmp(coarseSolver_, "SuperLU") )
   {
      targv[0] = (char *) &coarseSolverNSweeps_;
      targv[1] = (char *) coarseSolverWeights_ ;
      sprintf( paramString, "relaxWeight" );
      csolverPtr->setParams(paramString, 2, targv);
   }
   mli_Amat = mli->getSystemMatrix(level);
   csolverPtr->setup(mli_Amat);
   mli->setCoarseSolve(csolverPtr);
   totalTime_ = MLI_Utils_WTime() - totalTime_;

   /* --------------------------------------------------------------- */
   /* return the coarsest grid level number                           */
   /* --------------------------------------------------------------- */

   if ( outputLevel_ >= 2 ) printStatistics(mli);

#ifdef MLI_DEBUG_DETAILED
   printf("MLI_Method_AMGRS::setup ends.");
#endif
   return (level+1);
}

/* ********************************************************************* *
 * set diagnostics output level
 * --------------------------------------------------------------------- */

int MLI_Method_AMGRS::setOutputLevel( int level )
{
   outputLevel_ = level;
   return 0;
}

/* ********************************************************************* *
 * set number of levels 
 * --------------------------------------------------------------------- */

int MLI_Method_AMGRS::setNumLevels( int nlevels )
{
   if ( nlevels < maxLevels_ && nlevels > 0 ) numLevels_ = nlevels;
   return 0;
}

/* ********************************************************************* *
 * set smoother
 * --------------------------------------------------------------------- */

int MLI_Method_AMGRS::setSmoother(char *stype, int num, double *wgt)
{
   int i;

#ifdef MLI_DEBUG_DETAILED
   printf("MLI_Method_AMGRS::setSmoother - type = %s.\n", stype);
#endif

   strcpy( smoother_, stype );
   if ( num > 0 ) smootherNSweeps_ = num; else smootherNSweeps_ = 1;
   delete [] smootherWeights_;
   smootherWeights_ = new double[smootherNSweeps_];
   if ( wgt == NULL )
      for (i = 0; i < smootherNSweeps_; i++) smootherWeights_[i] = 0.;
   else
      for (i = 0; i < smootherNSweeps_; i++) smootherWeights_[i] = wgt[i];
   return 0;
}

/* ********************************************************************* *
 * set coarse solver 
 * --------------------------------------------------------------------- */

int MLI_Method_AMGRS::setCoarseSolver( char *stype, int num, double *wgt )
{
   int i;

#ifdef MLI_DEBUG_DETAILED
   printf("MLI_Method_AMGRS::setCoarseSolver - type = %s.\n", stype);
#endif

   strcpy( coarseSolver_, stype );
   if ( num > 0 ) coarseSolverNSweeps_ = num; else coarseSolverNSweeps_ = 1;
   delete [] coarseSolverWeights_ ;
   if ( wgt != NULL && strcmp(coarseSolver_, "SuperLU") )
   {
      coarseSolverWeights_ = new double[coarseSolverNSweeps_]; 
      for (i = 0; i < coarseSolverNSweeps_; i++) 
         coarseSolverWeights_ [i] = wgt[i];
   }
   else coarseSolverWeights_  = NULL;
   return 0;
}

/* ********************************************************************* *
 * set measure type 
 * --------------------------------------------------------------------- */

int MLI_Method_AMGRS::setMeasureType( int mtype )
{
   measureType_ = mtype;
   return 0;
}

/* ********************************************************************* *
 * set node degree of freedom 
 * --------------------------------------------------------------------- */

int MLI_Method_AMGRS::setNodeDOF( int dof )
{
   if ( dof > 0 && dof < 20 ) nodeDOF_ = dof;
   return 0;
}

/* ********************************************************************* *
 * set coarsening scheme 
 * --------------------------------------------------------------------- */

int MLI_Method_AMGRS::setCoarsenScheme( int scheme )
{
   if ( scheme == MLI_METHOD_AMGRS_CLJP ) 
   {
      coarsenScheme_ = MLI_METHOD_AMGRS_CLJP;
      return 0;
   }
   else if ( scheme == MLI_METHOD_AMGRS_RUGE ) 
   {
      coarsenScheme_ = MLI_METHOD_AMGRS_RUGE;
      return 0;
   }
   else if ( scheme == MLI_METHOD_AMGRS_FALGOUT ) 
   {
      coarsenScheme_ = MLI_METHOD_AMGRS_FALGOUT;
      return 0;
   }
   else
   {
      printf("MLI_Method_AMGRS::setCoarsenScheme - invalid scheme.\n");
      return 1;
   }
}

/* ********************************************************************* *
 * set minimum coarse size
 * --------------------------------------------------------------------- */

int MLI_Method_AMGRS::setMinCoarseSize( int coarse_size )
{
   if ( coarse_size > 0 ) minCoarseSize_ = coarse_size;
   return 0;
}

/* ********************************************************************* *
 * set coarsening threshold
 * --------------------------------------------------------------------- */

int MLI_Method_AMGRS::setStrengthThreshold( double thresh )
{
   if ( thresh > 0.0 ) threshold_ = thresh;
   else                threshold_ = 0.0;
   return 0;
}

/* ********************************************************************* *
 * print AMGRS information
 * --------------------------------------------------------------------- */

int MLI_Method_AMGRS::print()
{
   int      mypid;
   MPI_Comm comm = getComm();

   MPI_Comm_rank( comm, &mypid);
   if ( mypid == 0 )
   {
      printf("\t********************************************************\n");
      printf("\t*** method name             = %s\n", getName());
      printf("\t*** number of levels        = %d\n", numLevels_);
      printf("\t*** coarsen type            = %d\n", coarsenScheme_);
      printf("\t*** measure type            = %d\n", measureType_);
      printf("\t*** strength threshold      = %e\n", threshold_);
      printf("\t*** truncation factor       = %e\n", truncFactor_);
      printf("\t*** P max elments           = %d\n", mxelmtsP_);
      printf("\t*** nodal degree of freedom = %d\n", nodeDOF_);
      printf("\t*** symmetric flag          = %d\n", symmetric_);
      printf("\t*** R injection flag        = %d\n", useInjectionForR_);
      printf("\t*** minimum coarse size     = %d\n", minCoarseSize_);
      printf("\t*** smoother type           = %s\n", smoother_); 
      printf("\t*** smoother nsweeps        = %d\n", smootherNSweeps_);
      printf("\t*** coarse solver type      = %s\n", coarseSolver_); 
      printf("\t*** coarse solver nsweeps   = %d\n", coarseSolverNSweeps_);  
      printf("\t********************************************************\n");
   }
   return 0;
}

/* ********************************************************************* *
 * print AMGRS statistics information
 * --------------------------------------------------------------------- */

int MLI_Method_AMGRS::printStatistics(MLI *mli)
{
   int          mypid, level, globalNRows, totNRows, fineNRows;
   int          maxNnz, minNnz, fineNnz, totNnz, thisNnz, itemp;
   double       maxVal, minVal, dtemp;
   char         paramString[100];
   MLI_Matrix   *mli_Amat, *mli_Pmat;
   MPI_Comm     comm = getComm();

   /* --------------------------------------------------------------- */
   /* output header                                                   */
   /* --------------------------------------------------------------- */

   MPI_Comm_rank( comm, &mypid);
   if ( mypid == 0 )
      printf("\t****************** AMGRS Statistics ********************\n");

   /* --------------------------------------------------------------- */
   /* output processing time                                          */
   /* --------------------------------------------------------------- */

   if ( mypid == 0 )
   {
      printf("\t*** number of levels = %d\n", currLevel_+1);
      printf("\t*** total RAP   time = %e seconds\n", RAPTime_);
      printf("\t*** total GenML time = %e seconds\n", totalTime_);
      printf("\t******************** Amatrix ***************************\n");
      printf("\t*level   Nrows MaxNnz MinNnz TotalNnz  maxValue  minValue*\n");
   }

   /* --------------------------------------------------------------- */
   /* fine and coarse matrix complexity information                   */
   /* --------------------------------------------------------------- */

   totNnz = totNRows = 0;
   for ( level = 0; level <= currLevel_; level++ )
   {
      mli_Amat = mli->getSystemMatrix( level );
      sprintf(paramString, "nrows");
      mli_Amat->getMatrixInfo(paramString, globalNRows, dtemp);
      sprintf(paramString, "maxnnz");
      mli_Amat->getMatrixInfo(paramString, maxNnz, dtemp);
      sprintf(paramString, "minnnz");
      mli_Amat->getMatrixInfo(paramString, minNnz, dtemp);
      sprintf(paramString, "totnnz");
      mli_Amat->getMatrixInfo(paramString, thisNnz, dtemp);
      sprintf(paramString, "maxval");
      mli_Amat->getMatrixInfo(paramString, itemp, maxVal);
      sprintf(paramString, "minval");
      mli_Amat->getMatrixInfo(paramString, itemp, minVal);
      if ( mypid == 0 )
      {
         printf("\t*%3d %9d %5d  %5d %10d %8.3e %8.3e *\n",level,
                globalNRows, maxNnz, minNnz, thisNnz, maxVal, minVal);
      }
      if ( level == 0 ) fineNnz = thisNnz;
      totNnz += thisNnz;
      if ( level == 0 ) fineNRows = globalNRows;
      totNRows += globalNRows;
   }

   /* --------------------------------------------------------------- */
   /* prolongation operator complexity information                    */
   /* --------------------------------------------------------------- */

   if ( mypid == 0 )
   {
      printf("\t******************** Pmatrix ***************************\n");
      printf("\t*level   Nrows MaxNnz MinNnz TotalNnz  maxValue  minValue*\n");
      fflush(stdout);
   }
   for ( level = 1; level <= currLevel_; level++ )
   {
      mli_Pmat = mli->getProlongation( level );
      sprintf(paramString, "nrows");
      mli_Pmat->getMatrixInfo(paramString, globalNRows, dtemp);
      sprintf(paramString, "maxnnz");
      mli_Pmat->getMatrixInfo(paramString, maxNnz, dtemp);
      sprintf(paramString, "minnnz");
      mli_Pmat->getMatrixInfo(paramString, minNnz, dtemp);
      sprintf(paramString, "totnnz");
      mli_Pmat->getMatrixInfo(paramString, thisNnz, dtemp);
      sprintf(paramString, "maxval");
      mli_Pmat->getMatrixInfo(paramString, itemp, maxVal);
      sprintf(paramString, "minval");
      mli_Pmat->getMatrixInfo(paramString, itemp, minVal);
      if ( mypid == 0 )
      {
         printf("\t*%3d %9d %5d  %5d %10d %8.3e %8.3e *\n",level,
                globalNRows, maxNnz, minNnz, thisNnz, maxVal, minVal);
      }
   }

   /* --------------------------------------------------------------- */
   /* other complexity information                                    */
   /* --------------------------------------------------------------- */

   if ( mypid == 0 )
   {
      printf("\t********************************************************\n");
      dtemp = (double) totNnz / (double) fineNnz;
      printf("\t*** Amat complexity  = %e\n", dtemp);
      dtemp = (double) totNRows / (double) fineNRows;
      printf("\t*** grid complexity  = %e\n", dtemp);
      printf("\t********************************************************\n");
      fflush(stdout);
   }
   return 0;
}

/* ********************************************************************* *
 * perform compatible relaxation
 * --------------------------------------------------------------------- */

MLI_Matrix *MLI_Method_AMGRS::performCR(MLI_Matrix *mli_Amat, int *indepSet,
                                        MLI_Matrix **AfcMat, int numTrials,
                                        hypre_ParCSRMatrix *hypreS)
{
   int    nprocs, mypid, localNRows, iT, numFpts, irow, *reduceArray1;
   int    *reduceArray2, iP, FStartRow, FNRows, ierr, *rowLengs;
   int    startRow, rowIndex, colIndex, targc, numSweeps=4, rowCount;
   int    one=1, iC, *ADiagI, *ADiagJ, *fList, colInd;
   int    iV, ranSeed, jcol, rowCount2, CStartRow, CNRows;
   int    *rowStarts, *newRowStarts, *colStarts, *newColStarts, kcol;
   int    numVectors = 1, *SDiagI, *SDiagJ, count;
   int    numNew1, numNew2, numNew3, *sortIndices, stopRefine=1;
   double maxEigen, relaxWts[5], colValue, rnorm0, rnorm1, dOne=1.0;
   double *XaccData, *XData, *ADiagD;
   double aratio, ratio1, *ADiagA, targetMu=MU;
   char   paramString[200], *targv[2];
   HYPRE_IJMatrix     IJPFF, IJPFC;
   hypre_ParCSRMatrix *hypreA, *hypreAff, *hyprePFC;
   hypre_ParCSRMatrix *hypreAfc, *hyprePFF, *hyprePFFT, *hypreAPFC;
   hypre_CSRMatrix    *ADiag, *SDiag;
   HYPRE_IJVector     IJB, IJX, IJXacc;
   hypre_ParVector    *hypreB, *hypreX, *hypreXacc;
   MLI_Matrix *mli_PFFMat, *mli_AffMat, *mli_AfcMat;
#ifdef HAVE_TRANS
   MLI_Matrix *mli_AffTMat;
#endif
   MLI_Vector *mli_Xvec, *mli_Bvec;
   MLI_Solver *smootherPtr;
   MPI_Comm   comm;
   MLI_Function *funcPtr;

   /* ------------------------------------------------------ */
   /* get matrix and machine information                     */
   /* ------------------------------------------------------ */

   comm = getComm();
   MPI_Comm_size(comm, &nprocs);
   MPI_Comm_rank(comm, &mypid);
   hypreA = (hypre_ParCSRMatrix *) mli_Amat->getMatrix();
   ADiag = hypre_ParCSRMatrixDiag(hypreA);
   ADiagI = hypre_CSRMatrixI(ADiag);
   ADiagJ = hypre_CSRMatrixJ(ADiag);
   ADiagA = hypre_CSRMatrixData(ADiag);
   SDiag = hypre_ParCSRMatrixDiag(hypreS);
   SDiagI = hypre_CSRMatrixI(SDiag);
   SDiagJ = hypre_CSRMatrixJ(SDiag);
   localNRows = hypre_CSRMatrixNumRows(ADiag);
   startRow = hypre_ParCSRMatrixFirstRowIndex(hypreA);

   /* ------------------------------------------------------ */
   /* select initial set of fine points                      */
   /* ------------------------------------------------------ */

#if 0
   if (numTrials != 1)
   {
      for (irow = 0; irow < localNRows; irow++) indepSet[irow] = 1; 
      for (irow = 0; irow < localNRows; irow++) 
      {
         if (indepSet[irow] == 1)  /* if I am a C-point */
         {
            indepSet[irow] = 0;  /* set myself to be a F-pt */
            for (jcol = ADiagI[irow]; jcol < ADiagI[irow+1]; jcol++) 
            {
               colInd = ADiagJ[jcol]; /* for each of my neighbors */
               if (indepSet[colInd] == 1) /* if it is a C-point */ 
               {
                  /* if I depend strongly on it, leave it as C-pt */
                  for (kcol = SDiagI[irow]; kcol < SDiagI[irow+1]; kcol++) 
                  {
                     if (SDiagJ[kcol] == colInd) 
                     {
                        indepSet[colInd] = -1;
                        break;
                     }
                  }
                  /* if I don't depend strongly on it, see if it depends on me*/
                  if (kcol == SDiagI[irow+1]) 
                  {
                     for (kcol=SDiagI[colInd]; kcol < SDiagI[colInd+1]; kcol++) 
                     {
                        if (SDiagJ[kcol] == irow) 
                        {
                           indepSet[colInd] = -1;
                           break;
                        }
                     }
                  }
               }
            }
         }
      }
      for (irow = 0; irow < localNRows; irow++) 
         if (indepSet[irow] < 0) indepSet[irow] = 1;
      count = 0;
      for (irow = 0; irow < localNRows; irow++) 
         if (indepSet[irow] == 1) count++;

      /* ------------------------------------------------------ */
      /* select second set of fine points                       */
      /* ------------------------------------------------------ */

      for (irow = 0; irow < localNRows; irow++) 
      {
         if (indepSet[irow] == 1)  /* if I am a C-point */
         {
            count = 0;
            for (jcol = ADiagI[irow]; jcol < ADiagI[irow+1]; jcol++) 
            {
               colInd = ADiagJ[jcol]; /* for each of my neighbors */
               if (indepSet[colInd] == 0) /* if it is a F-point */ 
               {
                  /* if I depend strongly on it, increment counter */
                  for (kcol = SDiagI[irow]; kcol < SDiagI[irow+1]; kcol++) 
                  {
                     if (SDiagJ[kcol] == colInd) 
                     {
                        count++;
                        break;
                     }
                  }
                  /* if I don't depend strongly on it, see if it depends on me*/
                  if (kcol == SDiagI[irow+1]) 
                  {
                     for (kcol=SDiagI[colInd]; kcol < SDiagI[colInd+1]; kcol++) 
                     {
                        if (SDiagJ[kcol] == irow) 
                        {
                           count++;
                           break;
                        }
                     }
                  }
               }
            }
            if (count <= 1) indepSet[irow] = 0;
         }
      }
      count = 0;
      for (irow = 0; irow < localNRows; irow++) 
         if (indepSet[irow] == 1) count++;
   }
#endif

   /* ------------------------------------------------------ */
   /* loop over number of trials                             */
   /* ------------------------------------------------------ */

   printf("\tPerform compatible relaxation\n");
   ADiagD = new double[localNRows];
   for (irow = 0; irow < localNRows; irow++)
      for (jcol = ADiagI[irow]; jcol < ADiagI[irow+1]; jcol++)
         if (ADiagJ[jcol] == irow) {ADiagD[irow] = ADiagA[jcol]; break;}
   fList = new int[localNRows];
   aratio = 0.0;
   numNew1 = numNew2 = numNew3 = 0;
   for (iT = 0; iT < numTrials; iT++)
   {
      /* --------------------------------------------------- */
      /* get Aff and Afc matrices (get dimension)            */
      /* --------------------------------------------------- */

      numFpts = 0;
      for (irow = 0; irow < localNRows; irow++)
         if (indepSet[irow] != 1) fList[numFpts++] = irow;
      printf("\tTrial %3d (%3d) : number of F-points = %d\n", iT,
             numTrials, numFpts);
      reduceArray1 = new int[nprocs+1];
      reduceArray2 = new int[nprocs+1];
      for (iP = 0; iP < nprocs; iP++) reduceArray1[iP] = 0;
      for (iP = 0; iP < nprocs; iP++) reduceArray2[iP] = 0;
      reduceArray1[mypid] = numFpts;
      MPI_Allreduce(reduceArray1,reduceArray2,nprocs,MPI_INT,MPI_SUM,comm);
      for (iP = nprocs-1; iP >= 0; iP--) reduceArray2[iP+1] = reduceArray2[iP];
      reduceArray2[0] = 0;
      for (iP = 2; iP <= nprocs; iP++) reduceArray2[iP] += reduceArray2[iP-1];
      FStartRow = reduceArray2[mypid];
      FNRows = reduceArray2[mypid+1] - FStartRow;
      delete [] reduceArray1;
      delete [] reduceArray2;
      CStartRow = startRow - FStartRow;
      CNRows = localNRows - FNRows;

      /* --------------------------------------------------- */
      /* get Aff and Afc matrices (create permute matrices)  */
      /* --------------------------------------------------- */

      ierr = HYPRE_IJMatrixCreate(comm,startRow,startRow+localNRows-1,
                           FStartRow,FStartRow+FNRows-1,&IJPFF);
      ierr = HYPRE_IJMatrixSetObjectType(IJPFF, HYPRE_PARCSR);
      assert(!ierr);
      rowLengs = new int[localNRows];
      for (irow = 0; irow < localNRows; irow++) rowLengs[irow] = 1;
      ierr = HYPRE_IJMatrixSetRowSizes(IJPFF, rowLengs);
      ierr = HYPRE_IJMatrixInitialize(IJPFF);
      assert(!ierr);

      ierr = HYPRE_IJMatrixCreate(comm,startRow,startRow+localNRows-1,
                   CStartRow,CStartRow+CNRows-1, &IJPFC);
      ierr = HYPRE_IJMatrixSetObjectType(IJPFC, HYPRE_PARCSR);
      assert(!ierr);
      ierr = HYPRE_IJMatrixSetRowSizes(IJPFC, rowLengs);
      ierr = HYPRE_IJMatrixInitialize(IJPFC);
      assert(!ierr);
      delete [] rowLengs;

      /* --------------------------------------------------- */
      /* get Aff and Afc matrices (load permute matrices)    */
      /* --------------------------------------------------- */

      colValue = 1.0;
      rowCount = rowCount2 = 0;
      for (irow = 0; irow < localNRows; irow++)
      {
         rowIndex = startRow + irow;
         if (indepSet[irow] == 0) 
         {
            colIndex = FStartRow + rowCount;
            HYPRE_IJMatrixSetValues(IJPFF,1,&one,(const int *) &rowIndex,
                    (const int *) &colIndex, (const double *) &colValue);
            rowCount++;
         }
         else
         {
            colIndex = CStartRow + rowCount2;
            HYPRE_IJMatrixSetValues(IJPFC,1,&one,(const int *) &rowIndex,
                    (const int *) &colIndex, (const double *) &colValue);
            rowCount2++;
         }
      }
      ierr = HYPRE_IJMatrixAssemble(IJPFF);
      assert( !ierr );
      HYPRE_IJMatrixGetObject(IJPFF, (void **) &hyprePFF);
      //hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) hyprePFF);
      sprintf(paramString, "HYPRE_ParCSR" );
      mli_PFFMat = new MLI_Matrix((void *)hyprePFF,paramString,NULL);

      ierr = HYPRE_IJMatrixAssemble(IJPFC);
      assert( !ierr );
      HYPRE_IJMatrixGetObject(IJPFC, (void **) &hyprePFC);
      //hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) hyprePFC);
      hypreAPFC = hypre_ParMatmul(hypreA, hyprePFC);
      hypre_ParCSRMatrixTranspose(hyprePFF, &hyprePFFT, 1);
      hypreAfc = hypre_ParMatmul(hyprePFFT, hypreAPFC);
      rowStarts = hypre_ParCSRMatrixRowStarts(hyprePFFT);
      newRowStarts = (int *) malloc((nprocs+1) * sizeof(int));
      for (irow = 0; irow <= nprocs; irow++) 
         newRowStarts[irow] = rowStarts[irow];
      hypre_ParCSRMatrixRowStarts(hypreAfc) = newRowStarts;
      colStarts = hypre_ParCSRMatrixColStarts(hypreAPFC);
      newColStarts = (int *) malloc((nprocs+1) * sizeof(int));
      for (irow = 0; irow <= nprocs; irow++) 
         newColStarts[irow] = colStarts[irow];
      hypre_ParCSRMatrixColStarts(hypreAfc) = newColStarts;
      hypre_ParCSRMatrixOwnsRowStarts(hypreAfc) = 1;
      hypre_ParCSRMatrixOwnsColStarts(hypreAfc) = 1;

      funcPtr = new MLI_Function();
      MLI_Utils_HypreParCSRMatrixGetDestroyFunc(funcPtr);
      sprintf(paramString, "HYPRE_ParCSR" );
      mli_AfcMat = new MLI_Matrix((void *)hypreAfc,paramString,funcPtr);
      delete funcPtr;

      MLI_Matrix_ComputePtAP(mli_PFFMat, mli_Amat, &mli_AffMat);
      hypreAff  = (hypre_ParCSRMatrix *) mli_AffMat->getMatrix();
      hypre_ParCSRMatrixOwnsRowStarts(hyprePFF) = 0;
      hypre_ParCSRMatrixOwnsColStarts(hyprePFF) = 0;

      //if (aratio > targetMu || iT == numTrials-1) break;
      //if (((double)FNRows/(double)localNRows) > 0.75) break;

      HYPRE_IJVectorCreate(comm,FStartRow, FStartRow+FNRows-1,&IJX);
      HYPRE_IJVectorSetObjectType(IJX, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(IJX);
      HYPRE_IJVectorAssemble(IJX);
      HYPRE_IJVectorGetObject(IJX, (void **) &hypreX);
      sprintf(paramString, "HYPRE_ParVector" );
      mli_Xvec = new MLI_Vector((void *)hypreX,paramString,NULL);

      HYPRE_IJVectorCreate(comm,FStartRow, FStartRow+FNRows-1,&IJXacc);
      HYPRE_IJVectorSetObjectType(IJXacc, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(IJXacc);
      HYPRE_IJVectorAssemble(IJXacc);
      HYPRE_IJVectorGetObject(IJXacc, (void **) &hypreXacc);
      hypre_ParVectorSetConstantValues(hypreXacc, 0.0);

      HYPRE_IJVectorCreate(comm,FStartRow, FStartRow+FNRows-1,&IJB);
      HYPRE_IJVectorSetObjectType(IJB, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(IJB);
      HYPRE_IJVectorAssemble(IJB);
      HYPRE_IJVectorGetObject(IJB, (void **) &hypreB);
      hypre_ParVectorSetConstantValues(hypreB, 0.0);
      sprintf(paramString, "HYPRE_ParVector" );
      mli_Bvec = new MLI_Vector((void *)hypreB,paramString,NULL);

      /* --------------------------------------------------- */
      /* set up Jacobi smoother with 4 sweeps and weight=1   */
      /* --------------------------------------------------- */

      strcpy(paramString, "Jacobi");
      smootherPtr = MLI_Solver_CreateFromName(paramString);
      strcpy(paramString, "setModifiedDiag");
      smootherPtr->setParams(paramString, 0, NULL);
      targc = 2;
      numSweeps = 1;
      targv[0] = (char *) &numSweeps;
      for (iC = 0; iC < 5; iC++) relaxWts[iC] = 3.0/3.0;
      targv[1] = (char *) relaxWts;
      strcpy(paramString, "relaxWeight");
      smootherPtr->setParams(paramString, targc, targv);
      maxEigen = 1.0;
      targc = 1;
      targv[0] = (char *) &maxEigen;
      strcpy(paramString, "setMaxEigen");
      smootherPtr->setParams(paramString, targc, targv);
      smootherPtr->setup(mli_AffMat);

      /* --------------------------------------------------- */
      /* relaxation                                          */
      /* --------------------------------------------------- */

      targc = 2;
      targv[0] = (char *) &numSweeps;
      targv[1] = (char *) relaxWts;
      strcpy(paramString, "relaxWeight");
      aratio = 0.0;
      XData = (double *) hypre_VectorData(hypre_ParVectorLocalVector(hypreX));
      XaccData = (double *) 
               hypre_VectorData(hypre_ParVectorLocalVector(hypreXacc));
      for (iV = 0; iV < numVectors; iV++)
      {
         ranSeed = 9001 * 7901 * iV *iV * iV * iV + iV * iV * iV + 101;
         HYPRE_ParVectorSetRandomValues((HYPRE_ParVector) hypreX,ranSeed);
         for (irow = 0; irow < FNRows; irow++)
            XData[irow] = 0.5 * XData[irow] + 0.5;
         hypre_ParVectorAxpy(dOne, hypreX, hypreXacc);
         hypre_ParVectorSetConstantValues(hypreB, 0.0);
         hypre_ParCSRMatrixMatvec(-1.0, hypreAff, hypreX, 1.0, hypreB);
         rnorm0 = sqrt(hypre_ParVectorInnerProd(hypreB, hypreB));
         hypre_ParVectorSetConstantValues(hypreB, 0.0);

         numSweeps = 5;
         smootherPtr->setParams(paramString, targc, targv);
         smootherPtr->solve(mli_Bvec, mli_Xvec);
         hypre_ParCSRMatrixMatvec(-1.0, hypreAff, hypreX, 1.0, hypreB);
         rnorm1 = sqrt(hypre_ParVectorInnerProd(hypreB, hypreB));

         rnorm0 = rnorm1;
         hypre_ParVectorSetConstantValues(hypreB, 0.0);
         numSweeps = PDEGREE+1;
         smootherPtr->setParams(paramString, targc, targv);
         smootherPtr->solve(mli_Bvec, mli_Xvec);
         hypre_ParCSRMatrixMatvec(-1.0, hypreAff, hypreX, 1.0, hypreB);
         rnorm1 = sqrt(hypre_ParVectorInnerProd(hypreB, hypreB));

         aratio = 0;
         for (irow = 0; irow < FNRows; irow++)
         {
            ratio1 = habs(XData[irow]);
            XaccData[irow] = ratio1;
            if (ratio1 > aratio) aratio = ratio1;
         } 
         printf("\tTrial %3d : Jacobi norms = %16.8e %16.8e %16.8e\n",iT,
                rnorm0,rnorm1,aratio);
         if (rnorm0 > 1.0e-10) aratio = rnorm1 / rnorm0;
         else aratio = 0.0;
      }
      delete smootherPtr;

      /* --------------------------------------------------- */
      /* select fine points                                  */
      /* --------------------------------------------------- */
      
      if (iV == numVectors) aratio /= (double) numVectors; 
      if (aratio < targetMu) 
      {
         if (stopRefine == 0)
         {
            for (irow = 0; irow < localNRows; irow++) 
            {
               if (indepSet[irow] == 1)  /* if I am a C-point */
               {
                  count = 0;
                  for (jcol = ADiagI[irow]; jcol < ADiagI[irow+1]; jcol++) 
                  {
                     colInd = ADiagJ[jcol]; /* for each of my neighbors */
                     if (indepSet[colInd] == 0) /* if it is a F-point */ 
                     {
                        /* if I depend strongly on it, increment counter */
                        for (kcol= SDiagI[irow]; kcol < SDiagI[irow+1]; kcol++) 
                        {
                           if (SDiagJ[kcol] == colInd) 
                           {
                              count++;
                              break;
                           }
                        }
                        if (kcol == SDiagI[irow+1]) 
                        {
                          for (kcol=SDiagI[colInd];kcol<SDiagI[colInd+1];kcol++)
                          {
                             if (SDiagJ[kcol] == irow) 
                             {
                                count++;
                                break;
                             }
                          }
                        }
                     }
                  }
                  if (count <= 3*(iT+1)) indepSet[irow] = 0;
               }
            }
         }
      }
      else if (aratio > targetMu)
      {
         sortIndices = new int[localNRows];
         for (irow = 0; irow < localNRows; irow++) sortIndices[irow] = -1;
         iC = 0;
         for (irow = 0; irow < localNRows; irow++) 
            if (indepSet[irow] != 1) sortIndices[irow] = iC++;

         for (irow = 0; irow < localNRows; irow++) 
         {
            iC = sortIndices[irow];
            if (indepSet[irow] == 0 && (habs(XaccData[iC]) > 0.1))
            {
               aratio = targetMu;
               //stopRefine = 1;
               indepSet[irow] = 1; /* set it to a coarse point */
               for (jcol = ADiagI[irow]; jcol < ADiagI[irow+1]; jcol++) 
               {
                  colInd = ADiagJ[jcol];
                  if (indepSet[colInd] == 0) /* if it is a F-point */ 
                  {
                     for (kcol = SDiagI[irow]; kcol < SDiagI[irow+1]; kcol++) 
                     {
                        if (SDiagJ[kcol] == colInd) 
                        {
                           indepSet[colInd] = -1;
                           break;
                        }
                     }
                     if (kcol == SDiagI[irow+1]) 
                     {
                        for (kcol=SDiagI[colInd];kcol<SDiagI[colInd+1];kcol++) 
                        {
                           if (SDiagJ[kcol] == irow) 
                           {
                              indepSet[colInd] = -1;
                              break;
                           }
                        }
                     }
                  }
               }
            }
         }
         for (irow = 0; irow < localNRows; irow++) 
            if (indepSet[irow] == -1) indepSet[irow] = 0;
         delete [] sortIndices;
      } 

      /* --------------------------------------------------- */
      /* clean up                                            */
      /* --------------------------------------------------- */
      
      HYPRE_IJMatrixDestroy(IJPFF);
      hypre_ParCSRMatrixDestroy(hyprePFFT);
      hypre_ParCSRMatrixDestroy(hypreAPFC);
      delete mli_PFFMat;
#ifdef HAVE_TRANS
      delete mli_AffTMat;
#endif
      HYPRE_IJMatrixDestroy(IJPFC);
      HYPRE_IJVectorDestroy(IJX);
      HYPRE_IJVectorDestroy(IJB);
      HYPRE_IJVectorDestroy(IJXacc);
      delete mli_Bvec;
      delete mli_Xvec;
      //if (localNRows != FNRows && aratio < targetMu) break;
      //if (aratio < targetMu) break;
      if (numTrials == 1) break;
      numNew1 = numNew2;
      numNew2 = numNew3;
      numNew3 = 0;
      for (irow=0; irow < localNRows; irow++) if (indepSet[irow]==0) numNew3++;
      if (numNew3 == numNew1) break;
      delete mli_AffMat;
      delete mli_AfcMat;
      //hypre_ParCSRMatrixDestroy(hypreAfc);
   }

   /* ------------------------------------------------------ */
   /* final clean up                                         */
   /* ------------------------------------------------------ */

   delete [] ADiagD;
   delete [] fList;
   (*AfcMat) = mli_AfcMat;
   return mli_AffMat;
}

/* ********************************************************************* *
 * create the prolongation matrix
 * --------------------------------------------------------------------- */

MLI_Matrix *MLI_Method_AMGRS::createPmat(int *indepSet, MLI_Matrix *mli_Amat,
                               MLI_Matrix *mli_Affmat, MLI_Matrix *mli_Afcmat)
{
   int    *ADiagI, *ADiagJ, localNRows, AffNRows, AffStartRow, irow;
   int    *rowLengs, ierr, startRow, rowCount, rowIndex, colIndex;
   int    *colInd, rowSize, jcol, one=1, maxRowLeng, nnz, PDegree=PDEGREE;
   int    *tPDiagI, *tPDiagJ, cCount, fCount, ncount, *ADDiagI, *ADDiagJ;
   int    *AD2DiagI, *AD2DiagJ, *newColInd, newRowSize, *rowStarts;
   int    *newRowStarts, nprocs, AccStartRow, AccNRows;
   double *ADiagA, *colVal, colValue, *newColVal, *DDiagA;
   double *tPDiagA, *ADDiagA, *AD2DiagA, omega=2.0/3.0, dtemp;
   char   paramString[100];
   HYPRE_IJMatrix     IJInvD, IJP;
   hypre_ParCSRMatrix *hypreA, *hypreAff, *hypreInvD, *hypreP, *hypreAD;
   hypre_ParCSRMatrix *hypreAD2, *hypreAfc, *hypreTmp;
   hypre_CSRMatrix    *ADiag, *DDiag, *tPDiag, *ADDiag, *AD2Diag;
   MLI_Function       *funcPtr;
   MLI_Matrix         *mli_Pmat;
   MPI_Comm           comm;

   /* ------------------------------------------------------ */
   /* get matrix information                                 */
   /* ------------------------------------------------------ */

   comm = getComm();
   MPI_Comm_size(comm, &nprocs);
   hypreA = (hypre_ParCSRMatrix *) mli_Amat->getMatrix();
   startRow = hypre_ParCSRMatrixFirstRowIndex(hypreA);
   localNRows = hypre_ParCSRMatrixNumRows(hypreA);

   hypreAff = (hypre_ParCSRMatrix *) mli_Affmat->getMatrix();
   AffStartRow = hypre_ParCSRMatrixFirstRowIndex(hypreAff);
   AffNRows = hypre_ParCSRMatrixNumRows(hypreAff);

   /* ------------------------------------------------------ */
   /* create the diagonal matrix of A                        */
   /* ------------------------------------------------------ */

   ierr = HYPRE_IJMatrixCreate(comm,AffStartRow,AffStartRow+AffNRows-1,
                           AffStartRow,AffStartRow+AffNRows-1,&IJInvD);
   ierr = HYPRE_IJMatrixSetObjectType(IJInvD, HYPRE_PARCSR);
   assert(!ierr);
   rowLengs = new int[AffNRows];
   for (irow = 0; irow < AffNRows; irow++) rowLengs[irow] = 1;
   ierr = HYPRE_IJMatrixSetRowSizes(IJInvD, rowLengs);
   ierr = HYPRE_IJMatrixInitialize(IJInvD);
   assert(!ierr);
   delete [] rowLengs;

   /* ------------------------------------------------------ */
   /* load the diagonal matrix of A                          */
   /* ------------------------------------------------------ */

   rowCount = 0;
   for (irow = 0; irow < localNRows; irow++)
   {
      rowIndex = startRow + irow;
      if (indepSet[irow] == 0) 
      {
         HYPRE_ParCSRMatrixGetRow((HYPRE_ParCSRMatrix) hypreA, rowIndex, 
                                  &rowSize, &colInd, &colVal);
         colValue = 1.0;
         for (jcol = 0; jcol < rowSize; jcol++)
         {
            if (colInd[jcol] == rowIndex) 
            {
               colValue = colVal[jcol];
               break;
            }
         }
         if (colValue >= 0.0)
         {
            for (jcol = 0; jcol < rowSize; jcol++)
               if (colInd[jcol] != rowIndex && 
                   (indepSet[colInd[jcol]-startRow] == 0) && 
                   colVal[jcol] > 0.0) 
                  colValue += colVal[jcol];
         }
         else
         {
            for (jcol = 0; jcol < rowSize; jcol++)
               if (colInd[jcol] != rowIndex && 
                   (indepSet[colInd[jcol]-startRow] == 0) && 
                   colVal[jcol] < 0.0) 
                  colValue += colVal[jcol];
         }
         colValue = 1.0 / colValue;
         colIndex = AffStartRow + rowCount;
         HYPRE_IJMatrixSetValues(IJInvD,1,&one,(const int *) &colIndex,
                    (const int *) &colIndex, (const double *) &colValue);
         rowCount++;
         HYPRE_ParCSRMatrixRestoreRow((HYPRE_ParCSRMatrix) hypreA, rowIndex, 
                                      &rowSize, &colInd, &colVal);
      }
   }

   /* ------------------------------------------------------ */
   /* finally assemble the diagonal matrix of A              */
   /* ------------------------------------------------------ */

   ierr = HYPRE_IJMatrixAssemble(IJInvD);
   assert( !ierr );
   HYPRE_IJMatrixGetObject(IJInvD, (void **) &hypreInvD);
   ierr += HYPRE_IJMatrixSetObjectType(IJInvD, -1);
   ierr += HYPRE_IJMatrixDestroy(IJInvD);
   assert( !ierr );

   /* ------------------------------------------------------ */
   /* generate polynomial of Aff and invD                    */
   /* ------------------------------------------------------ */

   if (PDegree == 0)
   {
      hypreP = hypreInvD;
      hypreInvD = NULL;
      ADiag  = hypre_ParCSRMatrixDiag(hypreP);
      ADiagI = hypre_CSRMatrixI(ADiag);
      ADiagJ = hypre_CSRMatrixJ(ADiag);
      ADiagA = hypre_CSRMatrixData(ADiag);
      for (irow = 0; irow < AffNRows; irow++) 
         for (jcol = ADiagI[irow]; jcol < ADiagI[irow+1]; jcol++) 
            ADiagA[jcol] = - ADiagA[jcol];
   }
   else if (PDegree == 1)
   {
#if 1
      hypreP = hypre_ParMatmul(hypreAff, hypreInvD);
      DDiag  = hypre_ParCSRMatrixDiag(hypreInvD);
      DDiagA = hypre_CSRMatrixData(DDiag);
      ADiag  = hypre_ParCSRMatrixDiag(hypreP);
      ADiagI = hypre_CSRMatrixI(ADiag);
      ADiagJ = hypre_CSRMatrixJ(ADiag);
      ADiagA = hypre_CSRMatrixData(ADiag);
      for (irow = 0; irow < AffNRows; irow++) 
      {
         for (jcol = ADiagI[irow]; jcol < ADiagI[irow+1]; jcol++) 
         {
            if (ADiagJ[jcol] == irow) 
                 ADiagA[jcol] = - omega*DDiagA[irow]*(2.0-omega*ADiagA[jcol]);
            else ADiagA[jcol] = omega * omega * DDiagA[irow] * ADiagA[jcol];
         }
      }
      hypre_ParCSRMatrixOwnsColStarts(hypreInvD) = 0;
      rowStarts = hypre_ParCSRMatrixRowStarts(hypreA);
      newRowStarts = (int *) malloc((nprocs+1) * sizeof(int));
      for (irow = 0; irow <= nprocs; irow++) 
         newRowStarts[irow] = rowStarts[irow];
      hypre_ParCSRMatrixRowStarts(hypreP) = newRowStarts;
#else
      ierr = HYPRE_IJMatrixCreate(comm,AffStartRow,AffStartRow+AffNRows-1,
                           AffStartRow,AffStartRow+AffNRows-1,&IJP);
      ierr = HYPRE_IJMatrixSetObjectType(IJP, HYPRE_PARCSR);
      assert(!ierr);
      rowLengs = new int[AffNRows];
      maxRowLeng = 0;
      ADiag   = hypre_ParCSRMatrixDiag(hypreAff);
      ADiagI  = hypre_CSRMatrixI(ADiag);
      ADiagJ  = hypre_CSRMatrixJ(ADiag);
      ADiagA  = hypre_CSRMatrixData(ADiag);
      for (irow = 0; irow < AffNRows; irow++)
      {
         newRowSize = 1;
         for (jcol = ADiagI[irow]; jcol < ADiagI[irow+1]; jcol++)
            if (ADiagJ[jcol] == irow) {index = jcol; break;}
         for (jcol = ADiagI[irow]; jcol < ADiagI[irow+1]; jcol++)
            if (ADiagJ[jcol] != irow && ADiagA[jcol]*ADiagA[index] < 0.0)
               newRowSize++;
         rowLengs[irow] = newRowSize;
         if (newRowSize > maxRowLeng) maxRowLeng = newRowSize; 
      }
      ierr = HYPRE_IJMatrixSetRowSizes(IJP, rowLengs);
      ierr = HYPRE_IJMatrixInitialize(IJP);
      assert(!ierr);
      delete [] rowLengs;
      newColInd = new int[maxRowLeng];
      newColVal = new double[maxRowLeng];
      for (irow = 0; irow < AffNRows; irow++)
      {
         newRowSize = 0;
         index = -1;
         for (jcol = ADiagI[irow]; jcol < ADiagI[irow+1]; jcol++)
            if (ADiagJ[jcol] == irow) {index = jcol; break;}
         if (index == -1) printf("WARNING : zero diagonal.\n");
         newColInd[0] = AffStartRow + irow;
         newColVal[0] = ADiagA[index];
         newRowSize++;
         for (jcol = ADiagI[irow]; jcol < ADiagI[irow+1]; jcol++)
         {
            if (ADiagJ[jcol] != irow && ADiagA[jcol]*ADiagA[index] < 0.0)
            {
               newColInd[newRowSize] = AffStartRow + ADiagJ[jcol]; 
               newColVal[newRowSize++] = ADiagA[jcol];
            }
            else
            {
               newColVal[0] += ADiagA[jcol];
            }
         }
         for (jcol = 1; jcol < newRowSize; jcol++)
            newColVal[jcol] /= (-newColVal[0]);
         newColVal[0] = 2.0 - newColVal[0];
         rowIndex = AffStartRow + irow;
         ierr = HYPRE_IJMatrixSetValues(IJP, 1, &newRowSize,
                   (const int *) &rowIndex, (const int *) newColInd,
                   (const double *) newColVal);
         assert(!ierr);
      }
      delete [] newColInd;
      delete [] newColVal;
      ierr = HYPRE_IJMatrixAssemble(IJP);
      assert( !ierr );
      HYPRE_IJMatrixGetObject(IJP, (void **) &hypreAD);
      hypreP = hypre_ParMatmul(hypreAD, hypreInvD);
      hypre_ParCSRMatrixOwnsRowStarts(hypreP) = 1;
      hypre_ParCSRMatrixOwnsRowStarts(hypreAD) = 0;
      ierr += HYPRE_IJMatrixDestroy(IJP);
#endif
   }
   else if (PDegree == 2)
   {
      hypreAD  = hypre_ParMatmul(hypreAff, hypreInvD);
      hypreAD2 = hypre_ParMatmul(hypreAD, hypreAD);
      ADDiag   = hypre_ParCSRMatrixDiag(hypreAD);
      AD2Diag  = hypre_ParCSRMatrixDiag(hypreAD2);
      ADDiagI  = hypre_CSRMatrixI(ADDiag);
      ADDiagJ  = hypre_CSRMatrixJ(ADDiag);
      ADDiagA  = hypre_CSRMatrixData(ADDiag);
      AD2DiagI = hypre_CSRMatrixI(AD2Diag);
      AD2DiagJ = hypre_CSRMatrixJ(AD2Diag);
      AD2DiagA = hypre_CSRMatrixData(AD2Diag);
      DDiag    = hypre_ParCSRMatrixDiag(hypreInvD);
      DDiagA   = hypre_CSRMatrixData(DDiag);
      newColInd = new int[2*AffNRows];
      newColVal = new double[2*AffNRows];
      ierr = HYPRE_IJMatrixCreate(comm,AffStartRow,AffStartRow+AffNRows-1,
                           AffStartRow,AffStartRow+AffNRows-1,&IJP);
      ierr = HYPRE_IJMatrixSetObjectType(IJP, HYPRE_PARCSR);
      assert(!ierr);
      rowLengs = new int[AffNRows];
      maxRowLeng = 0;
      for (irow = 0; irow < AffNRows; irow++)
      {
         newRowSize = 0;
         for (jcol = ADDiagI[irow]; jcol < ADDiagI[irow+1]; jcol++)
            newColInd[newRowSize] = ADDiagJ[jcol]; 
         for (jcol = AD2DiagI[irow]; jcol < AD2DiagI[irow+1]; jcol++)
            newColInd[newRowSize] = AD2DiagJ[jcol]; 
         if (newRowSize > maxRowLeng) maxRowLeng = newRowSize; 
         qsort0(newColInd, 0, newRowSize-1);
         ncount = 0;
         for ( jcol = 0; jcol < newRowSize; jcol++ )
         {
            if ( newColInd[jcol] != newColInd[ncount] )
            {
               ncount++;
               newColInd[ncount] = newColInd[jcol];
            } 
         }
         newRowSize = ncount + 1;
         rowLengs[irow] = newRowSize;
      }
      ierr = HYPRE_IJMatrixSetRowSizes(IJP, rowLengs);
      ierr = HYPRE_IJMatrixInitialize(IJP);
      assert(!ierr);
      delete [] rowLengs;
      nnz = 0;
      for (irow = 0; irow < AffNRows; irow++)
      {
         rowIndex = AffStartRow + irow;
         newRowSize = 0;
         for (jcol = ADDiagI[irow]; jcol < ADDiagI[irow+1]; jcol++)
         {
            newColInd[newRowSize] = ADDiagJ[jcol]; 
            if (ADDiagJ[jcol] == irow) 
               newColVal[newRowSize++] = 3.0 * (1.0 - ADDiagA[jcol]); 
            else
               newColVal[newRowSize++] = - 3.0 * ADDiagA[jcol]; 
         }
         for (jcol = AD2DiagI[irow]; jcol < AD2DiagI[irow+1]; jcol++)
         {
            newColInd[newRowSize] = AD2DiagJ[jcol]; 
            newColVal[newRowSize++] = AD2DiagA[jcol]; 
         }
         qsort1(newColInd, newColVal, 0, newRowSize-1);
         ncount = 0;
         for ( jcol = 0; jcol < newRowSize; jcol++ )
         {
            if ( jcol != ncount && newColInd[jcol] == newColInd[ncount] )
               newColVal[ncount] += newColVal[jcol];
            else if ( newColInd[jcol] != newColInd[ncount] )
            {
               ncount++;
               newColVal[ncount] = newColVal[jcol];
               newColInd[ncount] = newColInd[jcol];
            } 
         }
         newRowSize = ncount + 1;
         for ( jcol = 0; jcol < newRowSize; jcol++ )
            newColVal[jcol] = - (DDiagA[irow] * newColVal[jcol]);
     
         ierr = HYPRE_IJMatrixSetValues(IJP, 1, &newRowSize,
                   (const int *) &rowIndex, (const int *) newColInd,
                   (const double *) newColVal);
         nnz += newRowSize;
         assert(!ierr);
      }
      delete [] newColInd;
      delete [] newColVal;
      ierr = HYPRE_IJMatrixAssemble(IJP);
      assert( !ierr );
      HYPRE_IJMatrixGetObject(IJP, (void **) &hypreP);
      ierr += HYPRE_IJMatrixSetObjectType(IJP, -1);
      ierr += HYPRE_IJMatrixDestroy(IJP);
      assert(!ierr);
      hypre_ParCSRMatrixDestroy(hypreAD);
      hypre_ParCSRMatrixDestroy(hypreAD2);
   }
   if (hypreInvD != NULL) hypre_ParCSRMatrixDestroy(hypreInvD);

   /* ------------------------------------------------------ */
   /* create the final P matrix (from hypreP)                */
   /* ------------------------------------------------------ */

   hypreAfc = (hypre_ParCSRMatrix *) mli_Afcmat->getMatrix();
   hypreTmp = hypre_ParMatmul(hypreP, hypreAfc);
   hypre_ParCSRMatrixOwnsRowStarts(hypreP) = 0;
   hypre_ParCSRMatrixOwnsColStarts(hypreAfc) = 0;
   hypre_ParCSRMatrixOwnsRowStarts(hypreTmp) = 1;
   hypre_ParCSRMatrixOwnsColStarts(hypreTmp) = 1;
   hypre_ParCSRMatrixDestroy(hypreP);
   hypreP = hypreTmp;
   tPDiag   = hypre_ParCSRMatrixDiag(hypreP);
   tPDiagI  = hypre_CSRMatrixI(tPDiag);
   tPDiagJ  = hypre_CSRMatrixJ(tPDiag);
   tPDiagA  = hypre_CSRMatrixData(tPDiag);
   AccStartRow = startRow - AffStartRow;
   AccNRows = localNRows - AffNRows;
   ierr = HYPRE_IJMatrixCreate(comm,startRow,startRow+localNRows-1,
                        AccStartRow,AccStartRow+AccNRows-1,&IJP);
   ierr = HYPRE_IJMatrixSetObjectType(IJP, HYPRE_PARCSR);
   assert(!ierr);
   rowLengs = new int[localNRows];
   maxRowLeng = 0;
   ncount = 0;
   for (irow = 0; irow < localNRows; irow++)
   {
      if (indepSet[irow] == 1) rowLengs[irow] = 1;
      else                     
      {
         rowLengs[irow] = tPDiagI[ncount+1] - tPDiagI[ncount];
         ncount++;
      }
      if (rowLengs[irow] > maxRowLeng) maxRowLeng = rowLengs[irow]; 
   }
   ierr = HYPRE_IJMatrixSetRowSizes(IJP, rowLengs);
   ierr = HYPRE_IJMatrixInitialize(IJP);
   assert(!ierr);
   delete [] rowLengs;
   fCount = 0;
   cCount = 0;
   newColInd = new int[maxRowLeng];
   newColVal = new double[maxRowLeng];
   for (irow = 0; irow < localNRows; irow++)
   {
      rowIndex = startRow + irow;
      if (indepSet[irow] == 1)
      {
         newRowSize = 1;
         newColInd[0] = AccStartRow + cCount;
         newColVal[0] = 1.0;
         cCount++;
      } 
      else
      {
         newRowSize = 0;
         for (jcol = tPDiagI[fCount]; jcol < tPDiagI[fCount+1]; jcol++)
         {
            newColInd[newRowSize] = tPDiagJ[jcol] + AccStartRow; 
            newColVal[newRowSize++] = tPDiagA[jcol]; 
         }
         fCount++;
      }

      /* pruning */
      if (irow == 0) printf("pruning and scaling\n");
      dtemp = 0.0;
      for (jcol = 0; jcol < newRowSize; jcol++)
         if (habs(newColVal[jcol]) > dtemp) dtemp = habs(newColVal[jcol]);
      dtemp *= pruneFactor;
      ncount = 0;
      for (jcol = 0; jcol < newRowSize; jcol++)
      {
         if (habs(newColVal[jcol]) > dtemp) 
         {
            newColInd[ncount] = newColInd[jcol];
            newColVal[ncount++] = newColVal[jcol];
         }
      }
      newRowSize = ncount;

      /* scaling */
      dtemp = 0.0;
      for (jcol = 0; jcol < newRowSize; jcol++)
         dtemp += habs(newColVal[jcol]);
      dtemp = 1.0 / dtemp;
      for (jcol = 0; jcol < newRowSize; jcol++)
         newColVal[jcol] *= dtemp;
      ierr = HYPRE_IJMatrixSetValues(IJP, 1, &newRowSize,
                   (const int *) &rowIndex, (const int *) newColInd,
                   (const double *) newColVal);
      assert(!ierr);
   }
   delete [] newColInd;
   delete [] newColVal;
   ierr = HYPRE_IJMatrixAssemble(IJP);
   assert( !ierr );
   hypre_ParCSRMatrixDestroy(hypreP);
   HYPRE_IJMatrixGetObject(IJP, (void **) &hypreP);
   ierr += HYPRE_IJMatrixSetObjectType(IJP, -1);
   ierr += HYPRE_IJMatrixDestroy(IJP);
   assert(!ierr);

   /* ------------------------------------------------------ */
   /* package the P matrix                                   */
   /* ------------------------------------------------------ */

   sprintf(paramString, "HYPRE_ParCSR");
   funcPtr = new MLI_Function();
   MLI_Utils_HypreParCSRMatrixGetDestroyFunc(funcPtr);
   mli_Pmat = new MLI_Matrix((void*) hypreP, paramString, funcPtr);
   delete funcPtr;
   return mli_Pmat;
}

