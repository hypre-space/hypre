/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

#ifdef WIN32
#define strcasecmp _stricmp
#endif

/* #define MLI_USE_HYPRE_MATMATMULT */

#include <string.h>
#include <assert.h>
#include "HYPRE.h"
#include "parcsr_ls/parcsr_ls.h"
#include "util/mli_utils.h"
#include "matrix/mli_matrix.h"
#include "matrix/mli_matrix_misc.h"
#include "vector/mli_vector.h"
#include "solver/mli_solver.h"
#include "amgs/mli_method_amgrs.h"

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
   coarsenScheme_    = 0;              /* default : CLJP */
   measureType_      = 0;              /* default : local measure */
   threshold_        = 0.5;
   nodeDOF_          = 1;
   minCoarseSize_    = 200;
   maxRowSum_        = 0.9;
   symmetric_        = 1;
   useInjectionForR_ = 0;
   truncFactor_      = 0.0;
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
   if ( !strcasecmp(param1, "setOutputLevel" ))
   {
      sscanf(in_name,"%s %d", param1, &level);
      return ( setOutputLevel( level ) );
   }
   else if ( !strcasecmp(param1, "setNumLevels" ))
   {
      sscanf(in_name,"%s %d", param1, &level);
      return ( setNumLevels( level ) );
   }
   else if ( !strcasecmp(param1, "setCoarsenScheme" ))
   {
      sscanf(in_name,"%s %s", param1, param2);
      if ( !strcasecmp(param2, "cljp" ) )
         return ( setCoarsenScheme( MLI_METHOD_AMGRS_CLJP ) );
      else if ( !strcasecmp(param2, "ruge" ) )
         return ( setCoarsenScheme( MLI_METHOD_AMGRS_RUGE ) );
      else if ( !strcasecmp(param2, "falgout" ) )
         return ( setCoarsenScheme( MLI_METHOD_AMGRS_FALGOUT ) );
      else
      {
         printf("MLI_Method_AMGRS::setParams ERROR : setCoarsenScheme not");
         printf(" valid.  Valid options are : cljp, ruge, and falgout \n");
         return 1;
      }
   }
   else if ( !strcasecmp(param1, "setMeasureType" ))
   {
      sscanf(in_name,"%s %s", param1, param2);
      if ( !strcasecmp(param2, "local" ) )
         return ( setMeasureType( 0 ) );
      else if ( !strcasecmp(param2, "global" ) )
         return ( setMeasureType( 1 ) );
      else
      {
         printf("MLI_Method_AMGRS::setParams ERROR : setMeasureType not");
         printf(" valid.  Valid options are : local or global\n");
         return 1;
      }
   }
   else if ( !strcasecmp(param1, "setStrengthThreshold" ))
   {
      sscanf(in_name,"%s %lg", param1, &thresh);
      return ( setStrengthThreshold( thresh ) );
   }
   else if ( !strcasecmp(param1, "setTruncationFactor" ))
   {
      sscanf(in_name,"%s %lg", param1, &truncFactor_);
      return 0;
   }
   else if ( !strcasecmp(param1, "setNodeDOF" ))
   {
      sscanf(in_name,"%s %d", param1, &size);
      return ( setNodeDOF( size ) );
   }
   else if ( !strcasecmp(param1, "setNullSpace" ))
   {
      size = *(int *) argv[0];
      return ( setNodeDOF( size ) );
   }
   else if ( !strcasecmp(param1, "setMinCoarseSize" ))
   {
      sscanf(in_name,"%s %d", param1, &size);
      return ( setMinCoarseSize( size ) );
   }
   else if ( !strcasecmp(param1, "nonsymmetric" ))
   {
      symmetric_ = 0;
      return 0;
   }
   else if ( !strcasecmp(param1, "useInjectionForR" ))
   {
      useInjectionForR_ = 1;
      return 0;
   }
   else if ( !strcasecmp(param1, "setSmoother" ) || 
             !strcasecmp(param1, "setPreSmoother" ))
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
   else if ( !strcasecmp(param1, "setSmootherPrintRNorm" ))
   {
      smootherPrintRNorm_ = 1;
      return 0;
   }
   else if ( !strcasecmp(param1, "setSmootherFindOmega" ))
   {
      smootherFindOmega_ = 1;
      return 0;
   }
   else if ( !strcasecmp(param1, "setCoarseSolver" ))
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
   else if ( !strcasecmp(param1, "print" ))
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
   int             globalCoarseNRows;
   double          startTime, elapsedTime, colVal=1.0;
   char            paramString[100], *targv[10];
   MLI_Matrix      *mli_Pmat, *mli_Rmat, *mli_APmat, *mli_Amat, *mli_cAmat;
   MLI_Matrix      *mli_ATmat;
   MLI_Solver      *smootherPtr, *csolverPtr;
   MPI_Comm        comm;
   MLI_Function    *funcPtr;
   HYPRE_IJMatrix  IJRmat;
   hypre_ParCSRMatrix *hypreA, *hypreS, *hypreAT, *hypreST, *hypreP, *hypreR;
   hypre_ParCSRMatrix *hypreRT;
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

      if ( symmetric_ == 0 )
      {
         zeroNRows = 0;
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
         if ( hypreS != NULL ) hypre_ParCSRMatrixDestroy(hypreS);
         break;
      }
      k = (int) (globalNRows * 0.75);
      if ( coarsenScheme_ > 0 && coarsePartition[nprocs] >= k )
         coarsenScheme_ = 0;

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

      hypre_BoomerAMGBuildInterp(hypreA, CFMarkers, hypreS, 
                  coarsePartition, nodeDOF_, dofArray, outputLevel_, 
                  truncFactor_, &hypreP);
      funcPtr = new MLI_Function();
      MLI_Utils_HypreParCSRMatrixGetDestroyFunc(funcPtr);
      sprintf(paramString, "HYPRE_ParCSR" ); 
      mli_Pmat = new MLI_Matrix( (void *) hypreP, paramString, funcPtr );
      mli->setProlongation(level+1, mli_Pmat);
      delete funcPtr;
      if ( hypreS != NULL ) hypre_ParCSRMatrixDestroy(hypreS);

      /* ------build and set the restriction operator, if needed-------- */

      if ( useInjectionForR_ == 1 )
      {
         reduceArray1 = new int[nprocs+1];
         reduceArray2 = new int[nprocs+1];
         for ( k = 0; k < nprocs; k++ ) reduceArray1[k] = 0;
         reduceArray1[mypid] = coarseNRows;
         MPI_Allreduce(reduceArray1,reduceArray2,nprocs,MPI_INT,MPI_SUM,comm);
         for ( k = 0; k < nprocs; k++ ) reduceArray2[k+1] = reduceArray2[k];
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
                     truncFactor_, &hypreRT);
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

      startTime = MLI_Utils_WTime();

      /* ------construct and set the coarse grid matrix----------------- */

      if ( mypid == 0 && outputLevel_ > 0 ) printf("\tComputing RAP\n");
      if ( symmetric_ == 1 )
      {
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

