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
   outputLevel_   = 0;
   maxLevels_     = 25;
   numLevels_     = 25;
   currLevel_     = 0;
   coarsenScheme_ = 0;              /* default : CLJP */
   measureType_   = 0;              /* default : local measure */
   threshold_     = 0.5;
   nodeDOF_       = 1;
   minCoarseSize_ = 200;
   maxRowSum_     = 0.9;
   symmetric_     = 1;
   truncFactor_   = 0.0;
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
      return ( setStrengthThreshold( thresh ) );
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
   int             k, level, irow, local_nrows, mypid, nprocs;
   int             num_nodes, one=1, global_nrows, *coarse_partition;
   int             *CF_markers, coarse_nrows, *dof_array, *cdof_array=NULL;
   int             reduceArray1[2], reduceArray2[2], zeroNRows;
   double          start_time, elapsed_time;
   char            param_string[100], *targv[10];
   MLI_Matrix      *mli_Pmat, *mli_Rmat, *mli_APmat, *mli_Amat, *mli_cAmat;
   MLI_Matrix      *mli_ATmat;
   MLI_Solver      *smoother_ptr, *csolve_ptr;
   MPI_Comm        comm;
   MLI_Function    *func_ptr;
   hypre_ParCSRMatrix *hypreA, *hypreS, *hypreAT, *hypreST, *hypreP, *hypreR;
   hypre_ParCSRMatrix *hypreRT;
#if 0 
   hypre_ParCSRMatrix *hypreAP, *hypreCA;
#endif

#ifdef MLI_DEBUG_DETAILED
   printf("MLI_Method_AMGRS::setup begins...\n");
#endif

   /* --------------------------------------------------------------- */
   /* traverse all levels                                             */
   /* --------------------------------------------------------------- */

   RAPTime_ = 0.0;
   comm     = getComm();
   MPI_Comm_rank( comm, &mypid );
   MPI_Comm_size( comm, &nprocs );
   level    = 0;
   mli_Amat   = mli->getSystemMatrix(level);
   totalTime_ = MLI_Utils_WTime();

   for (level = 0; level < numLevels_; level++ )
   {
      if ( mypid == 0 && outputLevel_ > 0 )
      {
         printf("\t*****************************************************\n");
         printf("\t*** RS AMG : level = %d\n", level);
         printf("\t-----------------------------------------------------\n");
      }
      currLevel_ = level;
      if ( level == numLevels_-1 ) break;

      /* ------fetch fine grid matrix----------------------------------- */

      mli_Amat = mli->getSystemMatrix(level);
      assert ( mli_Amat != NULL );
      hypreA = (hypre_ParCSRMatrix *) mli_Amat->getMatrix();
      local_nrows  = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(hypreA));
      global_nrows = hypre_ParCSRMatrixGlobalNumRows(hypreA);

      /* ------create strength matrix----------------------------------- */

      num_nodes = local_nrows / nodeDOF_;
      if ( level == 0 && (num_nodes * nodeDOF_) != local_nrows )
      {
         printf("\tMLI_Method_AMGRS::setup - nrows not divisible by dof.\n");
         printf("\tMLI_Method_AMGRS::setup - revert nodeDOF to 1.\n");
         nodeDOF_ = 1; 
         num_nodes = local_nrows / nodeDOF_;
      }
      if ( level == 0 )
      {
         if ( local_nrows > 0 ) dof_array = new int[local_nrows];
         else                   dof_array = NULL;
         for ( irow = 0; irow < local_nrows; irow+=nodeDOF_ )
            for ( k = 0; k < nodeDOF_; k++ ) dof_array[irow+k] = k;
      }
      else
      {
         if ( level > 0 && dof_array != NULL ) delete [] dof_array;
         dof_array = cdof_array;
      }
      hypre_BoomerAMGCreateS(hypreA, threshold_, maxRowSum_, nodeDOF_,
                             dof_array, &hypreS);

      /* ------perform coarsening--------------------------------------- */

      switch ( coarsenScheme_ )
      {
         case MLI_METHOD_AMGRS_CLJP :
              hypre_BoomerAMGCoarsen(hypreS, hypreA, 0, outputLevel_,
                            	     &CF_markers);
              break;
         case MLI_METHOD_AMGRS_RUGE :
              hypre_BoomerAMGCoarsenRuge(hypreS, hypreA, measureType_,
                            	 coarsenScheme_, outputLevel_, &CF_markers);
              break;
         case MLI_METHOD_AMGRS_FALGOUT :
              hypre_BoomerAMGCoarsenFalgout(hypreS, hypreA, measureType_, 
                                            outputLevel_, &CF_markers);
              break;
      }

      /* ------if nonsymmetric, compute S for R------------------------- */

      coarse_nrows = 0;
      for ( irow = 0; irow < local_nrows; irow++ )
         if ( CF_markers[irow] == 1 ) coarse_nrows++;

      if ( symmetric_ == 0 )
      {
         MLI_Matrix_Transpose( mli_Amat, &mli_ATmat );
         hypreAT = (hypre_ParCSRMatrix *) mli_ATmat->getMatrix();
         hypre_BoomerAMGCreateS(hypreAT, threshold_, maxRowSum_, nodeDOF_,
                                dof_array, &hypreST);
         hypre_BoomerAMGCoarsen(hypreST, hypreAT, 1, outputLevel_,
                            	&CF_markers);

         coarse_nrows = 0;
         for ( irow = 0; irow < local_nrows; irow++ )
            if ( CF_markers[irow] == 1 ) coarse_nrows++;
      }
      reduceArray1[0] = coarse_nrows;
      reduceArray1[1] = local_nrows;
      MPI_Allreduce(reduceArray1, reduceArray2, 2, MPI_INT, MPI_SUM, comm);
      if ( outputLevel_ > 1 && mypid == 0 )
         printf("\tMLI_Method_AMGRS::setup - # C dof = %d(%d)\n",
                reduceArray2[0],reduceArray2[1]);

      /* ------see if the coarsest level is reached--------------------- */

      coarse_partition = (int *) hypre_CTAlloc(int, nprocs+1);
      coarse_partition[0] = 0;
      MPI_Allgather(&coarse_nrows, 1, MPI_INT, &(coarse_partition[1]),
		    1, MPI_INT, comm);
      for ( irow = 2; irow < nprocs+1; irow++ )
         coarse_partition[irow] += coarse_partition[irow-1];

      /* ------if nonsymmetric, need to make sure local_nrows > 0 ------ */
      /* ------ or the matrixTranspose function will give problems ----- */

      if ( symmetric_ == 0 )
      {
         zeroNRows = 0;
         for ( irow = 0; irow < nprocs; irow++ )
         {
            if ( (coarse_partition[irow+1]-coarse_partition[irow]) <= 0 )
            {
               zeroNRows = 1;
               break;
            }
         }
      }
          
      /* ------ wrap up creating the multigrid hierarchy --------------- */

      if ( coarse_partition[nprocs] < minCoarseSize_ ||
           coarse_partition[nprocs] == global_nrows || zeroNRows == 1 ) 
      {
         if ( symmetric_ == 0 )
         {
            delete mli_ATmat;
            hypre_ParCSRMatrixDestroy(hypreST);
         }
         hypre_TFree( coarse_partition );
         if ( CF_markers != NULL ) hypre_TFree( CF_markers );
         if ( hypreS != NULL ) hypre_ParCSRMatrixDestroy(hypreS);
         break;
      }
      k = (int) (global_nrows * 0.75);
      if ( coarsenScheme_ > 0 && coarse_partition[nprocs] >= k )
         coarsenScheme_ = 0;

      /* ------create new dof array for coarse grid--------------------- */

      if ( coarse_nrows > 0 ) cdof_array = new int[coarse_nrows];
      else                    cdof_array = NULL;
      coarse_nrows = 0;
      for ( irow = 0; irow < local_nrows; irow++ )
      {
         if ( CF_markers[irow] == 1 )
            cdof_array[coarse_nrows++] = dof_array[irow];
      }

      /* ------build and set the interpolation operator----------------- */

      hypre_BoomerAMGBuildInterp(hypreA, CF_markers, hypreS, 
                  coarse_partition, nodeDOF_, dof_array, outputLevel_, 
                  truncFactor_, &hypreP);
      func_ptr = new MLI_Function();
      MLI_Utils_HypreParCSRMatrixGetDestroyFunc(func_ptr);
      sprintf(param_string, "HYPRE_ParCSR" ); 
      mli_Pmat = new MLI_Matrix( (void *) hypreP, param_string, func_ptr );
      mli->setProlongation(level+1, mli_Pmat);
      delete func_ptr;
      if ( hypreS != NULL ) hypre_ParCSRMatrixDestroy(hypreS);

      /* ------build and set the restriction operator, if needed-------- */

      if ( symmetric_ == 0 )
      {
         hypre_BoomerAMGBuildInterp(hypreAT, CF_markers, hypreST, 
                     coarse_partition, nodeDOF_, dof_array, outputLevel_, 
                     truncFactor_, &hypreRT);
         hypreRT->owns_col_starts = 0;
         hypre_ParCSRMatrixTranspose( hypreRT, &hypreR, one );
         func_ptr = new MLI_Function();
         MLI_Utils_HypreParCSRMatrixGetDestroyFunc(func_ptr);
         sprintf(param_string, "HYPRE_ParCSRT" ); 
         mli_Rmat = new MLI_Matrix( (void *) hypreRT, param_string, func_ptr );
         mli->setRestriction(level, mli_Rmat);
         delete func_ptr;
         delete mli_ATmat;
         hypre_ParCSRMatrixDestroy(hypreST);
         hypre_ParCSRMatrixDestroy(hypreRT);
      }
      else
      {
         sprintf(param_string, "HYPRE_ParCSRT");
         mli_Rmat = new MLI_Matrix(mli_Pmat->getMatrix(), param_string, NULL);
         mli->setRestriction(level, mli_Rmat);
      }
      if ( CF_markers != NULL ) hypre_TFree( CF_markers );

      start_time = MLI_Utils_WTime();

      /* ------construct and set the coarse grid matrix----------------- */

      if ( mypid == 0 && outputLevel_ > 0 ) printf("\tComputing RAP\n");
      if ( symmetric_ == 1 )
      {
         MLI_Matrix_ComputePtAP(mli_Pmat, mli_Amat, &mli_cAmat);
      }
      else
      {
#if 1
           MLI_Matrix_MatMatMult(mli_Amat, mli_Pmat, &mli_APmat);
           MLI_Matrix_MatMatMult(mli_Rmat, mli_APmat, &mli_cAmat);
           delete mli_APmat;
#else
           hypreP  = (hypre_ParCSRMatrix *) mli_Pmat->getMatrix();
           hypreR  = (hypre_ParCSRMatrix *) mli_Rmat->getMatrix();
           hypreAP = hypre_ParMatmul( hypreA, hypreP );
           hypreCA = hypre_ParMatmul( hypreR, hypreAP );
           hypre_ParCSRMatrixDestroy( hypreAP );
           func_ptr = new MLI_Function();
           MLI_Utils_HypreParCSRMatrixGetDestroyFunc(func_ptr);
           sprintf(param_string, "HYPRE_ParCSR" ); 
           mli_cAmat = new MLI_Matrix((void *) hypreCA, param_string, func_ptr);
           delete func_ptr;
#endif
      }
      mli->setSystemMatrix(level+1, mli_cAmat);
      elapsed_time = (MLI_Utils_WTime() - start_time);
      RAPTime_ += elapsed_time;
      if ( mypid == 0 && outputLevel_ > 0 ) 
         printf("\tRAP computed, time = %e seconds.\n", elapsed_time);

      /* ------set the smoothers---------------------------------------- */

      smoother_ptr = MLI_Solver_CreateFromName( smoother_ );
      targv[0] = (char *) &smootherNSweeps_;
      targv[1] = (char *) smootherWeights_;
      sprintf( param_string, "relaxWeight" );
      smoother_ptr->setParams(param_string, 2, targv);
      if ( smootherPrintRNorm_ == 1 )
      {
         sprintf( param_string, "printRNorm" );
         smoother_ptr->setParams(param_string, 0, NULL);
      }
      if ( smootherFindOmega_ == 1 )
      {
         sprintf( param_string, "findOmega" );
         smoother_ptr->setParams(param_string, 0, NULL);
      }
      smoother_ptr->setup(mli_Amat);
      mli->setSmoother( level, MLI_SMOOTHER_BOTH, smoother_ptr );
   }
   if ( dof_array != NULL ) delete [] dof_array;

   /* ------set the coarse grid solver---------------------------------- */

   if (mypid == 0 && outputLevel_ > 0) printf("\tCoarse level = %d\n",level);
   csolve_ptr = MLI_Solver_CreateFromName( coarseSolver_ );
   if ( strcmp(coarseSolver_, "SuperLU") )
   {
      targv[0] = (char *) &coarseSolverNSweeps_;
      targv[1] = (char *) coarseSolverWeights_ ;
      sprintf( param_string, "relaxWeight" );
      csolve_ptr->setParams(param_string, 2, targv);
   }
   mli_Amat = mli->getSystemMatrix(level);
   csolve_ptr->setup(mli_Amat);
   mli->setCoarseSolve(csolve_ptr);
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
   int          mypid, level, global_nrows, tot_nrows, fine_nrows;
   int          max_nnz, min_nnz, fine_nnz, tot_nnz, this_nnz, itemp;
   double       max_val, min_val, dtemp;
   char         param_string[100];
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

   tot_nnz = tot_nrows = 0;
   for ( level = 0; level <= currLevel_; level++ )
   {
      mli_Amat = mli->getSystemMatrix( level );
      sprintf(param_string, "nrows");
      mli_Amat->getMatrixInfo(param_string, global_nrows, dtemp);
      sprintf(param_string, "maxnnz");
      mli_Amat->getMatrixInfo(param_string, max_nnz, dtemp);
      sprintf(param_string, "minnnz");
      mli_Amat->getMatrixInfo(param_string, min_nnz, dtemp);
      sprintf(param_string, "totnnz");
      mli_Amat->getMatrixInfo(param_string, this_nnz, dtemp);
      sprintf(param_string, "maxval");
      mli_Amat->getMatrixInfo(param_string, itemp, max_val);
      sprintf(param_string, "minval");
      mli_Amat->getMatrixInfo(param_string, itemp, min_val);
      if ( mypid == 0 )
      {
         printf("\t*%3d %9d %5d  %5d %10d %8.3e %8.3e *\n",level,
                global_nrows, max_nnz, min_nnz, this_nnz, max_val, min_val);
      }
      if ( level == 0 ) fine_nnz = this_nnz;
      tot_nnz += this_nnz;
      if ( level == 0 ) fine_nrows = global_nrows;
      tot_nrows += global_nrows;
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
      sprintf(param_string, "nrows");
      mli_Pmat->getMatrixInfo(param_string, global_nrows, dtemp);
      sprintf(param_string, "maxnnz");
      mli_Pmat->getMatrixInfo(param_string, max_nnz, dtemp);
      sprintf(param_string, "minnnz");
      mli_Pmat->getMatrixInfo(param_string, min_nnz, dtemp);
      sprintf(param_string, "totnnz");
      mli_Pmat->getMatrixInfo(param_string, this_nnz, dtemp);
      sprintf(param_string, "maxval");
      mli_Pmat->getMatrixInfo(param_string, itemp, max_val);
      sprintf(param_string, "minval");
      mli_Pmat->getMatrixInfo(param_string, itemp, min_val);
      if ( mypid == 0 )
      {
         printf("\t*%3d %9d %5d  %5d %10d %8.3e %8.3e *\n",level,
                global_nrows, max_nnz, min_nnz, this_nnz, max_val, min_val);
      }
   }

   /* --------------------------------------------------------------- */
   /* other complexity information                                    */
   /* --------------------------------------------------------------- */

   if ( mypid == 0 )
   {
      printf("\t********************************************************\n");
      dtemp = (double) tot_nnz / (double) fine_nnz;
      printf("\t*** Amat complexity  = %e\n", dtemp);
      dtemp = (double) tot_nrows / (double) fine_nrows;
      printf("\t*** grid complexity  = %e\n", dtemp);
      printf("\t********************************************************\n");
      fflush(stdout);
   }
   return 0;
}

