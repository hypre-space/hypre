/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

#include <string.h>
#include <strings.h>
#include <assert.h>
#include "HYPRE.h"
#include "util/mli_utils.h"
#include "matrix/mli_matrix.h"
#include "vector/mli_vector.h"
#include "solver/mli_solver.h"
#include "base/mli_defs.h"
#include "amgs/mli_method_amgsa.h"

/* ********************************************************************* *
 * functions external to MLI 
 * --------------------------------------------------------------------- */

#ifdef MLI_ARPACK
extern "C"
{
   /* ARPACK function to compute eigenvalues/eigenvectors */
   void dnstev_(int *n, int *nev, char *which, double *sigmar, 
                double *sigmai, int *colptr, int *rowind, double *nzvals, 
                double *dr, double *di, double *z, int *ldz, int *info);
}
#endif

/* ********************************************************************* *
 * constructor
 * --------------------------------------------------------------------- */

MLI_Method_AMGSA::MLI_Method_AMGSA( MPI_Comm comm ) : MLI_Method( comm )
{
   char name[100];

   strcpy(name, "AMGSA");
   setName( name );
   setID( MLI_METHOD_AMGSA_ID );
   maxLevels_     = 40;
   numLevels_     = 40;
   currLevel_     = 0;
   outputLevel_   = 0;
   nodeDofs_      = 1;
   currNodeDofs_  = 1;
   threshold_     = 0.08;
   nullspaceDim_  = 1;
   nullspaceVec_  = NULL;
   nullspaceLen_  = 0;
   Pweight_       = 0.0;
   dropTolForP_   = 0.0;            /* tolerance to sparsify P*/
   saCounts_      = new int[40];    /* number of aggregates   */
   saData_        = new int*[40];   /* node to aggregate data */
   spectralNorms_ = new double[40]; /* calculated max eigen   */
   for ( int i = 0; i < 40; i++ ) 
   {
      saCounts_[i] = 0;
      saData_[i]   = NULL;
      spectralNorms_[i] = 0.0;
   }
   calcNormScheme_ = 0;              /* use matrix rowsum norm */
   minCoarseSize_  = 5;              /* smallest coarse grid   */
   coarsenScheme_  = MLI_METHOD_AMGSA_LOCAL;
   strcpy(preSmoother_, "Jacobi");
   strcpy(postSmoother_, "Jacobi");
   preSmootherNum_  = 2;
   postSmootherNum_  = 2;
   preSmootherWgt_  = new double[2];
   postSmootherWgt_  = new double[2];
   preSmootherWgt_[0] = preSmootherWgt_[1] = 0.667;
   postSmootherWgt_[0] = postSmootherWgt_[1] = 0.667;
   strcpy(coarseSolver_, "SGS");
   coarseSolverNum_    = 20;
   coarseSolverWgt_    = new double[20];
   for ( int j = 0; j < 20; j++ ) coarseSolverWgt_ [j] = 1.0;
   calibrationSize_    = 0;
   useSAMGeFlag_       = 0;
   RAPTime_            = 0.0;
   totalTime_          = 0.0;
   ddObj_              = NULL;
   ARPACKSuperLUExists_ = 0;
   saLabels_            = NULL;
   useSAMGDDFlag_       = 0;
   printNullSpace_      = 0;
   strcpy( paramFile_, "empty" );
}

/* ********************************************************************* *
 * destructor
 * --------------------------------------------------------------------- */

MLI_Method_AMGSA::~MLI_Method_AMGSA()
{
   char paramString[20];

   if ( nullspaceVec_ != NULL ) delete [] nullspaceVec_;
   if ( saCounts_ != NULL ) delete [] saCounts_;
   if ( saData_ != NULL )
   {
      for ( int i = 0; i < maxLevels_; i++ )
      {
         if ( saData_[i] != NULL ) delete [] saData_[i];
         else break;
      }
      delete [] saData_;
      saData_ = NULL;
   }
   if ( saLabels_ != NULL )
   {
      for ( int i = 0; i < maxLevels_; i++ )
      {
         if ( saLabels_[i] != NULL ) delete [] saLabels_[i];
         else break;
      }
      delete [] saLabels_;
      saLabels_ = NULL;
   }
   if ( spectralNorms_   != NULL ) delete [] spectralNorms_;
   if ( preSmootherWgt_  != NULL ) delete [] preSmootherWgt_;
   if ( postSmootherWgt_ != NULL ) delete [] postSmootherWgt_;
   if ( coarseSolverWgt_ != NULL ) delete [] coarseSolverWgt_ ;
   if ( ddObj_!= NULL ) 
   {
      if ( ddObj_->sendProcs != NULL ) delete [] ddObj_->sendProcs;
      if ( ddObj_->recvProcs != NULL ) delete [] ddObj_->recvProcs;
      if ( ddObj_->sendLengs != NULL ) delete [] ddObj_->sendLengs;
      if ( ddObj_->recvLengs != NULL ) delete [] ddObj_->recvLengs;
      if ( ddObj_->sendMap   != NULL ) delete [] ddObj_->sendMap;
      if ( ddObj_->ANodeEqnList != NULL ) delete [] ddObj_->ANodeEqnList;
      if ( ddObj_->SNodeEqnList != NULL ) delete [] ddObj_->SNodeEqnList;
      delete ddObj_;
   }
   if ( ARPACKSuperLUExists_ ) 
   {
      strcpy( paramString, "destroy" );
#ifdef MLI_ARPACK
      int  info;
      dnstev_(NULL, NULL, paramString, NULL, NULL, NULL, NULL, NULL, NULL, 
              NULL, NULL, NULL, &info);
#endif
   }
}

/* ********************************************************************* *
 * set parameters
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::setParams(char *in_name, int argc, char *argv[])
{
   int        level, size, nDOF, numNS, length, nSweeps=1, offset, nsDim;
   int        prePost, nnodes, nAggr, *aggrInfo, *labels, is, *indices;
   double     thresh, pweight, *nullspace, *weights=NULL, *coords, *scales;
   double     *nsAdjust;
   char       param1[256], param2[256], *param3;

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
   else if ( !strcasecmp(param1, "useSAMGe" ))
   {
      useSAMGeFlag_ = 1;
      return 0;
   }
   else if ( !strcasecmp(param1, "useSAMGDD" ))
   {
      useSAMGDDFlag_ = 1;
      return 0;
   }
   else if ( !strcasecmp(param1, "setCoarsenScheme" ))
   {
      sscanf(in_name,"%s %s", param1, param2);
      if ( !strcasecmp(param2, "local" ) )
         return ( setCoarsenScheme( MLI_METHOD_AMGSA_LOCAL ) );
      else      
      {
         printf("MLI_Method_AMGSA::setParams ERROR : setCoarsenScheme not");
         printf(" valid.  Valid options are : local \n");
         return 1;
      }
   }
   else if ( !strcasecmp(param1, "setMinCoarseSize" ))
   {
      sscanf(in_name,"%s %d", param1, &size);
      return ( setMinCoarseSize( size ) );
   }
   else if ( !strcasecmp(param1, "setStrengthThreshold" ))
   {
      sscanf(in_name,"%s %lg", param1, &thresh);
      return ( setStrengthThreshold( thresh ) );
   }
   else if ( !strcasecmp(param1, "setPweight" ))
   {
      sscanf(in_name,"%s %lg", param1, &pweight);
      return ( setPweight( pweight ) );
   }
   else if ( !strcasecmp(param1, "setCalcSpectralNorm" ))
   {
      return ( setCalcSpectralNorm() );
   }
   else if ( !strcasecmp(param1, "setAggregateInfo" ))
   {
      if ( argc != 4 )
      {
         printf("MLI_Method_AMGSA::setParams ERROR - setAggregateInfo");
         printf(" needs 4 args.\n");
         printf("     argument[0] : level number \n");
         printf("     argument[1] : number of aggregates \n");
         printf("     argument[2] : total degree of freedom \n");
         printf("     argument[3] : aggregate information \n");
         return 1;
      } 
      level    = *(int *) argv[0];
      nAggr    = *(int *) argv[1];
      length   = *(int *) argv[2];
      aggrInfo = (int *)  argv[3];
      return ( setAggregateInfo(level,nAggr,length,aggrInfo) );
   }
   else if ( !strcasecmp(param1, "setCalibrationSize" ))
   {
      sscanf(in_name,"%s %d", param1, &size);
      return ( setCalibrationSize( size ) );
   }
   else if ( !strcasecmp(param1, "setPreSmoother" ))
   {
      sscanf(in_name,"%s %s", param1, param2);
      if ( argc != 2 )
      {
         printf("MLI_Method_AMGSA::setParams ERROR - setPreSmoother needs");
         printf(" 2 arguments.\n");
         printf("     argument[0] : number of relaxation sweeps \n");
         printf("     argument[1] : relaxation weights\n");
         return 1;
      } 
      prePost = MLI_SMOOTHER_PRE;
      nSweeps = *(int *)   argv[0];
      weights = (double *) argv[1];
      return ( setSmoother(prePost, param2, nSweeps, weights) );
   }
   else if ( !strcasecmp(param1, "setPostSmoother" ))
   {
      sscanf(in_name,"%s %s", param1, param2);
      if ( argc != 2 )
      {
         printf("MLI_Method_AMGSA::setParams ERROR - setPostSmoother needs");
         printf(" 2 arguments.\n");
         printf("     argument[0] : number of relaxation sweeps \n");
         printf("     argument[1] : relaxation weights\n");
         return 1;
      } 
      prePost = MLI_SMOOTHER_POST;
      nSweeps = *(int *)   argv[0];
      weights = (double *) argv[1];
      return ( setSmoother(prePost, param2, nSweeps, weights) );
   }
   else if ( !strcasecmp(param1, "setCoarseSolver" ))
   {
      sscanf(in_name,"%s %s", param1, param2);
      if ( strcmp(param2, "SuperLU") && argc != 2 )
      {
         printf("MLI_Method_AMGSA::setParams ERROR - setCoarseSolver needs");
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
   else if ( !strcasecmp(param1, "setNullSpace" ))
   {
      if ( argc != 4 )
      {
         printf("MLI_Method_AMGSA::setParams ERROR - setNullSpace needs");
         printf(" 4 arguments.\n");
         printf("     argument[0] : node degree of freedom \n");
         printf("     argument[1] : number of null space vectors \n");
         printf("     argument[2] : null space information \n");
         printf("     argument[3] : vector length \n");
         return 1;
      } 
      nDOF      = *(int *)   argv[0];
      numNS     = *(int *)   argv[1];
      nullspace = (double *) argv[2];
      length    = *(int *)   argv[3];
      return ( setNullSpace(nDOF,numNS,nullspace,length) );
   }
   else if ( !strcasecmp(param1, "adjustNullSpace" ))
   {
      if ( argc != 1 )
      {
         printf("MLI_Method_AMGSA::setParams ERROR - adjustNullSpace needs");
         printf(" 1 argument.\n");
         printf("     argument[0] : adjustment vectors \n");
         return 1;
      } 
      nsAdjust = (double *) argv[0];
      return ( adjustNullSpace( nsAdjust ) );
   }
   else if ( !strcasecmp(param1, "resetNullSpaceComponents" ))
   {
      if ( argc != 3 )
      {
         printf("MLI_Method_AMGSA::setParams ERROR - resetNSComponents needs");
         printf(" 2 arguments.\n");
         printf("     argument[0] : number of equations \n");
         printf("     argument[1] : equation number offset \n");
         printf("     argument[2] : list of equation numbers \n");
         return 1;
      } 
      length  = *(int *) argv[0];
      offset  = *(int *) argv[1];
      indices =  (int *) argv[2];
      return ( resetNullSpaceComponents(length, offset, indices) );
   }
   else if ( !strcasecmp(param1, "setNodalCoord" ))
   {
      if ( argc != 5 && argc != 6 )
      {
         printf("MLI_Method_AMGSA::setParams ERROR - setNodalCoord needs");
         printf(" 4 arguments.\n");
         printf("     argument[0] : number of nodes \n");
         printf("     argument[1] : node degree of freedom \n");
         printf("     argument[2] : number of space dimension\n");
         printf("     argument[3] : coordinate information \n");
         printf("     argument[4] : null space dimension \n");
         printf("     argument[5] : scalings (can be null) \n");
         return 1;
      } 
      nnodes = *(int *)   argv[0];
      nDOF   = *(int *)   argv[1];
      nsDim  = *(int *)   argv[2];
      coords = (double *) argv[3];
      numNS  = *(int *)   argv[4];
      if ( argc == 6 ) scales = (double *) argv[5]; else scales = NULL;
      return ( setNodalCoordinates(nnodes,nDOF,nsDim,coords,numNS,scales) );
   }
   else if ( !strcasecmp(param1, "setLabels" ))
   {
      if ( argc != 3 )
      {
         printf("MLI_Method_AMGSA::setParams ERROR - setLabels needs");
         printf(" 3 arguments.\n");
         printf("     argument[0] : vector length \n");
         printf("     argument[1] : level number \n");
         printf("     argument[2] : label information \n");
         return 1;
      } 
      length = *(int *) argv[0];
      level  = *(int *) argv[1];
      labels =  (int *) argv[2];
      if ( saLabels_ == NULL ) 
      {
         saLabels_ = new int*[maxLevels_];
         for ( is = 0; is < maxLevels_; is++ ) saLabels_[is] = NULL;
      }
      if ( level < 0 || level >= maxLevels_ )
      {
         printf("MLI_Method_AMGSA::setParams ERROR - setLabels has \n");
         printf("invalid level number = %d (%d)\n", level, maxLevels_);
         return 1;
      }
      if ( saLabels_[level] != NULL ) delete [] saLabels_[level];
      saLabels_[level] = new int[length];
      for ( is = 0; is < length; is++ ) saLabels_[level][is] = labels[is];
      return 0;
   }
   else if ( !strcasecmp(param1, "setParamFile" ))
   {
      param3 = (char *) argv[0];
      strcpy( paramFile_, param3 ); 
      return 0;
   }
   else if ( !strcasecmp(param1, "printNullSpace" ))
   {
      printNullSpace_ = 1;
      return 0;
   }
   else if ( !strcasecmp(param1, "print" ))
   {
      return ( print() );
   }
   return 1;
}

/*****************************************************************************
 * get parameters 
 *--------------------------------------------------------------------------*/

int MLI_Method_AMGSA::getParams(char *in_name, int *argc, char *argv[])
{
   int    nDOF, numNS, length;
   double *nullspace;

   if ( !strcasecmp(in_name, "getNullSpace" ))
   {
      if ( (*argc) < 4 )
      {
         printf("MLI_Method_AMGSA::getParams ERROR - getNullSpace needs");
         printf(" 4 arguments.\n");
         exit(1);
      }
      getNullSpace(nodeDofs_,numNS,nullspace,length);
      argv[0] = (char *) &nDOF;
      argv[1] = (char *) &numNS;
      argv[2] = (char *) nullspace;
      argv[3] = (char *) &length;
      (*argc) = 4;
      return 0;
   }
   else
   {
      printf("MLI_Method_AMGSA::getParams ERROR - invalid param string.\n");
      return 1;
   }
}

/***********************************************************************
 * generate multilevel structure
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::setup( MLI *mli ) 
{
   int             level, mypid;
   double          start_time, elapsed_time, max_eigen;
   char            param_string[100], *targv[10];
   MLI_Matrix      *mli_Pmat, *mli_Rmat, *mli_Amat, *mli_cAmat;
   MLI_Solver      *smoother_ptr, *csolve_ptr;
   MPI_Comm        comm;

#ifdef MLI_DEBUG_DETAILED
   printf("MLI_Method_AMGSA::setup begins...\n");
#endif

   /* --------------------------------------------------------------- */
   /* clean up some mess made previously                              */
   /* --------------------------------------------------------------- */

   if ( saData_ != NULL )
   {
      for ( level = 1; level < maxLevels_; level++ )
      {
         if ( saData_[level] != NULL ) delete [] saData_[level];
         saData_[level] = NULL;
      }
   }

   /* --------------------------------------------------------------- */
   /* call SAe and/or setupDD if flag is set                          */
   /* --------------------------------------------------------------- */

   if ( useSAMGeFlag_ )  setupSubdomainNullSpaceUsingFEData(mli);
   if ( useSAMGDDFlag_ ) setupDDFormSubdomainAggregate(mli);

   /* --------------------------------------------------------------- */
   /* call calibration if calibration size > 0                        */
   /* --------------------------------------------------------------- */

   if ( calibrationSize_ > 0 ) return( setupCalibration( mli ) );
      
   /* --------------------------------------------------------------- */
   /* traverse all levels                                             */
   /* --------------------------------------------------------------- */

   RAPTime_ = 0.0;
   level    = 0;
   comm     = getComm();
   MPI_Comm_rank( comm, &mypid );
   mli_Amat   = mli->getSystemMatrix(level);
   totalTime_ = MLI_Utils_WTime();
   if ( nullspaceDim_ != nodeDofs_ && nullspaceVec_ == NULL )
      nullspaceDim_ = nodeDofs_;

#if HAVE_LOBPCG
   relaxNullSpaces(mli_Amat);
#endif
   
   for (level = 0; level < numLevels_; level++ )
   {
      if ( mypid == 0 && outputLevel_ > 0 )
      {
         printf("\t*****************************************************\n");
         printf("\t*** Aggregation (uncoupled) : level = %d\n", level);
         printf("\t-----------------------------------------------------\n");
      }
      currLevel_ = level;
      if ( level == numLevels_-1 ) break;

      /* ------fetch fine grid matrix----------------------------------- */

      mli_Amat = mli->getSystemMatrix(level);
      assert ( mli_Amat != NULL );

      /* ------perform coarsening--------------------------------------- */

      switch ( coarsenScheme_ )
      {
         case MLI_METHOD_AMGSA_LOCAL :
              if ( level == 0 )
                 max_eigen = genPLocal(mli_Amat, &mli_Pmat, saCounts_[0], 
                                       saData_[0]); 
              else
                 max_eigen = genPLocal(mli_Amat, &mli_Pmat, 0, NULL); 
              break;
      }
      if ( max_eigen != 0.0 ) spectralNorms_[level] = max_eigen;
      if ( mli_Pmat == NULL ) break;
      start_time = MLI_Utils_WTime();

      /* ------construct and set the coarse grid matrix----------------- */

      if ( mypid == 0 && outputLevel_ > 0 ) printf("\tComputing RAP\n");
      MLI_Matrix_ComputePtAP(mli_Pmat, mli_Amat, &mli_cAmat);
      mli->setSystemMatrix(level+1, mli_cAmat);
      elapsed_time = (MLI_Utils_WTime() - start_time);
      RAPTime_ += elapsed_time;
      if ( mypid == 0 && outputLevel_ > 0 ) 
         printf("\tRAP computed, time = %e seconds.\n", elapsed_time);

#if 0
      mli_Amat->print("Amat");
      mli_Pmat->print("Pmat");
      mli_cAmat->print("cAmat");
#endif

      /* ------set the prolongation matrix------------------------------ */

      mli->setProlongation(level+1, mli_Pmat);

      /* ------set the restriction matrix------------------------------- */

      sprintf(param_string, "HYPRE_ParCSRT");
      mli_Rmat = new MLI_Matrix(mli_Pmat->getMatrix(), param_string, NULL);
      mli->setRestriction(level, mli_Rmat);

      /* ------set the smoothers---------------------------------------- */

      if ( useSAMGDDFlag_ && numLevels_ == 2 && 
           !strcmp(preSmoother_, "ARPACKSuperLU") )
      {
         setupDDSuperLUSmoother(mli, level);
         smoother_ptr = MLI_Solver_CreateFromName(preSmoother_);
         targv[0] = (char *) ddObj_;
         sprintf( param_string, "ARPACKSuperLUObject" );
         smoother_ptr->setParams(param_string, 1, targv);
         smoother_ptr->setup(mli_Amat);
         mli->setSmoother( level, MLI_SMOOTHER_PRE, smoother_ptr );
#if 0
         smoother_ptr = MLI_Solver_CreateFromName(preSmoother_);
         smoother_ptr->setParams(param_string, 1, targv);
         smoother_ptr->setup(mli_Amat);
         mli->setSmoother( level, MLI_SMOOTHER_POST, smoother_ptr );
#endif
         continue;
      }
      smoother_ptr = MLI_Solver_CreateFromName( preSmoother_ );
      targv[0] = (char *) &preSmootherNum_;
      targv[1] = (char *) preSmootherWgt_;
      sprintf( param_string, "relaxWeight" );
      smoother_ptr->setParams(param_string, 2, targv);
      if ( !strcmp(preSmoother_, "MLS") ) 
      {
         sprintf( param_string, "maxEigen" );
         targv[0] = (char *) &max_eigen;
         smoother_ptr->setParams(param_string, 1, targv);
      }
      smoother_ptr->setup(mli_Amat);
      mli->setSmoother( level, MLI_SMOOTHER_PRE, smoother_ptr );

      if ( strcmp(preSmoother_, postSmoother_) )
      {
         smoother_ptr = MLI_Solver_CreateFromName( postSmoother_ );
         targv[0] = (char *) &postSmootherNum_;
         targv[1] = (char *) postSmootherWgt_;
         sprintf( param_string, "relaxWeight" );
         smoother_ptr->setParams(param_string, 2, targv);
         if ( !strcmp(postSmoother_, "MLS") ) 
         {
            sprintf( param_string, "maxEigen" );
            targv[0] = (char *) &max_eigen;
            smoother_ptr->setParams(param_string, 1, targv);
         }
         smoother_ptr->setup(mli_Amat);
      }
      mli->setSmoother( level, MLI_SMOOTHER_POST, smoother_ptr );
   }

   /* ------set the coarse grid solver---------------------------------- */

   if (mypid == 0 && outputLevel_ > 0) printf("\tCoarse level = %d\n",level);
   csolve_ptr = MLI_Solver_CreateFromName( coarseSolver_ );
   if ( strcmp(coarseSolver_, "SuperLU") )
   {
      targv[0] = (char *) &coarseSolverNum_;
      targv[1] = (char *) coarseSolverWgt_ ;
      sprintf( param_string, "relaxWeight" );
      csolve_ptr->setParams(param_string, 2, targv);
      if ( !strcmp(coarseSolver_, "MLS") )
      {
         sprintf( param_string, "maxEigen" );
         targv[0] = (char *) &max_eigen;
         csolve_ptr->setParams(param_string, 1, targv);
      }
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
   printf("MLI_Method_AMGSA::setup ends.");
#endif
   return (level+1);
}

/* ********************************************************************* *
 * set diagnostics output level
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::setOutputLevel( int level )
{
   outputLevel_ = level;
   return 0;
}

/* ********************************************************************* *
 * set number of levels 
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::setNumLevels( int nlevels )
{
   if ( nlevels < maxLevels_ && nlevels > 0 ) numLevels_ = nlevels;
   return 0;
}

/* ********************************************************************* *
 * set smoother
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::setSmoother(int prePost, char *stype, int num, 
                                  double *wgt)
{
   int i;

#ifdef MLI_DEBUG_DETAILED
   printf("MLI_Method_AMGSA::setSmoother - type = %s.\n", stype);
#endif

   if ( prePost != MLI_SMOOTHER_PRE && prePost != MLI_SMOOTHER_BOTH &&
        prePost != MLI_SMOOTHER_POST )
   {
      printf("MLI_Method_AMGSA::setSmoother ERROR - invalid info (1).\n");
      return 1;
   }
   if ( prePost == MLI_SMOOTHER_PRE || prePost != MLI_SMOOTHER_BOTH )
   {
      strcpy( preSmoother_, stype );
      if ( num > 0 ) preSmootherNum_ = num; else preSmootherNum_ = 1;
      delete [] preSmootherWgt_;
      preSmootherWgt_ = new double[preSmootherNum_];
      if ( wgt == NULL )
         for (i = 0; i < preSmootherNum_; i++) preSmootherWgt_[i] = 0.;
      else
         for (i = 0; i < preSmootherNum_; i++) preSmootherWgt_[i] = wgt[i];
   }
   if ( prePost == MLI_SMOOTHER_POST || prePost == MLI_SMOOTHER_BOTH )
   {
      strcpy( postSmoother_, stype );
      if ( num > 0 ) postSmootherNum_ = num; else postSmootherNum_ = 1;
      delete [] postSmootherWgt_;
      postSmootherWgt_ = new double[postSmootherNum_];
      if ( wgt == NULL )
         for (i = 0; i < postSmootherNum_; i++) postSmootherWgt_[i] = 0.;
      else
         for (i = 0; i < postSmootherNum_; i++) postSmootherWgt_[i] = wgt[i];
   }
   return 0;
}

/* ********************************************************************* *
 * set coarse solver 
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::setCoarseSolver( char *stype, int num, double *wgt )
{
   int i;

#ifdef MLI_DEBUG_DETAILED
   printf("MLI_Method_AMGSA::setCoarseSolver - type = %s.\n", stype);
#endif

   strcpy( coarseSolver_, stype );
   if ( num > 0 ) coarseSolverNum_ = num; else coarseSolverNum_ = 1;
   delete [] coarseSolverWgt_ ;
   if ( wgt != NULL && strcmp(coarseSolver_, "SuperLU") )
   {
      coarseSolverWgt_  = new double[coarseSolverNum_]; 
      for (i = 0; i < coarseSolverNum_; i++) coarseSolverWgt_ [i] = wgt[i];
   }
   else coarseSolverWgt_  = NULL;
   return 0;
}

/* ********************************************************************* *
 * set coarsening scheme 
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::setCoarsenScheme( int scheme )
{
   if ( scheme == MLI_METHOD_AMGSA_LOCAL ) 
   {
      coarsenScheme_ = MLI_METHOD_AMGSA_LOCAL;
      return 0;
   }
   else
   {
      printf("MLI_Method_AMGSA::setCoarsenScheme ERROR - invalid scheme.\n");
      return 1;
   }
}

/* ********************************************************************* *
 * set minimum coarse size
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::setMinCoarseSize( int coarse_size )
{
   if ( coarse_size > 0 ) minCoarseSize_ = coarse_size;
   return 0;
}

/* ********************************************************************* *
 * set coarsening threshold
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::setStrengthThreshold( double thresh )
{
   if ( thresh > 0.0 ) threshold_ = thresh;
   else                threshold_ = 0.0;
   return 0;
}

/* ********************************************************************* *
 * set damping factor for smoother prolongator
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::setPweight( double weight )
{
   if ( weight >= 0.0 && weight <= 2.0 ) Pweight_ = weight;
   return 0;
}

/* ********************************************************************* *
 * indicate spectral norm is to be calculated
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::setCalcSpectralNorm()
{
   calcNormScheme_ = 1;
   return 0;
}

/* ********************************************************************* *
 * load the initial aggregate information 
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::setAggregateInfo(int level, int aggrCnt, int length,
                                       int *aggrInfo) 
{
   if ( level != 0 )
   {
      printf("MLI_Method_AMGSA::setAggregateInfo ERROR : invalid level");
      printf(" number = %d.", level);
      return 1;
   }
   saCounts_[level] = aggrCnt;
   if ( saData_[level] != NULL ) delete [] saData_[level];
   saData_[level] = new int[length];
   for ( int i = 0; i < length; i++ ) saData_[level][i] = aggrInfo[i];
   return 0;
}

/* ********************************************************************* *
 * load the null space
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::setNullSpace( int nDOF, int ndim, double *nullvec, 
                                    int length ) 
{
#if 0
   if ( (nullvec == NULL) && (nDOF != ndim) )
   {
      printf("MLI_Method_AMGSA::setNullSpace WARNING -  When no nullspace\n");
      printf(" vector is specified, the nodal DOFS must be equal to the \n");
      printf("nullspace dimension.\n");
      ndim = nDOF;
   }
#endif
   nodeDofs_     = nDOF;
   currNodeDofs_ = nDOF;
   nullspaceDim_ = ndim;
   nullspaceLen_ = length;
   if ( nullspaceVec_ != NULL ) delete [] nullspaceVec_;
   if ( nullvec != NULL )
   {
      nullspaceVec_ = new double[length * ndim];
      for ( int i = 0; i < length*ndim; i++ )
         nullspaceVec_[i] = nullvec[i];
   }
   else nullspaceVec_ = NULL;
   return 0;
}

/* ********************************************************************* *
 * adjust null space vectors
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::adjustNullSpace(double *vecAdjust)
{
   int i;

   if ( useSAMGeFlag_ ) return 0;

   for ( i = 0; i < nullspaceLen_*nullspaceDim_; i++ )
      nullspaceVec_[i] += vecAdjust[i];

   return 0;
}

/* ********************************************************************* *
 * reset some entry in the null space vectors
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::resetNullSpaceComponents(int length, int start,
                                               int *eqnIndices)
{
   int i, j, index;

   if ( useSAMGeFlag_ ) return 0;

   for ( i = 0; i < length; i++ )
   {
      index = eqnIndices[i] - start;
      for ( j = 0; j < nullspaceDim_; j++ )
         nullspaceVec_[j*nullspaceLen_+index] = 0.;
   }
   return 0;
}

/* ********************************************************************* *
 * load nodal coordinates (translates into rigid body modes)
 * (abridged from similar function in ML)
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::setNodalCoordinates(int num_nodes, int nDOF, int nsDim, 
                         double *coords, int numNS, double *scalings)
{
   int  i, j, k, offset, voffset, mypid;
   char fname[100];
   FILE *fp;

   MPI_Comm comm = getComm();
   MPI_Comm_rank( comm, &mypid );

   if ( useSAMGeFlag_ ) return 0;

   if ( nDOF == 1 )
   {
      nodeDofs_     = 1;
      currNodeDofs_ = 1;
      nullspaceLen_ = num_nodes;
      nullspaceDim_ = numNS;
      if ( numNS != 1 && !(numNS == 4 && nsDim == 3) ) 
      {
         printf("setNodalCoordinates: nDOF,numNS,nsDim = %d %d %d\n",nDOF,
                numNS,nsDim);
         exit(1);
      }
   }
   else if ( nDOF == 3 )
   {
      nodeDofs_     = 3;
      currNodeDofs_ = 3;
      nullspaceLen_ = num_nodes * 3;
      nullspaceDim_ = numNS;
      if ( nullspaceDim_ <= 3 ) nullspaceDim_  = 6;
      if ( numNS != 3 && numNS != 6 && numNS != 9 && numNS != 12 ) 
      {
         printf("setNodalCoordinates: numNS %d not supported\n",numNS);
         exit(1);
      }
   }
   else
   {
      printf("setNodalCoordinates: nDOF = %d not supported\n",nDOF);
      exit(1);
   }
   if ( nullspaceVec_ != NULL ) delete [] nullspaceVec_;
   nullspaceVec_ = new double[nullspaceLen_ * nullspaceDim_];

   for( i = 0 ; i < num_nodes; i++ ) 
   {
      if ( nodeDofs_ == 1 )
      {
         nullspaceVec_[i] = 1.0;
         if ( nullspaceDim_ == 4 ) 
         {
            for( k = 0; k < nsDim; k++ )
               nullspaceVec_[(k+1)*nullspaceLen_+i] = coords[i*nsDim+k];
         }
      }
      else if ( nodeDofs_ == 3 )
      {
         voffset = i * nodeDofs_;
         for ( j = 0; j < 3; j++ )
         {
            for( k = 0; k < 3; k++ )
            {
               offset = k * nullspaceLen_ + voffset + j;
               if ( j == k ) nullspaceVec_[offset] = 1.0;
               else          nullspaceVec_[offset] = 0.0;
            }
         }
         for ( j = 0; j < 3; j++ )
         { 
            for ( k = 3; k < 6; k++ )
            {
               offset = k * nullspaceLen_ + voffset + j;
               if ( j == k-3 ) nullspaceVec_[offset] = 0.0;
               else 
               {
                  if      (j+k == 4) nullspaceVec_[offset] = coords[i*3+2];
                  else if (j+k == 5) nullspaceVec_[offset] = coords[i*3+1];
                  else if (j+k == 6) nullspaceVec_[offset] = coords[i*3];
                  else nullspaceVec_[offset] = 0.0;
               }
            }
         }
         j = 0; k = 5; offset = k * nullspaceLen_ + voffset + j; 
         nullspaceVec_[offset] *= -1.0;
         j = 1; k = 3; offset = k * nullspaceLen_ + voffset + j; 
         nullspaceVec_[offset] *= -1.0;
         j = 2; k = 4; offset = k * nullspaceLen_ + voffset + j; 
         nullspaceVec_[offset] *= -1.0;
         if ( nullspaceDim_ == 9 ) 
         {
            for ( j = 0; j < 3; j++ )
            { 
               for ( k = 6; k < 9; k++ )
               {
                  offset = k * nullspaceLen_ + voffset + j;
                  if ( j == k-6 ) nullspaceVec_[offset] = 0.0;
                  else 
                  {
                     if (j+k == 7) 
                        nullspaceVec_[offset] = coords[i*3+2] * coords[i*3+2];
                     else if (j+k == 8) 
                        nullspaceVec_[offset] = coords[i*3+1] * coords[i*3+1];
                     else if (j+k == 9) 
                        nullspaceVec_[offset] = coords[i*3] * coords[i*3];
                     else nullspaceVec_[offset] = 0.0;
                  }
               }
            }
            j = 0; k = 8; offset = k * nullspaceLen_ + voffset + j; 
            nullspaceVec_[offset] *= -1.0;
            j = 1; k = 6; offset = k * nullspaceLen_ + voffset + j; 
            nullspaceVec_[offset] *= -1.0;
            j = 2; k = 7; offset = k * nullspaceLen_ + voffset + j; 
            nullspaceVec_[offset] *= -1.0;
         }
         if ( nullspaceDim_ == 12 ) 
         {
            for ( j = 0; j < 3; j++ )
            { 
               for ( k = 9; k < 12; k++ )
               {
                  offset = k * nullspaceLen_ + voffset + j;
                  if ( j == k-9 ) nullspaceVec_[offset] = 0.0;
                  else 
                  {
                     if (j+k == 10) 
                        nullspaceVec_[offset] = 
                           coords[i*3+2] * coords[i*3+2] * coords[i*3+2];
                     else if (j+k == 11) 
                        nullspaceVec_[offset] = 
                           coords[i*3+1] * coords[i*3+1] * coords[i*3+1];
                     else if (j+k == 12) 
                        nullspaceVec_[offset] = 
                           coords[i*3] * coords[i*3] * coords[i*3];
                     else nullspaceVec_[offset] = 0.0;
                  }
               }
            }
            j = 0; k = 11; offset = k * nullspaceLen_ + voffset + j; 
            nullspaceVec_[offset] *= -1.0;
            j = 1; k = 9; offset = k * nullspaceLen_ + voffset + j; 
            nullspaceVec_[offset] *= -1.0;
            j = 2; k = 10; offset = k * nullspaceLen_ + voffset + j; 
            nullspaceVec_[offset] *= -1.0;
         }
      }
   }
   if ( scalings != NULL )
   {
      for ( i = 0 ; i < nullspaceDim_; i++ ) 
         for ( j = 0 ; j < nullspaceLen_; j++ ) 
            nullspaceVec_[i*nullspaceLen_+j] *= scalings[j];
   }

   if ( printNullSpace_ == 1 )
   {
      for ( i = 0 ; i < nullspaceDim_; i++ ) 
      {
         sprintf(fname, "nullspace%d.%d", i, mypid); 
         fp = fopen( fname, "w" );
         for ( j = 0 ; j < nullspaceLen_; j++ ) 
            fprintf(fp," %25.16e\n", nullspaceVec_[i*nullspaceLen_+j]);
         fclose(fp);
      }
   }
   return 0;
}

/* ********************************************************************* *
 * set parameter for calibration AMG 
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::setCalibrationSize( int size )
{
   if ( size > 0 ) calibrationSize_ = size;
   return 0;
}

/* ********************************************************************* *
 * print AMGSA information
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::print()
{
   int      mypid;
   MPI_Comm comm = getComm();

   MPI_Comm_rank( comm, &mypid);
   if ( mypid == 0 )
   {
      printf("\t********************************************************\n");
      printf("\t*** method name             = %s\n", getName());
      printf("\t*** number of levels        = %d\n", numLevels_);
      printf("\t*** coarsen scheme          = %d\n", coarsenScheme_);
      printf("\t*** nodal degree of freedom = %d\n", nodeDofs_);
      printf("\t*** null space dimension    = %d\n", nullspaceDim_);
      printf("\t*** strength threshold      = %e\n", threshold_);
      printf("\t*** Prolongator factor      = %e\n", Pweight_);
      printf("\t*** drop tolerance for P    = %e\n", dropTolForP_);
      printf("\t*** A-norm scheme           = %d\n", calcNormScheme_);
      printf("\t*** minimum coarse size     = %d\n", minCoarseSize_);
      printf("\t*** pre  smoother type      = %s\n", preSmoother_); 
      printf("\t*** pre  smoother nsweeps   = %d\n", preSmootherNum_);
      printf("\t*** post smoother type      = %s\n", postSmoother_); 
      printf("\t*** post smoother nsweeps   = %d\n", postSmootherNum_);
      printf("\t*** coarse solver type      = %s\n", coarseSolver_); 
      printf("\t*** coarse solver nsweeps   = %d\n", coarseSolverNum_);  
      printf("\t*** calibration size        = %d\n", calibrationSize_);
      printf("\t********************************************************\n");
   }
   return 0;
}

/* ********************************************************************* *
 * print AMGSA statistics information
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::printStatistics(MLI *mli)
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
      printf("\t****************** AMGSA Statistics ********************\n");

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

/* ********************************************************************* *
 * get the null space
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::getNullSpace(int &nDOF,int &ndim,double *&nullvec,
                                   int &leng) 
{
   nDOF    = currNodeDofs_;
   ndim    = nullspaceDim_;
   nullvec = nullspaceVec_;
   leng    = nullspaceLen_;
   return 0;
}

/* ********************************************************************* *
 * clone another object
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::copy( MLI_Method *new_obj )
{
   MLI_Method_AMGSA *new_amgsa;

   if ( ! strcasecmp(new_obj->getName(), "AMGSA" ) )
   {
      new_amgsa = (MLI_Method_AMGSA *) new_obj;
      new_amgsa->maxLevels_ = maxLevels_;
      new_amgsa->setOutputLevel( outputLevel_ );
      new_amgsa->setNumLevels( numLevels_ );
      new_amgsa->setSmoother( MLI_SMOOTHER_PRE, preSmoother_, 
                              preSmootherNum_, preSmootherWgt_ );
      new_amgsa->setSmoother( MLI_SMOOTHER_POST, postSmoother_, 
                              postSmootherNum_, postSmootherWgt_ );
      new_amgsa->setCoarseSolver(coarseSolver_,coarseSolverNum_,
                                 coarseSolverWgt_ ); 
      new_amgsa->setCoarsenScheme( coarsenScheme_ );
      new_amgsa->setMinCoarseSize( minCoarseSize_ );
      if ( calcNormScheme_ ) new_amgsa->setCalcSpectralNorm();
      new_amgsa->setPweight( Pweight_ );
      new_amgsa->setNullSpace(nodeDofs_,nullspaceDim_,nullspaceVec_,
                              nullspaceLen_);
      new_amgsa->setStrengthThreshold( threshold_ );
   }
   else
   {
      printf("MLI_Method_AMGSA::copy ERROR - incoming object not AMGSA.\n");
      exit(1);
   }
   return 0;
}

/* ********************************************************************* *
 * LOBPCG subroutine calls
 * ********************************************************************* */

#ifdef HAVE_LOBPCG
#ifdef __cplusplus
extern "C" {
#endif

#include "../../../eigen/lobpcg/lobpcg.h"
#include "../../../IJ_mv/IJ_mv.h"
#include "../../../parcsr_mv/parcsr_mv.h"
#include "../../../seq_mv/seq_mv.h"
#include "../../../parcsr_ls/parcsr_ls.h"
HYPRE_Solver	   lobHYPRESolver;
HYPRE_ParCSRMatrix lobHYPREA;

int Funct_Solve(HYPRE_ParVector b,HYPRE_ParVector x)
{
   int ierr=0;
   ierr=HYPRE_ParCSRPCGSolve(lobHYPRESolver,lobHYPREA,b,x);assert2(ierr);
   return 0;
}
int Func_Matvec(HYPRE_ParVector x,HYPRE_ParVector y)
{
   int ierr=0;
   ierr=HYPRE_ParCSRMatrixMatvec(1.0,lobHYPREA,x,0.0,y);assert2(ierr);
   return 0;
}
#ifdef __cplusplus
}
#endif
#endif

/* ********************************************************************* *
 * relax null spaces 
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::relaxNullSpaces(MLI_Matrix *mli_Amat)
{
#ifdef HAVE_LOBPCG
   int                mypid, *partitioning, startRow, endRow, localNRows;
   int                iV, i, offset, *cols;
   double             *eigval, *uData;
   MPI_Comm           comm;
   HYPRE_IJVector     tempIJ;
   hypre_ParVector    **lobVecs = new hypre_ParVector*[nullspaceDim_];
   hypre_ParCSRMatrix *hypreA;
   HYPRE_Solver       HYPrecon=NULL;
   HYPRE_LobpcgData   lobpcgdata;
   int (*FuncT)(HYPRE_ParVector x,HYPRE_ParVector y);

   comm     = getComm();
   MPI_Comm_rank( comm, &mypid );
   hypreA = (hypre_ParCSRMatrix *) mli_Amat->getMatrix();
   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) hypreA, 
                                        &partitioning );
   startRow   = partitioning[mypid];
   endRow     = partitioning[mypid+1] - 1;
   localNRows = endRow - startRow + 1;
   cols       = new int[localNRows];
   for ( i = startRow; i <= endRow; i++ ) cols[i] = startRow + i;

   for ( iV = 0; iV < nullspaceDim_; iV++ )
   {
      HYPRE_IJVectorCreate(comm, startRow, endRow, &tempIJ);
      HYPRE_IJVectorSetObjectType(tempIJ, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(tempIJ);
      HYPRE_IJVectorAssemble(tempIJ);
      offset = nullspaceLen_ * iV ;
      HYPRE_IJVectorSetValues(tempIJ, localNRows, (const int *) cols,
                              (const double *) &(nullspaceVec_[offset]));
      HYPRE_IJVectorGetObject(tempIJ, (void **) &(lobVecs[iV]));
//HYPRE_ParVectorSetRandomValues( (HYPRE_ParVector) lobVecs[iV], 9001*iV*7901 );
      HYPRE_IJVectorSetObjectType(tempIJ, -1);
      HYPRE_IJVectorDestroy(tempIJ);
   }
   delete [] cols;
   free(partitioning);

   printf("LOBPCG Solve\n");
   HYPRE_LobpcgCreate(&lobpcgdata);
   HYPRE_LobpcgSetVerbose(lobpcgdata);
   HYPRE_LobpcgSetBlocksize(lobpcgdata, nullspaceDim_);
   FuncT = Funct_Solve;
   HYPRE_LobpcgSetSolverFunction(lobpcgdata,FuncT);
   HYPRE_LobpcgSetup(lobpcgdata);
   lobHYPREA      = (HYPRE_ParCSRMatrix) hypreA;
   HYPRE_ParCSRPCGCreate(comm, &lobHYPRESolver);
   HYPRE_ParCSRPCGSetMaxIter(lobHYPRESolver, 10);
   HYPRE_ParCSRPCGSetTol(lobHYPRESolver, 1.0e-1);
   HYPRE_ParCSRPCGSetup(lobHYPRESolver, lobHYPREA, 
          (HYPRE_ParVector) lobVecs[0], (HYPRE_ParVector) lobVecs[1]);
   HYPRE_ParCSRPCGSetPrecond(lobHYPRESolver, HYPRE_ParCSRDiagScale,
                             HYPRE_ParCSRDiagScaleSetup, HYPrecon);
   HYPRE_LobpcgSetTolerance(lobpcgdata, 1.0e-1);

   HYPRE_LobpcgSolve(lobpcgdata,Func_Matvec,(HYPRE_ParVector*)lobVecs,&eigval);
   for ( iV = 0; iV < nullspaceDim_; iV++ )
   {
      uData = hypre_VectorData(
                 hypre_ParVectorLocalVector((hypre_ParVector *)lobVecs[iV]));
      offset = nullspaceLen_ * iV;
      for ( i = 0; i < nullspaceLen_; i++ )
         nullspaceVec_[offset+i] = uData[i];
      hypre_ParVectorDestroy(lobVecs[iV]);
   }
   HYPRE_LobpcgDestroy(lobpcgdata);
#endif
   return 0;
}

