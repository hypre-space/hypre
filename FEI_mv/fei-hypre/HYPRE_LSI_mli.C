/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/
/****************************************************************************/ 
/* HYPRE_LSI_MLI interface                                                  */
/*--------------------------------------------------------------------------*/
/*  local functions :
 * 
 *        HYPRE_LSI_MLICreate
 *        HYPRE_LSI_MLIDestroy
 *        HYPRE_LSI_MLISetup
 *        HYPRE_LSI_MLISolve
 *        HYPRE_LSI_MLISetParams
 *--------------------------------------------------------------------------
 *        HYPRE_LSI_MLICreateNodeEqnMap
 *        HYPRE_LSI_MLIAdjustNodeEqnMap
 *        HYPRE_LSI_MLISetFEData
 *        HYPRE_LSI_MLISetStrengthThreshold
 *        HYPRE_LSI_MLISetMethod
 *        HYPRE_LSI_MLISetSmoother
 *        HYPRE_LSI_MLISetCoarseSolver
 *        HYPRE_LSI_MLISetNodalCoordinates
 *--------------------------------------------------------------------------
 *        HYPRE_LSI_MLIFEDataCreate
 *        HYPRE_LSI_MLIFEDataDestroy
 *        HYPRE_LSI_MLIFEDataInitFields
 *        HYPRE_LSI_MLIFEDataInitElemBlock
 *        HYPRE_LSI_MLIFEDataInitElemNodeList
 *        HYPRE_LSI_MLIFEDataInitSharedNodes
 *        HYPRE_LSI_MLIFEDataInitComplete
 *        HYPRE_LSI_MLIFEDataLoadElemMatrix
 *        HYPRE_LSI_MLIFEDataWriteToFile
 ****************************************************************************/

/****************************************************************************/ 
/* system include files                                                     */
/*--------------------------------------------------------------------------*/

#include <string.h>
#include <iostream.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

/****************************************************************************/ 
/* MLI include files                                                        */
/*--------------------------------------------------------------------------*/

#ifdef HAVE_MLI
#include "base/mli_defs.h"
#include "base/mli.h"
#include "util/mli_utils.h"
#else
#define MLI_SOLVER_JACOBI_ID    0 
#define MLI_SOLVER_GS_ID        1 
#define MLI_SOLVER_SGS_ID       2 
#define MLI_SOLVER_PARASAILS_ID 3 
#define MLI_SOLVER_SCHWARZ_ID   4 
#define MLI_SOLVER_MLS_ID       5 
#define MLI_SOLVER_SUPERLU_ID   6 
#define MLI_METHOD_AMGSA_ID     7
#define MLI_METHOD_AMGSAE_ID    8
#endif
#ifdef NOFEI
#define GlobalID int
#endif
#include "HYPRE_LSI_mli.h"

/****************************************************************************/ 
/* HYPRE_LSI_MLI data structure                                             */
/*--------------------------------------------------------------------------*/

typedef struct HYPRE_LSI_MLI_Struct
{
#ifdef HAVE_MLI
   MLI        *mli_;
   MLI_FEData *feData_;           /* holds FE information */
   MLI_Mapper *mapper_;           /* holds mapping information */
#endif
   MPI_Comm mpiComm_;
   int      outputLevel_;         /* for diagnostics */
   int      nLevels_;             /* max number of levels */
   int      cycleType_;           /* 1 for V and 2 for W */
   int      maxIterations_;       /* default - 1 iteration */
   int      method_;              /* default - smoothed aggregation */
   int      preSmoother_;         /* default - symmetric Gauss Seidel */
   int      postSmoother_ ;       /* default - symmetric Gauss Seidel */
   int      preNSweeps_;          /* default - 2 smoothing steps */
   int      postNSweeps_;         /* default - 2 smoothing steps */
   double   *preSmootherWts_;     /* relaxation weights */
   double   *postSmootherWts_;    /* relaxation weights */
   double   strengthThreshold_;   /* strength threshold */
   int      coarseSolver_;        /* default = SuperLU */
   int      coarseSolverNSweeps_; /* number of sweeps (if iterative used) */
   double   *coarseSolverWts_;    /* relaxation weight (if Jacobi used) */
   int      minCoarseSize_;       /* minimum coarse grid size */
   int      nodeDOF_;             /* node degree of freedom */
   int      nSpaceDim_;           /* number of null vectors */
   double   *nSpaceVects_;        /* holder for null space information */
   int      localNEqns_;          /* number of equations locally */
   int      nCoordAccept_;        /* flag to accept nodal coordinate or not */ 
   double   *nCoordinates_;       /* for storing nodal coordinates */
   double   *nullScales_;         /* scaling vector for null space */
   int      calibrationSize_;     /* for calibration smoothed aggregation */
   double   Pweight_;
} 
HYPRE_LSI_MLI;

/****************************************************************************/ 
/* HYPRE_LSI_MLI data structure                                             */
/*--------------------------------------------------------------------------*/

typedef struct HYPRE_MLI_FEData_Struct
{
#ifdef HAVE_MLI
   MPI_Comm   comm_;              /* MPI communicator */
   MLI_FEData *fedata_;           /* holds FE information */
   int        fedataOwn_;         /* flag to indicate ownership */
   int        computeNull_;       /* flag - compute null space or not */
   int        nullDim_;           /* number of null space vectors */
#endif
}
HYPRE_MLI_FEData;

/****************************************************************************/
/* HYPRE_LSI_MLICreate                                                      */
/*--------------------------------------------------------------------------*/

extern "C"
int HYPRE_LSI_MLICreate( MPI_Comm comm, HYPRE_Solver *solver )
{
   HYPRE_LSI_MLI *mli_object = (HYPRE_LSI_MLI *) malloc(sizeof(HYPRE_LSI_MLI));
   *solver = (HYPRE_Solver) mli_object;
   mli_object->mpiComm_             = comm;
   mli_object->outputLevel_         = 0;
   mli_object->nLevels_             = 30;
   mli_object->maxIterations_       = 1;
   mli_object->cycleType_           = 1;
   mli_object->method_              = MLI_METHOD_AMGSA_ID;
   mli_object->preSmoother_         = MLI_SOLVER_SGS_ID;
   mli_object->postSmoother_        = MLI_SOLVER_SGS_ID;
   mli_object->preNSweeps_          = 2;
   mli_object->postNSweeps_         = 2;
   mli_object->preSmootherWts_      = NULL;
   mli_object->postSmootherWts_     = NULL;
   mli_object->strengthThreshold_   = 0.08;
   mli_object->coarseSolver_        = MLI_SOLVER_SUPERLU_ID;
   mli_object->coarseSolverNSweeps_ = 0;
   mli_object->coarseSolverWts_     = NULL;
   mli_object->minCoarseSize_       = 20;
   mli_object->nodeDOF_             = 1;
   mli_object->nSpaceDim_           = 1;
   mli_object->nSpaceVects_         = NULL;
   mli_object->localNEqns_          = 0;
   mli_object->nCoordinates_        = NULL;
   mli_object->nCoordAccept_        = 0;
   mli_object->nullScales_          = NULL;
   mli_object->calibrationSize_     = 0;
   mli_object->Pweight_             = 1.333;
#ifdef HAVE_MLI
   mli_object->mli_                 = NULL;
   mli_object->feData_              = NULL;
   mli_object->mapper_              = NULL;
   return 0;
#else
   printf("MLI not available.\n");
   exit(1);
   return -1;
#endif
}

/****************************************************************************/
/* HYPRE_LSI_MLIDestroy                                                     */
/*--------------------------------------------------------------------------*/

extern "C"
int HYPRE_LSI_MLIDestroy( HYPRE_Solver solver )
{
   HYPRE_LSI_MLI *mli_object = (HYPRE_LSI_MLI *) solver;

   if ( mli_object->preSmootherWts_ != NULL ) 
      delete [] mli_object->preSmootherWts_;
   if ( mli_object->postSmootherWts_ != NULL ) 
      delete [] mli_object->postSmootherWts_;
   if ( mli_object->coarseSolverWts_ != NULL ) 
      delete [] mli_object->coarseSolverWts_;
   if ( mli_object->nSpaceVects_ != NULL ) 
      delete [] mli_object->nSpaceVects_;
   if ( mli_object->nCoordinates_ != NULL ) 
      delete [] mli_object->nCoordinates_;
   if ( mli_object->nullScales_ != NULL ) 
      delete [] mli_object->nullScales_;
#ifdef HAVE_MLI
   if ( mli_object->feData_ != NULL ) delete mli_object->feData_;
   if ( mli_object->mli_ != NULL ) delete mli_object->mli_;
   free( mli_object );
   return 0;
#else
   free( mli_object );
   printf("MLI not available.\n");
   return -1;
#endif
}

/****************************************************************************/
/* HYPRE_LSI_MLISetup                                                       */
/*--------------------------------------------------------------------------*/

extern "C"
int HYPRE_LSI_MLISetup( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                        HYPRE_ParVector b,   HYPRE_ParVector x )
{
#ifdef HAVE_MLI
   int           targc, nNodes;
   double        tol=1.0e-8;
   char          *targv[4], paramString[100];;
   HYPRE_LSI_MLI *mli_object;
   MLI_Matrix    *mli_mat;   
   MLI_Method    *method;   
   MPI_Comm      mpiComm;
   MLI           *mli;

   /* -------------------------------------------------------- */ 
   /* create object                                            */
   /* -------------------------------------------------------- */ 

   mli_object = (HYPRE_LSI_MLI *) solver;
   mpiComm    = mli_object->mpiComm_;
   mli        = new MLI( mpiComm );
   if ( mli_object->mli_ != NULL ) delete mli_object->mli_;
   mli_object->mli_ = mli;

   /* -------------------------------------------------------- */ 
   /* set general parameters                                   */
   /* -------------------------------------------------------- */ 

   mli->setNumLevels( mli_object->nLevels_ );
   mli->setTolerance( tol );

   /* -------------------------------------------------------- */ 
   /* set method specific parameters                           */
   /* -------------------------------------------------------- */ 

   switch ( mli_object->method_ )
   {
      case MLI_METHOD_AMGSA_ID : 
           method = MLI_Method_CreateFromID(mli_object->method_,mpiComm);
           break;
      case MLI_METHOD_AMGSAE_ID : 
           method = MLI_Method_CreateFromID(mli_object->method_,mpiComm);
           break;
      default :
           printf("HYPRE_LSI_MLISetup : method not valid.\n");
           exit(1);
   }
   sprintf(paramString, "setNumLevels %d", mli_object->nLevels_);
   method->setParams( paramString, 0, NULL );
   sprintf(paramString, "setStrengthThreshold %f",
           mli_object->strengthThreshold_);
   method->setParams( paramString, 0, NULL );

   /* -------------------------------------------------------- */ 
   /* set up presmoother                                       */
   /* -------------------------------------------------------- */ 

   switch ( mli_object->preSmoother_ )
   {
      case MLI_SOLVER_JACOBI_ID    : 
           sprintf(paramString, "setPreSmoother Jacobi" ); break;
      case MLI_SOLVER_GS_ID        : 
           sprintf(paramString, "setPreSmoother GS" ); break;
      case MLI_SOLVER_SGS_ID       : 
           sprintf(paramString, "setPreSmoother SGS" ); break;
      case MLI_SOLVER_PARASAILS_ID : 
           sprintf(paramString, "setPreSmoother ParaSails" ); break;
      case MLI_SOLVER_SCHWARZ_ID   : 
           sprintf(paramString, "setPreSmoother Schwarz" ); break;
      case MLI_SOLVER_MLS_ID       : 
           sprintf(paramString, "setPreSmoother MLS" ); break;
   }
   targc    = 2;
   targv[0] = (char *) &mli_object->preNSweeps_;
   targv[1] = (char *) mli_object->preSmootherWts_;
   method->setParams( paramString, targc, targv );

   /* -------------------------------------------------------- */ 
   /* set up postsmoother                                      */
   /* -------------------------------------------------------- */ 

   switch ( mli_object->postSmoother_ )
   {
      case MLI_SOLVER_JACOBI_ID    : 
           sprintf(paramString, "setPostSmoother Jacobi" ); break;
      case MLI_SOLVER_GS_ID        : 
           sprintf(paramString, "setPostSmoother GS" ); break;
      case MLI_SOLVER_SGS_ID       : 
           sprintf(paramString, "setPostSmoother SGS" ); break;
      case MLI_SOLVER_PARASAILS_ID : 
           sprintf(paramString, "setPostSmoother ParaSails" ); break;
      case MLI_SOLVER_SCHWARZ_ID   : 
           sprintf(paramString, "setPostSmoother Schwarz" ); break;
      case MLI_SOLVER_MLS_ID       : 
           sprintf(paramString, "setPostSmoother MLS" ); break;
   }
   targc    = 2;
   targv[0] = (char *) &mli_object->postNSweeps_;
   targv[1] = (char *) mli_object->postSmootherWts_;
   method->setParams( paramString, targc, targv );

   /* -------------------------------------------------------- */ 
   /* set up coarse solver                                     */
   /* -------------------------------------------------------- */ 

   switch ( mli_object->coarseSolver_ )
   {
      case MLI_SOLVER_JACOBI_ID    : 
           sprintf(paramString, "setCoarseSolver Jacobi" ); break;
      case MLI_SOLVER_GS_ID        : 
           sprintf(paramString, "setCoarseSolver GS" ); break;
      case MLI_SOLVER_SGS_ID       : 
           sprintf(paramString, "setCoarseSolver SGS" ); break;
      case MLI_SOLVER_PARASAILS_ID : 
           sprintf(paramString, "setCoarseSolver ParaSails" ); break;
      case MLI_SOLVER_SCHWARZ_ID   : 
           sprintf(paramString, "setCoarseSolver Schwarz" ); break;
      case MLI_SOLVER_MLS_ID       : 
           sprintf(paramString, "setCoarseSolver MLS" ); break;
      case MLI_SOLVER_SUPERLU_ID       : 
           sprintf(paramString, "setCoarseSolver SuperLU" ); break;
   }
   targc    = 2;
   targv[0] = (char *) &(mli_object->coarseSolverNSweeps_);
   targv[1] = (char *) mli_object->coarseSolverWts_;
   method->setParams( paramString, targc, targv );

   /* -------------------------------------------------------- */ 
   /* load minimum coarse grid size                            */
   /* -------------------------------------------------------- */ 

   sprintf( paramString, "setMinCoarseSize %d", mli_object->minCoarseSize_ );
   method->setParams( paramString, 0, NULL );
   sprintf( paramString, "setPweight %e", mli_object->Pweight_ );
   method->setParams( paramString, 0, NULL );

   /* -------------------------------------------------------- */ 
   /* load calibration size                                    */
   /* -------------------------------------------------------- */ 

   sprintf(paramString, "setCalibrationSize %d", mli_object->calibrationSize_);
   method->setParams( paramString, 0, NULL );

   /* -------------------------------------------------------- */ 
   /* load FEData, if there is any                             */
   /* -------------------------------------------------------- */ 

   mli->setFEData( 0, mli_object->feData_, mli_object->mapper_ );
   mli_object->mapper_ = NULL;
   mli_object->feData_ = NULL;

   /* -------------------------------------------------------- */ 
   /* load null space, if there is any                         */
   /* -------------------------------------------------------- */ 

   if ( mli_object->nCoordinates_ != NULL )
   {
      nNodes = mli_object->localNEqns_ / mli_object->nodeDOF_;
      targv[0] = (char *) &nNodes;
      targv[1] = (char *) &(mli_object->nodeDOF_);
      targv[2] = (char *) mli_object->nCoordinates_;
      targv[3] = (char *) mli_object->nullScales_;
      targc    = 4;
      strcpy( paramString, "setNodalCoord" );
      method->setParams( paramString, targc, targv );
   }
   else 
   {
      targv[0] = (char *) &(mli_object->nodeDOF_);
      targv[1] = (char *) &(mli_object->nSpaceDim_);
      targv[2] = (char *) mli_object->nSpaceVects_;
      targv[3] = (char *) &(mli_object->localNEqns_);
      targc    = 4;
      strcpy( paramString, "setNullSpace" );
      method->setParams( paramString, targc, targv );
   }

   /* -------------------------------------------------------- */ 
   /* finally, set up                                          */
   /* -------------------------------------------------------- */ 

   mli_mat = new MLI_Matrix((void*) A, "HYPRE_ParCSR", NULL);
   mli->setMethod( method );
   mli->setSystemMatrix( 0, mli_mat );
   mli->setOutputLevel( mli_object->outputLevel_ );
   mli->setup();
   mli->setMaxIterations( mli_object->maxIterations_ );
   mli->setCyclesAtLevel( -1, mli_object->cycleType_ );
   if ( mli_object->outputLevel_ >= 1 )
   {
      strcpy( paramString, "print" );
      method->setParams( paramString, 0, NULL );
   }
   return 0;
#else
   printf("MLI not available.\n");
   return -1;
#endif
}

/****************************************************************************/
/* HYPRE_LSI_MLISolve                                                       */
/*--------------------------------------------------------------------------*/

extern "C"
int HYPRE_LSI_MLISolve( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                        HYPRE_ParVector b, HYPRE_ParVector x )
{
#ifdef HAVE_MLI
   HYPRE_LSI_MLI *mli_object;
   MLI_Vector    *sol, *rhs;

   sol = new MLI_Vector( (void *) x, "HYPRE_ParVector", NULL);
   rhs = new MLI_Vector( (void *) b, "HYPRE_ParVector", NULL);

   mli_object = (HYPRE_LSI_MLI *) solver;
   if ( mli_object->mli_ == NULL )
   {
      printf("HYPRE_LSI_MLISolve ERROR : mli not instantiated.\n");
      exit(1);
   }
   mli_object->mli_->solve( sol, rhs);

   return 0;
#else
   printf("MLI not available.\n");
   return -1;
#endif
}

/****************************************************************************/
/* HYPRE_LSI_MLISetParams                                                   */
/*--------------------------------------------------------------------------*/

extern "C"
int HYPRE_LSI_MLISetParams( HYPRE_Solver solver, char *paramString )
{
   int           mypid;
   MPI_Comm      mpiComm;
   HYPRE_LSI_MLI *mli_object;
   char          param1[256], param2[256], param3[256];

   mli_object = (HYPRE_LSI_MLI *) solver;
   sscanf(paramString,"%s", param1);
   if ( strcmp(param1, "MLI") )
   {
      printf("HYPRE_LSI_MLI::parameters not for me.\n");
      return 1;
   }
   MPI_Comm_rank( mli_object->mpiComm_, &mypid );
   sscanf(paramString,"%s %s", param1, param2);
   if ( !strcmp(param2, "help") )
   {
      if ( mypid == 0 )
      {
         printf("%4d : Available options for MLI are : \n", mypid);
         printf("\t      outputLevel <d> \n");
         printf("\t      maxIterations <d> \n");
         printf("\t      cycleType <'V','W'> \n");
         printf("\t      strengthThreshold <f> \n");
         printf("\t      method <AMGSA, AMGSAe> \n");
         printf("\t      smoother <Jacobi,GS,...> \n");
         printf("\t      coarseSolver <Jacobi,GS,...> \n");
         printf("\t      numSweeps <d> \n");
         printf("\t      minCoarseSize <d> \n");
         printf("\t      Pweight <f> \n");
         printf("\t      nodeDOF <d> \n");
         printf("\t      nullSpaceDim <d> \n");
         printf("\t      useNodalCoord <on,off> \n");
      }
   }
   else if ( !strcmp(param2, "outputLevel") )
   {
      sscanf(paramString,"%s %s %d",param1,param2,&(mli_object->outputLevel_));
   }
   else if ( !strcmp(param2, "numLevels") )
   {
      sscanf(paramString,"%s %s %d", param1, param2,
             &(mli_object->nLevels_));
      if ( mli_object->nLevels_ <= 0 ) mli_object->nLevels_ = 1;
   }
   else if ( !strcmp(param2, "maxIterations") )
   {
      sscanf(paramString,"%s %s %d", param1, param2,
             &(mli_object->maxIterations_));
      if ( mli_object->maxIterations_ <= 0 ) mli_object->maxIterations_ = 1;
   }
   else if ( !strcmp(param2, "cycleType") )
   {
      sscanf(paramString,"%s %s %s", param1, param2, param3);
      if ( ! strcmp( param3, "V" ) )      mli_object->cycleType_ = 1;
      else if ( ! strcmp( param3, "W" ) ) mli_object->cycleType_ = 2;
   }
   else if ( !strcmp(param2, "strengthThreshold") )
   {
      sscanf(paramString,"%s %s %lg", param1, param2,
             &(mli_object->strengthThreshold_));
      if ( mli_object->strengthThreshold_ < 0.0 )
         mli_object->strengthThreshold_ = 0.0;
   }
   else if ( !strcmp(param2, "method") )
   {
      sscanf(paramString,"%s %s %s", param1, param2, param3);
      if ( ! strcmp( param3, "AMGSA" ) )
         mli_object->method_ = MLI_METHOD_AMGSA_ID;
      else if ( ! strcmp( param3, "AMGSAe" ) )
         mli_object->method_ = MLI_METHOD_AMGSAE_ID;
   }
   else if ( !strcmp(param2, "smoother") )
   {
      sscanf(paramString,"%s %s %s", param1, param2, param3);
      if ( ! strcmp( param3, "Jacobi" ) )
      {
         mli_object->preSmoother_  = MLI_SOLVER_JACOBI_ID;
         mli_object->postSmoother_ = MLI_SOLVER_JACOBI_ID;
      }
      else if ( ! strcmp( param3, "GS" ) )
      {
         mli_object->preSmoother_  = MLI_SOLVER_GS_ID;
         mli_object->postSmoother_ = MLI_SOLVER_GS_ID;
      }
      else if ( ! strcmp( param3, "SGS" ) )
      {
         mli_object->preSmoother_  = MLI_SOLVER_SGS_ID;
         mli_object->postSmoother_ = MLI_SOLVER_SGS_ID;
      }
      else if ( ! strcmp( param3, "ParaSails" ) )
      {
         mli_object->preSmoother_  = MLI_SOLVER_PARASAILS_ID;
         mli_object->postSmoother_ = MLI_SOLVER_PARASAILS_ID;
      }
      else if ( ! strcmp( param3, "Schwarz" ) )
      {
         mli_object->preSmoother_  = MLI_SOLVER_SCHWARZ_ID;
         mli_object->postSmoother_ = MLI_SOLVER_SCHWARZ_ID;
      }
      else if ( ! strcmp( param3, "MLS" ) )
      {
         mli_object->preSmoother_  = MLI_SOLVER_MLS_ID;
         mli_object->postSmoother_ = MLI_SOLVER_MLS_ID;
      }
      else 
      {
         printf("HYPRE_LSI_MLISetParams ERROR : unrecognized smoother.\n");
         exit(1);
      }
   }
   else if ( !strcmp(param2, "coarseSolver") )
   {
      sscanf(paramString,"%s %s %s", param1, param2, param3);
      if ( ! strcmp( param3, "Jacobi" ) )
         mli_object->coarseSolver_ = MLI_SOLVER_JACOBI_ID;
      else if ( ! strcmp( param3, "GS" ) )
         mli_object->coarseSolver_ = MLI_SOLVER_GS_ID;
      else if ( ! strcmp( param3, "SGS" ) )
         mli_object->coarseSolver_ = MLI_SOLVER_SGS_ID;
      else if ( ! strcmp( param3, "ParaSails" ) )
         mli_object->coarseSolver_ = MLI_SOLVER_PARASAILS_ID;
      else if ( ! strcmp( param3, "Schwarz" ) )
         mli_object->coarseSolver_ = MLI_SOLVER_SCHWARZ_ID;
      else if ( ! strcmp( param3, "MLS" ) )
         mli_object->coarseSolver_ = MLI_SOLVER_MLS_ID;
      else 
      {
         printf("HYPRE_LSI_MLISetParams ERROR : unrecognized coarseSolver.\n");
         exit(1);
      }
   }
   else if ( !strcmp(param2, "numSweeps") )
   {
      sscanf(paramString,"%s %s %d",param1,param2,&(mli_object->preNSweeps_));
      if ( mli_object->preNSweeps_ <= 0 ) mli_object->preNSweeps_ = 1;
      mli_object->postNSweeps_ = mli_object->preNSweeps_; 
   }
   else if ( !strcmp(param2, "minCoarseSize") )
   {
      sscanf(paramString,"%s %s %d",param1,param2,
             &(mli_object->minCoarseSize_));
      if ( mli_object->minCoarseSize_ <= 0 ) mli_object->minCoarseSize_ = 20;
   }
   else if ( !strcmp(param2, "Pweight") )
   {
      sscanf(paramString,"%s %s %lg",param1,param2,
             &(mli_object->Pweight_));
      if ( mli_object->Pweight_ < 0. ) mli_object->Pweight_ = 1.333;
   }
   else if ( !strcmp(param2, "nodeDOF") )
   {
      sscanf(paramString,"%s %s %d",param1,param2,
             &(mli_object->nodeDOF_));
      if ( mli_object->nodeDOF_ <= 0 ) mli_object->nodeDOF_ = 1;
   }
   else if ( !strcmp(param2, "nullSpaceDim") )
   {
      sscanf(paramString,"%s %s %d",param1,param2,
             &(mli_object->nSpaceDim_));
      if ( mli_object->nSpaceDim_ <= 0 ) mli_object->nSpaceDim_ = 1;
   }
   else if ( !strcmp(param2, "useNodalCoord") )
   {
      sscanf(paramString,"%s %s %s",param1,param2,param3);
      if ( !strcmp(param3, "on") ) mli_object->nCoordAccept_ = 1;
      else                         mli_object->nCoordAccept_ = 0;
   }
   else if ( !strcmp(param2, "saAMGCalibrationSize") )
   {
      sscanf(paramString,"%s %s %d",param1,param2,
             &(mli_object->calibrationSize_));
      if ( mli_object->calibrationSize_ < 0 ) 
         mli_object->calibrationSize_ = 0; 
   }
   else 
   {
      if ( mypid == 0 )
      {
         printf("%4d : HYPRE_LSI_MLISetParams ERROR : unrecognized request.\n",
                mypid);
         printf("\t    offending request = %s.\n", paramString);
         printf("\tAvailable options for MLI are : \n");
         printf("\t      outputLevel <d> \n");
         printf("\t      maxIterations <d> \n");
         printf("\t      cycleType <'V','W'> \n");
         printf("\t      strengthThreshold <f> \n");
         printf("\t      method <AMGSA, AMGSAe> \n");
         printf("\t      smoother <Jacobi,GS,...> \n");
         printf("\t      coarseSolver <Jacobi,GS,...> \n");
         printf("\t      numSweeps <d> \n");
         printf("\t      minCoarseSize <d> \n");
         printf("\t      nodeDOF <d> \n");
         printf("\t      nullSpaceDim <d> \n");
         printf("\t      useNodalCoord <on,off> \n");
         printf("\t      saAMGCalibrationSize <d> \n");
         exit(1);
      }
   }
   return 0;
}

/****************************************************************************/
/* HYPRE_LSI_MLICreateNodeEqnMap                                            */
/*--------------------------------------------------------------------------*/

extern "C"
int HYPRE_LSI_MLICreateNodeEqnMap(HYPRE_Solver solver, int nNodes, 
                                  int *nodeNumbers, int *eqnNumbers,
                                  int *procNRows)
{
#ifdef HAVE_MLI
   int           iN, iP, mypid, nprocs, *procMapArray, *iTempArray;
   int           iS, nSends, *sendLengs, *sendProcs, **iSendBufs;
   int           iR, nRecvs, *recvLengs, *recvProcs, **iRecvBufs, *procList;
   int           newNumNodes, *newNodeNumbers, *newEqnNumbers, procIndex;
   MPI_Comm      mpiComm;
   MPI_Request   *mpiRequests;
   MPI_Status    mpiStatus;
   MLI_Mapper    *mapper;
   HYPRE_LSI_MLI *mli_object = (HYPRE_LSI_MLI *) solver;

   /* -------------------------------------------------------- */ 
   /* fetch processor information                              */
   /* -------------------------------------------------------- */ 

   if ( mli_object == NULL ) return 1;
   if ( mli_object->mapper_ != NULL ) delete mli_object->mapper_;
   mpiComm = mli_object->mpiComm_;
   MPI_Comm_rank( mpiComm, &mypid );
   MPI_Comm_size( mpiComm, &nprocs );

   /* -------------------------------------------------------- */ 
   /* construct node to processor map array                    */
   /* -------------------------------------------------------- */ 

   procMapArray = new int[nNodes]; 
   for ( iN = 0; iN < nNodes; iN++ )
   {
      procMapArray[iN] = -1;
      if ( eqnNumbers[iN] < procNRows[mypid] || 
           eqnNumbers[iN] >= procNRows[mypid+1] )
      {
         for ( iP = 0; iP < nprocs; iP++ )
            if ( eqnNumbers[iN] < procNRows[iP] ) break;
         procMapArray[iN] = iP - 1;
      }
   } 

   /* -------------------------------------------------------- */ 
   /* construct send information                               */
   /* -------------------------------------------------------- */ 

   procList = new int[nprocs];
   for ( iP = 0; iP < nprocs; iP++ ) procList[iP] = 0;
   for ( iN = 0; iN < nNodes; iN++ )
      if ( procMapArray[iN] >= 0 ) procList[procMapArray[iN]]++;
   nSends = 0;
   for ( iP = 0; iP < nprocs; iP++ ) if ( procList[iP] > 0 ) nSends++;
   if ( nSends > 0 )
   {
      sendProcs = new int[nSends]; 
      sendLengs = new int[nSends]; 
      iSendBufs = new int*[nSends]; 
   }
   nSends = 0;
   for ( iP = 0; iP < nprocs; iP++ ) 
   {
      if ( procList[iP] > 0 ) 
      {
         sendLengs[nSends] = procList[iP];
         sendProcs[nSends++] = iP;
      }
   }

   /* -------------------------------------------------------- */ 
   /* construct recv information                               */
   /* -------------------------------------------------------- */ 

   for ( iP = 0; iP < nprocs; iP++ ) procList[iP] = 0;
   for ( iP = 0; iP < nSends; iP++ ) procList[sendProcs[iP]]++; 
   iTempArray = new int[nprocs]; 
   MPI_Allreduce(procList,iTempArray,nprocs,MPI_INT,MPI_SUM,mpiComm);
   nRecvs = iTempArray[mypid];
   delete [] procList;
   delete [] iTempArray;
   if ( nRecvs > 0 )
   {
      recvLengs = new int[nRecvs];
      recvProcs = new int[nRecvs];
      iRecvBufs = new int*[nRecvs];
      mpiRequests = new MPI_Request[nRecvs];
   }
   for ( iP = 0; iP < nRecvs; iP++ ) 
      MPI_Irecv(&(recvLengs[iP]), 1, MPI_INT, MPI_ANY_SOURCE, 29421, 
                mpiComm, &(mpiRequests[iP]));
   for ( iP = 0; iP < nSends; iP++ ) 
      MPI_Send(&(sendLengs[iP]), 1, MPI_INT, sendProcs[iP], 29421, mpiComm);
   for ( iP = 0; iP < nRecvs; iP++ ) 
   {
      MPI_Wait(&(mpiRequests[iP]), &mpiStatus);
      recvProcs[iP] = mpiStatus.MPI_SOURCE;
   }

   /* -------------------------------------------------------- */ 
   /* communicate node and equation information                */
   /* -------------------------------------------------------- */ 

   for ( iP = 0; iP < nRecvs; iP++ ) 
   {
      iRecvBufs[iP] = new int[recvLengs[iP]*2];
      MPI_Irecv(iRecvBufs[iP], recvLengs[iP]*2, MPI_INT, recvProcs[iP], 
                29422, mpiComm, &(mpiRequests[iP]));
   }
   for ( iP = 0; iP < nSends; iP++ ) 
   {
      iSendBufs[iP] = new int[sendLengs[iP]*2];
      sendLengs[iP] = 0;
   }
   for ( iN = 0; iN < nNodes; iN++ )
   {
      if ( procMapArray[iN] >= 0 ) 
      {
         procIndex = procMapArray[iN];
         iSendBufs[procIndex][sendLengs[procIndex]++] = nodeNumbers[iN];
         iSendBufs[procIndex][sendLengs[procIndex]++] = eqnNumbers[iN];
      }
   }
   for ( iP = 0; iP < nSends; iP++ ) 
   {
      sendLengs[iP] /= 2;
      MPI_Send(iSendBufs[iP], sendLengs[iP]*2, MPI_INT, sendProcs[iP], 
               29422, mpiComm);
   }
   for ( iP = 0; iP < nRecvs; iP++ ) MPI_Wait(&(mpiRequests[iP]), &mpiStatus);

   /* -------------------------------------------------------- */ 
   /* form node and equation map                               */
   /* -------------------------------------------------------- */ 

   newNumNodes = nNodes;
   for ( iP = 0; iP < nRecvs; iP++ ) newNumNodes += recvLengs[iP];
   newNodeNumbers = new int[newNumNodes];
   newEqnNumbers  = new int[newNumNodes];
   newNumNodes = 0;
   for (iN = 0; iN < nNodes; iN++) 
   {
      newNodeNumbers[newNumNodes]  = nodeNumbers[iN];
      newEqnNumbers[newNumNodes++] = eqnNumbers[iN];
   }
   for ( iP = 0; iP < nRecvs; iP++ ) 
   {
      for ( iR = 0; iR < recvLengs[iP]; iR++ ) 
      {
         newNodeNumbers[newNumNodes]  = iRecvBufs[iP][iR*2];
         newEqnNumbers[newNumNodes++] = iRecvBufs[iP][iR*2+1];
      }
   }
   mapper = new MLI_Mapper();
   mapper->setMap( newNumNodes, newNodeNumbers, newEqnNumbers );
   mli_object->mapper_ = mapper;

   /* -------------------------------------------------------- */ 
   /* clean up and return                                      */
   /* -------------------------------------------------------- */ 

   delete [] procMapArray; 
   if ( nSends > 0 )
   {
      delete [] sendProcs;
      delete [] sendLengs;
      for ( iS = 0; iS < nSends; iS++ ) delete [] iSendBufs[iS];
      delete [] iSendBufs;
   } 
   if ( nRecvs > 0 )
   {
      delete [] recvProcs;
      delete [] recvLengs;
      for ( iR = 0; iR < nRecvs; iR++ ) delete [] iRecvBufs[iR];
      delete [] iRecvBufs;
      delete [] mpiRequests;
   } 
   delete [] newNodeNumbers;
   delete [] newEqnNumbers;
   return 0;
#else
   (void) solver;
   (void) nNodes;
   (void) nodeNumbers;
   (void) eqnNumbers;
   (void) procNRows;
   return 1;
#endif
}

/****************************************************************************/
/* HYPRE_LSI_MLIAdjustNodeEqnMap                                            */
/*--------------------------------------------------------------------------*/

extern "C"
int HYPRE_LSI_MLIAdjustNodeEqnMap(HYPRE_Solver solver, int *procNRows,
                                  int *procOffsets)
{
#ifdef HAVE_MLI
   HYPRE_LSI_MLI *mli_object = (HYPRE_LSI_MLI *) solver;
   if ( mli_object == NULL ) return 1;
   if ( mli_object->mapper_ == NULL ) return 1;
   mli_object->mapper_->adjustMapOffset( mli_object->mpiComm_, procNRows, 
                                         procOffsets );
   return 0;
#else
   (void) solver;
   (void) procNRows;
   (void) procOffsets;
   return 1;
#endif
}

/****************************************************************************/
/* HYPRE_LSI_MLISetFEData                                                   */
/*--------------------------------------------------------------------------*/

extern "C"
int HYPRE_LSI_MLISetFEData(HYPRE_Solver solver, void *object)
{
#ifdef HAVE_MLI
   HYPRE_LSI_MLI *mli_object = (HYPRE_LSI_MLI *) solver;
   HYPRE_MLI_FEData *hypre_fedata = (HYPRE_MLI_FEData *) object;
   mli_object->feData_      = hypre_fedata->fedata_; 
   hypre_fedata->fedata_    = NULL; 
   hypre_fedata->fedataOwn_ = 0;
   return 0;
#else
   (void) solver;
   (void) object;
   return 1;
#endif
}

/****************************************************************************/
/* HYPRE_LSI_MLISetStrengthThreshold                                        */
/*--------------------------------------------------------------------------*/

extern "C"
int HYPRE_LSI_MLISetStrengthThreshold(HYPRE_Solver solver,
                                      double strengthThreshold)
{
   HYPRE_LSI_MLI *mli_object = (HYPRE_LSI_MLI *) solver;
  
   if ( strengthThreshold < 0.0 )
   {
      printf("HYPRE_LSI_MLISetStrengthThreshold ERROR : reset to 0.\n");
      mli_object->strengthThreshold_ = 0.0;
   } 
   else mli_object->strengthThreshold_ = strengthThreshold;
   return 0;
}

/****************************************************************************/
/* HYPRE_LSI_MLISetMethod                                                   */
/*--------------------------------------------------------------------------*/

extern "C"
int HYPRE_LSI_MLI_SetMethod( HYPRE_Solver solver, char *paramString )
{
   HYPRE_LSI_MLI *mli_object = (HYPRE_LSI_MLI *) solver;

   if ( ! strcmp( paramString, "AMGSA" ) )
      mli_object->method_ = MLI_METHOD_AMGSA_ID;
   else if ( ! strcmp( paramString, "AMGSAe" ) )
      mli_object->method_ = MLI_METHOD_AMGSAE_ID;
   else
   {
      printf("HYPRE_LSI_MLISetMethod ERROR : method unrecognized.\n");
      exit(1);
   }
   return 0;
}

/****************************************************************************/
/* HYPRE_LSI_MLISetSmoother                                                 */
/* smoother type : 0 (Jacobi)                                               */
/*                 1 (GS)                                                   */
/*                 2 (SGS)                                                  */
/*                 3 (ParaSails)                                            */
/*                 4 (Schwarz)                                              */
/*                 5 (MLS)                                                  */
/*--------------------------------------------------------------------------*/

extern "C"
int HYPRE_LSI_MLISetSmoother( HYPRE_Solver solver, int pre_post,
                              int smoother_type, int argc, char **argv  )
{
   int           i, nsweeps, stype;
   double        *relax_wgts;
   HYPRE_LSI_MLI *mli_object = (HYPRE_LSI_MLI *) solver;

   stype = smoother_type;
   if ( stype < 0 || stype > 5 )
   {
      printf("HYPRE_LSI_MLI_SetSmoother WARNING : set to Jacobi.\n");
      stype = 0;
   }  
   stype += MLI_SOLVER_JACOBI_ID;
   if ( argc > 0 ) nsweeps = *(int *) argv[0];
   else            nsweeps = 1;
   if ( nsweeps < 0 ) nsweeps = 1;
   if ( argc > 1 ) relax_wgts = (double *) argv[1];
   else            relax_wgts = NULL;

   switch ( pre_post )
   {
      case 0 : mli_object->preSmoother_ = stype;
               mli_object->preNSweeps_  = nsweeps;
               if ( mli_object->preSmootherWts_ != NULL ) 
                  delete [] mli_object->preSmootherWts_;
               mli_object->preSmootherWts_ = NULL;
               if ( argc > 1 )
               {
                  mli_object->preSmootherWts_ = new double[nsweeps];
                  for ( i = 0; i < nsweeps; i++ )
                  {
                     if ( relax_wgts[i] > 0.0 && relax_wgts[i] < 2.0 )
                        mli_object->preSmootherWts_[i] = relax_wgts[i];
                     else
                        mli_object->preSmootherWts_[i] = 1.0;
                  } 
               } 
               break;

      case 1 : mli_object->postSmoother_ = stype;
               mli_object->postNSweeps_  = nsweeps;
               if ( mli_object->postSmootherWts_ != NULL ) 
                  delete [] mli_object->postSmootherWts_;
               mli_object->postSmootherWts_ = NULL;
               if ( argc > 1 )
               {
                  mli_object->postSmootherWts_ = new double[nsweeps];
                  for ( i = 0; i < nsweeps; i++ )
                  {
                     if ( relax_wgts[i] > 0.0 && relax_wgts[i] < 2.0 )
                        mli_object->postSmootherWts_[i] = relax_wgts[i];
                     else
                        mli_object->postSmootherWts_[i] = 1.0;
                  } 
              } 
               break;

      case 2 : mli_object->preSmoother_  = stype;
               mli_object->postSmoother_ = stype;
               mli_object->preNSweeps_   = nsweeps;
               mli_object->postNSweeps_  = nsweeps;
               if ( mli_object->preSmootherWts_ != NULL ) 
                  delete [] mli_object->preSmootherWts_;
               if ( mli_object->postSmootherWts_ != NULL ) 
                  delete [] mli_object->postSmootherWts_;
               mli_object->preSmootherWts_ = NULL;
               mli_object->postSmootherWts_ = NULL;
               if ( argc > 1 )
               {
                  mli_object->preSmootherWts_ = new double[nsweeps];
                  mli_object->postSmootherWts_ = new double[nsweeps];
                  for ( i = 0; i < nsweeps; i++ )
                  {
                     if ( relax_wgts[i] > 0.0 && relax_wgts[i] < 2.0 )
                     {
                        mli_object->preSmootherWts_[i] = relax_wgts[i];
                        mli_object->postSmootherWts_[i] = relax_wgts[i];
                     } 
                     else
                     {
                        mli_object->preSmootherWts_[i] = 1.0;
                        mli_object->postSmootherWts_[i] = 1.0;
                     } 
                  } 
               } 
               break;
   }
   return 0;
}

/****************************************************************************/
/* HYPRE_LSI_MLISetCoarseSolver                                             */
/* solver ID = 0  (superlu)                                                 */
/* solver ID = 1  (aggregation)                                             */
/*--------------------------------------------------------------------------*/

extern "C"
int HYPRE_LSI_MLISetCoarseSolver( HYPRE_Solver solver, int solver_id,
                                  int argc, char **argv )
{
   int           i, stype, nsweeps;
   double        *relax_wgts;
   HYPRE_LSI_MLI *mli_object = (HYPRE_LSI_MLI *) solver;

   stype = solver_id;
   if ( stype < 0 || stype > 6 )
   {
      printf("HYPRE_LSI_MLISetCoarseSolver WARNING : set to Jacobi.\n");
      stype = 0;
   } 
   stype += MLI_SOLVER_JACOBI_ID;

   if ( argc > 0 ) nsweeps = *(int *) argv[0];
   else            nsweeps = 0;
   if ( nsweeps < 0 ) nsweeps = 1;
 
   mli_object->coarseSolver_        = stype;
   mli_object->coarseSolverNSweeps_ = nsweeps;
   if ( mli_object->coarseSolverWts_ != NULL ) 
      delete [] mli_object->coarseSolverWts_;
   mli_object->coarseSolverWts_ = NULL;
   if ( argc > 1 )
   {
      relax_wgts = (double *) argv[1];
      mli_object->coarseSolverWts_ = new double[nsweeps];
      for ( i = 0; i < nsweeps; i++ )
      {
         if ( relax_wgts[i] > 0.0 && relax_wgts[i] < 2.0 )
            mli_object->coarseSolverWts_[i] = relax_wgts[i];
         else
            mli_object->coarseSolverWts_[i] = 1.0;
       } 
   } 
   return 0;
}

/****************************************************************************/
/* HYPRE_LSI_MLILoadNodalCoordinates                                        */
/* (The nodal coordinates loaded in here conforms to the nodal labeling in  */
/* FEI, so the lookup object can be used to find the equation number.       */
/* In addition, node numbers and coordinates need to be shuffled between    */
/* processors)
/*--------------------------------------------------------------------------*/

extern "C"
int HYPRE_LSI_MLILoadNodalCoordinates(HYPRE_Solver solver, int nNodes,
              int nodeDOF, int *eqnNumbers, int endRow, double *coords)
{
   int           iN, iP, mypid, nprocs, *nodeProcMap, *iTempArray, eqnInd;
   int           iS, nSends, *sendLengs, *sendProcs, **iSendBufs, procIndex;
   int           iR, nRecvs, *recvLengs, *recvProcs, **iRecvBufs, *procList;
   int           iD, *procNRows, numNodes;
   double        **dSendBufs, **dRecvBufs, *nCoords;
   MPI_Request   *mpiRequests;
   MPI_Status    mpiStatus;
   MPI_Comm      mpiComm;
   HYPRE_LSI_MLI *mli_object = (HYPRE_LSI_MLI *) solver;

   /* -------------------------------------------------------- */ 
   /* if nodal coordinates flag off, do not take coordinates   */
   /* -------------------------------------------------------- */ 

   if ( ! mli_object->nCoordAccept_ ) return 1;

   /* -------------------------------------------------------- */ 
   /* clean up previously allocated space                      */
   /* -------------------------------------------------------- */ 

   if ( mli_object->nCoordinates_ != NULL )
      delete [] mli_object->nCoordinates_;
   if ( mli_object->nullScales_ != NULL )
      delete [] mli_object->nullScales_;
   if ( mli_object->nSpaceVects_ != NULL )
      delete [] mli_object->nSpaceVects_; 
   mli_object->nCoordinates_ = NULL;
   mli_object->nullScales_   = NULL;
   mli_object->nSpaceVects_  = NULL; 

   /* -------------------------------------------------------- */ 
   /* fetch machine information                                */
   /* -------------------------------------------------------- */ 

   mpiComm = mli_object->mpiComm_;
   MPI_Comm_rank( mpiComm, &mypid );
   MPI_Comm_size( mpiComm, &nprocs );

   /* -------------------------------------------------------- */
   /* construct procNRows array                                */
   /* -------------------------------------------------------- */

   procNRows = new int[nprocs+1];
   iTempArray = new int[nprocs];
   for ( iP = 0; iP < nprocs; iP++ ) iTempArray[iP] = 0;
   iTempArray[mypid] = endRow + 1;
   MPI_Allreduce(iTempArray,&(procNRows[1]),nprocs,MPI_INT,MPI_SUM,mpiComm);
   procNRows[0] = 0;

   /* -------------------------------------------------------- */
   /* construct node to processor map                          */
   /* -------------------------------------------------------- */

   nodeProcMap = new int[nNodes];
   for ( iN = 0; iN < nNodes; iN++ )
   {
      nodeProcMap[iN] = -1;
      if ( eqnNumbers[iN] < procNRows[mypid] ||
           eqnNumbers[iN] >= procNRows[mypid+1] )
      {
         for ( iP = 0; iP < nprocs; iP++ )
            if ( eqnNumbers[iN] < procNRows[iP] ) break;
         nodeProcMap[iN] = iP - 1;
      }
   }

   /* -------------------------------------------------------- */
   /* construct send information                               */
   /* -------------------------------------------------------- */

   procList = new int[nprocs];
   for ( iP = 0; iP < nprocs; iP++ ) procList[iP] = 0;
   for ( iN = 0; iN < nNodes; iN++ )
      if ( nodeProcMap[iN] >= 0 ) procList[nodeProcMap[iN]]++;
   nSends = 0;
   for ( iP = 0; iP < nprocs; iP++ ) if ( procList[iP] > 0 ) nSends++;
   if ( nSends > 0 )
   {
      sendProcs = new int[nSends];
      sendLengs = new int[nSends];
      iSendBufs = new int*[nSends];
      dSendBufs = new double*[nSends];
   }
   nSends = 0;
   for ( iP = 0; iP < nprocs; iP++ )
   {
      if ( procList[iP] > 0 )
      {
         sendLengs[nSends] = procList[iP];
         sendProcs[nSends++] = iP;
      }
   }

   /* -------------------------------------------------------- */
   /* construct recv information                               */
   /* -------------------------------------------------------- */

   for ( iP = 0; iP < nprocs; iP++ ) procList[iP] = 0;
   for ( iP = 0; iP < nSends; iP++ ) procList[sendProcs[iP]]++;
   iTempArray = new int[nprocs];
   MPI_Allreduce(procList,iTempArray,nprocs,MPI_INT,MPI_SUM,mpiComm);
   nRecvs = iTempArray[mypid];
   delete [] procList;
   delete [] iTempArray;
   if ( nRecvs > 0 )
   {
      recvLengs = new int[nRecvs];
      recvProcs = new int[nRecvs];
      iRecvBufs = new int*[nRecvs];
      dRecvBufs = new double*[nRecvs];
      mpiRequests = new MPI_Request[nRecvs];
   }
   for ( iP = 0; iP < nRecvs; iP++ )
      MPI_Irecv(&(recvLengs[iP]), 1, MPI_INT, MPI_ANY_SOURCE, 29421,
                mpiComm, &(mpiRequests[iP]));
   for ( iP = 0; iP < nSends; iP++ )
      MPI_Send(&(sendLengs[iP]), 1, MPI_INT, sendProcs[iP], 29421, mpiComm);
   for ( iP = 0; iP < nRecvs; iP++ )
   {
      MPI_Wait(&(mpiRequests[iP]), &mpiStatus);
      recvProcs[iP] = mpiStatus.MPI_SOURCE;
   }

   /* -------------------------------------------------------- */
   /* communicate equation numbers information                */
   /* -------------------------------------------------------- */

   for ( iP = 0; iP < nRecvs; iP++ )
   {
      iRecvBufs[iP] = new int[recvLengs[iP]];
      MPI_Irecv(iRecvBufs[iP], recvLengs[iP], MPI_INT, recvProcs[iP],
                29422, mpiComm, &(mpiRequests[iP]));
   }
   for ( iP = 0; iP < nSends; iP++ )
   {
      iSendBufs[iP] = new int[sendLengs[iP]];
      sendLengs[iP] = 0;
   }
   for ( iN = 0; iN < nNodes; iN++ )
   {
      if ( nodeProcMap[iN] >= 0 )
      {
         procIndex = nodeProcMap[iN];
         iSendBufs[procIndex][sendLengs[procIndex]++] = eqnNumbers[iN];
      }
   }
   for ( iP = 0; iP < nSends; iP++ )
   {
      MPI_Send(iSendBufs[iP], sendLengs[iP], MPI_INT, sendProcs[iP],
               29422, mpiComm);
   }
   for ( iP = 0; iP < nRecvs; iP++ ) MPI_Wait(&(mpiRequests[iP]), &mpiStatus);

   /* -------------------------------------------------------- */
   /* communicate coordinate information                       */
   /* -------------------------------------------------------- */

   for ( iP = 0; iP < nRecvs; iP++ )
   {
      dRecvBufs[iP] = new double[recvLengs[iP]*nodeDOF];
      MPI_Irecv(dRecvBufs[iP], recvLengs[iP]*nodeDOF, MPI_DOUBLE, 
                recvProcs[iP], 29425, mpiComm, &(mpiRequests[iP]));
   }
   for ( iP = 0; iP < nSends; iP++ )
   {
      dSendBufs[iP] = new double[sendLengs[iP]*nodeDOF];
      sendLengs[iP] = 0;
   }
   for ( iN = 0; iN < nNodes; iN++ )
   {
      if ( nodeProcMap[iN] >= 0 )
      {
         procIndex = nodeProcMap[iN];
         for ( iD = 0; iD < nodeDOF; iD++ )
            dSendBufs[procIndex][sendLengs[procIndex]++]=coords[iN*nodeDOF+iD];
      }
   }
   for ( iP = 0; iP < nSends; iP++ )
   {
      sendLengs[iP] /= nodeDOF;
      MPI_Send(dSendBufs[iP], sendLengs[iP]*nodeDOF, MPI_DOUBLE, 
               sendProcs[iP], 29425, mpiComm);
   }
   for ( iP = 0; iP < nRecvs; iP++ ) MPI_Wait(&(mpiRequests[iP]), &mpiStatus);

   /* -------------------------------------------------------- */
   /* set up nodal coordinate information in correct order     */
   /* -------------------------------------------------------- */

   mli_object->nodeDOF_      = nodeDOF;
   mli_object->localNEqns_   = procNRows[mypid+1] - procNRows[mypid];
   mli_object->nCoordinates_ = new double[mli_object->localNEqns_];
   nCoords                   = mli_object->nCoordinates_;
   for (iN = 0; iN < mli_object->localNEqns_; iN++) nCoords[iN] = -999999.0;  

   numNodes = 0;
   for ( iN = 0; iN < nNodes; iN++ )  
   {
      if ( nodeProcMap[iN] < 0 ) 
      {
         eqnInd = eqnNumbers[iN] - procNRows[mypid];
         if ( nCoords[eqnInd] == -999999.0 ) numNodes++;
         for ( iD = 0; iD < nodeDOF; iD++ )  
            nCoords[eqnInd+iD] = coords[iN*nodeDOF+iD]; 
      }
   }
   for ( iP = 0; iP < nRecvs; iP++ )
   {
      for ( iR = 0; iR < recvLengs[iP]; iR++ )
      {
         eqnInd = iRecvBufs[iP][iR] - procNRows[mypid];
         if ( nCoords[eqnInd] == -999999.0 ) numNodes++;
         for ( iD = 0; iD < nodeDOF; iD++ )  
            nCoords[eqnInd+iD] = dRecvBufs[iP][iR*nodeDOF+iD];
      }
   }
   mli_object->localNEqns_ = numNodes * nodeDOF;

   /* -------------------------------------------------------- */
   /* clean up                                                 */
   /* -------------------------------------------------------- */

   delete [] nodeProcMap;
   if ( nSends > 0 )
   {
      delete [] sendProcs;
      delete [] sendLengs;
      for ( iS = 0; iS < nSends; iS++ ) delete [] iSendBufs[iS];
      for ( iS = 0; iS < nSends; iS++ ) delete [] dSendBufs[iS];
      delete [] dSendBufs;
      delete [] iSendBufs;
   }
   if ( nRecvs > 0 )
   {
      delete [] recvProcs;
      delete [] recvLengs;
      for ( iR = 0; iR < nRecvs; iR++ ) delete [] iRecvBufs[iR];
      for ( iR = 0; iR < nRecvs; iR++ ) delete [] dRecvBufs[iR];
      delete [] iRecvBufs;
      delete [] dRecvBufs;
      delete [] mpiRequests;
   }
   return 0;
} 

/****************************************************************************/
/* HYPRE_LSI_MLILoadMatrixScalings                                          */
/*--------------------------------------------------------------------------*/

extern "C"
int HYPRE_LSI_MLILoadMatrixScalings(HYPRE_Solver solver, int nEqns,
                                    double *scalings)
{
   HYPRE_LSI_MLI *mli_object = (HYPRE_LSI_MLI *) solver;
   if ( scalings != NULL )
   {
      mli_object->nullScales_ = new double[nEqns];
      for ( int i = 0; i < nEqns; i++ )  
         mli_object->nullScales_[i] = scalings[i];
   }
   return 0;
}

/****************************************************************************/
/* HYPRE_LSI_MLIFEDataCreate                                                */
/*--------------------------------------------------------------------------*/

extern "C"
void *HYPRE_LSI_MLIFEDataCreate( MPI_Comm mpi_comm )
{
#ifdef HAVE_MLI
   HYPRE_MLI_FEData *hypre_fedata;
   hypre_fedata = (HYPRE_MLI_FEData *) malloc( sizeof(HYPRE_MLI_FEData) );  
   hypre_fedata->comm_          = mpi_comm;
   hypre_fedata->fedata_        = NULL;
   hypre_fedata->fedataOwn_     = 0;
   hypre_fedata->computeNull_   = 0;
   hypre_fedata->nullDim_       = 1;
   return ((void *) hypre_fedata);
#else
   return NULL;
#endif 
}

/****************************************************************************/
/* HYPRE_LSI_MLIFEDataDestroy                                               */
/*--------------------------------------------------------------------------*/

extern "C"
int HYPRE_LSI_MLIFEDataDestroy( void *object )
{
#ifdef HAVE_MLI
   HYPRE_MLI_FEData *hypre_fedata = (HYPRE_MLI_FEData *) object;
   if ( hypre_fedata == NULL ) return 1;
   if ( hypre_fedata->fedataOwn_ && hypre_fedata->fedata_ != NULL ) 
      delete hypre_fedata->fedata_;
   hypre_fedata->fedata_        = NULL;
   free( hypre_fedata );
   return 0;
#else
   return 1;
#endif 
}

/****************************************************************************/
/* HYPRE_LSI_MLIFEDataInitFields                                            */
/*--------------------------------------------------------------------------*/

extern "C"
int HYPRE_LSI_MLIFEDataInitFields( void *object, int nFields, int *fieldSizes,
                                   int *fieldIDs )
{
#ifdef HAVE_MLI
   HYPRE_MLI_FEData *hypre_fedata = (HYPRE_MLI_FEData *) object;
   MLI_FEData       *fedata;

   if ( hypre_fedata == NULL ) return 1;
   if ( hypre_fedata->fedata_ != NULL ) delete hypre_fedata->fedata_;
   hypre_fedata->fedata_    = new MLI_FEData(hypre_fedata->comm_);
   hypre_fedata->fedataOwn_ = 1;
   fedata = (MLI_FEData *) hypre_fedata->fedata_;
   fedata->initFields( nFields, fieldSizes, fieldIDs );
   return 0;
#else
   (void) object;
   (void) nFields;
   (void) fieldSizes;
   (void) fieldIDs;
   return 1;
#endif
}

/****************************************************************************/
/* HYPRE_LSI_MLIFEDataInitElemBlock                                         */
/*--------------------------------------------------------------------------*/

extern "C"
int HYPRE_LSI_MLIFEDataInitElemBlock(void *object, int nElems, 
                                     int nNodesPerElem, int numNodeFields,
                                     int *nodeFieldIDs) 
{
#ifdef HAVE_MLI
   HYPRE_MLI_FEData *hypre_fedata = (HYPRE_MLI_FEData *) object;
   MLI_FEData       *fedata;

   if ( hypre_fedata == NULL ) return 1;
   fedata = (MLI_FEData *) hypre_fedata->fedata_;
   if ( fedata == NULL ) return 1;
   if ( numNodeFields != 1 ) return 1;
   hypre_fedata->fedata_->initElemBlock(nElems,nNodesPerElem,numNodeFields,
                                        nodeFieldIDs,0,NULL);
   return 0;
#else
   (void) object;
   (void) nElems;
   (void) nNodesPerElem;
   (void) numNodeFields;
   (void) nodeFieldIDs;
   return 1;
#endif
}

/****************************************************************************/
/* HYPRE_LSI_MLIFEDataInitElemNodeList                                      */
/*--------------------------------------------------------------------------*/

extern "C"
int HYPRE_LSI_MLIFEDataInitElemNodeList( void *object, int elemID, 
                       int nNodesPerElem, int *elemNodeList)
{
#ifdef HAVE_MLI
   HYPRE_MLI_FEData *hypre_fedata = (HYPRE_MLI_FEData *) object;
   MLI_FEData       *fedata;

   if ( hypre_fedata == NULL ) return 1;
   fedata = (MLI_FEData *) hypre_fedata->fedata_;
   if ( fedata == NULL ) return 1;
   fedata->initElemNodeList(elemID,nNodesPerElem,elemNodeList,3,NULL);
   return 0;
#else
   return 1;
#endif
}

/****************************************************************************/
/* HYPRE_LSI_MLIFEDataInitSharedNodes                                       */
/*--------------------------------------------------------------------------*/

extern "C"
int HYPRE_LSI_MLIFEDataInitSharedNodes(void *object, int nSharedNodes,
                  int *sharedNodeIDs, int *sharedProcLengs,
                  int **sharedProcIDs) 
{
#ifdef HAVE_MLI
   HYPRE_MLI_FEData *hypre_fedata = (HYPRE_MLI_FEData *) object;
   MLI_FEData       *fedata;

   if ( hypre_fedata == NULL ) return 1;
   fedata = (MLI_FEData *) hypre_fedata->fedata_;
   if ( fedata == NULL ) return 1;
   if ( nSharedNodes > 0 )
      fedata->initSharedNodes(nSharedNodes, sharedNodeIDs, sharedProcLengs, 
                              sharedProcIDs);
   return 0;
#else
   (void) object;
   (void) nSharedNodes;
   (void) sharedNodeIDs;
   (void) sharedProcLengs;
   (void) sharedProcIDs;
   return 1;
#endif
}

/****************************************************************************/
/* HYPRE_LSI_MLIFEDataInitComplete                                          */
/*--------------------------------------------------------------------------*/

extern "C"
int HYPRE_LSI_MLIFEDataInitComplete( void *object )
{
#ifdef HAVE_MLI
   HYPRE_MLI_FEData *hypre_fedata = (HYPRE_MLI_FEData *) object;
   MLI_FEData       *fedata;

   if ( hypre_fedata == NULL ) return 1;
   fedata = (MLI_FEData *) hypre_fedata->fedata_;
   if ( fedata == NULL ) return 1;
   fedata->initComplete();
   return 0;
#else
   (void) object;
   return 1;
#endif
}

/****************************************************************************/
/* HYPRE_LSI_MLIFEDataLoadElemMatrix                                        */
/*--------------------------------------------------------------------------*/

extern "C"
int HYPRE_LSI_MLIFEDataLoadElemMatrix(void *object, int elemID, int nNodes,
                            int *nodeList, int matDim, double **inMat)
{
   (void) nNodes;
   (void) nodeList;
#ifdef HAVE_MLI
   int              i, j;
   double           *elemMat;
   HYPRE_MLI_FEData *hypre_fedata = (HYPRE_MLI_FEData *) object;
   MLI_FEData       *fedata;

   /* -------------------------------------------------------- */ 
   /* error checking                                           */
   /* -------------------------------------------------------- */ 

   if ( hypre_fedata == NULL ) return 1;
   fedata = (MLI_FEData *) hypre_fedata->fedata_;
   if ( fedata == NULL ) return 1;

   /* -------------------------------------------------------- */ 
   /* load the element matrix                                  */
   /* -------------------------------------------------------- */ 

   elemMat = new double[matDim*matDim];
   for ( i = 0; i < matDim; i++ )
      for ( j = 0; j < matDim; j++ )
         elemMat[i+j*matDim] = inMat[i][j];
   fedata->loadElemMatrix(elemID, matDim, elemMat);
   delete [] elemMat;
   return 0;

#else
   (void) object;
   (void) elemID;
   (void) matDim;
   (void) inMat;
   return 1;
#endif
}

/****************************************************************************/
/* HYPRE_LSI_MLIFEDataWriteToFile                                           */
/*--------------------------------------------------------------------------*/

extern "C"
int HYPRE_LSI_MLIFEDataWriteToFile( void *object, char *filename )
{
#ifdef HAVE_MLI
   MLI_FEData       *fedata;
   HYPRE_MLI_FEData *hypre_fedata = (HYPRE_MLI_FEData *) object;

   /* -------------------------------------------------------- */ 
   /* error checking                                           */
   /* -------------------------------------------------------- */ 

   if ( hypre_fedata == NULL ) return 1;
   fedata = (MLI_FEData *) hypre_fedata->fedata_;
   if ( fedata == NULL ) return 1;

   /* -------------------------------------------------------- */ 
   /* write to file                                            */
   /* -------------------------------------------------------- */ 

   fedata->writeToFile( filename );
   return 0;
#else
   (void) object;
   (void) filename;
   return 1;
#endif
}


