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
 *        HYPRE_LSI_MLISetStrengthThreshold
 *        HYPRE_LSI_MLISetMethod
 *        HYPRE_LSI_MLISetSmoother
 *        HYPRE_LSI_MLISetCoarseSolver
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
#include <mpi.h>

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
#endif
#define GlobalID int
#include "HYPRE_LSI_mli.h"

extern "C" 
{
void mli_computespectrum_(int *,int *,double *, double *, int *, double *, 
                     double *, double *, int *);
}

/****************************************************************************/ 
/* HYPRE_LSI_MLI data structure                                             */
/*--------------------------------------------------------------------------*/

typedef struct HYPRE_LSI_MLI_Struct
{
#ifdef HAVE_MLI
   MLI      *mli_;
#endif
   MPI_Comm mpiComm_;
   int      outputLevel_;         /* for diagnostics */
   int      nLevels_;             /* max number of levels */
   int      cycleType_;           /* 1 for V and 2 for W */
   int      maxIterations_;       /* default - 1 iteration */
   int      method_;              /* default - smoothed aggregation */
   int      numPDEs_;             /* default - 1 */
   int      preSmoother_;         /* default - Gauss Seidel */
   int      postSmoother_ ;
   int      preNSweeps_;          /* default - 2 smoothing steps */
   int      postNSweeps_;         /* default - 2 smoothing steps */
   double   *preSmootherWts_;
   double   *postSmootherWts_;
   double   strengthThreshold_;   /* strength threshold */
   int      coarseSolver_;        /* default = SuperLU */
   int      coarseSolverNSweeps_;
   double   *coarseSolverWts_;
   int      minCoarseSize_;       /* minimum coarse grid size */
   int      nodeDOF_;
   int      nSpaceDim_;           /* number of null vectors */
   double   *nSpaceVects_;
   int      localNEqns_;
} 
HYPRE_LSI_MLI;

/****************************************************************************/ 
/* HYPRE_LSI_MLI data structure                                             */
/*--------------------------------------------------------------------------*/

typedef struct HYPRE_MLI_FEData_Struct
{
#ifdef HAVE_MLI
   MPI_Comm   comm;
   MLI_FEData *fedata_;
   int        *fedataElemIDs_;
   int        fedataElemCnt_;
   int        fedataRowCnt_;
   int        fedataMatDim_;
   double     *fedataValues_;
   int        nullLeng_;
   int        nullDim_;
   int        *nullCnts_;
   double     *nullSpaces_;
   Lookup     *lookup_;
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
   mli_object->numPDEs_             = 1;
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
#ifdef HAVE_MLI
   mli_object->mli_                 = NULL;
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

#ifdef HAVE_MLI
   if ( mli_object->mli_ != NULL ) delete mli_object->mli_;
#endif
 
   if ( mli_object->preSmootherWts_ != NULL ) 
      delete [] mli_object->preSmootherWts_;
   if ( mli_object->postSmootherWts_ != NULL ) 
      delete [] mli_object->postSmootherWts_;
   if ( mli_object->coarseSolverWts_ != NULL ) 
      delete [] mli_object->coarseSolverWts_;
   if ( mli_object->nSpaceVects_ != NULL ) 
      delete [] mli_object->nSpaceVects_;
   free( mli_object );
#ifdef HAVE_MLI
   return 0;
#else
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
   int           targc;
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
           method = MLI_Method_CreateFromID(mli_object->method_, mpiComm );
           break;
      default :
           printf("HYPRE_LSI_MLISetup : method not valid.\n");
           exit(1);
   }
   sprintf(paramString, "setNumLevels %d", mli_object->nLevels_);
   method->setParams( paramString, 0, NULL );
   sprintf(paramString,"setStrengthThreshold %f",mli_object->strengthThreshold_);
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

   /* -------------------------------------------------------- */ 
   /* load null space, if there is any                         */
   /* -------------------------------------------------------- */ 

   if ( mli_object->nSpaceDim_ > 0 )
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
   HYPRE_LSI_MLI *mli_object;
   char          param1[256], param2[256], param3[256];

   mli_object = (HYPRE_LSI_MLI *) solver;
   sscanf(paramString,"%s", param1);
   if ( strcmp(param1, "MLI") )
   {
      printf("HYPRE_LSI_MLI::parameters not for me.\n");
      return 1;
   }
   sscanf(paramString,"%s %s", param1, param2);
   if ( !strcmp(param2, "outputLevel") )
   {
      sscanf(paramString,"%s %s %d",param1,param2,&(mli_object->outputLevel_));
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
   else 
   {
      printf("HYPRE_LSI_MLISetParams ERROR : unrecognized request.\n");
      printf("    offending request = %s.\n", paramString);
      printf("Available ones : \n");
      printf("      outputLevel <d> \n");
      printf("      maxIterations <d> \n");
      printf("      cycleType <'V','W'> \n");
      printf("      strengthThreshold <f> \n");
      printf("      method AMGSA \n");
      printf("      smoother <Jacobi,GS,...> \n");
      printf("      coarseSolver <Jacobi,GS,...> \n");
      printf("      numSweeps <d> \n");
      printf("      minCoarseSize <d> \n");
      printf("      nodeDOF <d> \n");
      printf("      nullSpaceDim <d> \n");
      exit(1);
   }
   return 0;
}

/****************************************************************************/
/* HYPRE_LSI_MLISetStrengthhreshold                                         */
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
   else
   {
      printf("HYPRE_LSI_MLISetMethod ERROR : method unrecognized.\n");
      exit(1);
   }
   return 0;
}

/****************************************************************************/
/* HYPRE_LSI_MLISetNumPDEs                                                  */
/*--------------------------------------------------------------------------*/

extern "C"
int HYPRE_LSI_MLISetNumPDEs( HYPRE_Solver solver, int numPDE )
{
   HYPRE_LSI_MLI *mli_object = (HYPRE_LSI_MLI *) solver;

   if ( numPDE > 1 ) mli_object->numPDEs_ = numPDE;
   else              mli_object->numPDEs_ = 1;
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
/* HYPRE_LSI_MLISetNullSpaces                                               */
/*--------------------------------------------------------------------------*/

extern "C"
int HYPRE_LSI_MLISetNullSpaces( HYPRE_Solver solver, int nodeDOF,
                                int numNS, double *NSpaces, int numEqns)
{
   HYPRE_LSI_MLI *mli_object = (HYPRE_LSI_MLI *) solver;
   mli_object->nodeDOF_     = nodeDOF;
   mli_object->nSpaceDim_   = numNS;
   mli_object->nSpaceVects_ = new double[numEqns*numNS];
   for ( int i = 0; i < numEqns*numNS; i++ )  
      mli_object->nSpaceVects_[i] = NSpaces[i];
   mli_object->nSpaceVects_ = NULL;
   mli_object->localNEqns_  = numEqns;
   return 0;
} 

/****************************************************************************/
/* HYPRE_LSI_MLICreateFEData                                                */
/*--------------------------------------------------------------------------*/

extern "C"
void *HYPRE_LSI_MLIFEDataCreate( MPI_Comm mpi_comm )
{
#ifdef HAVE_MLI
   HYPRE_MLI_FEData *hypre_fedata;
   hypre_fedata = (HYPRE_MLI_FEData *) malloc( sizeof(HYPRE_MLI_FEData) );  
   hypre_fedata->fedataElemIDs_ = NULL;
   hypre_fedata->fedataElemCnt_ = 0;
   hypre_fedata->fedataRowCnt_  = 0;
   hypre_fedata->fedataMatDim_  = 0;
   hypre_fedata->fedataValues_  = NULL;
   hypre_fedata->comm           = mpi_comm;
   hypre_fedata->fedata_        = new MLI_FEData(mpi_comm);
   hypre_fedata->lookup_        = NULL;
   hypre_fedata->nullLeng_      = 0;
   hypre_fedata->nullCnts_      = NULL;
   hypre_fedata->nullDim_       = 0;
   hypre_fedata->nullSpaces_    = NULL;
   return ((void *) hypre_fedata);
#else
   return NULL;
#endif 
}

/****************************************************************************/
/* HYPRE_LSI_MLIDestroyFEData                                               */
/*--------------------------------------------------------------------------*/

extern "C"
int HYPRE_LSI_MLIFEDataDestroy( void *object )
{
#ifdef HAVE_MLI
   HYPRE_MLI_FEData *hypre_fedata = (HYPRE_MLI_FEData *) object;
   if ( hypre_fedata->fedata_ != NULL ) delete hypre_fedata->fedata_;
   if ( hypre_fedata->fedataElemIDs_ != NULL ) 
      delete [] hypre_fedata->fedataElemIDs_;
   if ( hypre_fedata->fedataValues_ != NULL ) 
      delete [] hypre_fedata->fedataValues_;
   if ( hypre_fedata->nullSpaces_ != NULL )
      delete [] hypre_fedata->nullSpaces_;
   if ( hypre_fedata->nullCnts_ != NULL )
      delete [] hypre_fedata->nullCnts_;
   hypre_fedata->fedata_        = NULL;
   hypre_fedata->fedataElemIDs_ = NULL;
   hypre_fedata->fedataValues_  = NULL;
   hypre_fedata->nullSpaces_    = NULL;
   hypre_fedata->nullCnts_      = NULL;
   free( hypre_fedata );
   return 0;
#else
   return 1;
#endif 
}

/****************************************************************************/
/* HYPRE_LSI_MLIDestroyFEData                                               */
/* (1) setLookup (called by FEI_initComplete) which calls this function     */ 
/* (2) setGlobalOffsets in HYPRE_LinSysCore.C (next sequence of calls)      */
/* (3) setConnectivies in HYPRE_LinSysCore.C (next sequence of calls)      */
/*--------------------------------------------------------------------------*/

extern "C"
int HYPRE_LSI_MLIFEDataInit( void *object, Lookup *lookup )
{
#ifdef HAVE_MLI
   int              i, nFields, blockID=0, interleaveStrategy, lumpingStrategy;
   int              numElemDOF, numElemBlocks, numElements, numNodesPerElem;
   int              numEqnsPerElem, *nodeFieldIDs, nodeNumFields, numProcs;
   int              nSharedNodes, *sharedProcLengs, **sharedProcIDs;
   const int        *intArray, *fieldIDs, *fieldSizes, *sharedNodeIDs;
   const int* const *intArray2;
   HYPRE_MLI_FEData *hypre_fedata = (HYPRE_MLI_FEData *) object;
   MLI_FEData       *fedata = (MLI_FEData *) hypre_fedata->fedata_;

   hypre_fedata->lookup_ = lookup;
   if ( lookup == NULL ) return 1;
   numElemBlocks = lookup->getNumElemBlocks();
   nFields       = lookup->getNumFields();
   if ( fedata != NULL && numElemBlocks == 1 && nFields == 1 )
   {
      fieldIDs      = lookup->getFieldIDsPtr();
      fieldSizes    = lookup->getFieldSizesPtr();
      lookup->getElemBlockInfo(blockID, interleaveStrategy, lumpingStrategy,
                               numElemDOF, numElements, numNodesPerElem,
                               numEqnsPerElem);
      intArray = lookup->getNumFieldsPerNode(blockID);
      nodeNumFields = intArray[0];
      intArray2 = lookup->getFieldIDsTable(blockID);
      nodeFieldIDs = new int[nodeNumFields];
      for ( i = 0;i < nodeNumFields; i++ ) nodeFieldIDs[i] = intArray2[0][i];
      fedata->initFields(nFields, fieldSizes, fieldIDs);
      fedata->initElemBlock(numElements, numNodesPerElem, nodeNumFields,
                            nodeFieldIDs, 0, NULL);
      delete [] nodeFieldIDs;
      hypre_fedata->fedataElemIDs_ = new int[numElements];
      hypre_fedata->fedataElemCnt_ = 0;
   }
   return 0;
#else
   return 1;
#endif
}

/****************************************************************************/
/* HYPRE_LSI_MLIFEDataSetElemNodeList                                       */
/*--------------------------------------------------------------------------*/

extern "C"
int HYPRE_LSI_MLIFEDataSetElemNodeList( void *object, int numElements,
                       int numNodesPerElem, const int* elemIDs,
                       const int* const* connNodes)
{
#ifdef HAVE_MLI
   HYPRE_MLI_FEData *hypre_fedata = (HYPRE_MLI_FEData *) object;
   MLI_FEData *fedata = (MLI_FEData *) hypre_fedata->fedata_;
   if ( fedata != NULL )
   {
      for ( int i = 0; i < numElements; i++ )
      {
         fedata->initElemNodeList(elemIDs[i], numNodesPerElem,
                                  connNodes[i], 3, NULL);
         hypre_fedata->fedataElemIDs_[hypre_fedata->fedataElemCnt_++]=elemIDs[i];
      }
   }
   return 0;
#else
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
   int              i, j;
   int              numProcs, nSharedNodes, *sharedProcLengs, **sharedProcIDs;
   const int        *intArray, *sharedNodeIDs;
   HYPRE_MLI_FEData *hypre_fedata = (HYPRE_MLI_FEData *) object;
   MLI_FEData       *fedata = (MLI_FEData *) hypre_fedata->fedata_;
   Lookup           *lookup;

   if ( fedata != NULL )
   {
      MPI_Comm_size( hypre_fedata->comm, &numProcs );
      lookup = hypre_fedata->lookup_;
      if ( numProcs > 1 )
      {
         nSharedNodes    = hypre_fedata->lookup_->getNumSharedNodes();
         sharedNodeIDs   = hypre_fedata->lookup_->getSharedNodeNumbers();
         sharedProcIDs   = new int*[nSharedNodes];
         sharedProcLengs = new int[nSharedNodes];
         for ( i = 0; i < nSharedNodes; i++ )
         {
            sharedProcLengs[i] = lookup->getNumSharingProcs(sharedNodeIDs[i]);
            sharedProcIDs[i] = new int[sharedProcLengs[i]];
            intArray = lookup->getSharedNodeProcs( sharedNodeIDs[i] );
            for ( j = 0; j < sharedProcLengs[i]; j++ )
               sharedProcIDs[i][j] = intArray[j];
         }
         fedata->initSharedNodes(nSharedNodes, sharedNodeIDs, sharedProcLengs,
                                 sharedProcIDs);
         delete [] sharedProcLengs;
         for ( i = 0; i < nSharedNodes; i++ ) delete [] sharedProcIDs[i];
         delete [] sharedProcIDs;
      }
      fedata->initComplete();
      hypre_fedata->fedataElemCnt_ = 0;
      hypre_fedata->fedataRowCnt_  = 0;
      hypre_fedata->fedataValues_  = NULL;
      hypre_fedata->fedataMatDim_  = 0;
   }
   return 0;
#else
   return 1;
#endif
}

/****************************************************************************/
/* HYPRE_LSI_MLIFEDataAccumulateElemMatrix                                  */
/*--------------------------------------------------------------------------*/

extern "C"
int HYPRE_LSI_MLIFEDataAccumulateElemMatrix(void *object, int nrows, int ncols,
                                            const int * cols,
                                            const double * const *inValues)
{
#ifdef HAVE_MLI
   int              nElems, elemID, i, j, rowCnt, matz=0, ierr, nNodes, itmp;
   int              index, nDim, length, *nCnts;
   double           *elemMat, *evalues, *evectors, *dAux1, *dAux2, *nSpace;
   char             paramString[100], *targv[1];
   HYPRE_MLI_FEData *hypre_fedata = (HYPRE_MLI_FEData *) object;
   MLI_FEData       *fedata = (MLI_FEData *) hypre_fedata->fedata_;

   if (hypre_fedata->fedataMatDim_ != 0 && hypre_fedata->fedataMatDim_ != ncols)
   {
      printf("HYPRE_LSI_MLIFEDataAccumulateElemMatrix ERROR:MatDim mismatch\n");
      exit(1);
   }
   if (hypre_fedata->fedataMatDim_ == 0 )
   {
      hypre_fedata->fedataMatDim_ = ncols;
      if ( hypre_fedata->fedataValues_ != NULL )
         delete [] hypre_fedata->fedataValues_;
      hypre_fedata->fedataValues_ = new double[ncols*ncols];
   }
   rowCnt  = hypre_fedata->fedataRowCnt_;
   elemMat = hypre_fedata->fedataValues_;
   for ( i = 0; i < nrows; i++ )
   {
      for ( j = 0; j < ncols; j++ )
         elemMat[j*ncols+rowCnt] = inValues[i][j];
      rowCnt++;
   }
   hypre_fedata->fedataRowCnt_ = rowCnt;
   if ( rowCnt == ncols )
   {
      fedata->getNumElements(nElems);
      if ( hypre_fedata->fedataElemCnt_ >= nElems )
      {
         printf("HYPRE_LSI_MLIFEDataAccumulateElemMatrix ERROR:nElems > max \n");
         exit(1);
      }
      elemID = hypre_fedata->fedataElemIDs_[hypre_fedata->fedataElemCnt_++];
      fedata->loadElemMatrix(elemID, ncols, elemMat);
      hypre_fedata->fedataRowCnt_ = 0;
      evalues  = new double[ncols];
      evectors = new double[ncols*ncols];
      dAux1    = new double[ncols];
      dAux2    = new double[ncols];
      mli_computespectrum_(&ncols, &ncols, elemMat, evalues, &matz, evectors,
                           dAux1, dAux2, &ierr);
      if ( hypre_fedata->nullSpaces_ == NULL ) 
      {
         fedata->getNumNodes(nNodes);
         strcpy( paramString, "getNumExtNodes" );
         targv[0] = (char *) &itmp;
         fedata->impSpecificRequests( paramString, 1, targv );
         nNodes -= itmp;
         hypre_fedata->nullSpaces_ = new double[nNodes*3*6];
         hypre_fedata->nullCnts_   = new int[nNodes*3*6];
         hypre_fedata->nullDim_    = 6;
         hypre_fedata->nullLeng_   = nNodes * 3;
         for ( i = 0; i < nNodes*3*6; i++ ) hypre_fedata->nullSpaces_[i] = 0.0;
         for ( i = 0; i < nNodes*3*6; i++ ) hypre_fedata->nullCnts_[i] = 0;
      }
      nSpace = hypre_fedata->nullSpaces_;
      length = hypre_fedata->nullLeng_;
      nDim   = hypre_fedata->nullDim_;
      nCnts  = hypre_fedata->nullCnts_;
   
      for ( i = 0; i < ncols; i++ ) 
      {
         index = cols[i];
         for ( j = 0; j < nDim; j++ ) 
         {
            nSpace[index+length*j] += evectors[j*ncols+i];
            nCnts[index+length*j]++;
         }
      }
      for ( i = 0; i < ncols; i++ )
         printf("Element %5d : eigenvalue = %e\n", elemID, evalues[i]);
      delete [] evalues;
      delete [] evectors;
      delete [] dAux1;
      delete [] dAux2;
   }
   return 0;
#else
   return 1;
#endif
}

/****************************************************************************/
/* HYPRE_LSI_MLIFEDataConstructKernel                                       */
/*--------------------------------------------------------------------------*/

extern "C"
int HYPRE_LSI_MLIFEDataConstructKernel( void *object )
{
#ifdef HAVE_MLI
   int    i, j, length, nDim, *nCnts;
   double *nSpace;
   HYPRE_MLI_FEData *hypre_fedata = (HYPRE_MLI_FEData *) object;

   nSpace = hypre_fedata->nullSpaces_;
   length = hypre_fedata->nullLeng_;
   nDim   = hypre_fedata->nullDim_;
   nCnts  = hypre_fedata->nullCnts_;
   if ( nSpace == NULL || nCnts == NULL )
   {
      printf("HYPRE_LSI_MLIFEDataConstructKernel ERROR : no kernel.\n");
      exit(1);
   }
   for ( i = 0; i < nDim; i++ ) 
   {
      for ( j = 0; j < length; j++ ) 
      {
         if ( nCnts[j+length*i] != 0.0 ) 
            nSpace[j+length*i] /= nCnts[j+length*i];
         else
         {
            printf("HYPRE_LSI_MLIFEDataConstructKernel ERROR : nCnts = 0.\n");
            exit(1);
         }
      }
   }
   delete [] hypre_fedata->nullCnts_;
   hypre_fedata->nullCnts_ = NULL;
   return 0;
#else
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
   HYPRE_MLI_FEData *hypre_fedata = (HYPRE_MLI_FEData *) object;
   MLI_FEData       *fedata = (MLI_FEData *) hypre_fedata->fedata_;

   if ( fedata != NULL ) fedata->writeToFile( filename );
   return 0;
#else
   return 1;
#endif
}

