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

/****************************************************************************/ 
/* HYPRE_LSI_MLI data structure                                             */
/*--------------------------------------------------------------------------*/

typedef struct HYPRE_LSI_MLI_Struct
{
#ifdef HAVE_MLI
   MLI      *mli_;
#endif
   MPI_Comm mpiComm_;
   int      nLevels_;             /* max number of levels */
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
} 
HYPRE_LSI_MLI;

/****************************************************************************/
/* HYPRE_LSI_MLICreate                                                      */
/*--------------------------------------------------------------------------*/

int HYPRE_LSI_MLICreate( MPI_Comm comm, HYPRE_Solver *solver )
{
   HYPRE_LSI_MLI *mli_object = (HYPRE_LSI_MLI *) malloc(sizeof(HYPRE_LSI_MLI));
   *solver = (HYPRE_Solver) mli_object;
   mli_object->mpiComm_             = comm;
   mli_object->nLevels_             = 30;
   mli_object->method_              = 1;
   mli_object->numPDEs_             = 1;
   mli_object->preSmoother_         = MLI_SOLVER_SGS_ID;
   mli_object->postSmoother_        = MLI_SOLVER_SGS_ID;
   mli_object->preSmoother_         = 0;
   mli_object->postSmoother_        = 0;
   mli_object->preNSweeps_          = 2;
   mli_object->postNSweeps_         = 2;
   mli_object->preSmootherWts_      = NULL;
   mli_object->postSmootherWts_     = NULL;
   mli_object->strengthThreshold_   = 0.08;
   mli_object->coarseSolver_        = 0;
   mli_object->coarseSolverNSweeps_ = 0;
   mli_object->coarseSolverWts_     = NULL;
#ifdef HAVE_MLI
   mli_object->mli_                 = NULL;
   return 0;
#else
   printf("MLI not available.\n");
   return -1;
#endif
}

/****************************************************************************/
/* HYPRE_LSI_MLIDestroy                                                     */
/*--------------------------------------------------------------------------*/

int HYPRE_LSI_MLIDestroy( HYPRE_Solver solver )
{
   HYPRE_LSI_MLI *mli_object = (HYPRE_LSI_MLI *) solver;

#ifdef HAVE_MLI
   if ( mli_object->mli_ != NULL ) delete [] mli_object->mli_;
#endif
 
   if ( mli_object->preSmootherWts_ != NULL ) 
      delete [] mli_object->preSmootherWts_;
   if ( mli_object->postSmootherWts_ != NULL ) 
      delete [] mli_object->postSmootherWts_;
   if ( mli_object->coarseSolverWts_ != NULL ) 
      delete [] mli_object->coarseSolverWts_;
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

int HYPRE_LSI_MLISetup( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                        HYPRE_ParVector b,   HYPRE_ParVector x )
{
#ifdef HAVE_MLI
   int           targc, maxIterations=1;
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
   mli_object->mli_ = mli;

   /* -------------------------------------------------------- */ 
   /* set general parameters                                   */
   /* -------------------------------------------------------- */ 

   mli->setNumLevels( mli_object->nLevels );
   mli->setMaxIterations( maxIterations );
   mli->setTolerance( tol );

   /* -------------------------------------------------------- */ 
   /* set method specific parameters                           */
   /* -------------------------------------------------------- */ 

   switch ( mli_object->method_ )
   {
      case MLI_METHOD_AMGSA_ID : 
           method = MLI_Method_CreateFromID(mli_object->method, mpiComm );
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
      case MLI_SOLVER_SUPERLU_ID       : 
           sprintf(paramString, "setPostSmoother SuperLU" ); break;
   }
   targc    = 2;
   targv[0] = (char *) &mli_object->coarseSolverNSweeps_;
   targv[1] = (char *) mli_object->coarseSolverWts_;
   method->setParams( paramString, targc, targv );

   /* -------------------------------------------------------- */ 
   /* load null space, if there is any                         */
   /* -------------------------------------------------------- */ 

   if ( mli_object->NSpaceDim_ > 0 )
   {
      targv[0] = (char *) &mli_object->nodeDOF_;
      targv[1] = (char *) &mli_object->nSpaceDim_;
      targv[2] = (char *) nSpaceVects_;
      targv[3] = (char *) &localNEqns_;
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
   free( func_ptr );
   mli->setup();

   return 0;
#else
   printf("MLI not available.\n");
   return -1;
#endif
}

/****************************************************************************/
/* HYPRE_LSI_MLISolve                                                       */
/*--------------------------------------------------------------------------*/

int HYPRE_LSI_MLISolve( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                        HYPRE_ParVector b, HYPRE_ParVector x )
{
#ifdef HAVE_MLI
   HYPRE_LSI_MLI *mli_object;
   MLI_Vector    *sol, *rhs;

   sol = new MLI_Vector( (void *) x, "HYPRE_ParVector", NULL);
   rhs = new MLI_Vector( (void *) b, "HYPRE_ParVector", NULL);

   mli_object = (HYPRE_LSI_MLI *) solver;
   mli_object->solve( sol, rhs);

   delete [] sol;
   delete [] rhs;

   return 0;
#else
   printf("MLI not available.\n");
   return -1;
#endif
}

/****************************************************************************/
/* HYPRE_LSI_MLISetParams                                                   */
/*--------------------------------------------------------------------------*/

int HYPRE_LSI_MLISetParams( HYPRE_Solver solver, char *paramString )
{
   HYPRE_LSI_MLI *mli_object;
   char          param1[256], param2[256], param3[256];

   mli_object = (HYPRE_LSI_MLI *) solver;
   sscanf(params,"%s", param1);
   if ( strcmp(param1, "MLI") )
   {
      printf("HYPRE_LSI_MLI::parameters not for me.\n");
      return 1;
   }
   sscanf(params,"%s %s", param1, param2);
   if ( !strcmp(param2, "strengthThreshold") )
   {
      sscanf(params,"%s %s %lg",param1,param2,&(mli_object->strengthThreshold_));
      if ( mli_object->strengthThreshold_ < 0.0 )
         mli_object->strengthThreshold_ = 0.0;
   }
   else if ( !strcmp(param2, "method") )
   {
      sscanf(params,"%s %s %s", param1, param2, param3);
      if ( ! strcmp( param3, "AMGSA" ) )
         mli_object->method_ = MLI_METHOD_AMGSA_ID;
   }
   else if ( !strcmp(param2, "smoother") )
   {
      sscanf(params,"%s %s %s", param1, param2, param3);
      if ( ! strcmp( param3, "Jacobi" ) )
      {
         mli_object->preSmoother_  = MLI_METHOD_JACOBI_ID;
         mli_object->postSmoother_ = MLI_METHOD_JACOBI_ID;
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
      sscanf(params,"%s %s %s", param1, param2, param3);
      if ( ! strcmp( param3, "Jacobi" ) )
         mli_object->coarseSolver_ = MLI_METHOD_JACOBI_ID;
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
      sscanf(params,"%s %s %d",param1,param2,&(mli_object->preNSweeps_));
      if ( mli_object->preNSweeps_ <= 0 ) mli_object->preNSweeps_ = 1;
      mli_object->postNSweeps_ = mli_object->preNSweeps_; 
   }
   else 
   {
      printf("HYPRE_LSI_MLISetParams ERROR : unrecognized request.\n");
      exit(1);
   }
   return 0;
}

/****************************************************************************/
/* HYPRE_LSI_MLISetStrengthhreshold                                         */
/*--------------------------------------------------------------------------*/

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

int HYPRE_LSI_MLISetCoarseSolver( HYPRE_Solver solver, int solver_id, 
                                  int argc, char **argv )
{
   int           i, stype, nsweeps;
   double        *relax_wgts;
   HYPRE_LSI_MLI *mli_object = (HYPRE_LSI_MLI *) solver;

   stype = solver_id;
   if ( stype < 0 || stype > 6 )
   {
      printf("HYPRE_LSI_MLISetCoarseSolver WARNING : set to Jacobi.\n";
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

