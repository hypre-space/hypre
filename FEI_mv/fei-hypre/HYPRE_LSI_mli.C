/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/
/****************************************************************************/ 
/* HYPRE_MLI interface                                                      */
/*--------------------------------------------------------------------------*/
/*  local functions :
 * 
 *        HYPRE_ParCSRMLI_Create
 *        HYPRE_ParCSRMLI_Destroy
 *        HYPRE_ParCSRMLI_Setup
 *        HYPRE_ParCSRMLI_Solve
 *        HYPRE_ParCSRMLI_SetStrengthThreshold
 *        HYPRE_ParCSRMLI_SetMethod
 *        HYPRE_ParCSRMLI_SetSmoother
 *        HYPRE_ParCSRMLI_SetCoarseSolver
 ****************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#ifdef HAVE_MLI
#include "base/mli_defs.h"
#include "base/mli.h"
#include "util/mli_utils.h"
#endif

//******************************************************************************
//******************************************************************************
// C-Interface data structure 
//------------------------------------------------------------------------------

typedef struct HYPRE_MLI_Struct
{
#ifdef HAVE_MLI
   MLI *mli_;
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
HYPRE_MLI;

/****************************************************************************/
/* HYPRE_ParCSRMLI_Create                                                     */
/*--------------------------------------------------------------------------*/

int HYPRE_ParCSRMLI_Create( MPI_Comm comm, HYPRE_Solver *solver )
{
   HYPRE_MLI *mli_object = (HYPRE_MLI *) malloc( sizeof(HYPRE_MLI) );
   *solver = (HYPRE_Solver) mli_object;
   mli_object->mpiComm_             = comm;
   mli_object->nLevels_             = 30;
   mli_object->method_              = 1;
   mli_object->numPDEs_             = 1;
   mli_object->preSmoother_         = MLI_SOLVER_SGS_ID;
   mli_object->postSmoother_        = MLI_SOLVER_SGS_ID;
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
   cout << "MLI not available.\n");
   return -1;
#endif
}

/****************************************************************************/
/* HYPRE_ParCSRMLI_Destroy                                                    */
/*--------------------------------------------------------------------------*/

int HYPRE_ParCSRMLI_Destroy( HYPRE_Solver solver )
{
   HYPRE_MLI *mli_object = (HYPRE_MLI *) solver;

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
   cout << "MLI not available.\n");
   return -1;
#endif

}

/****************************************************************************/
/* HYPRE_ParCSRMLSetup                                                      */
/*--------------------------------------------------------------------------*/

int HYPRE_ParCSRMLSetup( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                         HYPRE_ParVector b,   HYPRE_ParVector x      )
{
#ifdef HAVE_MLI
   int        targc, maxIterations=1;
   double     tol=1.0e-8;
   char       *targv[4], paramString[100];;
   HYPRE_MLI  *mli_object;
   MLI_Matrix *mli_mat;   
   MLI_Method *method;   
   MPI_Comm   mpiComm;
   MLI        *mli;

   /* -------------------------------------------------------- */ 
   /* create object                                            */
   /* -------------------------------------------------------- */ 

   mli_object = (HYPRE_MLI *) solver;
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
   cout << "MLI not available.\n";
   return -1;
#endif
}

/****************************************************************************/
/* HYPRE_ParCSRMLSolve                                                      */
/*--------------------------------------------------------------------------*/

int HYPRE_ParCSRMLSolve( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                         HYPRE_ParVector b,   HYPRE_ParVector x      )
{
#ifdef HAVE_MLI
   MLI_Vector *sol, *rhs;

   sol = new MLI_Vector( (void *) x, "HYPRE_ParVector", NULL);
   rhs = new MLI_Vector( (void *) b, "HYPRE_ParVector", NULL);

   mli_object->solve( sol, rhs);

   delete [] sol;
   delete [] rhs;

   return 0;
#else
   cout << "MLI not available.\n";
   return -1;
#endif
}

/****************************************************************************/
/* HYPRE_ParCSRMLSetStrongThreshold                                         */
/*--------------------------------------------------------------------------*/

int HYPRE_ParCSRMLI_SetStrengthThreshold(HYPRE_Solver solver,
                                         double strength_threshold)
{
   HYPRE_MLI *mli_object = (HYPRE_MLI *) solver;
  
   if ( strength_threshold < 0.0 )
   {
      cout << "HYPRE_ParCSRMLI_SetStrengthThreshold ERROR : reset to 0.\n";
      mli_object->strength_threshold = 0.0;
   } 
   else mli_object->strength_threshold = strength_threshold;
   return( 0 );
}

/****************************************************************************/
/* HYPRE_ParCSRMLI_SetMethod                                                  */
/*--------------------------------------------------------------------------*/

int HYPRE_ParCSRMLI_SetMethod( HYPRE_Solver solver, char *paramString )
{
   HYPRE_MLI *mli_object = (HYPRE_MLI *) solver;

   if ( ! strcmp( paramString, "AMGSA" ) )
      mli_object->method = MLI_METHOD_AMGSA_ID;
   else
   {
      cout << "HYPRE_ParCSRMLI_SetMethod ERROR : method unrecognized.\n";
      exit(1);
   }
   return( 0 );
}

/****************************************************************************/
/* HYPRE_ParCSRMLI_SetNumPDEs                                                 */
/*--------------------------------------------------------------------------*/

int HYPRE_ParCSRMLI_SetNumPDEs( HYPRE_Solver solver, int numPDE )
{
   HYPRE_MLI *mli_object = (HYPRE_MLI *) solver;

   if ( numPDE > 1 ) mli_object->num_PDEs = numPDE;
   else              mli_object->num_PDEs = 1;
   return( 0 );
}

/****************************************************************************/
/* HYPRE_ParCSRMLI_SetSmoother                                              */
/* smoother type : 0 (Jacobi)                                               */
/*                 1 (GS)                                                   */
/*                 2 (SGS)                                                  */
/*                 3 (ParaSails)                                            */
/*                 4 (Schwarz)                                              */
/*                 5 (MLS)                                                  */
/*--------------------------------------------------------------------------*/

int HYPRE_ParCSRMLI_SetSmoother( HYPRE_Solver solver, int pre_post,
                                 int smoother_type, int argc, char **argv  )
{
   int       i, nsweeps, stype;
   double    *relax_wgts;
   HYPRE_MLI *mli_object = (HYPRE_MLI *) solver;

   stype = smoother_type;
   if ( stype < 0 || stype > 5 )
   {
      printf("HYPRE_ParCSRMLI_SetSmoother WARNING : set to Jacobi.\n");
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
      case 0 : mli_object->pre_smoother = stype;
               mli_object->pre_nsweeps  = nsweeps;
               if ( mli_object->pre_relax_wts != NULL ) 
                  delete [] mli_object->pre_relax_wts;
               mli_object->pre_relax_wts = NULL;
               if ( argc > 1 )
               {
                  mli_object->pre_relax_wts = new double[nsweeps];
                  for ( i = 0; i < nsweeps; i++ )
                  {
                     if ( relax_wgts[i] > 0.0 && relax_wgts[i] < 2.0 )
                        mli_object->pre_relax_wts[i] = relax_wgts[i];
                     else
                        mli_object->pre_relax_wts[i] = 1.0;
                  } 
               } 
               break;

      case 1 : mli_object->postsmoother = stype;
               mli_object->postnsweeps  = nsweeps;
               if ( mli_object->postrelax_wts != NULL ) 
                  delete [] mli_object->postrelax_wts;
               mli_object->postrelax_wts = NULL;
               if ( argc > 1 )
               {
                  mli_object->postrelax_wts = new double[nsweeps];
                  for ( i = 0; i < nsweeps; i++ )
                  {
                     if ( relax_wgts[i] > 0.0 && relax_wgts[i] < 2.0 )
                        mli_object->postrelax_wts[i] = relax_wgts[i];
                     else
                        mli_object->postrelax_wts[i] = 1.0;
                  } 
              } 
               break;

      case 2 : mli_object->pre_smoother = stype;
               mli_object->postsmoother = stype;
               mli_object->pre_nsweeps  = nsweeps;
               mli_object->postnsweeps  = nsweeps;
               if ( mli_object->pre_relax_wts != NULL ) 
                  delete [] mli_object->pre_relax_wts;
               if ( mli_object->postrelax_wts != NULL ) 
                  delete [] mli_object->postrelax_wts;
               mli_object->pre_relax_wts = NULL;
               mli_object->postrelax_wts = NULL;
               if ( argc > 1 )
               {
                  mli_object->pre_relax_wts = new double[nsweeps];
                  mli_object->postrelax_wts = new double[nsweeps];
                  for ( i = 0; i < nsweeps; i++ )
                  {
                     if ( relax_wgts[i] > 0.0 && relax_wgts[i] < 2.0 )
                     {
                        mli_object->pre_relax_wts[i] = relax_wgts[i];
                        mli_object->postrelax_wts[i] = relax_wgts[i];
                     } 
                     else
                     {
                        mli_object->pre_relax_wts[i] = 1.0;
                        mli_object->postrelax_wts[i] = 1.0;
                     } 
                  } 
               } 
               break;
   }
   return( 0 );
}

/****************************************************************************/
/* HYPRE_ParCSRMLI_SetCoarseSolver                                          */
/* solver ID = 0  (superlu)                                                 */
/* solver ID = 1  (aggregation)                                             */
/*--------------------------------------------------------------------------*/

int HYPRE_ParCSRMLI_SetCoarseSolver( HYPRE_Solver solver, int solver_id, 
                                     int argc, char **argv )
{
   int       i, stype, nsweeps;
   HYPRE_MLI *mli_object = (HYPRE_MLI *) solver;

   stype = solver_id;
   if ( stype < 0 || stype > 8 )
   {
      cout << ("HYPRE_ParCSRMLI_SetCoarseSolver WARNING : set to Jacobi.\n");
      stype = 0;
   } 
   stype += MLI_SOLVER_JACOBI_ID;

   if ( argc > 0 ) nsweeps = *(int *) argv[0];
   else            nsweeps = 0;
   if ( nsweeps < 0 ) nsweeps = 1;
 
   mli_object->coarseSolver_        = stype;
   mli_object->CoarseSolverNSweeps_ = nsweeps;
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

