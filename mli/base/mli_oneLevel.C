/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * functions for each grid level
 *
 *****************************************************************************/

#include <string.h>
#include <iostream.h>
#include "utilities/utilities.h"
#include "util/mli_utils.h"
#include "base/mli_oneLevel.h"

/*****************************************************************************
 * constructor 
 *--------------------------------------------------------------------------*/

MLI_OneLevel::MLI_OneLevel( MLI *mli )
{
#ifdef MLI_DEBUG_DETAILED
   cout << "MLI_OneLevel::MLI_OneLevel" << endl;
   cout.flush();
#endif
   mli_object    = mli;
   level_num     = -1;
   fedata        = NULL;
   Amat          = NULL;
   Rmat          = NULL;
   Pmat          = NULL;
   vec_sol       = NULL;
   vec_rhs       = NULL;
   vec_res       = NULL;
   pre_smoother  = NULL;
   postsmoother  = NULL;
   coarse_solver = NULL;
   next_level    = NULL;
   prev_level    = NULL;
   ncycles       = 1;
}

/*****************************************************************************
 * destructor 
 *--------------------------------------------------------------------------*/

MLI_OneLevel::~MLI_OneLevel()
{
#ifdef MLI_DEBUG_DETAILED
   cout << "MLI_OneLevel::~MLI_OneLevel" << endl;
   cout.flush();
#endif
   if ( fedata != NULL ) delete fedata;
   if ( Amat   != NULL ) delete Amat;
   if ( Rmat   != NULL ) delete Rmat;
   if ( Pmat   != NULL ) delete Pmat;
   if ( vec_sol != NULL ) delete vec_sol;
   if ( vec_rhs != NULL ) delete vec_rhs;
   if ( vec_res != NULL ) delete vec_res;
   if ( pre_smoother == postsmoother ) postsmoother = NULL; 
   if ( pre_smoother != NULL ) delete pre_smoother;
   if ( postsmoother != NULL ) delete postsmoother;
   if ( coarse_solver != NULL ) delete coarse_solver;
}

/*****************************************************************************
 * set A matrix 
 *--------------------------------------------------------------------------*/

int MLI_OneLevel::setAmat( MLI_Matrix *A )
{
#ifdef MLI_DEBUG_DETAILED
   cout << "MLI_OneLevel::setAmat" << endl;
   cout.flush();
#endif
   if ( Amat != NULL ) delete Amat;
   Amat = A;
   return 0;
}

/*****************************************************************************
 * set R matrix 
 *--------------------------------------------------------------------------*/

int MLI_OneLevel::setRmat( MLI_Matrix *R )
{
#ifdef MLI_DEBUG_DETAILED
   cout << "MLI_OneLevel::setRmat at level " << level_num << endl;
   cout.flush();
#endif
   if ( Rmat != NULL ) delete Rmat;
   Rmat = R;
   return 0;
}

/*****************************************************************************
 * set P matrix 
 *--------------------------------------------------------------------------*/

int MLI_OneLevel::setPmat( MLI_Matrix *P )
{
#ifdef MLI_DEBUG_DETAILED
   cout << "MLI_OneLevel::setPmat at level " << level_num << endl;
   cout.flush();
#endif
   if ( Pmat != NULL ) delete Pmat;
   Pmat = P;
   return 0;
}

/*****************************************************************************
 * set solution vector
 *--------------------------------------------------------------------------*/

int MLI_OneLevel::setSolutionVector( MLI_Vector *sol )
{
#ifdef MLI_DEBUG_DETAILED
   cout << "MLI_OneLevel::setSolutionVector" << endl;
   cout.flush();
#endif
   if ( vec_sol != NULL ) delete vec_sol;
   vec_sol = sol;
   return 0;
}

/*****************************************************************************
 * set right hand side vector
 *--------------------------------------------------------------------------*/

int MLI_OneLevel::setRHSVector( MLI_Vector *rhs )
{
#ifdef MLI_DEBUG_DETAILED
   cout << "MLI_OneLevel::setRHSVector" << endl;
   cout.flush();
#endif
   if ( vec_rhs != NULL ) delete vec_rhs;
   vec_rhs = rhs;
   return 0;
}

/*****************************************************************************
 * set residual vector
 *--------------------------------------------------------------------------*/

int MLI_OneLevel::setResidualVector( MLI_Vector *res )
{
#ifdef MLI_DEBUG_DETAILED
   cout << "MLI_OneLevel::setResidualVector" << endl;
   cout.flush();
#endif
   if ( vec_res != NULL ) delete vec_res;
   vec_res = res;
   return 0;
}

/*****************************************************************************
 * set the smoother 
 *--------------------------------------------------------------------------*/

int MLI_OneLevel::setSmoother( int pre_post, MLI_Solver *smoother )
{
#ifdef MLI_DEBUG_DETAILED
   cout << "MLI_OneLevel::setSmoother, pre_post = " << pre_post << endl;
   cout.flush();
#endif
   if      ( pre_post == MLI_SMOOTHER_PRE  ) pre_smoother = smoother;
   else if ( pre_post == MLI_SMOOTHER_POST ) postsmoother = smoother;
   else if ( pre_post == MLI_SMOOTHER_BOTH )
   {
      pre_smoother = smoother;
      postsmoother = smoother;
   }
   return 0;
}

/*****************************************************************************
 * set the coarse solver 
 *--------------------------------------------------------------------------*/

int MLI_OneLevel::setCoarseSolve( MLI_Solver *solver )
{
#ifdef MLI_DEBUG_DETAILED
   cout << "MLI_OneLevel::setCoarseSolve" << endl;
   cout.flush();
#endif
   if ( coarse_solver != NULL ) delete coarse_solver;
   coarse_solver = solver;
   return 0;
}

/*****************************************************************************
 * set finite element information object 
 *--------------------------------------------------------------------------*/

int MLI_OneLevel::setFEData( MLI_FEData *data, MLI_Mapper *map )
{
#ifdef MLI_DEBUG_DETAILED
   cout << "MLI_OneLevel::setFEData" << endl;
   cout.flush();
#endif
   if ( fedata != NULL ) delete fedata;
   fedata = data;
   if ( nodeEqnMap != NULL ) delete nodeEqnMap;
   nodeEqnMap = map;
   return 0;
}

/*****************************************************************************
 * setup 
 *--------------------------------------------------------------------------*/

int MLI_OneLevel::setup()
{
#ifdef MLI_DEBUG_DETAILED
   cout << "MLI_OneLevel::setup at level " << level_num << endl;
   cout.flush();
#endif
   if ( Amat == NULL )
   {
      cout << "MLI_OneLevel::setup at level " << level_num << " - no Amat\n";
      exit(1);
   } 
   if ( level_num != 0 && Pmat == NULL )
   {
      cout << "MLI_OneLevel::setup at level " << level_num << " - no Pmat\n";
      exit(1);
   } 
   if ( !strcmp(Amat->getName(),"HYPRE_ParCSR") && 
        !strcmp(Amat->getName(),"HYPRE_ParCSRT"))
   {
      cout << "MLI_OneLevel::setup ERROR : Amat not HYPRE_ParCSR." << endl;
      exit(1);
   }
   if ( vec_res != NULL ) delete vec_res;
   vec_res = Amat->createVector();
   if ( level_num > 0 )
   {
      if ( level_num > 0 && vec_rhs != NULL ) delete vec_rhs;
      if ( level_num > 0 && vec_sol != NULL ) delete vec_sol;
      vec_sol = vec_res->clone();
      vec_rhs = vec_res->clone();
   }
   return 0;
}

/*****************************************************************************
 * perform one cycle
 *--------------------------------------------------------------------------*/

int MLI_OneLevel::solve1Cycle()
{
   int        i;
   MLI_Vector *sol, *rhs, *res;
#ifdef MLI_DEBUG_DETAILED
   cout << "MLI_OneLevel::solve1Cycle" << endl;
   cout.flush();
#endif

   sol = vec_sol;
   rhs = vec_rhs;
   res = vec_res;

   if ( Rmat == NULL )
   {
      /* ---------------------------------------------------------------- */
      /* coarsest level - perform coarse solve                            */
      /* ---------------------------------------------------------------- */

      if ( coarse_solver != NULL ) 
      {
#ifdef MLI_DEBUG_DETAILED
         cout << "MLI_OneLevel::solve1Cycle - coarse solve at level " 
              << level_num << endl;
         cout.flush();
#endif
         coarse_solver->solve( rhs, sol );
      }
      else 
      {
         if      ( pre_smoother != NULL ) pre_smoother->solve( rhs, sol );
         else if ( postsmoother != NULL ) postsmoother->solve( rhs, sol );
         else                             rhs->copy(sol);
      }
      return 0;
   }
   else
   {
      for ( i = 0; i < ncycles; i++ )
      {
         /* ------------------------------------------------------------- */
         /* smooth and compute residual                                   */
         /* ------------------------------------------------------------- */

         if ( pre_smoother != NULL ) 
         {
#ifdef MLI_DEBUG_DETAILED
         cout << "MLI_OneLevel::solve1Cycle - presmoothing at level "
              << level_num << endl;
         cout.flush();
#endif
            pre_smoother->solve( rhs, sol );
         }

         Amat->apply( -1.0, sol, 1.0, rhs, res );
 
         /* ------------------------------------------------------------- */
         /* transfer to coarse level                                      */
         /* ------------------------------------------------------------- */

#ifdef MLI_DEBUG_DETAILED
         cout << "MLI_OneLevel::solve1Cycle - restriction to level "
              << level_num+1 << endl;
         cout.flush();
#endif
         Rmat->apply(1.0, res, 0.0, NULL, next_level->vec_rhs);
         next_level->vec_sol->setConstantValue(0.0e0);
         next_level->solve1Cycle();

         /* ------------------------------------------------------------- */
         /* transfer solution back to fine level                          */
         /* ------------------------------------------------------------- */

#ifdef MLI_DEBUG_DETAILED
         cout << "MLI_OneLevel::solve1Cycle - interpolate to level "
              << level_num << endl;
         cout.flush();
#endif
         next_level->Pmat->apply(1.0, next_level->vec_sol, 1.0, sol, sol);

         /* ------------------------------------------------------------- */
         /* postsmoothing                                                 */
         /* ------------------------------------------------------------- */

         if ( postsmoother != NULL ) 
         {
            postsmoother->solve( rhs, sol );
#ifdef MLI_DEBUG_DETAILED
            cout << "MLI_OneLevel::solve1Cycle - postsmoothing at level "
                 << level_num << endl;
            cout.flush();
#endif
         }
      }
   }
   return 0;
}

/*****************************************************************************
 * wipe out Amatrix for this level (but not destroy it)
 *--------------------------------------------------------------------------*/

int MLI_OneLevel::resetAmat()
{
#ifdef MLI_DEBUG_DETAILED
   cout << "MLI_OneLevel::resetAmat" << endl;
   cout.flush();
#endif
   Amat = NULL;
   return 0;
}

