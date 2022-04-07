/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * functions for each grid level
 *
 *****************************************************************************/

#include <string.h>
#include "_hypre_utilities.h"
#include "mli_utils.h"
#include "mli_oneLevel.h"

/*****************************************************************************
 * constructor 
 *--------------------------------------------------------------------------*/

MLI_OneLevel::MLI_OneLevel( MLI *mli )
{
#ifdef MLI_DEBUG_DETAILED
   printf("MLI_OneLevel::MLI_OneLevel\n");
#endif
   mliObject_    = mli;
   levelNum_     = -1;
   fedata_       = NULL;
   sfei_         = NULL;
   nodeEqnMap_   = NULL;
   Amat_         = NULL;
   Rmat_         = NULL;
   Pmat_         = NULL;
   vecSol_       = NULL;
   vecRhs_       = NULL;
   vecRes_       = NULL;
   preSmoother_  = NULL;
   postSmoother_ = NULL;
   coarseSolver_ = NULL;
   nextLevel_    = NULL;
   prevLevel_    = NULL;
   ncycles_      = 1;
}

/*****************************************************************************
 * destructor 
 *--------------------------------------------------------------------------*/

MLI_OneLevel::~MLI_OneLevel()
{
#ifdef MLI_DEBUG_DETAILED
   printf("MLI_OneLevel::~MLI_OneLevel\n");
#endif
   if ( Amat_   != NULL ) delete Amat_;
   if ( Rmat_   != NULL ) delete Rmat_;
   if ( Pmat_   != NULL ) delete Pmat_;
   if ( vecSol_ != NULL ) delete vecSol_;
   if ( vecRhs_ != NULL ) delete vecRhs_;
   if ( vecRes_ != NULL ) delete vecRes_;
   if ( preSmoother_  == postSmoother_ ) postSmoother_ = NULL; 
   if ( preSmoother_  != NULL ) delete preSmoother_;
   if ( postSmoother_ != NULL ) delete postSmoother_;
   if ( coarseSolver_ != NULL ) delete coarseSolver_;
}

/*****************************************************************************
 * set A matrix 
 *--------------------------------------------------------------------------*/

int MLI_OneLevel::setAmat( MLI_Matrix *A )
{
#ifdef MLI_DEBUG_DETAILED
   printf("MLI_OneLevel::setAmat\n");
#endif
   if ( Amat_ != NULL ) delete Amat_;
   Amat_ = A;
   return 0;
}

/*****************************************************************************
 * set R matrix 
 *--------------------------------------------------------------------------*/

int MLI_OneLevel::setRmat( MLI_Matrix *R )
{
#ifdef MLI_DEBUG_DETAILED
   printf("MLI_OneLevel::setRmat at level %d\n", levelNum_);
#endif
   if ( Rmat_ != NULL ) delete Rmat_;
   Rmat_ = R;
   return 0;
}

/*****************************************************************************
 * set P matrix 
 *--------------------------------------------------------------------------*/

int MLI_OneLevel::setPmat( MLI_Matrix *P )
{
#ifdef MLI_DEBUG_DETAILED
   printf("MLI_OneLevel::setPmat at level %d\n", levelNum_);
#endif
   if ( Pmat_ != NULL ) delete Pmat_;
   Pmat_ = P;
   return 0;
}

/*****************************************************************************
 * set solution vector
 *--------------------------------------------------------------------------*/

int MLI_OneLevel::setSolutionVector( MLI_Vector *sol )
{
#ifdef MLI_DEBUG_DETAILED
   printf("MLI_OneLevel::setSolutionVector\n");
#endif
   if ( vecSol_ != NULL ) delete vecSol_;
   vecSol_ = sol;
   return 0;
}

/*****************************************************************************
 * set right hand side vector
 *--------------------------------------------------------------------------*/

int MLI_OneLevel::setRHSVector( MLI_Vector *rhs )
{
#ifdef MLI_DEBUG_DETAILED
   printf("MLI_OneLevel::setRHSVector\n");
#endif
   if ( vecRhs_ != NULL ) delete vecRhs_;
   vecRhs_ = rhs;
   return 0;
}

/*****************************************************************************
 * set residual vector
 *--------------------------------------------------------------------------*/

int MLI_OneLevel::setResidualVector( MLI_Vector *res )
{
#ifdef MLI_DEBUG_DETAILED
   printf("MLI_OneLevel::setResidualVector\n");
#endif
   if ( vecRes_ != NULL ) delete vecRes_;
   vecRes_ = res;
   return 0;
}

/*****************************************************************************
 * set the smoother 
 *--------------------------------------------------------------------------*/

int MLI_OneLevel::setSmoother( int pre_post, MLI_Solver *smoother )
{
#ifdef MLI_DEBUG_DETAILED
   printf("MLI_OneLevel::setSmoother, pre_post = %d\n", pre_post);
#endif
   if      ( pre_post == MLI_SMOOTHER_PRE  ) preSmoother_  = smoother;
   else if ( pre_post == MLI_SMOOTHER_POST ) postSmoother_ = smoother;
   else if ( pre_post == MLI_SMOOTHER_BOTH )
   {
      preSmoother_  = smoother;
      postSmoother_ = smoother;
   }
   return 0;
}

/*****************************************************************************
 * set the coarse solver 
 *--------------------------------------------------------------------------*/

int MLI_OneLevel::setCoarseSolve( MLI_Solver *solver )
{
#ifdef MLI_DEBUG_DETAILED
   printf("MLI_OneLevel::setCoarseSolve\n");
#endif
   if ( coarseSolver_ != NULL ) delete coarseSolver_;
   coarseSolver_ = solver;
   return 0;
}

/*****************************************************************************
 * set finite element information object 
 *--------------------------------------------------------------------------*/

int MLI_OneLevel::setFEData( MLI_FEData *data, MLI_Mapper *map )
{
#ifdef MLI_DEBUG_DETAILED
   printf("MLI_OneLevel::setFEData\n");
#endif
   fedata_ = data;
   if ( nodeEqnMap_ != NULL ) delete nodeEqnMap_;
   nodeEqnMap_ = map;
   return 0;
}

/*****************************************************************************
 * set finite element information object 
 *--------------------------------------------------------------------------*/

int MLI_OneLevel::setSFEI( MLI_SFEI *data )
{
#ifdef MLI_DEBUG_DETAILED
   printf("MLI_OneLevel::setSFEI\n");
#endif
   sfei_ = data;
   return 0;
}

/*****************************************************************************
 * setup 
 *--------------------------------------------------------------------------*/

int MLI_OneLevel::setup()
{
#ifdef MLI_DEBUG_DETAILED
   printf("MLI_OneLevel::setup at level %d\n", levelNum_);
#endif
   if ( Amat_ == NULL )
   {
      printf("MLI_OneLevel::setup at level %d\n", levelNum_);
      exit(1);
   } 
   if ( levelNum_ != 0 && Pmat_ == NULL )
   {
      printf("MLI_OneLevel::setup at level %d - no Pmat\n", levelNum_);
      exit(1);
   } 
   if ( !strcmp(Amat_->getName(),"HYPRE_ParCSR") && 
        !strcmp(Amat_->getName(),"HYPRE_ParCSRT"))
   {
      printf("MLI_OneLevel::setup ERROR : Amat not HYPRE_ParCSR.\n");
      exit(1);
   }
   if ( vecRes_ != NULL ) delete vecRes_;
   vecRes_ = Amat_->createVector();
   if ( levelNum_ > 0 )
   {
      if ( levelNum_ > 0 && vecRhs_ != NULL ) delete vecRhs_;
      if ( levelNum_ > 0 && vecSol_ != NULL ) delete vecSol_;
      vecSol_ = vecRes_->clone();
      vecRhs_ = vecRes_->clone();
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
   printf("MLI_OneLevel::solve1Cycle\n");
#endif

   sol = vecSol_;
   rhs = vecRhs_;
   res = vecRes_;

   if ( Rmat_ == NULL )
   {
      /* ---------------------------------------------------------------- */
      /* coarsest level - perform coarse solve                            */
      /* ---------------------------------------------------------------- */

      if ( coarseSolver_ != NULL ) 
      {
#ifdef MLI_DEBUG_DETAILED
         printf("MLI_OneLevel::solve1Cycle - coarse solve at level %d\n",
                levelNum_);
#endif
         coarseSolver_->solve( rhs, sol );
      }
      else 
      {
         if      (preSmoother_  != NULL) preSmoother_->solve(rhs, sol);
         else if (postSmoother_ != NULL) postSmoother_->solve(rhs, sol);
         else                            rhs->copy(sol);
      }
      return 0;
   }
   else
   {
      for ( i = 0; i < ncycles_; i++ )
      {
         /* ------------------------------------------------------------- */
         /* smooth and compute residual                                   */
         /* ------------------------------------------------------------- */

         if ( preSmoother_ != NULL ) 
         {
#ifdef MLI_DEBUG_DETAILED
         printf("MLI_OneLevel::solve1Cycle - presmoothing at level %d\n",
                levelNum_);
#endif
            preSmoother_->solve( rhs, sol );
         }

         Amat_->apply( -1.0, sol, 1.0, rhs, res );
 
         /* ------------------------------------------------------------- */
         /* transfer to coarse level                                      */
         /* ------------------------------------------------------------- */

#ifdef MLI_DEBUG_DETAILED
         printf("MLI_OneLevel::solve1Cycle - restriction to level %d\n",
                levelNum_+1);
#endif
         Rmat_->apply(1.0, res, 0.0, NULL, nextLevel_->vecRhs_);
         nextLevel_->vecSol_->setConstantValue(0.0e0);
         nextLevel_->solve1Cycle();

         /* ------------------------------------------------------------- */
         /* transfer solution back to fine level                          */
         /* ------------------------------------------------------------- */

#ifdef MLI_DEBUG_DETAILED
         printf("MLI_OneLevel::solve1Cycle - interpolate to level %d\n",
                levelNum_);
#endif
         nextLevel_->Pmat_->apply(1.0, nextLevel_->vecSol_, 1.0, sol, sol);

         /* ------------------------------------------------------------- */
         /* postsmoothing                                                 */
         /* ------------------------------------------------------------- */

         if ( postSmoother_ != NULL ) 
         {
            postSmoother_->solve( rhs, sol );
#ifdef MLI_DEBUG_DETAILED
            printf("MLI_OneLevel::solve1Cycle - postsmoothing at level %d\n",
                   levelNum_);
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
   printf("MLI_OneLevel::resetAmat\n");
#endif
   Amat_ = NULL;
   return 0;
}

/*****************************************************************************
 * wipe out solution vector for this level (but not destroy it)
 *--------------------------------------------------------------------------*/

int MLI_OneLevel::resetSolutionVector()
{
#ifdef MLI_DEBUG_DETAILED
   printf("MLI_OneLevel::resetSolutionVector\n");
#endif
   vecSol_ = NULL;
   return 0;
}

/*****************************************************************************
 * wipe out rhs vector for this level (but not destroy it)
 *--------------------------------------------------------------------------*/

int MLI_OneLevel::resetRHSVector()
{
#ifdef MLI_DEBUG_DETAILED
   printf("MLI_OneLevel::resetRHSVector\n");
#endif
   vecRhs_ = NULL;
   return 0;
}

