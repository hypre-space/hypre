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

#include "mli_onelevel.h"

/*--------------------------------------------------------------------------
 * constructor 
 *--------------------------------------------------------------------------*/

MLI_OneLevel *MLI_OneLevelCreate( MLI *mli )
{
   MLI_OneLevel *one_level;
   one_level = (MLI_OneLevel *) calloc( MLI_OneLevel, 1 );
   one_level->mli  = mli;
   one_level->Amat = NULL;
   one_level->Rmat = NULL;
   one_level->Pmat = NULL;
   one_level->pre_smoother = NULL;
   one_level->postsmoother = NULL;
   one_level->next_level   = NULL;
   one_level->prev_level   = NULL;
   return one_level;
}

/*--------------------------------------------------------------------------
 * destructor 
 *--------------------------------------------------------------------------*/

int MLI_OneLevelDestroy( MLI_OneLevel *one_level )
{
   MLI_MatDestroy( one_level->Amat );
   MLI_MatDestroy( one_level->Rmat );
   MLI_MatDestroy( one_level->Pmat );
   MLI_SolverDestroy( one_level->pre_smoother );
   MLI_SolverDestroy( one_level->postsmoother );
   free( one_level );
   return 0;
}

/*--------------------------------------------------------------------------
 * set the A matrix  
 *--------------------------------------------------------------------------*/

int MLI_OneLevelSetAmat( MLI_OneLevel *one_level, MLI_Matrix *Amat)
{
   one_level->Amat = Amat;
   return 0;
}

/*--------------------------------------------------------------------------
 * set the R matrix  
 *--------------------------------------------------------------------------*/

int MLI_OneLevelSetRmat( MLI_OneLevel *one_level, MLI_Matrix *Rmat)
{
   one_level->Rmat = Rmat;
   return 0;
}

/*--------------------------------------------------------------------------
 * set the P matrix  
 *--------------------------------------------------------------------------*/

int MLI_OneLevelSetPmat( MLI_OneLevel *one_level, MLI_Matrix *Pmat)
{
   one_level->Pmat = Pmat;
   return 0;
}

/*--------------------------------------------------------------------------
 * set the smoother 
 *--------------------------------------------------------------------------*/

int MLI_OneLevelSetSmoother( MLI_OneLevel *one_level, int pre_post, 
                             MLI_Solver *smoother )
{
   if ( pre_post == MLI_SMOOTHER_PRE )
      one_level->pre_smoother = smoother;
   else if ( pre_post == MLI_SMOOTHER_POST )
      one_level->postsmoother = smoother;
   else if ( pre_post == MLI_SMOOTHER_BOTH )
   {
      one_level->pre_smoother = smoother;
      one_level->postsmoother = smoother;
   }
   return 0;
}
 
