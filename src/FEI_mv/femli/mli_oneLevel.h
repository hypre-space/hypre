/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 * Header info each MLI level
 *****************************************************************************/

#ifndef __MLIONELEVELH__
#define __MLIONELEVELH__

#include "_hypre_utilities.h"
#include "mli.h"

class MLI;

/*--------------------------------------------------------------------------
 * MLI data structure declaration
 *--------------------------------------------------------------------------*/

class MLI_OneLevel
{
   MLI            *mliObject_;      /* pointer to the top level structure */
   MLI_FEData     *fedata_;         /* pointer to finite element data     */
   MLI_SFEI       *sfei_;           /* pointer to finite element data     */
   MLI_Mapper     *nodeEqnMap_;     /* pointer to node to equation map    */
   MLI_Matrix     *Amat_;           /* pointer to Amat                    */
   MLI_Matrix     *Rmat_;           /* pointer to Rmat                    */
   MLI_Matrix     *Pmat_;           /* pointer to Pmat                    */
   MLI_Solver     *preSmoother_;    /* pointer to pre-smoother            */
   MLI_Solver     *postSmoother_;   /* pointer to postsmoother            */
   MLI_Solver     *coarseSolver_;   /* pointer to coarse grid solver      */
   MLI_OneLevel   *nextLevel_;      /* point to next coarse level         */
   MLI_OneLevel   *prevLevel_;      /* point to next coarse level         */
   MLI_Vector     *vecSol_;         /* pointer to solution vector         */
   MLI_Vector     *vecRhs_;         /* pointer to right hand side vector  */
   MLI_Vector     *vecRes_;         /* pointer to residual vector         */
   int            ncycles_;         /* V, W, or others                    */
   int            levelNum_;        /* level number                       */

public :

   MLI_OneLevel( MLI *mli );
   ~MLI_OneLevel();
   int  setCycles( int cycles )     { ncycles_ = cycles; return 0; }
   int  setLevelNum( int num )      { levelNum_ = num; return 0; }
   int  setAmat( MLI_Matrix *A );
   int  setRmat( MLI_Matrix *R );
   int  setPmat( MLI_Matrix *P );
   int  setSolutionVector( MLI_Vector *sol );
   int  setRHSVector( MLI_Vector *rhs );
   int  setResidualVector( MLI_Vector *res );
   int  setSmoother( int pre_post, MLI_Solver *solver );
   int  setCoarseSolve( MLI_Solver *solver );
   int  setFEData( MLI_FEData *data, MLI_Mapper *map );
   int  setSFEI( MLI_SFEI *data );
   int  setNextLevel( MLI_OneLevel *next )   {nextLevel_ = next; return 0;}
   int  setPrevLevel( MLI_OneLevel *prev )   {prevLevel_ = prev; return 0;}
   int  setup();
   int  solve1Cycle();
   int  resetAmat();
   int  resetSolutionVector();
   int  resetRHSVector();
   MLI_Matrix *getAmat()                     { return Amat_; }
   MLI_Matrix *getPmat()                     { return Pmat_; }
   MLI_Matrix *getRmat()                     { return Rmat_; }
   MLI_Solver *getPreSmoother()              { return preSmoother_; }     
   MLI_Solver *getPostSmoother()             { return postSmoother_; }     
   MLI_Vector *getRHSVector()                { return vecRhs_; }
   MLI_Vector *getResidualVector()           { return vecRes_; }
   MLI_Vector *getSolutionVector()           { return vecSol_; }
   MLI_FEData *getFEData()                   { return fedata_; }
   MLI_SFEI   *getSFEI()                     { return sfei_; }
   MLI_Mapper *getNodeEqnMap()               { return nodeEqnMap_; }
};

#endif

