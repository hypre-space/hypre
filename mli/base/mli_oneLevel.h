/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * Header info each MLI level
 *
 *****************************************************************************/

#ifndef __MLIONELEVELH__
#define __MLIONELEVELH__

#include "utilities/utilities.h"
#include "../solver/mli_solver.h"
#include "../amgs/mli_method.h"
#include "../matrix/mli_matrix.h"
#include "../vector/mli_vector.h"
#include "../fedata/mli_fedata.h"
#include "mli.h"

class MLI;

/*--------------------------------------------------------------------------
 * MLI data structure declaration
 *--------------------------------------------------------------------------*/

class MLI_OneLevel
{
   MLI            *mli_object;      /* pointer to the top level structure */
   MLI_FEData     *fedata;          /* pointer to finite element data     */
   MLI_Matrix     *Amat;            /* pointer to Amat                    */
   MLI_Matrix     *Rmat;            /* pointer to Rmat                    */
   MLI_Matrix     *Pmat;            /* pointer to Pmat                    */
   MLI_Solver     *pre_smoother;    /* pointer to pre-smoother            */
   MLI_Solver     *postsmoother;    /* pointer to postsmoother            */
   MLI_Solver     *coarse_solver;   /* pointer to coarse grid solver      */
   MLI_OneLevel   *next_level;      /* point to next coarse level         */
   MLI_OneLevel   *prev_level;      /* point to next coarse level         */
   MLI_Vector     *vec_sol;         /* pointer to solution vector         */
   MLI_Vector     *vec_rhs;         /* pointer to right hand side vector  */
   MLI_Vector     *vec_res;         /* pointer to residual vector         */
   int            ncycles;          /* V, W, or others                    */
   int            level_num;        /* level number                       */

public :

   MLI_OneLevel( MLI *mli );
   ~MLI_OneLevel();
   int  setCycles( int cycles )              { ncycles = cycles; return 0; }
   int  setLevelNum( int num )               { level_num = num; return 0; }
   int  setAmat( MLI_Matrix *A );
   int  setRmat( MLI_Matrix *R );
   int  setPmat( MLI_Matrix *P );
   int  setSolutionVector( MLI_Vector *sol );
   int  setRHSVector( MLI_Vector *rhs );
   int  setResidualVector( MLI_Vector *res );
   int  setSmoother( int pre_post, MLI_Solver *solver );
   int  setCoarseSolve( MLI_Solver *solver );
   int  setFEData( MLI_FEData *data );
   int  setNextLevel( MLI_OneLevel *next )   { next_level = next; return 0; }
   int  setPrevLevel( MLI_OneLevel *prev )   { prev_level = prev; return 0; }
   int  setup();
   int  solve1Cycle();
   int  resetAmat();
   MLI_Matrix *getAmat()                     { return Amat; }
   MLI_Matrix *getPmat()                     { return Pmat; }
   MLI_Vector *getRHSVector()                { return vec_rhs; }
   MLI_Vector *getResidualVector()           { return vec_res; }
   MLI_Vector *getSolutionVector()           { return vec_sol; }
};

#endif
