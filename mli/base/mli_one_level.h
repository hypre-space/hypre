/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * Header info each grid level
 *
 *****************************************************************************/

#ifndef __MLIONELEVELH__
#define __MLIONELEVELH__

#include "utilities.h"
#include "mli_onelevel.h"
#include "../solver/mli_solver.h"
#include "../amgs/mli_method.h"
#include "../matrix/mli_matrix.h"
#include "../vector/mli_vector.h"

typedef struct MLI_OneLevel_Struct MLI_OneLevel;

/*--------------------------------------------------------------------------
 * MLI data structure declaration
 *--------------------------------------------------------------------------*/

typedef struct MLI_OneLevel_Struct
{
   MLI            *mli;             /* pointer to the top level structure */
   MLI_Matrix     *Amat;            /* pointer to Amat                    */
   MLI_Matrix     *Rmat;            /* pointer to Rmat                    */
   MLI_Matrix     *Pmat;            /* pointer to Pmat                    */
   MLI_Solver     *pre_smoother;    /* pointer to pre-smoother            */
   MLI_Solver     *postsmoother;    /* pointer to postsmoother            */
   MLI_OneLevel   *next_level;      /* point to next coarse level         */
   MLI_OneLevel   *prev_level;      /* point to next coarse level         */
};

/*--------------------------------------------------------------------------
 * functions for MLI 
 *--------------------------------------------------------------------------*/

#ifdef __cplusplus
extern "C"
{
#endif

#ifdef __cplusplus
extern "C" {
#endif

extern MLI_OneLevel *MLI_OneLevelCreate( MLI *mli );
extern int MLI_OneLevelDestroy( MLI_OneLevel *object );
extern int MLI_OneLevelSetAmat( MLI_OneLevel *object, MLI_Matrix *Amat);
extern int MLI_OneLevelSetRmat( MLI_OneLevel *object, MLI_Matrix *Rmat);
extern int MLI_OneLevelSetPmat( MLI_OneLevel *object, MLI_Matrix *Pmat);
extern int MLI_OneLevelSetSmoother( MLI_OneLevel *object, int pre_post, 
                                    MLI_Solver *solver );

#ifdef __cplusplus
}
#endif

#endif

