/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for the MLI_Solver data structure
 *
 *****************************************************************************/

#ifndef __MLI_SOLVER_H__
#define __MLI_SOLVER_H__

/*--------------------------------------------------------------------------
 * defines 
 *--------------------------------------------------------------------------*/

#define MLI_SOLVER_JACOBI_ID        301
#define MLI_SOLVER_BJACOBI_ID       302
#define MLI_SOLVER_GS_ID            303
#define MLI_SOLVER_SGS_ID           304
#define MLI_SOLVER_BSGS_ID          305
#define MLI_SOLVER_PARASAILS_ID     306
#define MLI_SOLVER_MLS_ID           307
#define MLI_SOLVER_CHEBYSHEV_ID     308
#define MLI_SOLVER_CG_ID            309
#define MLI_SOLVER_SUPERLU_ID       310
#define MLI_SOLVER_SEQSUPERLU_ID    311
#define MLI_SOLVER_ARPACKSUPERLU_ID 312
#define MLI_SOLVER_KARCMARZ_ID      313
#define MLI_SOLVER_GMRES_ID         314
#define MLI_SOLVER_MLI_ID           315
#define MLI_SOLVER_ILU_ID           316
#define MLI_SOLVER_AMG_ID           317

/*--------------------------------------------------------------------------
 * include files 
 *--------------------------------------------------------------------------*/

#include <string.h>
#include "_hypre_utilities.h"

#include "mli_matrix.h"
#include "mli_vector.h"

/*--------------------------------------------------------------------------
 * MLI_Solver data structure declaration
 *--------------------------------------------------------------------------*/

class MLI_Solver
{
   char solver_name[100];

public :

   MLI_Solver(char *name);
   virtual ~MLI_Solver();

   char*   getName();
   virtual int setup(MLI_Matrix *)=0;
   virtual int solve(MLI_Vector *, MLI_Vector *)=0;
   virtual int setParams(char *paramString,int argc,char **argv);
   virtual int getParams(char *paramString,int *argc,char **argv);
};

/*--------------------------------------------------------------------------
 * MLI_Solver functions 
 *--------------------------------------------------------------------------*/

extern MLI_Solver *MLI_Solver_CreateFromName(char *);

#endif

