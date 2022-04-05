/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/****************************************************************************/ 
/* data structures for local matrix and ml object                           */
/*--------------------------------------------------------------------------*/

#ifndef _MLMAXWELL_
#define _MLMAXWELL_

/* #define HAVE_MLMAXWELL */

#ifdef HAVE_MLMAXWELL
#include "ml_include.h"
#endif

#include "HYPRE_MLMatrix.h"

typedef struct
{
    HYPRE_ML_Matrix *Amat;
    MPI_Comm comm;
    int globalEqns;
    int *partition;
}
MLMaxwell_Context;
    
typedef struct
{
    MPI_Comm comm;
#ifdef HAVE_MLMAXWELL
    ML       *ml_ee;
    ML       *ml_nn;
#endif
    int      nlevels;
    int      smoothP_flag;
    double   ag_threshold;
    void     *edge_smoother;
    void     *node_smoother;
    HYPRE_ParCSRMatrix hypreG;
    HYPRE_ParCSRMatrix hypreAnn;
#ifdef HAVE_MLMAXWELL
    ML_Aggregate *ml_ag;
    ML_Operator  *Annmat;
    ML_Operator  *Gmat;
    ML_Operator  *GTmat;
    ML_Operator  **Gmat_array;
    ML_Operator  **GTmat_array;

#endif
    MLMaxwell_Context *Aee_contxt;
    MLMaxwell_Context *G_contxt;
    MLMaxwell_Context *Ann_contxt;
    void              **node_args;
    void              **edge_args;
} 
MLMaxwell_Link;

#endif

