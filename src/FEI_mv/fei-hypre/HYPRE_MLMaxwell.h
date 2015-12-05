/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.8 $
 ***********************************************************************EHEADER*/





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

