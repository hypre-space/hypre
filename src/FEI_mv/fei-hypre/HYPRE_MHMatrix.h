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
/* data structures  for local matrix                                        */
/*--------------------------------------------------------------------------*/

#ifndef _MHMAT_
#define _MHMAT_

#ifdef HAVE_ML
#include "ml_struct.h"
#include "ml_aggregate.h"
#include "ml_amg.h"
#endif

typedef struct
{
    int      Nrows;
    int      *rowptr;
    int      *colnum;
    int      *map;
    double   *values;
    int      sendProcCnt;
    int      *sendProc;
    int      *sendLeng;
    int      **sendList;
    int      recvProcCnt;
    int      *recvProc;
    int      *recvLeng;
}
MH_Matrix;

typedef struct
{
    MH_Matrix   *Amat;
    MPI_Comm    comm;
    int         globalEqns;
    int         *partition;
}
MH_Context;
    
typedef struct
{
    MPI_Comm     comm;
#ifdef HAVE_ML
    ML           *ml_ptr;
#endif
    int          nlevels;
    int          method;
    int          num_PDEs;
    int          pre, post;
    int          pre_sweeps, post_sweeps;
    int          BGS_blocksize;
    double       jacobi_wt;
    double       ag_threshold;
    int          coarse_solver;
    int          coarsen_scheme;
#ifdef HAVE_ML
    ML_Aggregate *ml_ag;
    ML_AMG       *ml_amg;
#endif
    MH_Context   *contxt;
} 
MH_Link;

#endif
