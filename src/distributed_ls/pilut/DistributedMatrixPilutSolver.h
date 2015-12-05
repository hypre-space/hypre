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





#ifndef _DISTRIBUTED_MATRIX_PILUT_SOLVER_HEADER
#define _DISTRIBUTED_MATRIX_PILUT_SOLVER_HEADER

#include "HYPRE_config.h"
#include "general.h"
#include "_hypre_utilities.h"
/*
#ifdef HYPRE_DEBUG
#include <gmalloc.h>
#endif
*/

#include "HYPRE.h"

#include "HYPRE_DistributedMatrixPilutSolver_types.h"

#include "HYPRE_distributed_matrix_types.h"
#include "HYPRE_distributed_matrix_protos.h"

#include "macros.h" /*contains some macros that are used here */

/*--------------------------------------------------------------------------
 * Global variables for the pilut solver
 *--------------------------------------------------------------------------*/

typedef struct
{
MPI_Comm hypre_MPI_communicator;
HYPRE_Int mype, npes;
double _secpertick;
HYPRE_Int Mfactor;
HYPRE_Int *jr, *jw, lastjr, *lr, lastlr;	/* Work space */
double *w;				/* Work space */
HYPRE_Int firstrow, lastrow;			/* Matrix distribution parameters */
timer SerTmr, ParTmr;
HYPRE_Int nrows, lnrows, ndone, ntogo, nleft; /* Various values used throught out */
HYPRE_Int maxnz;
HYPRE_Int *map;			        /* Map used for marking rows in the set */

HYPRE_Int *vrowdist;

/* Buffers for point to point communication */
HYPRE_Int pilu_recv[MAX_NPES];
HYPRE_Int pilu_send[MAX_NPES];
HYPRE_Int lu_recv[MAX_NPES];

#ifdef HYPRE_TIMING
  /* factorization */
HYPRE_Int CCI_timer;
HYPRE_Int SS_timer;
HYPRE_Int SFR_timer;
HYPRE_Int CR_timer;
HYPRE_Int FL_timer;
HYPRE_Int SLUD_timer;
HYPRE_Int SLUM_timer;
HYPRE_Int UL_timer;
HYPRE_Int FNR_timer;
HYPRE_Int SDSeptimer;
HYPRE_Int SDKeeptimer;
HYPRE_Int SDUSeptimer;
HYPRE_Int SDUKeeptimer;

  /* solves */
HYPRE_Int Ll_timer;
HYPRE_Int Lp_timer;
HYPRE_Int Up_timer;
HYPRE_Int Ul_timer;
#endif

} hypre_PilutSolverGlobals;

/* DEFINES for global variables */
#define pilut_comm (globals->hypre_MPI_communicator)
#define mype (globals->mype)
#define npes (globals->npes)
#define _secpertick (globals->_secpertick)
#define Mfactor (globals->Mfactor)
#define jr (globals->jr)
#define jw (globals->jw)
#define lastjr (globals->lastjr)
#define lr (globals->lr)
#define lastlr (globals->lastlr)
#define w (globals->w)
#define firstrow (globals->firstrow)
#define lastrow (globals->lastrow)
#define SerTmr (globals->SerTmr)
#define ParTmr (globals->ParTmr)
#define nrows (globals->nrows)
#define lnrows (globals->lnrows)
#define ndone (globals->ndone)
#define ntogo (globals->ntogo)
#define nleft (globals->nleft)
#define global_maxnz (globals->maxnz)
#define pilut_map (globals->map)
#define vrowdist (globals->vrowdist)
#define pilu_recv (globals->pilu_recv)
#define pilu_send (globals->pilu_send)
#define lu_recv (globals->lu_recv)



#include "./const.h"


/* prototype definitions for BLAS calls that are used */
double SNRM2( HYPRE_Int *, double *, HYPRE_Int *);
double SDOT(HYPRE_Int *, double *, HYPRE_Int *, double *, HYPRE_Int *);


/*--------------------------------------------------------------------------
 * pilut structures
 *--------------------------------------------------------------------------*/

#include "./struct.h"

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixPilutSolver
 *--------------------------------------------------------------------------*/

typedef struct
{

  /* Input parameters */
  MPI_Comm               comm;
  HYPRE_DistributedMatrix  Matrix;
  HYPRE_Int                    gmaxnz;
  double                 tol;
  HYPRE_Int                    max_its;

  /* Structure that is used internally and built from matrix */
  DataDistType          *DataDist;

  /* Data that is passed from the factor to the solve */
  FactorMatType         *FactorMat;

  hypre_PilutSolverGlobals *globals;

} hypre_DistributedMatrixPilutSolver;

/*--------------------------------------------------------------------------
 * Accessor functions for the hypre_DistributedMatrixPilutSolver structure
 *--------------------------------------------------------------------------*/

#define hypre_DistributedMatrixPilutSolverComm(solver)            ((solver) -> comm)
#define hypre_DistributedMatrixPilutSolverDataDist(solver)        ((solver) -> DataDist)
#define hypre_DistributedMatrixPilutSolverMatrix(solver)          ((solver) -> Matrix)
#define hypre_DistributedMatrixPilutSolverGmaxnz(solver)          ((solver) -> gmaxnz)
#define hypre_DistributedMatrixPilutSolverTol(solver)             ((solver) -> tol)
#define hypre_DistributedMatrixPilutSolverMaxIts(solver)          ((solver) -> max_its)
#define hypre_DistributedMatrixPilutSolverFactorMat(solver)       ((solver) -> FactorMat)
#define hypre_DistributedMatrixPilutSolverGlobals(solver)         ((solver) -> globals)

/* Include internal prototypes */
#include "./internal_protos.h"

#endif
