
#ifndef _DISTRIBUTED_MATRIX_PILUT_SOLVER_HEADER
#define _DISTRIBUTED_MATRIX_PILUT_SOLVER_HEADER


#include "../../utilities/general.h"
#include "../../utilities/memory.h"
#ifdef HYPRE_DEBUG
#include <gmalloc.h>
#endif

#include "HYPRE.h"

#include "./macros.h" /*contains some macros that are used here */

/*--------------------------------------------------------------------------
 * Global variables for the pilut solver
 *--------------------------------------------------------------------------*/

typedef struct
{
MPI_Comm MPI_communicator;
int mype, npes;
double _secpertick;
int Mfactor;
int *jr, *jw, lastjr, *lr, lastlr;	/* Work space */
double *w;				/* Work space */
int firstrow, lastrow;			/* Matrix distribution parameters */
timer SerTmr, ParTmr;
int nrows, lnrows, ndone, ntogo, nleft; /* Various values used throught out */
int maxnz;
int *map;			        /* Map used for marking rows in the set */

int *vrowdist;

/* Buffers for point to point communication */
int pilu_recv[MAX_NPES];
int pilu_send[MAX_NPES];
int lu_recv[MAX_NPES];

} hypre_PilutSolverGlobals;

/* DEFINES for global variables */
#define pilut_comm (globals->MPI_communicator)
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
#define map (globals->map)
#define vrowdist (globals->vrowdist)
#define pilu_recv (globals->pilu_recv)
#define pilu_send (globals->pilu_send)
#define lu_recv (globals->lu_recv)



#include "./const.h"

/* Include HYPRE library prototypes */
#include "HYPRE.h"

/* prototype definitions for BLAS calls that are used */
double SNRM2( int *, double *, int *);
double SDOT(int *, double *, int *, double *, int *);


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
  int                    gmaxnz;
  double                 tol;
  int                    max_its;

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
#include "./hypre_protos.h"
#include "./internal_protos.h"

#endif
