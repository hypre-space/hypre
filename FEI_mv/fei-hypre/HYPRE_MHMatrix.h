/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

/****************************************************************************/ 
/* data structures  for local matrix                                        */
/*--------------------------------------------------------------------------*/

#ifndef _MHMAT_
#define _MHMAT_

#ifdef MLPACK
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
#ifdef MLPACK
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
#ifdef MLPACK
    ML_Aggregate *ml_ag;
    ML_AMG       *ml_amg;
#endif
    MH_Context   *contxt;
} 
MH_Link;

#endif
