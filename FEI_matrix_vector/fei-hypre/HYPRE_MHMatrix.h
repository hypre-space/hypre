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
    ML           *ml_ptr;
    int          nlevels;
    int          pre, post;
    int          pre_sweeps, post_sweeps;
    int          BGS_blocksize;
    double       jacobi_wt;
    double       ag_threshold;
    ML_Aggregate *ml_ag;
    MH_Context   *contxt;
} 
MH_Link;

#endif
