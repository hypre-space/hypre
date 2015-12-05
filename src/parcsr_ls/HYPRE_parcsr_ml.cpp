/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.5 $
 ***********************************************************************EHEADER*/




/****************************************************************************/ 
/* HYPRE_ParCSRML interface                                                 */
/*--------------------------------------------------------------------------*/
/*  local functions :
 * 
 *        MH_Irecv
 *        MH_Send
 *        MH_Wait
 *        MH_ExchBdry
 *        MH_MatVec
 *        MH_GetRow
 *        HYPRE_ParCSRMLCreate
 *        HYPRE_ParCSRMLDestroy
 *        HYPRE_ParCSRMLSetup
 *        HYPRE_ParCSRMLSolve
 *        HYPRE_ParCSRMLSetStrongThreshold
 *        HYPRE_ParCSRMLSetNumPreSmoothings
 *        HYPRE_ParCSRMLSetNumPostSmoothings
 *        HYPRE_ParCSRMLSetPreSmoother
 *        HYPRE_ParCSRMLSetPostSmoother
 *        HYPRE_ParCSRMLSetDampingFactor
 *        HYPRE_ParCSRMLConstructMHMatrix
 ****************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "./HYPRE_parcsr_ls.h"

#include "../utilities/utilities.h"
#include "../distributed_matrix/HYPRE_distributed_matrix_types.h"
#include "../distributed_matrix/HYPRE_distributed_matrix_protos.h"

#include "../matrix_matrix/HYPRE_matrix_matrix_protos.h"

#include "../seq_mv/vector.h"
#include "../parcsr_mv/par_vector.h"

#ifdef MLPACK

#include "../../ML/ml/src/ml_struct.h"
#include "../../ML/ml/src/ml_aggregate.h"

extern void qsort0(HYPRE_Int *, HYPRE_Int, HYPRE_Int);

/****************************************************************************/ 
/* local data structures                                                    */
/*--------------------------------------------------------------------------*/

typedef struct
{
    HYPRE_Int      Nrows;
    HYPRE_Int      *rowptr;
    HYPRE_Int      *colnum;
    HYPRE_Int      *map;
    double   *values;
    HYPRE_Int      sendProcCnt;
    HYPRE_Int      *sendProc;
    HYPRE_Int      *sendLeng;
    HYPRE_Int      **sendList;
    HYPRE_Int      recvProcCnt;
    HYPRE_Int      *recvProc;
    HYPRE_Int      *recvLeng;
}
MH_Matrix;

typedef struct
{
    MH_Matrix   *Amat;
    MPI_Comm    comm;
    HYPRE_Int         globalEqns;
    HYPRE_Int         *partition;
}
MH_Context;
    
typedef struct
{
    MPI_Comm     comm;
    ML           *ml_ptr;
    HYPRE_Int          nlevels;
    HYPRE_Int          pre, post;
    HYPRE_Int          pre_sweeps, post_sweeps;
    HYPRE_Int          BGS_blocksize;
    double       jacobi_wt;
    double       ag_threshold;
    ML_Aggregate *ml_ag;
    MH_Context   *contxt;
} 
MH_Link;

extern HYPRE_Int  HYPRE_ParCSRMLConstructMHMatrix(HYPRE_ParCSRMatrix, MH_Matrix *,
                                            MPI_Comm, HYPRE_Int *,MH_Context*); 

/****************************************************************************/ 
/* communication functions on parallel platforms                            */
/*--------------------------------------------------------------------------*/

HYPRE_Int MH_Irecv(void* buf, hypre_uint count, HYPRE_Int *src, HYPRE_Int *mid,
            MPI_Comm comm, hypre_MPI_Request *request )
{
   HYPRE_Int my_id, lsrc, retcode;

   if ( *src < 0 ) lsrc = hypre_MPI_ANY_SOURCE; else lsrc = (*src); 
   retcode = hypre_MPI_Irecv( buf, (HYPRE_Int) count, hypre_MPI_BYTE, lsrc, *mid, comm, request);
   if ( retcode != 0 )
   {
      hypre_MPI_Comm_rank(comm, &my_id);
      hypre_printf("%d : MH_Irecv warning : retcode = %d\n", my_id, retcode);
   }
   return 0;
}

HYPRE_Int MH_Wait(void* buf, hypre_uint count, HYPRE_Int *src, HYPRE_Int *mid,
            MPI_Comm comm, hypre_MPI_Request *request )
{
   hypre_MPI_Status status;
   HYPRE_Int        my_id, incount, retcode;

   retcode = hypre_MPI_Wait( request, &status );
   if ( retcode != 0 )
   {
      hypre_MPI_Comm_rank(comm, &my_id);
      hypre_printf("%d : MH_Wait warning : retcode = %d\n", my_id, retcode);
   }
   hypre_MPI_Get_count(&status, hypre_MPI_BYTE, &incount);
   if ( *src < 0 ) *src = status.hypre_MPI_SOURCE; 
   return incount;
}

HYPRE_Int MH_Send(void* buf, hypre_uint count, HYPRE_Int dest, HYPRE_Int mid, MPI_Comm comm )
{
   HYPRE_Int my_id;
   HYPRE_Int retcode = hypre_MPI_Send( buf, (HYPRE_Int) count, hypre_MPI_BYTE, dest, mid, comm);
   if ( retcode != 0 )
   {
      hypre_MPI_Comm_rank(comm, &my_id);
      hypre_printf("%d : MH_Send warning : retcode = %d\n", my_id, retcode);
   }
   return 0;
}

/****************************************************************************/ 
/* wrapper function for interprocessor communication for matvec and getrow  */
/*--------------------------------------------------------------------------*/

HYPRE_Int MH_ExchBdry(double *vec, void *obj)
{
   HYPRE_Int         i, j, msgid, leng, src, dest, offset, *tempList;
   double      *dbuf;
   MH_Context  *context;
   MH_Matrix   *Amat;
   MPI_Comm    comm;
   hypre_MPI_Request *request; 

   HYPRE_Int sendProcCnt, recvProcCnt;
   HYPRE_Int *sendProc, *recvProc;
   HYPRE_Int *sendLeng, *recvLeng;
   HYPRE_Int **sendList, nRows;

   context     = (MH_Context *) obj;
   Amat        = (MH_Matrix  *) context->Amat;
   comm        = context->comm;
   sendProcCnt = Amat->sendProcCnt;
   recvProcCnt = Amat->recvProcCnt;
   sendProc    = Amat->sendProc;
   recvProc    = Amat->recvProc;
   sendLeng    = Amat->sendLeng;
   recvLeng    = Amat->recvLeng;
   sendList    = Amat->sendList;
   nRows       = Amat->Nrows;

   request = (hypre_MPI_Request *) malloc( recvProcCnt * sizeof( hypre_MPI_Request ));
   msgid = 234;
   offset = nRows;
   for ( i = 0; i < recvProcCnt; i++ )
   {
      leng = recvLeng[i] * sizeof( double );
      src  = recvProc[i];
      MH_Irecv((void*) &(vec[offset]), leng, &src, &msgid, comm, &request[i]);
      offset += recvLeng[i];
   }
   msgid = 234;
   for ( i = 0; i < sendProcCnt; i++ )
   {
      dest = sendProc[i];
      leng = sendLeng[i] * sizeof( double );
      dbuf = (double *) malloc( leng * sizeof(double) );
      tempList = sendList[i];
      for ( j = 0; j < sendLeng[i]; j++ ) {
         dbuf[j] = vec[tempList[j]];
      }
      MH_Send((void*) dbuf, leng, dest, msgid, comm);
      if ( dbuf != NULL ) free( dbuf );
   }
   offset = nRows;
   for ( i = 0; i < recvProcCnt; i++ )
   {
      leng = recvLeng[i] * sizeof( double );
      src  = recvProc[i];
      MH_Wait((void*) &(vec[offset]), leng, &src, &msgid, comm, &request[i]);
      offset += recvLeng[i];
   }
   free ( request );
   return 1;
}

/****************************************************************************/ 
/* matvec function for local matrix structure MH_Matrix                     */
/*--------------------------------------------------------------------------*/

HYPRE_Int MH_MatVec(void *obj, HYPRE_Int leng1, double p[], HYPRE_Int leng2, double ap[])
{
    MH_Context *context;
    MPI_Comm   comm;
    MH_Matrix *Amat;

    HYPRE_Int    i, j, length, nRows, ibeg, iend, k;
    double *dbuf, sum;
    HYPRE_Int    *rowptr, *colnum;
    double *values;

    context = (MH_Context *) obj;
    comm    = context->comm;
    Amat    = (MH_Matrix*) context->Amat;
    nRows = Amat->Nrows;
    rowptr  = Amat->rowptr;
    colnum  = Amat->colnum;
    values  = Amat->values;

    length = nRows;
    for ( i = 0; i < Amat->recvProcCnt; i++ ) length += Amat->recvLeng[i];
    dbuf = (double *) malloc( length * sizeof( double ) );
    for ( i = 0; i < nRows; i++ ) dbuf[i] = p[i];
    MH_ExchBdry(dbuf, obj);
    for ( i = 0 ; i < nRows; i++ ) 
    {
       sum = 0.0;
       ibeg = rowptr[i];
       iend = rowptr[i+1];
       for ( j = ibeg; j < iend; j++ )
       { 
          k = colnum[j];
          sum += ( values[j] * dbuf[k] );
       }
       ap[i] = sum;
    }
    if ( dbuf != NULL ) free( dbuf );
    return 1;
}

/****************************************************************************/
/* getrow function for local matrix structure MH_Matrix (ML compatible)     */
/*--------------------------------------------------------------------------*/

HYPRE_Int MH_GetRow(void *obj, HYPRE_Int N_requested_rows, HYPRE_Int requested_rows[],
   HYPRE_Int allocated_space, HYPRE_Int columns[], double values[], HYPRE_Int row_lengths[])
{
    HYPRE_Int        i, j, ncnt, colindex, rowLeng, rowindex;
    MH_Context *context = (MH_Context *) obj;
    MH_Matrix *Amat     = (MH_Matrix*) context->Amat;
    HYPRE_Int    nRows        = Amat->Nrows;
    HYPRE_Int    *rowptr      = Amat->rowptr;
    HYPRE_Int    *colInd      = Amat->colnum;
    double *colVal      = Amat->values;
    HYPRE_Int    *mapList     = Amat->map;

    ncnt = 0;
    for ( i = 0; i < N_requested_rows; i++ )
    {
       rowindex = requested_rows[i];
       rowLeng = rowptr[rowindex+1] - rowptr[rowindex];
       if ( ncnt+rowLeng > allocated_space ) return 0;
       row_lengths[i] = rowLeng;
       colindex = rowptr[rowindex];
       for ( j = 0; j < rowLeng; j++ )
       {
          columns[ncnt] = colInd[colindex];
          values[ncnt++] = colVal[colindex++];
       }
    }
    return 1;
}

/****************************************************************************/
/* HYPRE_ParCSRMLCreate                                                     */
/*--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_ParCSRMLCreate( MPI_Comm comm, HYPRE_Solver *solver)
{
    /* create an internal ML data structure */

    MH_Link *link = (MH_Link *) malloc( sizeof( MH_Link ) );
    if ( link == NULL ) return 1;   

    /* fill in all other default parameters */

    link->comm         = comm;
    link->nlevels      = 20;   /* max number of levels */
    link->pre          = 0;    /* default is Gauss Seidel */
    link->post         = 0;    /* default is Gauss Seidel */
    link->pre_sweeps   = 2;    /* default is 2 smoothing steps */
    link->post_sweeps  = 2;
    link->BGS_blocksize = 3;
    link->jacobi_wt    = 0.5;  /* default damping factor */
    link->ml_ag        = NULL;
    link->ag_threshold = 0.18; /* threshold for aggregation */
    link->contxt       = NULL; /* context for matvec */
  
    /* create the ML structure */

    ML_Create( &(link->ml_ptr), link->nlevels );

    *solver = (HYPRE_Solver) link;

    return 0;
}

/****************************************************************************/
/* HYPRE_ParCSRMLDestroy                                                    */
/*--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_ParCSRMLDestroy( HYPRE_Solver solver )
{
    HYPRE_Int       i;
    MH_Matrix *Amat;
    MH_Link   *link = (MH_Link *) solver;

    if ( link->ml_ag  != NULL ) ML_Aggregate_Destroy( &(link->ml_ag) );
    ML_Destroy( &(link->ml_ptr) );
    if ( link->contxt->partition != NULL ) free( link->contxt->partition );
    if ( link->contxt->Amat != NULL )
    {
       Amat = (MH_Matrix *) link->contxt->Amat;
       if ( Amat->sendProc != NULL ) free (Amat->sendProc);
       if ( Amat->sendLeng != NULL ) free (Amat->sendLeng);
       if ( Amat->sendList != NULL ) 
       {
          for (i = 0; i < Amat->sendProcCnt; i++ )
             if (Amat->sendList[i] != NULL) free (Amat->sendList[i]);
          free (Amat->sendList);
       }
       if ( Amat->recvProc != NULL ) free (Amat->recvProc);
       if ( Amat->recvLeng != NULL ) free (Amat->recvLeng);
       if ( Amat->map      != NULL ) free (Amat->map);
       free( Amat );
    }
    if ( link->contxt != NULL ) free( link->contxt );
    free( link );

    return 0;
}

/****************************************************************************/
/* HYPRE_ParCSRMLSetup                                                      */
/*--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_ParCSRMLSetup( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                         HYPRE_ParVector b,   HYPRE_ParVector x      )
{
    HYPRE_Int        i, my_id, nprocs, coarsest_level, level, sweeps, nlevels;
    HYPRE_Int        *row_partition, localEqns, length;
    HYPRE_Int        Nblocks, *blockList;
    double     wght;
    MH_Context *context;
    MH_Matrix  *mh_mat;

    /* -------------------------------------------------------- */ 
    /* fetch the ML pointer                                     */
    /* -------------------------------------------------------- */ 

    MH_Link *link = (MH_Link *) solver;
    ML      *ml   = link->ml_ptr;
    nlevels       = link->nlevels;
   
    /* -------------------------------------------------------- */ 
    /* set up the parallel environment                          */
    /* -------------------------------------------------------- */ 

    hypre_MPI_Comm_rank(link->comm, &my_id);
    hypre_MPI_Comm_size(link->comm, &nprocs);

    /* -------------------------------------------------------- */ 
    /* fetch the matrix row partition information and put it    */
    /* into the matrix data object (for matvec and getrow)      */
    /* -------------------------------------------------------- */ 

    HYPRE_ParCSRMatrixGetRowPartitioning( A, &row_partition );
    localEqns  = row_partition[my_id+1] - row_partition[my_id];
    context = (MH_Context *) malloc(sizeof(MH_Context));
    link->contxt = context;
    context->comm = link->comm;
    context->globalEqns = row_partition[nprocs];
    context->partition = (HYPRE_Int *) malloc(sizeof(HYPRE_Int)*(nprocs+1));
    for (i=0; i<=nprocs; i++) context->partition[i] = row_partition[i];
    hypre_TFree( row_partition );
    mh_mat = ( MH_Matrix * ) malloc( sizeof( MH_Matrix) );
    context->Amat = mh_mat;
    HYPRE_ParCSRMLConstructMHMatrix(A,mh_mat,link->comm,
                                    context->partition,context); 

    /* -------------------------------------------------------- */ 
    /* set up the ML communicator information                   */
    /* -------------------------------------------------------- */ 

    ML_Set_Comm_Communicator(ml, link->comm);
    ML_Set_Comm_MyRank(ml, my_id);
    ML_Set_Comm_Nprocs(ml, nprocs);
    ML_Set_Comm_Send(ml, MH_Send);
    ML_Set_Comm_Recv(ml, MH_Irecv);
    ML_Set_Comm_Wait(ml, MH_Wait);

    /* -------------------------------------------------------- */ 
    /* set up the ML matrix information                         */
    /* -------------------------------------------------------- */ 

    ML_Init_Amatrix(ml, nlevels-1, localEqns, localEqns, (void *) context);
    ML_Set_Amatrix_Matvec(ml, nlevels-1, MH_MatVec);
    length = localEqns;
    for (i=0; i<mh_mat->recvProcCnt; i++ ) length += mh_mat->recvLeng[i];
    ML_Set_Amatrix_Getrow(ml, nlevels-1, MH_GetRow, MH_ExchBdry, length);

    /* -------------------------------------------------------- */ 
    /* create an aggregate context                              */
    /* -------------------------------------------------------- */ 

    ML_Aggregate_Create(&(link->ml_ag));
    link->ml_ag->max_levels = link->nlevels;
    ML_Aggregate_Set_Threshold( link->ml_ag, link->ag_threshold );

    /* -------------------------------------------------------- */ 
    /* perform aggregation                                      */
    /* -------------------------------------------------------- */ 

    coarsest_level = ML_Gen_MGHierarchy_UsingAggregation(ml, nlevels-1, 
                                        ML_DECREASING, link->ml_ag);
    if ( my_id == 0 )
       hypre_printf("ML : number of levels = %d\n", coarsest_level);

    coarsest_level = nlevels - coarsest_level;

    /* -------------------------------------------------------- */ 
    /* set up smoother and coarse solver                        */
    /* -------------------------------------------------------- */ 

    for (level = nlevels-1; level > coarsest_level; level--) 
    {
       sweeps = link->pre_sweeps;
       wght   = link->jacobi_wt;
       switch ( link->pre )
       {
          case 0 :
             ML_Gen_SmootherJacobi(ml, level, ML_PRESMOOTHER, sweeps, wght);
             break;
          case 1 :
             ML_Gen_SmootherGaussSeidel(ml, level, ML_PRESMOOTHER, sweeps);
             break;
          case 2 :
             ML_Gen_SmootherSymGaussSeidel(ml,level,ML_PRESMOOTHER,sweeps,1.0);
             break;
          case 3 :
             Nblocks = ML_Aggregate_Get_AggrCount( link->ml_ag, level );
             ML_Aggregate_Get_AggrMap( link->ml_ag, level, &blockList );
             ML_Gen_SmootherVBlockGaussSeidel(ml,level,ML_PRESMOOTHER,
                                              sweeps, Nblocks, blockList);
             break;
          case 4 :
             Nblocks = ML_Aggregate_Get_AggrCount( link->ml_ag, level );
             ML_Aggregate_Get_AggrMap( link->ml_ag, level, &blockList );
             ML_Gen_SmootherVBlockJacobi(ml,level,ML_PRESMOOTHER,
                                         sweeps, wght, Nblocks, blockList);
             break;
       }

       sweeps = link->post_sweeps;
       switch ( link->post )
       {
          case 0 :
             ML_Gen_SmootherJacobi(ml, level, ML_POSTSMOOTHER, sweeps, wght);
             break;
          case 1 :
             ML_Gen_SmootherGaussSeidel(ml, level, ML_POSTSMOOTHER, sweeps);
             break;
          case 2 :
             ML_Gen_SmootherSymGaussSeidel(ml,level,ML_POSTSMOOTHER,sweeps,1.0);
             break;
          case 3 :
             Nblocks = ML_Aggregate_Get_AggrCount( link->ml_ag, level );
             ML_Aggregate_Get_AggrMap( link->ml_ag, level, &blockList );
             ML_Gen_SmootherVBlockGaussSeidel(ml,level,ML_POSTSMOOTHER,
                                              sweeps, Nblocks, blockList);
             break;
          case 4 :
             Nblocks = ML_Aggregate_Get_AggrCount( link->ml_ag, level );
             ML_Aggregate_Get_AggrMap( link->ml_ag, level, &blockList );
             ML_Gen_SmootherVBlockJacobi(ml,level,ML_POSTSMOOTHER,
                                         sweeps, wght, Nblocks, blockList);
             break;
       }
    }

    ML_Gen_CoarseSolverSuperLU(ml, coarsest_level);
    //ML_Gen_SmootherGaussSeidel(ml, coarsest_level, ML_PRESMOOTHER, 100);

    ML_Gen_Solver(ml, ML_MGV, nlevels-1, coarsest_level);
   
    return 0;
}

/****************************************************************************/
/* HYPRE_ParCSRMLSolve                                                      */
/*--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_ParCSRMLSolve( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                         HYPRE_ParVector b,   HYPRE_ParVector x      )
{
    double  *rhs, *sol;
    MH_Link *link = (MH_Link *) solver;
    ML      *ml = link->ml_ptr;
    HYPRE_Int     leng, level = ml->ML_num_levels - 1;
    ML_Operator *Amat = &(ml->Amat[level]);
    ML_Krylov *ml_kry;

    rhs = hypre_VectorData(hypre_ParVectorLocalVector((hypre_ParVector *) b));
    sol = hypre_VectorData(hypre_ParVectorLocalVector((hypre_ParVector *) x));

    /*
    ml_kry = ML_Krylov_Create(ml->comm);
    ML_Krylov_Set_Method(ml_kry, 1);
    ML_Krylov_Set_Amatrix(ml_kry, Amat);
    ML_Krylov_Set_Precon(ml_kry, ml);
    ML_Krylov_Set_PreconFunc(ml_kry, ML_AMGVSolve_Wrapper);
    leng = Amat->outvec_leng;
    ML_Krylov_Solve(ml_kry, leng, rhs, sol);
    ML_Krylov_Destroy(&ml_kry);
    */
  
    ML_Solve_AMGV(ml, rhs, sol);
    //ML_Iterate(ml, sol, rhs);

    return 0;
}

/****************************************************************************/
/* HYPRE_ParCSRMLSetStrongThreshold                                         */
/*--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_ParCSRMLSetStrongThreshold(HYPRE_Solver solver,
                                     double strong_threshold)
{
    MH_Link *link = (MH_Link *) solver;
  
    if ( strong_threshold < 0.0 )
    {
       hypre_printf("HYPRE_ParCSRMLSetStrongThreshold error : reset to 0.\n");
       link->ag_threshold = 0.0;
    } 
    else
    {
       link->ag_threshold = strong_threshold;
    } 
    return( 0 );
}

/****************************************************************************/
/* HYPRE_ParCSRMLSetNumPreSmoothings                                        */
/*--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_ParCSRMLSetNumPreSmoothings( HYPRE_Solver solver, HYPRE_Int num_sweeps  )
{
    MH_Link *link = (MH_Link *) solver;

    if ( num_sweeps < 0 )
    {
       hypre_printf("HYPRE_ParCSRMLSetNumPreSmoothings error : reset to 0.\n");
       link->pre_sweeps = 0;
    } 
    else
    {
       link->pre_sweeps = num_sweeps;
    } 
    return( 0 );
}

/****************************************************************************/
/* HYPRE_ParMLSetNumPostSmoothings                                          */
/*--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_ParCSRMLSetNumPostSmoothings( HYPRE_Solver solver, HYPRE_Int num_sweeps  )
{
    MH_Link *link = (MH_Link *) solver;

    if ( num_sweeps < 0 )
    {
       hypre_printf("HYPRE_ParCSRMLSetNumPostSmoothings error : reset to 0.\n");
       link->post_sweeps = 0;
    } 
    else
    {
       link->post_sweeps = num_sweeps;
    } 
    return( 0 );
}

/****************************************************************************/
/* HYPRE_ParCSRMLSetPreSmoother                                             */
/*--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_ParCSRMLSetPreSmoother( HYPRE_Solver solver, HYPRE_Int smoother_type  )
{
    MH_Link *link = (MH_Link *) solver;

    if ( smoother_type < 0 || smoother_type > 4 )
    {
       hypre_printf("HYPRE_ParCSRMLSetPreSmoother error : set to Jacobi.\n");
       link->pre = 0;
    } 
    else
    {
       link->pre = smoother_type;
    } 
    return( 0 );
}

/****************************************************************************/
/* HYPRE_ParCSRMLSetPostSmoother                                            */
/*--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_ParCSRMLSetPostSmoother( HYPRE_Solver solver, HYPRE_Int smoother_type  )
{
    MH_Link *link = (MH_Link *) solver;

    if ( smoother_type < 0 || smoother_type > 4 )
    {
       hypre_printf("HYPRE_ParCSRMLSetPostSmoother error : set to Jacobi.\n");
       link->post = 0;
    } 
    else
    {
       link->post = smoother_type;
    } 
    return( 0 );
}

/****************************************************************************/
/* HYPRE_ParCSRMLSetDampingFactor                                           */
/*--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_ParCSRMLSetDampingFactor( HYPRE_Solver solver, double factor  )
{
    MH_Link *link = (MH_Link *) solver;

    if ( factor < 0.0 || factor > 1.0 )
    {
       hypre_printf("HYPRE_ParCSRMLSetDampingFactor error : set to 0.5.\n");
       link->jacobi_wt = 0.5;
    } 
    else
    {
       link->jacobi_wt = factor;
    } 
    return( 0 );
}

/****************************************************************************/
/* HYPRE_ParCSRMLSetDampingFactor                                           */
/*--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_ParCSRMLSetBGSBlockSize( HYPRE_Solver solver, HYPRE_Int size  )
{
    MH_Link *link = (MH_Link *) solver;

    if ( size < 0 )
    {
       hypre_printf("HYPRE_ParCSRMLSetBGSBlockSize error : reset to 1.\n");
       link->BGS_blocksize = 1;
    } 
    else
    {
       link->BGS_blocksize = size;
    } 
    return( 0 );
}

/****************************************************************************/
/* HYPRE_ParCSRMLConstructMHMatrix                                          *
/*--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_ParCSRMLConstructMHMatrix(HYPRE_ParCSRMatrix A, MH_Matrix *mh_mat,
                             MPI_Comm comm, HYPRE_Int *partition,MH_Context *obj) 
{
    HYPRE_Int         i, j, index, my_id, nprocs, msgid, *tempCnt;
    HYPRE_Int         sendProcCnt, *sendLeng, *sendProc, **sendList;
    HYPRE_Int         recvProcCnt, *recvLeng, *recvProc;
    HYPRE_Int         rowLeng, *colInd, startRow, endRow, localEqns;
    HYPRE_Int         *diagSize, *offdiagSize, externLeng, *externList, ncnt, nnz;
    HYPRE_Int         *rowptr, *columns, num_bdry;
    double      *colVal, *values;
    hypre_MPI_Request *Request;
    hypre_MPI_Status  status;

    /* -------------------------------------------------------- */
    /* get machine information and local matrix information     */
    /* -------------------------------------------------------- */
    
    hypre_MPI_Comm_rank(comm, &my_id);
    hypre_MPI_Comm_size(comm, &nprocs);

    startRow  = partition[my_id];
    endRow    = partition[my_id+1] - 1;
    localEqns = endRow - startRow + 1;

    /* -------------------------------------------------------- */
    /* probe A to find out about diagonal and off-diagonal      */
    /* block information                                        */
    /* -------------------------------------------------------- */

    diagSize    = (HYPRE_Int*) malloc( sizeof(HYPRE_Int) * localEqns );
    offdiagSize = (HYPRE_Int*) malloc( sizeof(HYPRE_Int) * localEqns );
    num_bdry = 0;
    for ( i = startRow; i <= endRow; i++ )
    {
       diagSize[i-startRow] = offdiagSize[i-startRow] = 0;
       HYPRE_ParCSRMatrixGetRow(A, i, &rowLeng, &colInd, &colVal);
       for (j = 0; j < rowLeng; j++)
          if ( colInd[j] < startRow || colInd[j] > endRow )
          {
             //if ( colVal[j] != 0.0 ) offdiagSize[i-startRow]++;
             offdiagSize[i-startRow]++;
          }
          else
          {
             //if ( colVal[j] != 0.0 ) diagSize[i-startRow]++;
             diagSize[i-startRow]++;
          }
       HYPRE_ParCSRMatrixRestoreRow(A, i, &rowLeng, &colInd, &colVal);
       if ( diagSize[i-startRow] + offdiagSize[i-startRow] == 1 ) num_bdry++;
    }

    /* -------------------------------------------------------- */
    /* construct external node list in global eqn numbers       */
    /* -------------------------------------------------------- */

    externLeng = 0;
    for ( i = 0; i < localEqns; i++ ) externLeng += offdiagSize[i];
    if ( externLeng > 0 )
         externList = (HYPRE_Int *) malloc( sizeof(HYPRE_Int) * externLeng);
    else externList = NULL;
    externLeng = 0;
    for ( i = startRow; i <= endRow; i++ )
    {
       HYPRE_ParCSRMatrixGetRow(A, i, &rowLeng, &colInd, &colVal);
       for (j = 0; j < rowLeng; j++)
       {
          if ( colInd[j] < startRow || colInd[j] > endRow )
             //if ( colVal[j] != 0.0 ) externList[externLeng++] = colInd[j];
             externList[externLeng++] = colInd[j];
       }
       HYPRE_ParCSRMatrixRestoreRow(A, i, &rowLeng, &colInd, &colVal);
    }
    qsort0( externList, 0, externLeng-1 );
    ncnt = 0;
    for ( i = 1; i < externLeng; i++ )
    {
       if ( externList[i] != externList[ncnt] ) 
          externList[++ncnt] = externList[i];
    }
    externLeng = ncnt + 1;

    /* -------------------------------------------------------- */
    /* allocate the CSR matrix                                  */
    /* -------------------------------------------------------- */ 

    nnz = 0; 
    for ( i = 0; i < localEqns; i++ ) nnz += diagSize[i] + offdiagSize[i]; 
    rowptr  = (HYPRE_Int *)    malloc( (localEqns + 1) * sizeof(HYPRE_Int) ); 
    columns = (HYPRE_Int *)    malloc( nnz * sizeof(HYPRE_Int) ); 
    values  = (double *) malloc( nnz * sizeof(double) ); 
    rowptr[0] = 0; 
    for ( i = 1; i <= localEqns; i++ ) 
       rowptr[i] = rowptr[i-1] + diagSize[i-1] + offdiagSize[i-1];
    free( diagSize );
    free( offdiagSize );

    /* -------------------------------------------------------- */ 
    /* put the matrix data in the CSR matrix                    */
    /* -------------------------------------------------------- */ 

    rowptr[0] = 0; 
    ncnt      = 0;
    for ( i = startRow; i <= endRow; i++ )
    {
       HYPRE_ParCSRMatrixGetRow(A, i, &rowLeng, &colInd, &colVal);
       for (j = 0; j < rowLeng; j++)
       {
          index = colInd[j];
          //if ( colVal[j] != 0.0 ) 
          {
             if ( index < startRow || index > endRow )
             {
                columns[ncnt] = hypre_BinarySearch(externList,index,
                                                   externLeng );
                columns[ncnt] += localEqns;
                values [ncnt++] = colVal[j];
             }
             else
             {
                columns[ncnt] = index - startRow;
                values[ncnt++] = colVal[j];
             }
          }
       }
       rowptr[i-startRow+1] = ncnt;
       HYPRE_ParCSRMatrixRestoreRow(A, i, &rowLeng, &colInd, &colVal);
    }
    assert( ncnt == nnz );
   
    /* -------------------------------------------------------- */ 
    /* initialize the MH_Matrix data structure                  */
    /* -------------------------------------------------------- */ 

    mh_mat->Nrows       = localEqns;
    mh_mat->rowptr      = rowptr;
    mh_mat->colnum      = columns;
    mh_mat->values      = values;
    mh_mat->sendProcCnt = 0;
    mh_mat->recvProcCnt = 0;
    mh_mat->sendLeng    = NULL;
    mh_mat->recvLeng    = NULL;
    mh_mat->sendProc    = NULL;
    mh_mat->recvProc    = NULL;
    mh_mat->sendList    = NULL;
    mh_mat->map         = externList;
 
    /* -------------------------------------------------------- */ 
    /* form the remote portion of the matrix                    */
    /* -------------------------------------------------------- */ 

    if ( nprocs > 1 ) 
    {
       /* ----------------------------------------------------- */ 
       /* count number of elements to be received from each     */
       /* remote processor (assume sequential mapping)          */
       /* ----------------------------------------------------- */ 

       tempCnt = (HYPRE_Int *) malloc( sizeof(HYPRE_Int) * nprocs );
       for ( i = 0; i < nprocs; i++ ) tempCnt[i] = 0;
       for ( i = 0; i < externLeng; i++ )
       {
          for ( j = 0; j < nprocs; j++ )
          {
             if ( externList[i] >= partition[j] && 
                  externList[i] < partition[j+1] )
             {
                tempCnt[j]++;
                break;
             }
          }
       }

       /* ----------------------------------------------------- */ 
       /* compile a list processors data is to be received from */
       /* ----------------------------------------------------- */ 

       recvProcCnt = 0;
       for ( i = 0; i < nprocs; i++ )
          if ( tempCnt[i] > 0 ) recvProcCnt++;
       recvLeng = (HYPRE_Int*) malloc( sizeof(HYPRE_Int) * recvProcCnt );
       recvProc = (HYPRE_Int*) malloc( sizeof(HYPRE_Int) * recvProcCnt );
       recvProcCnt = 0;
       for ( i = 0; i < nprocs; i++ )
       {
          if ( tempCnt[i] > 0 ) 
          {
             recvProc[recvProcCnt]   = i;
             recvLeng[recvProcCnt++] = tempCnt[i];
          }
       }

       /* ----------------------------------------------------- */ 
       /* each processor has to find out how many processors it */
       /* has to send data to                                   */
       /* ----------------------------------------------------- */ 

       sendLeng = (HYPRE_Int *) malloc( nprocs * sizeof(HYPRE_Int) );
       for ( i = 0; i < nprocs; i++ ) tempCnt[i] = 0;
       for ( i = 0; i < recvProcCnt; i++ ) tempCnt[recvProc[i]] = 1;
       hypre_MPI_Allreduce(tempCnt, sendLeng, nprocs, HYPRE_MPI_INT, hypre_MPI_SUM, comm );
       sendProcCnt = sendLeng[my_id];
       free( sendLeng );
       if ( sendProcCnt > 0 )
       {
          sendLeng = (HYPRE_Int *)  malloc( sendProcCnt * sizeof(HYPRE_Int) );
          sendProc = (HYPRE_Int *)  malloc( sendProcCnt * sizeof(HYPRE_Int) );
          sendList = (HYPRE_Int **) malloc( sendProcCnt * sizeof(HYPRE_Int*) );
       }
       else 
       {
          sendLeng = sendProc = NULL;
          sendList = NULL;
       }

       /* ----------------------------------------------------- */ 
       /* each processor sends to all processors it expects to  */
       /* receive data about the lengths of data expected       */
       /* ----------------------------------------------------- */ 

       msgid = 539;
       for ( i = 0; i < recvProcCnt; i++ ) 
       {
          hypre_MPI_Send((void*) &recvLeng[i],1,HYPRE_MPI_INT,recvProc[i],msgid,comm);
       }
       for ( i = 0; i < sendProcCnt; i++ ) 
       {
          hypre_MPI_Recv((void*) &sendLeng[i],1,HYPRE_MPI_INT,hypre_MPI_ANY_SOURCE,msgid,
                   comm,&status);
          sendProc[i] = status.hypre_MPI_SOURCE;
          sendList[i] = (HYPRE_Int *) malloc( sendLeng[i] * sizeof(HYPRE_Int) );
          if ( sendList[i] == NULL ) 
             hypre_printf("allocate problem %d \n", sendLeng[i]);
       }

       /* ----------------------------------------------------- */ 
       /* each processor sends to all processors it expects to  */
       /* receive data about the equation numbers               */
       /* ----------------------------------------------------- */ 

       for ( i = 0; i < nprocs; i++ ) tempCnt[i] = 0; 
       ncnt = 1;
       for ( i = 0; i < externLeng; i++ ) 
       {
          if ( externList[i] >= partition[ncnt] )
          {
             tempCnt[ncnt-1] = i;
             i--;
             ncnt++;
          }
       }    
       for ( i = ncnt-1; i < nprocs; i++ ) tempCnt[i] = externLeng; 

       /* ----------------------------------------------------- */ 
       /* send the global equation numbers                      */
       /* ----------------------------------------------------- */ 

       msgid = 540;
       for ( i = 0; i < recvProcCnt; i++ ) 
       {
          if ( recvProc[i] == 0 ) j = 0;
          else                    j = tempCnt[recvProc[i]-1];
          rowLeng = recvLeng[i];
          hypre_MPI_Send((void*) &externList[j],rowLeng,HYPRE_MPI_INT,recvProc[i],
                    msgid,comm);
       }
       for ( i = 0; i < sendProcCnt; i++ ) 
       {
          rowLeng = sendLeng[i];
          hypre_MPI_Recv((void*)sendList[i],rowLeng,HYPRE_MPI_INT,sendProc[i],
                   msgid,comm,&status);
       }

       /* ----------------------------------------------------- */ 
       /* convert the send list from global to local numbers    */
       /* ----------------------------------------------------- */ 

       for ( i = 0; i < sendProcCnt; i++ )
       { 
          for ( j = 0; j < sendLeng[i]; j++ )
          {
             index = sendList[i][j] - startRow;
             if ( index < 0 || index >= localEqns )
             {
                hypre_printf("%d : Construct MH matrix Error - index out ");
                hypre_printf("of range%d\n", my_id, index);
             }
             sendList[i][j] = index;
          }
       }

       /* ----------------------------------------------------- */ 
       /* convert the send list from global to local numbers    */
       /* ----------------------------------------------------- */ 

       mh_mat->sendProcCnt = sendProcCnt;
       mh_mat->recvProcCnt = recvProcCnt;
       mh_mat->sendLeng    = sendLeng;
       mh_mat->recvLeng    = recvLeng;
       mh_mat->sendProc    = sendProc;
       mh_mat->recvProc    = recvProc;
       mh_mat->sendList    = sendList;

       /* ----------------------------------------------------- */ 
       /* clean up                                              */
       /* ----------------------------------------------------- */ 

       free( tempCnt );
    }

    return 0;
}

#endif

