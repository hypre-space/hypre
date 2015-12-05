/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.18 $
 ***********************************************************************EHEADER*/




/****************************************************************************/ 
/* HYPRE_LSI_MLMaxwell interface                                            */
/*--------------------------------------------------------------------------*/
/*  local functions :
 * 
 *        ML_ExchBdry
 *        ML_MatVec
 *        ML_GetRow
 *        HYPRE_LSI_MLMaxwellCreate
 *        HYPRE_LSI_MLMaxwellDestroy
 *        HYPRE_LSI_MLMaxwellSetup
 *        HYPRE_LSI_MLMaxwellSolve
 *        HYPRE_LSI_MLMaxwellSetStrongThreshold
 *        HYPRE_LSI_MLMaxwellSetGMatrix
 *        HYPRE_LSI_MLMaxwellSetANNMatrix
 *        HYPRE_LSI_ConstructMLMatrix
 ****************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "../../parcsr_ls/HYPRE_parcsr_ls.h"
#include "../../utilities/_hypre_utilities.h"
#include "../../seq_mv/vector.h"
/* #include "../../parcsr_mv/par_vector.h" */
#include "../../parcsr_mv/_hypre_parcsr_mv.h"
#include "HYPRE_MLMatrix.h"
#include "HYPRE_MLMaxwell.h"

extern void qsort0(int *, int, int);
extern int  HYPRE_LSI_MLConstructMLMatrix(HYPRE_ParCSRMatrix,
                  HYPRE_ML_Matrix *, int *, MPI_Comm, MLMaxwell_Context*); 

/****************************************************************************/
/* communication functions on parallel platforms                            */
/*--------------------------------------------------------------------------*/
                                                                                
int ML_Irecv(void* buf, unsigned int count, int *src, int *mid,
             MPI_Comm comm, MPI_Request *requests)
{
#ifdef HYPRE_SEQUENTIAL
   return 0;
#else
   int mypid, lsrc, retcode;
                                                                                
   if (*src < 0) lsrc = MPI_ANY_SOURCE; else lsrc = (*src);
   retcode = MPI_Irecv(buf, (int) count,MPI_BYTE,lsrc,*mid,comm,requests);
   if (retcode != 0)
   {
      MPI_Comm_rank(comm, &mypid);
      printf("%d : ML_Irecv warning : retcode = %d\n", mypid, retcode);
   }
   return 0;
#endif
}
                                                                                
int ML_Wait(void* buf, unsigned int count, int *src, int *mid,
            MPI_Comm comm, MPI_Request *requests)
{
#ifdef HYPRE_SEQUENTIAL
   return count;
#else
   MPI_Status status;
   int        mypid, incount, retcode;
                                                                                
   retcode = MPI_Wait(requests, &status);
   if (retcode != 0)
   {
      MPI_Comm_rank(comm, &mypid);
      printf("%d : ML_Wait warning : retcode = %d\n", mypid, retcode);
   }
   MPI_Get_count(&status, MPI_BYTE, &incount);
   if (*src < 0) *src = status.MPI_SOURCE;
   return incount;
#endif
}

int ML_Send(void* buf, unsigned int count, int dest, int mid, MPI_Comm comm)
{
#ifdef HYPRE_SEQUENTIAL
   return 0;
#else
   int mypid;
   int retcode = MPI_Send(buf, (int) count, MPI_BYTE, dest, mid, comm);
   if (retcode != 0)
   {
      MPI_Comm_rank(comm, &mypid);
      printf("%d : ML_Send warning : retcode = %d\n", mypid, retcode);
   }
   return 0;
#endif
}
                                                                                
/****************************************************************************/ 
/* wrapper function for interprocessor communication for matvec and getrow  */
/*--------------------------------------------------------------------------*/

int ML_ExchBdry(double *vec, void *obj)
{
#ifdef HYPRE_SEQUENTIAL
   return 0;
#else
   int         i, j, msgid, leng, src, dest, offset, *tempList;
   int         sendProcCnt, *sendProc, *sendLeng, **sendList;
   int         recvProcCnt, *recvProc, *recvLeng, nRows;
   double      *dbuf;
   HYPRE_ML_Matrix   *Amat;
   MPI_Comm    comm;
   MPI_Request *requests; 
   MLMaxwell_Context *context;

   context     = (MLMaxwell_Context *) obj;
   Amat        = (HYPRE_ML_Matrix  *) context->Amat;
   comm        = context->comm;
   sendProcCnt = Amat->sendProcCnt;
   recvProcCnt = Amat->recvProcCnt;
   sendProc    = Amat->sendProc;
   recvProc    = Amat->recvProc;
   sendLeng    = Amat->sendLeng;
   recvLeng    = Amat->recvLeng;
   sendList    = Amat->sendList;
   nRows       = Amat->Nrows;

   if (recvProcCnt > 0)
      requests = (MPI_Request *) malloc(recvProcCnt*sizeof(MPI_Request));
   msgid = 234;
   offset = nRows;
   for (i = 0; i < recvProcCnt; i++)
   {
      leng = recvLeng[i] * sizeof(double);
      src  = recvProc[i];
      ML_Irecv((void*) &(vec[offset]),leng,&src,&msgid,comm,&requests[i]);
      offset += recvLeng[i];
   }
   msgid = 234;
   for (i = 0; i < sendProcCnt; i++)
   {
      dest = sendProc[i];
      leng = sendLeng[i] * sizeof(double);
      dbuf = (double *) malloc(leng * sizeof(double));
      tempList = sendList[i];
      for (j = 0; j < sendLeng[i]; j++) dbuf[j] = vec[tempList[j]];
      ML_Send((void*) dbuf, leng, dest, msgid, comm);
      if (dbuf != NULL) free(dbuf);
   }
   offset = nRows;
   for (i = 0; i < recvProcCnt; i++)
   {
      leng = recvLeng[i] * sizeof(double);
      src  = recvProc[i];
      ML_Wait((void*) &(vec[offset]), leng, &src, &msgid, comm, &requests[i]);
      offset += recvLeng[i];
   }
   if (recvProcCnt > 0) free (requests);
   return 1;
#endif
}

/****************************************************************************/ 
/* matvec function for local matrix structure HYPRE_ML_Matrix               */
/*--------------------------------------------------------------------------*/

#ifdef HAVE_MLMAXWELL
int ML_MatVec(ML_Operator *obj, int leng1, double p[], int leng2, double ap[])
#else
int ML_MatVec(void *obj, int leng1, double p[], int leng2, double ap[])
#endif
{
#ifdef HAVE_MLMAXWELL
    int               i, j, length, nRows, ibeg, iend, k, *rowptr, *colInd;
    double            *dbuf, sum, *colVal;
    HYPRE_ML_Matrix   *Amat;
    MLMaxwell_Context *context;

    ML_Operator *ml_op = (ML_Operator *) obj;
    context = (MLMaxwell_Context *) ML_Get_MyGetrowData(ml_op);
    Amat    = (HYPRE_ML_Matrix*) context->Amat;
    nRows   = Amat->Nrows;
    rowptr  = Amat->rowptr;
    colInd  = Amat->colnum;
    colVal  = Amat->values;
    length = nRows;
    for (i = 0; i < Amat->recvProcCnt; i++) length += Amat->recvLeng[i];
    dbuf = (double *) malloc(length * sizeof(double));
    for (i = 0; i < nRows; i++) dbuf[i] = p[i];
    ML_ExchBdry(dbuf, (void *) context);
    for (i = 0 ; i < nRows; i++) 
    {
       sum = 0.0;
       ibeg = rowptr[i];
       iend = rowptr[i+1];
       for (j = ibeg; j < iend; j++)
       { 
          k = colInd[j];
          sum += (colVal[j] * dbuf[k]);
       }
       ap[i] = sum;
    }
    if (dbuf != NULL) free(dbuf);
    return 1;

#else
    printf("ML_MatVec : MLMaxwell not activated.\n");
    return -1;
#endif
}

/****************************************************************************/
/* getrow function for local matrix structure HYPRE_ML_Matrix(ML compatible)*/
/*--------------------------------------------------------------------------*/

#ifdef HAVE_MLMAXWELL
int ML_GetRow(ML_Operator *obj, int N_requested_rows, int requested_rows[],
   int allocated_space, int columns[], double values[], int row_lengths[])
#else
int ML_GetRow(void *obj, int N_requested_rows, int requested_rows[],
   int allocated_space, int columns[], double values[], int row_lengths[])
#endif
{
    int               i, j, ncnt, colindex, rowLeng, rowindex;
    int               nRows, *rowptr, *colInd;
    double            *colVal;

#ifdef HAVE_MLMAXWELL
    MLMaxwell_Context *context;
    HYPRE_ML_Matrix   *Amat;

    ML_Operator *ml_op = (ML_Operator *) obj;
    context = (MLMaxwell_Context *) ML_Get_MyGetrowData(ml_op);
    Amat    = (HYPRE_ML_Matrix*) context->Amat;
    nRows   = Amat->Nrows;
    rowptr  = Amat->rowptr;
    colInd  = Amat->colnum;
    colVal  = Amat->values;
#else
    printf("ML_GetRow : MLMaxwell not activated.\n");
    return -1;
#endif

    ncnt = 0;
    for (i = 0; i < N_requested_rows; i++)
    {
       rowindex = requested_rows[i];
       if (rowindex < 0 || rowindex >= nRows)
          printf("Invalid row request in GetRow : %d (%d)\n",rowindex,nRows);
       rowLeng = rowptr[rowindex+1] - rowptr[rowindex];
       if (ncnt+rowLeng > allocated_space) {row_lengths[i]=-9; return 0;}
       row_lengths[i] = rowLeng;
       colindex = rowptr[rowindex];
       for (j = 0; j < rowLeng; j++)
       {
          columns[ncnt] = colInd[colindex];
          values[ncnt++] = colVal[colindex++];
       }
    }
    return 1;
}

/****************************************************************************/
/* HYPRE_LSI_MLMaxwellCreate                                                */
/*--------------------------------------------------------------------------*/

int HYPRE_LSI_MLMaxwellCreate(MPI_Comm comm, HYPRE_Solver *solver)
{
#ifdef HAVE_MLMAXWELL
    /* create an internal ML data structure */

    MLMaxwell_Link *link = (MLMaxwell_Link *) malloc(sizeof(MLMaxwell_Link));
    if (link == NULL) return 1;   

    /* fill in all other default parameters */

    link->comm          = comm;
    link->nlevels       = 6;    /* max number of levels */
    link->smoothP_flag  = ML_YES;
    link->edge_smoother = (void *) ML_Gen_Smoother_MLS;
    link->node_smoother = (void *) ML_Gen_Smoother_MLS;
    link->ml_ag         = NULL;
    link->ml_ee         = NULL;
    link->ml_nn         = NULL;
    link->Aee_contxt    = NULL;
    link->Ann_contxt    = NULL;
    link->G_contxt      = NULL;
    link->ag_threshold  = 0.0;  /* threshold for aggregation */
    link->Annmat        = NULL;
    link->Gmat          = NULL;
    link->GTmat         = NULL;
    link->Gmat_array    = NULL;
    link->GTmat_array   = NULL;
    link->node_args     = NULL;
    link->edge_args     = NULL;
  
    ML_Create(&(link->ml_ee), link->nlevels);
    ML_Create(&(link->ml_nn), link->nlevels);

    *solver = (HYPRE_Solver) link;

    return 0;
#else
    printf("ML not linked.\n");
    return -1;
#endif
}

/****************************************************************************/
/* HYPRE_LSI_MLMaxwellDestroy                                               */
/*--------------------------------------------------------------------------*/

int HYPRE_LSI_MLMaxwellDestroy(HYPRE_Solver solver)
{
#ifdef HAVE_MLMAXWELL
    int             i;
    HYPRE_ML_Matrix *Amat;
    MLMaxwell_Link  *link = (MLMaxwell_Link *) solver;

    if (link->ml_ag != NULL) ML_Aggregate_Destroy(&(link->ml_ag));
    if (link->ml_ee != NULL) ML_Destroy(&(link->ml_ee));
    if (link->ml_nn != NULL) ML_Destroy(&(link->ml_nn));
    if (link->Aee_contxt->partition != NULL) free(link->Aee_contxt->partition);
    if (link->Ann_contxt->partition != NULL) free(link->Ann_contxt->partition);
    if (link->Aee_contxt->Amat != NULL)
    {
       Amat = (HYPRE_ML_Matrix *) link->Aee_contxt->Amat;
       if (Amat->sendProc != NULL ) free (Amat->sendProc);
       if (Amat->sendLeng != NULL ) free (Amat->sendLeng);
       if (Amat->sendList != NULL ) 
       {
          for (i = 0; i < Amat->sendProcCnt; i++)
             if (Amat->sendList[i] != NULL) free (Amat->sendList[i]);
          free (Amat->sendList);
       }
       if (Amat->recvProc != NULL) free (Amat->recvProc);
       if (Amat->recvLeng != NULL) free (Amat->recvLeng);
       if (Amat->map      != NULL) free (Amat->map);
       free(Amat);
    }
    if (link->Aee_contxt != NULL) free(link->Aee_contxt);

    if (link->Ann_contxt->Amat != NULL)
    {
       Amat = (HYPRE_ML_Matrix *) link->Ann_contxt->Amat;
       if (Amat->sendProc != NULL ) free (Amat->sendProc);
       if (Amat->sendLeng != NULL ) free (Amat->sendLeng);
       if (Amat->sendList != NULL ) 
       {
          for (i = 0; i < Amat->sendProcCnt; i++)
             if (Amat->sendList[i] != NULL) free (Amat->sendList[i]);
          free (Amat->sendList);
       }
       if (Amat->recvProc != NULL) free (Amat->recvProc);
       if (Amat->recvLeng != NULL) free (Amat->recvLeng);
       if (Amat->map      != NULL) free (Amat->map);
       free(Amat);
    }
    if (link->Ann_contxt != NULL) free(link->Ann_contxt);

    if (link->G_contxt->Amat != NULL)
    {
       Amat = (HYPRE_ML_Matrix *) link->G_contxt->Amat;
       if (Amat->sendProc != NULL ) free (Amat->sendProc);
       if (Amat->sendLeng != NULL ) free (Amat->sendLeng);
       if (Amat->sendList != NULL ) 
       {
          for (i = 0; i < Amat->sendProcCnt; i++)
             if (Amat->sendList[i] != NULL) free (Amat->sendList[i]);
          free (Amat->sendList);
       }
       if (Amat->recvProc != NULL) free (Amat->recvProc);
       if (Amat->recvLeng != NULL) free (Amat->recvLeng);
       if (Amat->map      != NULL) free (Amat->map);
       free(Amat);
    }
    if (link->G_contxt != NULL) free(link->G_contxt);

    if (link->Gmat  != NULL) ML_Operator_Destroy(&(link->Gmat));
    if (link->GTmat != NULL) ML_Operator_Destroy(&(link->GTmat));
    if (link->Gmat_array != NULL)
       ML_MGHierarchy_ReitzingerDestroy(link->nlevels-2, 
                       &(link->Gmat_array), &(link->GTmat_array));

    if (link->node_args != NULL)
       ML_Smoother_Arglist_Delete(&(link->node_args));
    if (link->edge_args != NULL)
       ML_Smoother_Arglist_Delete(&(link->edge_args));

    free(link);

    return 0;
#else
    printf("ML not linked.\n");
    return -1;
#endif
}

/****************************************************************************/
/* HYPRE_LSI_MLMaxwellSetup                                                 */
/*--------------------------------------------------------------------------*/

int HYPRE_LSI_MLMaxwellSetup(HYPRE_Solver solver, HYPRE_ParCSRMatrix A_ee,
                             HYPRE_ParVector x, HYPRE_ParVector b)
{
#ifdef HAVE_MLMAXWELL
   int         i, mypid, nprocs, coarsest_level, level, nlevels;
   int         *row_partition, nodeNEqns, edgeNEqns, length;
   int         edge_its = 3, node_its = 3, Nfine_node, Nfine_edge, itmp;
   int         hiptmair_type=HALF_HIPTMAIR, Nits_per_presmooth=1;
   int         Ncoarse_edge, Ncoarse_node;
   double      edge_coarsening_rate, node_coarsening_rate;
   double      node_omega = ML_DDEFAULT, edge_omega = ML_DDEFAULT; 
   ML          *ml_ee, *ml_nn;
   ML_Operator *Gmat, *GTmat;
   MLMaxwell_Link    *link;
   HYPRE_ML_Matrix   *mh_Aee, *mh_G, *mh_Ann;
   MLMaxwell_Context *Aee_context, *G_context, *Ann_context;

   /* -------------------------------------------------------- */ 
   /* set up the parallel environment                          */
   /* -------------------------------------------------------- */ 

   link = (MLMaxwell_Link *) solver;
   MPI_Comm_rank(link->comm, &mypid);
   MPI_Comm_size(link->comm, &nprocs);

   /* -------------------------------------------------------- */ 
   /* create ML structures                                     */
   /* -------------------------------------------------------- */ 

   nlevels = link->nlevels;
   ML_Create(&(link->ml_ee), nlevels);
   ML_Create(&(link->ml_nn), nlevels);
   ml_ee   = link->ml_ee;
   ml_nn   = link->ml_nn;
   
   /* -------------------------------------------------------- */ 
   /* fetch the matrix row partition information and put it    */
   /* into the matrix data object (for matvec and getrow)      */
   /* -------------------------------------------------------- */ 

   Aee_context = (MLMaxwell_Context *) malloc(sizeof(MLMaxwell_Context));
   link->Aee_contxt = Aee_context;
   Aee_context->comm = link->comm;
   HYPRE_ParCSRMatrixGetRowPartitioning(A_ee, &row_partition);
   edgeNEqns = row_partition[mypid+1] - row_partition[mypid];
   Aee_context->globalEqns = row_partition[nprocs];
   Aee_context->partition = (int *) malloc(sizeof(int)*(nprocs+1));
   for (i=0; i<=nprocs; i++) Aee_context->partition[i] = row_partition[i];
   hypre_TFree(row_partition);
   mh_Aee = (HYPRE_ML_Matrix *) malloc(sizeof(HYPRE_ML_Matrix));
   HYPRE_LSI_MLConstructMLMatrix(A_ee,mh_Aee,Aee_context->partition,
                                 link->comm,Aee_context); 
   Aee_context->Amat = mh_Aee;

   Ann_context = (MLMaxwell_Context *) malloc(sizeof(MLMaxwell_Context));
   link->Ann_contxt = Ann_context;
   Ann_context->comm = link->comm;
   HYPRE_ParCSRMatrixGetRowPartitioning(link->hypreAnn, &row_partition);
   nodeNEqns  = row_partition[mypid+1] - row_partition[mypid];
   Ann_context->globalEqns = row_partition[nprocs];
   Ann_context->partition = (int *) malloc(sizeof(int)*(nprocs+1));
   for (i=0; i<=nprocs; i++) Ann_context->partition[i] = row_partition[i];
   hypre_TFree(row_partition);
   mh_Ann = (HYPRE_ML_Matrix *) malloc(sizeof(HYPRE_ML_Matrix));
   HYPRE_LSI_MLConstructMLMatrix(link->hypreAnn,mh_Ann,Ann_context->partition,
                                 link->comm,Ann_context); 
   Ann_context->Amat = mh_Ann;

   G_context = (MLMaxwell_Context *) malloc(sizeof(MLMaxwell_Context));
   link->G_contxt = G_context;
   G_context->comm = link->comm;
   HYPRE_ParCSRMatrixGetRowPartitioning(link->hypreG, &row_partition);
   G_context->globalEqns = row_partition[nprocs];
   G_context->partition = (int *) malloc(sizeof(int)*(nprocs+1));
   for (i=0; i<=nprocs; i++) G_context->partition[i] = row_partition[i];
   hypre_TFree(row_partition);
   mh_G = (HYPRE_ML_Matrix *) malloc(sizeof(HYPRE_ML_Matrix));
   HYPRE_LSI_MLConstructMLMatrix(link->hypreG,mh_G,G_context->partition,
                                 link->comm,G_context); 
   G_context->Amat = mh_G;

   /* -------------------------------------------------------- */ 
   /* Build A_ee directly as an ML matrix                      */
   /* -------------------------------------------------------- */ 

   ML_Init_Amatrix(ml_ee,nlevels-1,edgeNEqns,edgeNEqns,(void *)Aee_context);
   length = edgeNEqns;
   for (i=0; i<mh_Aee->recvProcCnt; i++) length += mh_Aee->recvLeng[i];
   ML_Set_Amatrix_Getrow(ml_ee, nlevels-1, ML_GetRow, ML_ExchBdry, length);
   ML_Operator_Set_ApplyFunc(&(ml_ee->Amat[nlevels-1]), ML_MatVec);

   /* -------------------------------------------------------- */ 
   /* Build A_nn directly as an ML matrix                      */
   /* -------------------------------------------------------- */ 

   ML_Init_Amatrix(ml_nn, nlevels-1,nodeNEqns,nodeNEqns,(void *)Ann_context);
   length = nodeNEqns;
   for (i=0; i<mh_Ann->recvProcCnt; i++) length += mh_Ann->recvLeng[i];
   ML_Set_Amatrix_Getrow(ml_nn, nlevels-1, ML_GetRow, ML_ExchBdry, length);
   ML_Operator_Set_ApplyFunc(&(ml_nn->Amat[nlevels-1]), ML_MatVec);

   /* -------------------------------------------------------- */ 
   /* Build G matrix and its transpose                         */
   /* -------------------------------------------------------- */ 

   Gmat = ML_Operator_Create(ml_ee->comm);
   ML_Operator_Set_Getrow(Gmat, edgeNEqns, ML_GetRow);
   ML_Operator_Set_ApplyFuncData(Gmat, nodeNEqns, edgeNEqns,
                         (void *) G_context, edgeNEqns, ML_MatVec, 0);
   length = 0;
   for (i=0; i<mh_Ann->recvProcCnt; i++) length += mh_Ann->recvLeng[i];
   ML_CommInfoOP_Generate(&(Gmat->getrow->pre_comm), ML_ExchBdry,
                          G_context, ml_ee->comm, nodeNEqns, length);

   GTmat = ML_Operator_Create(ml_ee->comm);
   ML_Operator_Transpose_byrow(Gmat, GTmat);
   link->GTmat = GTmat;

   /* -------------------------------------------------------- */ 
   /* create an AMG or aggregate context                       */
   /* -------------------------------------------------------- */ 

   ML_Set_PrintLevel(2);
   ML_Set_Tolerance(ml_ee, 1.0e-8);
   ML_Aggregate_Create(&(link->ml_ag));
   ML_Aggregate_Set_CoarsenScheme_Uncoupled(link->ml_ag);
   ML_Aggregate_Set_DampingFactor(link->ml_ag, 0.0); /* must be 0 */
   ML_Aggregate_Set_MaxCoarseSize(link->ml_ag, 30);
   ML_Aggregate_Set_Threshold(link->ml_ag, link->ag_threshold);

   coarsest_level = ML_Gen_MGHierarchy_UsingReitzinger(ml_ee, &ml_nn,
                       nlevels-1, ML_DECREASING, link->ml_ag, Gmat,
                       GTmat, &(link->Gmat_array), &(link->GTmat_array),
                       link->smoothP_flag, 1.5, 0, ML_DDEFAULT);

   /* -------------------------------------------------------- */ 
   /* Set the Hiptmair subsmoothers                            */
   /* -------------------------------------------------------- */ 
                                                                                
   if (link->node_smoother == (void *) ML_Gen_Smoother_SymGaussSeidel)
   {
      link->node_args = ML_Smoother_Arglist_Create(2);
      ML_Smoother_Arglist_Set(link->node_args, 0, &node_its);
      ML_Smoother_Arglist_Set(link->node_args, 1, &node_omega);
   }
   if (link->edge_smoother == (void *) ML_Gen_Smoother_SymGaussSeidel)
   {
      link->edge_args = ML_Smoother_Arglist_Create(2);
      ML_Smoother_Arglist_Set(link->edge_args, 0, &edge_its);
      ML_Smoother_Arglist_Set(link->edge_args, 1, &edge_omega);
   }
   if (link->node_smoother == (void *) ML_Gen_Smoother_MLS)
   {
      link->node_args = ML_Smoother_Arglist_Create(2);
      ML_Smoother_Arglist_Set(link->node_args, 0, &node_its);
      Nfine_node = link->Gmat_array[nlevels-1]->invec_leng;
      ML_gsum_scalar_int(&Nfine_node, &itmp, ml_ee->comm);
   }
   if (link->edge_smoother == (void *) ML_Gen_Smoother_MLS)
   {
      link->edge_args = ML_Smoother_Arglist_Create(2);
      ML_Smoother_Arglist_Set(link->edge_args, 0, &edge_its);
      Nfine_edge = link->Gmat_array[nlevels-1]->outvec_leng;
      ML_gsum_scalar_int(&Nfine_edge, &itmp, ml_ee->comm);
   }

   /* -------------------------------------------------------- */ 
   /* perform aggregation                                      */
   /* -------------------------------------------------------- */ 

   if (mypid == 0)
      printf("HYPRE_MLMaxwell : number of levels = %d\n", coarsest_level);

   coarsest_level = nlevels - coarsest_level;

   /* -------------------------------------------------------- */
   /* set up at all levels                                     */
   /* -------------------------------------------------------- */

   for (level = nlevels-1; level >= coarsest_level; level--) 
   {
      if (link->edge_smoother == (void *) ML_Gen_Smoother_MLS)
      {
         if (level != coarsest_level)
         {
            Ncoarse_edge = link->Gmat_array[level-1]->outvec_leng;
            ML_gsum_scalar_int(&Ncoarse_edge, &itmp, ml_ee->comm);
            edge_coarsening_rate =  2.*((double) Nfine_edge)/
                                    ((double) Ncoarse_edge);
         }
         else edge_coarsening_rate =  (double) Nfine_edge;

         ML_Smoother_Arglist_Set(link->edge_args,1,&edge_coarsening_rate);
         Nfine_edge = Ncoarse_edge;
      }
      if (link->node_smoother == (void *) ML_Gen_Smoother_MLS)
      {
         if (level != coarsest_level)
         {
            Ncoarse_node = link->Gmat_array[level-1]->invec_leng;
            ML_gsum_scalar_int(&Ncoarse_node, &itmp, ml_ee->comm);
            node_coarsening_rate = 2.*((double) Nfine_node)/
                                   ((double) Ncoarse_node);
         }
         else node_coarsening_rate = (double) Nfine_node;

         ML_Smoother_Arglist_Set(link->node_args,1,&node_coarsening_rate);
         Nfine_node = Ncoarse_node;
      }
      ML_Gen_Smoother_Hiptmair(ml_ee, level, ML_BOTH, Nits_per_presmooth,
                     link->Gmat_array, link->GTmat_array, NULL,
                     link->edge_smoother, link->edge_args, 
                     link->node_smoother, link->node_args, hiptmair_type);
   }
                                                                                
   /* -------------------------------------------------------- */ 
   /* set up smoother and coarse solver                        */
   /* -------------------------------------------------------- */ 

   ML_Gen_Solver(ml_ee, ML_MGV, nlevels-1, coarsest_level);
   
   return 0;
#else
   printf("ML not linked.\n");
   return -1;
#endif
}

/****************************************************************************/
/* HYPRE_LSI_MLSolve                                                        */
/*--------------------------------------------------------------------------*/

int HYPRE_LSI_MLMaxwellSolve(HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                             HYPRE_ParVector b, HYPRE_ParVector x )
{
#ifdef HAVE_MLMAXWELL
    double  *rhs, *sol;
    MLMaxwell_Link *link = (MLMaxwell_Link *) solver;
    ML      *ml_ee = link->ml_ee;

    rhs = hypre_VectorData(hypre_ParVectorLocalVector((hypre_ParVector *) b));
    sol = hypre_VectorData(hypre_ParVectorLocalVector((hypre_ParVector *) x));
  
    ML_Solve_AMGV(ml_ee, rhs, sol);

    return 0;
#else
    printf("ML not linked.\n");
    return -1;
#endif
}

/****************************************************************************/
/* HYPRE_LSI_MLMaxwellSetStrongThreshold                                    */
/*--------------------------------------------------------------------------*/

int HYPRE_LSI_MLMaxwellSetStrengthThreshold(HYPRE_Solver solver,
                                     double strength_threshold)
{
    MLMaxwell_Link *link = (MLMaxwell_Link *) solver;
  
    if (strength_threshold < 0.0)
    {
       printf("HYPRE_LSI_MLMaxwellSetStrengthThreshold WARNING: set to 0.\n");
       link->ag_threshold = 0.0;
    } 
    else
    {
       link->ag_threshold = strength_threshold;
    } 
    return( 0 );
}

/****************************************************************************/
/* HYPRE_LSI_MLMaxwellSetGMatrix                                            */
/*--------------------------------------------------------------------------*/

int HYPRE_LSI_MLMaxwellSetGMatrix(HYPRE_Solver solver, HYPRE_ParCSRMatrix G)
{
    MLMaxwell_Link *link = (MLMaxwell_Link *) solver;
    link->hypreG = G;
    return( 0 );
}

/****************************************************************************/
/* HYPRE_LSI_MLMaxwellSetANNMatrix                                          */
/*--------------------------------------------------------------------------*/

int HYPRE_LSI_MLMaxwellSetANNMatrix(HYPRE_Solver solver, HYPRE_ParCSRMatrix ANN)
{
    MLMaxwell_Link *link = (MLMaxwell_Link *) solver;
    link->hypreAnn = ANN;
    return( 0 );
}

/****************************************************************************/
/* HYPRE_LSI_MLConstructMLMatrix                                            */
/*--------------------------------------------------------------------------*/

int HYPRE_LSI_MLConstructMLMatrix(HYPRE_ParCSRMatrix A,
                                  HYPRE_ML_Matrix *ml_mat, int *partition,
                                  MPI_Comm comm, MLMaxwell_Context *obj) 
{
    int         i, j, index, mypid, nprocs;
    int         rowLeng, *colInd, startRow, endRow, localEqns;
    int         *diagSize, *offdiagSize, externLeng, *externList, ncnt, nnz;
    int         *rowptr, *columns, num_bdry;
    double      *colVal, *values;
#ifndef HYPRE_SEQUENTIAL
    int         sendProcCnt, *sendLeng, *sendProc, **sendList;
    int         recvProcCnt, *recvLeng, *recvProc, *tempCnt, msgid;
    MPI_Request *requests;
    MPI_Status  status;
#endif

    /* -------------------------------------------------------- */
    /* get machine information and local matrix information     */
    /* -------------------------------------------------------- */
    
#ifdef HYPRE_SEQUENTIAL
    mypid = 0;
    nprocs = 1;
#else
    MPI_Comm_rank(comm, &mypid);
    MPI_Comm_size(comm, &nprocs);
#endif

    startRow  = partition[mypid];
    endRow    = partition[mypid+1] - 1;
    localEqns = endRow - startRow + 1;

    /* -------------------------------------------------------- */
    /* probe A to find out about diagonal and off-diagonal      */
    /* block information                                        */
    /* -------------------------------------------------------- */

    diagSize    = (int*) malloc(sizeof(int) * localEqns);
    offdiagSize = (int*) malloc(sizeof(int) * localEqns);
    num_bdry = 0;
    for (i = startRow; i <= endRow; i++)
    {
       diagSize[i-startRow] = offdiagSize[i-startRow] = 0;
       HYPRE_ParCSRMatrixGetRow(A, i, &rowLeng, &colInd, &colVal);
       for (j = 0; j < rowLeng; j++)
       {
          if (colInd[j] < startRow || colInd[j] > endRow)
          {
             if ( colVal[j] != 0.0 ) offdiagSize[i-startRow]++;
             /*offdiagSize[i-startRow]++;*/
          }
          else
          {
             if ( colVal[j] != 0.0 ) diagSize[i-startRow]++;
             /*diagSize[i-startRow]++;*/
          }
       }
       HYPRE_ParCSRMatrixRestoreRow(A, i, &rowLeng, &colInd, &colVal);
       if (diagSize[i-startRow] + offdiagSize[i-startRow] == 1) num_bdry++;
    }

    /* -------------------------------------------------------- */
    /* construct external node list in global eqn numbers       */
    /* -------------------------------------------------------- */

    externLeng = 0;
    for (i = 0; i < localEqns; i++) externLeng += offdiagSize[i];
    if (externLeng > 0)
         externList = (int *) malloc( sizeof(int) * externLeng);
    else externList = NULL;
    externLeng = 0;
    for (i = startRow; i <= endRow; i++)
    {
       HYPRE_ParCSRMatrixGetRow(A, i, &rowLeng, &colInd, &colVal);
       for (j = 0; j < rowLeng; j++)
       {
          if (colInd[j] < startRow || colInd[j] > endRow)
             if (colVal[j] != 0.0) externList[externLeng++] = colInd[j];
/*
             externList[externLeng++] = colInd[j];
*/
       }
       HYPRE_ParCSRMatrixRestoreRow(A, i, &rowLeng, &colInd, &colVal);
    }
    if (externLeng > 1) qsort0(externList, 0, externLeng-1);
    ncnt = 0;
    for (i = 1; i < externLeng; i++)
    {
       if (externList[i] != externList[ncnt])
          externList[++ncnt] = externList[i];
    }
    if (externLeng > 0) externLeng = ncnt + 1;

    /* -------------------------------------------------------- */
    /* allocate the CSR matrix                                  */
    /* -------------------------------------------------------- */ 

    nnz = 0; 
    for (i = 0; i < localEqns; i++) nnz += diagSize[i] + offdiagSize[i]; 
    rowptr  = (int *)    malloc((localEqns + 1) * sizeof(int)); 
    columns = (int *)    malloc(nnz * sizeof(int)); 
    values  = (double *) malloc(nnz * sizeof(double)); 
    rowptr[0] = 0; 
    for (i = 1; i <= localEqns; i++)
       rowptr[i] = rowptr[i-1] + diagSize[i-1] + offdiagSize[i-1];
    free(diagSize);
    free(offdiagSize);

    /* -------------------------------------------------------- */ 
    /* put the matrix data in the CSR matrix                    */
    /* -------------------------------------------------------- */ 

    rowptr[0] = 0; 
    ncnt      = 0;
    for (i = startRow; i <= endRow; i++)
    {
       HYPRE_ParCSRMatrixGetRow(A, i, &rowLeng, &colInd, &colVal);
       for (j = 0; j < rowLeng; j++)
       {
          index = colInd[j];
          if (colVal[j] != 0.0)
          {
             if (index < startRow || index > endRow)
             {
                columns[ncnt] = hypre_BinarySearch(externList,index,
                                                   externLeng);
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
    assert(ncnt == nnz);
   
    /* -------------------------------------------------------- */ 
    /* initialize the MH_Matrix data structure                  */
    /* -------------------------------------------------------- */ 

    ml_mat->Nrows       = localEqns;
    ml_mat->rowptr      = rowptr;
    ml_mat->colnum      = columns;
    ml_mat->values      = values;
    ml_mat->sendProcCnt = 0;
    ml_mat->recvProcCnt = 0;
    ml_mat->sendLeng    = NULL;
    ml_mat->recvLeng    = NULL;
    ml_mat->sendProc    = NULL;
    ml_mat->recvProc    = NULL;
    ml_mat->sendList    = NULL;
    ml_mat->map         = externList;
 
    /* -------------------------------------------------------- */ 
    /* form the remote portion of the matrix                    */
    /* -------------------------------------------------------- */ 

#ifndef HYPRE_SEQUENTIAL
    if (nprocs > 1) 
    {
       /* ----------------------------------------------------- */ 
       /* count number of elements to be received from each     */
       /* remote processor (assume sequential mapping)          */
       /* ----------------------------------------------------- */ 

       tempCnt = (int *) malloc(sizeof(int) * nprocs);
       for (i = 0; i < nprocs; i++) tempCnt[i] = 0;
       for (i = 0; i < externLeng; i++)
       {
          for ( j = 0; j < nprocs; j++)
          {
             if (externList[i] >= partition[j] && 
                 externList[i] < partition[j+1])
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
       for (i = 0; i < nprocs; i++)
          if (tempCnt[i] > 0) recvProcCnt++;
       recvLeng = (int*) malloc(sizeof(int) * recvProcCnt);
       recvProc = (int*) malloc(sizeof(int) * recvProcCnt);
       recvProcCnt = 0;
       for (i = 0; i < nprocs; i++)
       {
          if (tempCnt[i] > 0) 
          {
             recvProc[recvProcCnt]   = i;
             recvLeng[recvProcCnt++] = tempCnt[i];
          }
       }

       /* ----------------------------------------------------- */ 
       /* each processor has to find out how many processors it */
       /* has to send data to                                   */
       /* ----------------------------------------------------- */ 

       sendLeng = (int *) malloc(nprocs * sizeof(int));
       for (i = 0; i < nprocs; i++) tempCnt[i] = 0;
       for (i = 0; i < recvProcCnt; i++) tempCnt[recvProc[i]] = 1;
       MPI_Allreduce(tempCnt, sendLeng, nprocs, MPI_INT, MPI_SUM, comm);
       sendProcCnt = sendLeng[mypid];
       free(sendLeng);
       if (sendProcCnt > 0)
       {
          sendLeng = (int *)  malloc(sendProcCnt * sizeof(int));
          sendProc = (int *)  malloc(sendProcCnt * sizeof(int));
          sendList = (int **) malloc(sendProcCnt * sizeof(int*));
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
       for (i = 0; i < recvProcCnt; i++)
       {
          MPI_Send((void*) &recvLeng[i],1,MPI_INT,recvProc[i],msgid,comm);
       }
       for (i = 0; i < sendProcCnt; i++) 
       {
          MPI_Recv((void*) &sendLeng[i],1,MPI_INT,MPI_ANY_SOURCE,msgid,
                   comm,&status);
          sendProc[i] = status.MPI_SOURCE;
          sendList[i] = (int *) malloc(sendLeng[i] * sizeof(int));
          if (sendList[i] == NULL) 
             printf("allocate problem %d \n", sendLeng[i]);
       }

       /* ----------------------------------------------------- */ 
       /* each processor sends to all processors it expects to  */
       /* receive data about the equation numbers               */
       /* ----------------------------------------------------- */ 

       for (i = 0; i < nprocs; i++) tempCnt[i] = 0; 
       ncnt = 1;
       for (i = 0; i < externLeng; i++) 
       {
          if ( externList[i] >= partition[ncnt] )
          {
             tempCnt[ncnt-1] = i;
             i--;
             ncnt++;
          }
       }    
       for (i = ncnt-1; i < nprocs; i++) tempCnt[i] = externLeng; 

       /* ----------------------------------------------------- */ 
       /* send the global equation numbers                      */
       /* ----------------------------------------------------- */ 

       if (sendProcCnt > 0)
          requests = (MPI_Request *) malloc(sendProcCnt*sizeof(MPI_Request));

       msgid = 540;
       for (i = 0; i < sendProcCnt; i++) 
       {
          MPI_Irecv((void*)sendList[i],sendLeng[i],MPI_INT,sendProc[i],
                    msgid,comm,&requests[i]);
       }
       for (i = 0; i < recvProcCnt; i++) 
       {
          if (recvProc[i] == 0) j = 0;
          else                  j = tempCnt[recvProc[i]-1];
          rowLeng = recvLeng[i];
          MPI_Send((void*) &externList[j], rowLeng, MPI_INT, recvProc[i],
                   msgid, comm);
       }
       for (i = 0; i < sendProcCnt; i++) 
       {
          MPI_Wait( &requests[i], &status );
       }
       if (sendProcCnt > 0) free(requests);

       /* ----------------------------------------------------- */ 
       /* convert the send list from global to local numbers    */
       /* ----------------------------------------------------- */ 

       for (i = 0; i < sendProcCnt; i++)
       { 
          for (j = 0; j < sendLeng[i]; j++)
          {
             index = sendList[i][j] - startRow;
             if (index < 0 || index >= localEqns)
             {
                printf("%d : Construct ML matrix Error - index out ",mypid);
                printf("of range %d\n", index);
             }
             sendList[i][j] = index;
          }
       }

       /* ----------------------------------------------------- */ 
       /* convert the send list from global to local numbers    */
       /* ----------------------------------------------------- */ 

       ml_mat->sendProcCnt = sendProcCnt;
       ml_mat->recvProcCnt = recvProcCnt;
       ml_mat->sendLeng    = sendLeng;
       ml_mat->recvLeng    = recvLeng;
       ml_mat->sendProc    = sendProc;
       ml_mat->recvProc    = recvProc;
       ml_mat->sendList    = sendList;

       /* ----------------------------------------------------- */ 
       /* clean up                                              */
       /* ----------------------------------------------------- */ 

       free(tempCnt);
    }
    return 0;
#else
    nprocs = 1;
    return (nprocs-1);
#endif
}

