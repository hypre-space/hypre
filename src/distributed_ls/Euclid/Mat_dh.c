/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_Euclid.h"
/* #include "Mat_dh.h" */
/* #include "getRow_dh.h" */
/* #include "SubdomainGraph_dh.h" */
/* #include "TimeLog_dh.h" */
/* #include "Mem_dh.h" */
/* #include "Numbering_dh.h" */
/* #include "Parser_dh.h" */
/* #include "mat_dh_private.h" */
/* #include "io_dh.h" */
/* #include "Hash_i_dh.h" */

static void setup_matvec_sends_private(Mat_dh mat, HYPRE_Int *inlist);
static void setup_matvec_receives_private(Mat_dh mat, HYPRE_Int *beg_rows, HYPRE_Int *end_rows,
                           HYPRE_Int reqlen, HYPRE_Int *reqind, HYPRE_Int *outlist);

#if 0

partial (?) implementation below; not used anyplace, I think;
for future expansion?  [mar 21, 2K+1]

static void Mat_dhAllocate_getRow_private(Mat_dh A);
#endif

static bool commsOnly = false;  /* experimental, for matvec functions */

#undef __FUNC__
#define __FUNC__ "Mat_dhCreate"
void Mat_dhCreate(Mat_dh *mat)
{
  START_FUNC_DH
  struct _mat_dh* tmp = (struct _mat_dh*)MALLOC_DH(sizeof(struct _mat_dh)); CHECK_V_ERROR;
  *mat = tmp;

  commsOnly = Parser_dhHasSwitch(parser_dh, "-commsOnly");
  if (myid_dh == 0 && commsOnly == true) {
/*     hypre_printf("\n@@@ commsOnly == true for matvecs! @@@\n"); */
    fflush(stdout);
  }

  tmp->m = 0;
  tmp->n = 0;
  tmp->beg_row = 0;
  tmp->bs = 1;

  tmp->rp = NULL;
  tmp->len = NULL;
  tmp->cval = NULL;
  tmp->aval = NULL;
  tmp->diag = NULL;
  tmp->fill = NULL;
  tmp->owner = true;

  tmp->len_private = 0;
  tmp->rowCheckedOut = -1;
  tmp->cval_private = NULL;
  tmp->aval_private = NULL;

  tmp->row_perm = NULL;

  tmp->num_recv = 0;
  tmp->num_send = 0;
  tmp->recv_req = NULL;
  tmp->send_req = NULL;
  tmp->status = NULL;
  tmp->recvbuf = NULL;
  tmp->sendbuf = NULL;
  tmp->sendind = NULL;
  tmp->sendlen = 0;
  tmp->recvlen = 0;
  tmp->numb = NULL;
  tmp->matvecIsSetup = false;

  Mat_dhZeroTiming(tmp); CHECK_V_ERROR;
  tmp->matvec_timing = true;

  tmp->debug = Parser_dhHasSwitch(parser_dh, "-debug_Mat");
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "Mat_dhDestroy"
void Mat_dhDestroy(Mat_dh mat)
{
  START_FUNC_DH
  HYPRE_Int i;

  if (mat->owner) {
    if (mat->rp != NULL) { FREE_DH(mat->rp); CHECK_V_ERROR; }
    if (mat->len != NULL) { FREE_DH(mat->len); CHECK_V_ERROR; }
    if (mat->cval != NULL) { FREE_DH(mat->cval); CHECK_V_ERROR; }
    if (mat->aval != NULL) { FREE_DH(mat->aval); CHECK_V_ERROR; }
    if (mat->diag != NULL) { FREE_DH(mat->diag); CHECK_V_ERROR; }
    if (mat->fill != NULL) { FREE_DH(mat->fill); CHECK_V_ERROR; }
    if (mat->cval_private != NULL) { FREE_DH(mat->cval_private); CHECK_V_ERROR; }
    if (mat->aval_private != NULL) { FREE_DH(mat->aval_private); CHECK_V_ERROR; }
    if (mat->row_perm != NULL) { FREE_DH(mat->row_perm); CHECK_V_ERROR; }
  }

  for (i=0; i<mat->num_recv; i++) hypre_MPI_Request_free(&mat->recv_req[i]);
  for (i=0; i<mat->num_send; i++) hypre_MPI_Request_free(&mat->send_req[i]);
  if (mat->recv_req != NULL) { FREE_DH(mat->recv_req); CHECK_V_ERROR; }
  if (mat->send_req != NULL) { FREE_DH(mat->send_req); CHECK_V_ERROR; }
  if (mat->status != NULL) { FREE_DH(mat->status); CHECK_V_ERROR; }
  if (mat->recvbuf != NULL) { FREE_DH(mat->recvbuf); CHECK_V_ERROR; }
  if (mat->sendbuf != NULL) { FREE_DH(mat->sendbuf); CHECK_V_ERROR; }
  if (mat->sendind != NULL) { FREE_DH(mat->sendind); CHECK_V_ERROR; }

  if (mat->matvecIsSetup) {
    Mat_dhMatVecSetdown(mat); CHECK_V_ERROR;
  }
  if (mat->numb != NULL) { Numbering_dhDestroy(mat->numb); CHECK_V_ERROR; }
  FREE_DH(mat); CHECK_V_ERROR;
  END_FUNC_DH
}


/* this should put the cval array back the way it was! */
#undef __FUNC__
#define __FUNC__ "Mat_dhMatVecSetDown"
void Mat_dhMatVecSetdown(Mat_dh mat)
{
  HYPRE_UNUSED_VAR(mat);

  START_FUNC_DH
  if (ignoreMe) SET_V_ERROR("not implemented");
  END_FUNC_DH
}


/* adopted from Edmond Chow's ParaSails */
#undef __FUNC__
#define __FUNC__ "Mat_dhMatVecSetup"
void Mat_dhMatVecSetup(Mat_dh mat)
{
  START_FUNC_DH
  if (np_dh == 1) {
    goto DO_NOTHING;
  }

  else {
    HYPRE_Int *outlist, *inlist;
    HYPRE_Int ierr, i, row, *rp = mat->rp, *cval = mat->cval;
    Numbering_dh numb;
    HYPRE_Int m = mat->m;
    HYPRE_Int firstLocal = mat->beg_row;
    HYPRE_Int lastLocal = firstLocal+m;
    HYPRE_Int *beg_rows, *end_rows;

    mat->recv_req = (hypre_MPI_Request *)MALLOC_DH(np_dh * sizeof(hypre_MPI_Request)); CHECK_V_ERROR;
    mat->send_req = (hypre_MPI_Request *)MALLOC_DH(np_dh * sizeof(hypre_MPI_Request)); CHECK_V_ERROR;
    mat->status = (hypre_MPI_Status *)MALLOC_DH(np_dh * sizeof(hypre_MPI_Status)); CHECK_V_ERROR;
    beg_rows = (HYPRE_Int*)MALLOC_DH(np_dh*sizeof(HYPRE_Int)); CHECK_V_ERROR;
    end_rows = (HYPRE_Int*)MALLOC_DH(np_dh*sizeof(HYPRE_Int)); CHECK_V_ERROR;

    if (np_dh == 1) { /* this is for debugging purposes in some of the drivers */
      beg_rows[0] = 0;
      end_rows[0] = m;
    } else {
      ierr = hypre_MPI_Allgather(&firstLocal, 1, HYPRE_MPI_INT, beg_rows, 1, HYPRE_MPI_INT, comm_dh);

  CHECK_MPI_V_ERROR(ierr);

      ierr = hypre_MPI_Allgather(&lastLocal, 1, HYPRE_MPI_INT, end_rows, 1, HYPRE_MPI_INT, comm_dh); CHECK_MPI_V_ERROR(ierr);
    }

    outlist = (HYPRE_Int *)MALLOC_DH(np_dh*sizeof(HYPRE_Int)); CHECK_V_ERROR;
    inlist  = (HYPRE_Int *)MALLOC_DH(np_dh*sizeof(HYPRE_Int)); CHECK_V_ERROR;
    for (i=0; i<np_dh; ++i) {
      outlist[i] = 0;
      inlist[i] = 0;
    }

    /* Create Numbering object */
    Numbering_dhCreate(&(mat->numb)); CHECK_V_ERROR;
    numb = mat->numb;
    Numbering_dhSetup(numb, mat); CHECK_V_ERROR;

    setup_matvec_receives_private(mat, beg_rows, end_rows, numb->num_ext,
           numb->idx_ext, outlist); CHECK_V_ERROR;

    if (np_dh == 1) { /* this is for debugging purposes in some of the drivers */
      inlist[0] = outlist[0];
    } else {
      ierr = hypre_MPI_Alltoall(outlist, 1, HYPRE_MPI_INT, inlist, 1, HYPRE_MPI_INT, comm_dh); CHECK_MPI_V_ERROR(ierr);
    }

    setup_matvec_sends_private(mat, inlist); CHECK_V_ERROR;

    /* Convert to local indices */
    for (row=0; row<m; row++) {
      HYPRE_Int len = rp[row+1]-rp[row];
      HYPRE_Int *ind = cval+rp[row];
      Numbering_dhGlobalToLocal(numb, len, ind, ind); CHECK_V_ERROR;
    }

    FREE_DH(outlist); CHECK_V_ERROR;
    FREE_DH(inlist); CHECK_V_ERROR;
    FREE_DH(beg_rows); CHECK_V_ERROR;
    FREE_DH(end_rows); CHECK_V_ERROR;
  }

DO_NOTHING: ;

  END_FUNC_DH
}

/* adopted from Edmond Chow's ParaSails */
#undef __FUNC__
#define __FUNC__ "setup_matvec_receives_private"
void setup_matvec_receives_private(Mat_dh mat, HYPRE_Int *beg_rows, HYPRE_Int *end_rows,
                           HYPRE_Int reqlen, HYPRE_Int *reqind, HYPRE_Int *outlist)
{
  START_FUNC_DH
  HYPRE_Int ierr, i, j, this_pe;
  hypre_MPI_Request request;
  HYPRE_Int m = mat->m;

  mat->num_recv = 0;

  /* Allocate recvbuf */
  /* recvbuf has numlocal entries saved for local part of x, used in matvec */
  mat->recvbuf = (HYPRE_Real*)MALLOC_DH((reqlen+m) * sizeof(HYPRE_Real));

  for (i=0; i<reqlen; i=j) { /* j is set below */
    /* The processor that owns the row with index reqind[i] */
    this_pe = mat_find_owner(beg_rows, end_rows, reqind[i]); CHECK_V_ERROR;

    /* Figure out other rows we need from this_pe */
    for (j=i+1; j<reqlen; j++) {
      /* if row is on different pe */
      if (reqind[j] < beg_rows[this_pe] ||
             reqind[j] > end_rows[this_pe])
        break;
    }

    /* Request rows in reqind[i..j-1] */
    ierr = hypre_MPI_Isend(&reqind[i], j-i, HYPRE_MPI_INT, this_pe, 444, comm_dh, &request); CHECK_MPI_V_ERROR(ierr);
    ierr = hypre_MPI_Request_free(&request); CHECK_MPI_V_ERROR(ierr);

    /* Count of number of number of indices needed from this_pe */
    outlist[this_pe] = j-i;

    ierr = hypre_MPI_Recv_init(&mat->recvbuf[i+m], j-i, hypre_MPI_REAL, this_pe, 555,
            comm_dh, &mat->recv_req[mat->num_recv]); CHECK_MPI_V_ERROR(ierr);

    mat->num_recv++;
    mat->recvlen += j-i;  /* only used for statistical reporting */
  }
  END_FUNC_DH
}


/* adopted from Edmond Chow's ParaSails */
#undef __FUNC__
#define __FUNC__ "setup_matvec_sends_private"
void setup_matvec_sends_private(Mat_dh mat, HYPRE_Int *inlist)
{
  START_FUNC_DH
  HYPRE_Int ierr, i, j, sendlen, first = mat->beg_row;
  hypre_MPI_Request *requests;
  hypre_MPI_Status  *statuses;

  requests = (hypre_MPI_Request *) MALLOC_DH(np_dh * sizeof(hypre_MPI_Request)); CHECK_V_ERROR;
  statuses = (hypre_MPI_Status *)  MALLOC_DH(np_dh * sizeof(hypre_MPI_Status)); CHECK_V_ERROR;

  /* Determine size of and allocate sendbuf and sendind */
  sendlen = 0;
  for (i=0; i<np_dh; i++) sendlen += inlist[i];
  mat->sendlen = sendlen;
  mat->sendbuf = (HYPRE_Real *)MALLOC_DH(sendlen * sizeof(HYPRE_Real)); CHECK_V_ERROR;
  mat->sendind = (HYPRE_Int *)MALLOC_DH(sendlen * sizeof(HYPRE_Int)); CHECK_V_ERROR;

  j = 0;
  mat->num_send = 0;
  for (i=0; i<np_dh; i++) {
    if (inlist[i] != 0) {
      /* Post receive for the actual indices */
      ierr = hypre_MPI_Irecv(&mat->sendind[j], inlist[i], HYPRE_MPI_INT, i, 444, comm_dh,
                            &requests[mat->num_send]); CHECK_MPI_V_ERROR(ierr);
      /* Set up the send */
      ierr = hypre_MPI_Send_init(&mat->sendbuf[j], inlist[i], hypre_MPI_REAL, i, 555, comm_dh,
                       &mat->send_req[mat->num_send]); CHECK_MPI_V_ERROR(ierr);

      mat->num_send++;
      j += inlist[i];
    }
  }

  /* total bytes to be sent during matvec */
  mat->time[MATVEC_WORDS] = j;


  ierr = hypre_MPI_Waitall(mat->num_send, requests, statuses); CHECK_MPI_V_ERROR(ierr);
  /* convert global indices to local indices */
  /* these are all indices on this processor */
  for (i=0; i<mat->sendlen; i++) mat->sendind[i] -= first;

  FREE_DH(requests);
  FREE_DH(statuses);
  END_FUNC_DH
}


/* unthreaded MPI version */
#undef __FUNC__
#define __FUNC__ "Mat_dhMatVec"
void Mat_dhMatVec(Mat_dh mat, HYPRE_Real *x, HYPRE_Real *b)
{
  START_FUNC_DH
  if (np_dh == 1) {
    Mat_dhMatVec_uni(mat, x, b); CHECK_V_ERROR;
  }

  else {
    HYPRE_Int    ierr, i, row, m = mat->m;
    HYPRE_Int    *rp = mat->rp, *cval = mat->cval;
    HYPRE_Real *aval = mat->aval;
    HYPRE_Int    *sendind = mat->sendind;
    HYPRE_Int    sendlen = mat->sendlen;
    HYPRE_Real *sendbuf = mat->sendbuf;
    HYPRE_Real *recvbuf = mat->recvbuf;
    HYPRE_Real t1 = 0, t2 = 0, t3 = 0, t4 = 0;
    bool   timeFlag = mat->matvec_timing;


    if (timeFlag) t1 = hypre_MPI_Wtime();

    /* Put components of x into the right outgoing buffers */
    if (! commsOnly) {
      for (i=0; i<sendlen; i++) sendbuf[i] = x[sendind[i]];
    }

    if (timeFlag) {
      t2 = hypre_MPI_Wtime();
      mat->time[MATVEC_TIME] += (t2 - t1);

    }

    ierr = hypre_MPI_Startall(mat->num_recv, mat->recv_req); CHECK_MPI_V_ERROR(ierr);
    ierr = hypre_MPI_Startall(mat->num_send, mat->send_req); CHECK_MPI_V_ERROR(ierr);
    ierr = hypre_MPI_Waitall(mat->num_recv, mat->recv_req, mat->status); CHECK_MPI_V_ERROR(ierr);
    ierr = hypre_MPI_Waitall(mat->num_send, mat->send_req, mat->status); CHECK_MPI_V_ERROR(ierr);


    if (timeFlag) {
      t3 = hypre_MPI_Wtime();
      mat->time[MATVEC_MPI_TIME] += (t3 - t2);
    }

   /* Copy local part of x into top part of recvbuf */
   if (! commsOnly) {
      for (i=0; i<m; i++) recvbuf[i] = x[i];

    /* do the multiply */
    for (row=0; row<m; row++) {
      HYPRE_Int len = rp[row+1] - rp[row];
      HYPRE_Int * ind = cval+rp[row];
      HYPRE_Real * val = aval+rp[row];
      HYPRE_Real temp = 0.0;
      for (i=0; i<len; i++) {
        temp += (val[i] * recvbuf[ind[i]]);
      }
      b[row] = temp;
    }
  } /* if (! commsOnly) */

    if (timeFlag) {
      t4 = hypre_MPI_Wtime();
      mat->time[MATVEC_TOTAL_TIME] += (t4 - t1);
      mat->time[MATVEC_TIME] += (t4 - t3);
    }
  }
  END_FUNC_DH
}

/* OpenMP/MPI version */
#undef __FUNC__
#define __FUNC__ "Mat_dhMatVec_omp"
void Mat_dhMatVec_omp(Mat_dh mat, HYPRE_Real *x, HYPRE_Real *b)
{
  START_FUNC_DH
  HYPRE_Int    ierr, i, row, m = mat->m;
  HYPRE_Int    *rp = mat->rp, *cval = mat->cval;
  HYPRE_Real *aval = mat->aval;
  HYPRE_Int    *sendind = mat->sendind;
  HYPRE_Int    sendlen = mat->sendlen;
  HYPRE_Real *sendbuf = mat->sendbuf;
  HYPRE_Real *recvbuf = mat->recvbuf;
  HYPRE_Real t1 = 0, t2 = 0, t3 = 0, t4 = 0, tx = 0;
  HYPRE_Real *val, temp;
  HYPRE_Int len, *ind;
  bool   timeFlag = mat->matvec_timing;

  if (timeFlag) t1 = hypre_MPI_Wtime();

  /* Put components of x into the right outgoing buffers */
#ifdef USING_OPENMP_DH
#pragma omp parallel  for schedule(runtime) private(i)
#endif
  for (i=0; i<sendlen; i++) sendbuf[i] = x[sendind[i]];

  if (timeFlag) {
    t2 = hypre_MPI_Wtime();
    mat->time[MATVEC_TIME] += (t2 - t1);
  }

  ierr = hypre_MPI_Startall(mat->num_recv, mat->recv_req); CHECK_MPI_V_ERROR(ierr);
  ierr = hypre_MPI_Startall(mat->num_send, mat->send_req); CHECK_MPI_V_ERROR(ierr);
  ierr = hypre_MPI_Waitall(mat->num_recv, mat->recv_req, mat->status); CHECK_MPI_V_ERROR(ierr);
  ierr = hypre_MPI_Waitall(mat->num_send, mat->send_req, mat->status); CHECK_MPI_V_ERROR(ierr);

  if (timeFlag) {
    t3 = hypre_MPI_Wtime();
    mat->time[MATVEC_MPI_TIME] += (t3 - t2);
  }

  /* Copy local part of x into top part of recvbuf */
#ifdef USING_OPENMP_DH
#pragma omp parallel  for schedule(runtime) private(i)
#endif
  for (i=0; i<m; i++) recvbuf[i] = x[i];

  if (timeFlag) {
    tx = hypre_MPI_Wtime();
    mat->time[MATVEC_MPI_TIME2] += (tx - t1);
  }


  /* do the multiply */
#ifdef USING_OPENMP_DH
#pragma omp parallel  for schedule(runtime) private(row,i,len,ind,val,temp)
#endif
  for (row=0; row<m; row++) {
    len = rp[row+1] - rp[row];
    ind = cval+rp[row];
    val = aval+rp[row];
    temp = 0.0;
    for (i=0; i<len; i++) {
      temp += (val[i] * recvbuf[ind[i]]);
    }
    b[row] = temp;
  }

  if (timeFlag) {
    t4 = hypre_MPI_Wtime();
    mat->time[MATVEC_TOTAL_TIME] += (t4 - t1);
    mat->time[MATVEC_TIME] += (t4 - t3);
  }

  END_FUNC_DH
}


/* OpenMP/single primary task version */
#undef __FUNC__
#define __FUNC__ "Mat_dhMatVec_uni_omp"
void Mat_dhMatVec_uni_omp(Mat_dh mat, HYPRE_Real *x, HYPRE_Real *b)
{
  START_FUNC_DH
  HYPRE_Int    i, row, m = mat->m;
  HYPRE_Int    *rp = mat->rp, *cval = mat->cval;
  HYPRE_Real *aval = mat->aval;
  HYPRE_Real t1 = 0, t2 = 0;
  bool   timeFlag = mat->matvec_timing;

  if (timeFlag) { t1 = hypre_MPI_Wtime(); }

  /* do the multiply */
#ifdef USING_OPENMP_DH
#pragma omp parallel  for schedule(runtime) private(row,i)
#endif
  for (row=0; row<m; row++) {
    HYPRE_Int len = rp[row+1] - rp[row];
    HYPRE_Int * ind = cval+rp[row];
    HYPRE_Real * val = aval+rp[row];
    HYPRE_Real temp = 0.0;
    for (i=0; i<len; i++) {
      temp += (val[i] * x[ind[i]]);
    }
    b[row] = temp;
  }

  if (timeFlag) {
    t2 = hypre_MPI_Wtime();
    mat->time[MATVEC_TIME] += (t2 - t1);
    mat->time[MATVEC_TOTAL_TIME] += (t2 - t1);
  }

  END_FUNC_DH
}


/* unthreaded, single-task version */
#undef __FUNC__
#define __FUNC__ "Mat_dhMatVec_uni"
void Mat_dhMatVec_uni(Mat_dh mat, HYPRE_Real *x, HYPRE_Real *b)
{
  START_FUNC_DH
  HYPRE_Int    i, row, m = mat->m;
  HYPRE_Int    *rp = mat->rp, *cval = mat->cval;
  HYPRE_Real *aval = mat->aval;
  HYPRE_Real t1 = 0, t2 = 0;
  bool   timeFlag = mat->matvec_timing;

  if (timeFlag) t1 = hypre_MPI_Wtime();

  for (row=0; row<m; row++) {
    HYPRE_Int len = rp[row+1] - rp[row];
    HYPRE_Int * ind = cval+rp[row];
    HYPRE_Real * val = aval+rp[row];
    HYPRE_Real temp = 0.0;
    for (i=0; i<len; i++) {
      temp += (val[i] * x[ind[i]]);
    }
    b[row] = temp;
  }

  if (timeFlag)  {
    t2 = hypre_MPI_Wtime();
    mat->time[MATVEC_TIME] += (t2 - t1);
    mat->time[MATVEC_TOTAL_TIME] += (t2 - t1);
  }

  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "Mat_dhReadNz"
HYPRE_Int Mat_dhReadNz(Mat_dh mat)
{
  START_FUNC_DH
  HYPRE_Int ierr, retval = mat->rp[mat->m];
  HYPRE_Int nz = retval;
  ierr = hypre_MPI_Allreduce(&nz, &retval, 1, HYPRE_MPI_INT, hypre_MPI_SUM, comm_dh); CHECK_MPI_ERROR(ierr);
  END_FUNC_VAL(retval)
}



#if 0

#undef __FUNC__
#define __FUNC__ "Mat_dhAllocate_getRow_private"
void Mat_dhAllocate_getRow_private(Mat_dh A)
{
  START_FUNC_DH
  HYPRE_Int i, *rp = A->rp, len = 0;
  HYPRE_Int m = A->m;

  /* find longest row in matrix */
  for (i=0; i<m; ++i) len = MAX(len, rp[i+1]-rp[i]);
  len *= A->bs;

  /* free any previously allocated private storage */
  if (len > A->len_private) {
    if (A->cval_private != NULL) { FREE_DH(A->cval_private); CHECK_V_ERROR; }
    if (A->aval_private != NULL) { FREE_DH(A->aval_private); CHECK_V_ERROR; }
  }

  /* allocate private storage */
  A->cval_private = (HYPRE_Int*)MALLOC_DH(len*sizeof(HYPRE_Int)); CHECK_V_ERROR;
  A->aval_private = (HYPRE_Real*)MALLOC_DH(len*sizeof(HYPRE_Real)); CHECK_V_ERROR;
  A->len_private = len;
  END_FUNC_DH
}

#endif

#undef __FUNC__
#define __FUNC__ "Mat_dhZeroTiming"
void Mat_dhZeroTiming(Mat_dh mat)
{
  START_FUNC_DH
  HYPRE_Int i;

  for (i=0; i<MAT_DH_BINS; ++i) {
    mat->time[i] = 0;
    mat->time_max[i] = 0;
    mat->time_min[i] = 0;
  }
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "Mat_dhReduceTiming"
void Mat_dhReduceTiming(Mat_dh mat)
{
  START_FUNC_DH
  if (mat->time[MATVEC_MPI_TIME]) {
    mat->time[MATVEC_RATIO] = mat->time[MATVEC_TIME] / mat->time[MATVEC_MPI_TIME];
  }
  hypre_MPI_Allreduce(mat->time, mat->time_min, MAT_DH_BINS, hypre_MPI_REAL, hypre_MPI_MIN, comm_dh);
  hypre_MPI_Allreduce(mat->time, mat->time_max, MAT_DH_BINS, hypre_MPI_REAL, hypre_MPI_MAX, comm_dh);
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "Mat_dhPermute"
void Mat_dhPermute(Mat_dh A, HYPRE_Int *n2o, Mat_dh *Bout)
{
  START_FUNC_DH
  Mat_dh B;
  HYPRE_Int  i, j, *RP = A->rp, *CVAL = A->cval;
  HYPRE_Int  *o2n, *rp, *cval, m = A->m, nz = RP[m];
  HYPRE_Real *aval, *AVAL = A->aval;

  Mat_dhCreate(&B); CHECK_V_ERROR;
  B->m = B->n = m;
  *Bout = B;

  /* form inverse permutation */
  o2n = (HYPRE_Int*)MALLOC_DH(m*sizeof(HYPRE_Int)); CHECK_V_ERROR;
  for (i=0; i<m; ++i) o2n[n2o[i]] = i;

  /* allocate storage for permuted matrix */
  rp = B->rp = (HYPRE_Int*)MALLOC_DH((m+1)*sizeof(HYPRE_Int)); CHECK_V_ERROR;
  cval = B->cval = (HYPRE_Int*)MALLOC_DH(nz*sizeof(HYPRE_Int)); CHECK_V_ERROR;
  aval = B->aval = (HYPRE_Real*)MALLOC_DH(nz*sizeof(HYPRE_Real)); CHECK_V_ERROR;

  /* form new rp array */
  rp[0] = 0;
  for (i=0; i<m; ++i) {
    HYPRE_Int oldRow = n2o[i];
    rp[i+1] = RP[oldRow+1]-RP[oldRow];
  }
  for (i=1; i<=m; ++i) rp[i] = rp[i] + rp[i-1];

  for (i=0; i<m; ++i) {
    HYPRE_Int oldRow = n2o[i];
    HYPRE_Int idx = rp[i];
    for (j=RP[oldRow]; j<RP[oldRow+1]; ++j) {
      cval[idx] = o2n[CVAL[j]];
      aval[idx] = AVAL[j];
      ++idx;
    }
  }

  FREE_DH(o2n); CHECK_V_ERROR;
  END_FUNC_DH
}


/*----------------------------------------------------------------------
 * Print methods
 *----------------------------------------------------------------------*/

/* seq or mpi */
#undef __FUNC__
#define __FUNC__ "Mat_dhPrintGraph"
void Mat_dhPrintGraph(Mat_dh A, SubdomainGraph_dh sg, FILE *fp)
{
  START_FUNC_DH
  HYPRE_Int pe, id = myid_dh;
  HYPRE_Int ierr;

  if (sg != NULL) {
    id = sg->o2n_sub[id];
  }

  for (pe=0; pe<np_dh; ++pe) {
    ierr = hypre_MPI_Barrier(comm_dh); CHECK_MPI_V_ERROR(ierr);
    if (id == pe) {
      if (sg == NULL) {
        mat_dh_print_graph_private(A->m, A->beg_row, A->rp, A->cval,
                  A->aval, NULL, NULL, NULL, fp); CHECK_V_ERROR;
      } else {
        HYPRE_Int beg_row = sg->beg_rowP[myid_dh];
        mat_dh_print_graph_private(A->m, beg_row, A->rp, A->cval,
                  A->aval, sg->n2o_row, sg->o2n_col, sg->o2n_ext, fp); CHECK_V_ERROR;
      }
    }
  }
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "Mat_dhPrintRows"
void Mat_dhPrintRows(Mat_dh A, SubdomainGraph_dh sg, FILE *fp)
{
  START_FUNC_DH
  bool noValues;
  HYPRE_Int m = A->m, *rp = A->rp, *cval = A->cval;
  HYPRE_Real *aval = A->aval;

  noValues = (Parser_dhHasSwitch(parser_dh, "-noValues"));
  if (noValues) aval = NULL;

  /*----------------------------------------------------------------
   * case 1: print local portion of unpermuted matrix
   *----------------------------------------------------------------*/
  if (sg == NULL) {
    HYPRE_Int i, j;
    HYPRE_Int beg_row = A->beg_row;

    hypre_fprintf(fp, "\n----- A, unpermuted ------------------------------------\n");
    for (i=0; i<m; ++i) {
      hypre_fprintf(fp, "%i :: ", 1+i+beg_row);
      for (j=rp[i]; j<rp[i+1]; ++j) {
        if (noValues) {
          hypre_fprintf(fp, "%i ", 1+cval[j]);
        } else {
          hypre_fprintf(fp, "%i,%g ; ", 1+cval[j], aval[j]);
        }
      }
      hypre_fprintf(fp, "\n");
    }
  }

  /*----------------------------------------------------------------
   * case 2: single mpi task, with multiple subdomains
   *----------------------------------------------------------------*/
  else if (np_dh == 1) {
    HYPRE_Int i, k, idx = 1;
    HYPRE_Int oldRow;

    for (i=0; i<sg->blocks; ++i) {
      HYPRE_Int oldBlock = sg->n2o_sub[i];

      /* here, 'beg_row' and 'end_row' refer to rows in the
         original ordering of A.
      */
      HYPRE_Int beg_row = sg->beg_row[oldBlock];
      HYPRE_Int end_row = beg_row + sg->row_count[oldBlock];

      hypre_fprintf(fp, "\n");
      hypre_fprintf(fp, "\n----- A, permuted, single mpi task  ------------------\n");
      hypre_fprintf(fp, "---- new subdomain: %i;  old subdomain: %i\n", i, oldBlock);
      hypre_fprintf(fp, "     old beg_row:   %i;  new beg_row:   %i\n",
                                sg->beg_row[oldBlock], sg->beg_rowP[oldBlock]);
      hypre_fprintf(fp, "     local rows in this block: %i\n", sg->row_count[oldBlock]);
      hypre_fprintf(fp, "     bdry rows in this block:  %i\n", sg->bdry_count[oldBlock]);
      hypre_fprintf(fp, "     1st bdry row= %i \n", 1+end_row-sg->bdry_count[oldBlock]);

      for (oldRow=beg_row; oldRow<end_row; ++oldRow) {
        HYPRE_Int len = 0, *cval;
        HYPRE_Real *aval;

        hypre_fprintf(fp, "%3i (old= %3i) :: ", idx, 1+oldRow);
        ++idx;
        Mat_dhGetRow(A, oldRow, &len, &cval, &aval); CHECK_V_ERROR;

        for (k=0; k<len; ++k) {
          if (noValues) {
            hypre_fprintf(fp, "%i ", 1+sg->o2n_col[cval[k]]);
          } else {
            hypre_fprintf(fp, "%i,%g ; ", 1+sg->o2n_col[cval[k]], aval[k]);
          }
        }

        hypre_fprintf(fp, "\n");
        Mat_dhRestoreRow(A, oldRow, &len, &cval, &aval); CHECK_V_ERROR;
      }
    }
  }

  /*----------------------------------------------------------------
   * case 3: multiple mpi tasks, one subdomain per task
   *----------------------------------------------------------------*/
  else {
    Hash_i_dh hash = sg->o2n_ext;
    HYPRE_Int *o2n_col = sg->o2n_col, *n2o_row = sg->n2o_row;
    HYPRE_Int beg_row = sg->beg_row[myid_dh];
    HYPRE_Int beg_rowP = sg->beg_rowP[myid_dh];
    HYPRE_Int i, j;

    for (i=0; i<m; ++i) {
      HYPRE_Int row = n2o_row[i];
      hypre_fprintf(fp, "%3i (old= %3i) :: ", 1+i+beg_rowP, 1+row+beg_row);
      for (j=rp[row]; j<rp[row+1]; ++j) {
        HYPRE_Int col = cval[j];

        /* find permuted (old-to-new) value for the column */
        /* case i: column is locally owned */
        if (col >= beg_row && col < beg_row+m) {
          col = o2n_col[col-beg_row] + beg_rowP;
        }

        /* case ii: column is external */
        else {
          HYPRE_Int tmp = col;
          tmp = Hash_i_dhLookup(hash, col); CHECK_V_ERROR;
          if (tmp == -1) {
            hypre_sprintf(msgBuf_dh, "nonlocal column= %i not in hash table", 1+col);
            SET_V_ERROR(msgBuf_dh);
          } else {
            col = tmp;
          }
        }

        if (noValues) {
          hypre_fprintf(fp, "%i ", 1+col);
        } else {
          hypre_fprintf(fp, "%i,%g ; ", 1+col, aval[j]);
        }
      }
      hypre_fprintf(fp, "\n");
    }
  }
  END_FUNC_DH
}



#undef __FUNC__
#define __FUNC__ "Mat_dhPrintTriples"
void Mat_dhPrintTriples(Mat_dh A, SubdomainGraph_dh sg, char *filename)
{
  START_FUNC_DH
  HYPRE_Int m = A->m, *rp = A->rp, *cval = A->cval;
  HYPRE_Real *aval = A->aval;
  bool noValues;
  bool matlab;
  FILE *fp;

  noValues = (Parser_dhHasSwitch(parser_dh, "-noValues"));
  if (noValues) aval = NULL;
  matlab = (Parser_dhHasSwitch(parser_dh, "-matlab"));

  /*----------------------------------------------------------------
   * case 1: unpermuted matrix, single or multiple mpi tasks
   *----------------------------------------------------------------*/
  if (sg == NULL) {
    HYPRE_Int i, j, pe;
    HYPRE_Int beg_row = A->beg_row;
    HYPRE_Real val;

    for (pe=0; pe<np_dh; ++pe) {
      hypre_MPI_Barrier(comm_dh);
      if (pe == myid_dh) {
        if (pe == 0) {
          fp=openFile_dh(filename, "w"); CHECK_V_ERROR;
        } else {
          fp=openFile_dh(filename, "a"); CHECK_V_ERROR;
        }

        for (i=0; i<m; ++i) {
          for (j=rp[i]; j<rp[i+1]; ++j) {
            if (noValues) {
              hypre_fprintf(fp, "%i %i\n", 1+i+beg_row, 1+cval[j]);
            } else {
              val = aval[j];
              if (val == 0.0 && matlab) val = _MATLAB_ZERO_;
              hypre_fprintf(fp, TRIPLES_FORMAT, 1+i+beg_row, 1+cval[j], val);
            }
          }
        }
        closeFile_dh(fp); CHECK_V_ERROR;
      }
    }
  }

  /*----------------------------------------------------------------
   * case 2: single mpi task, with multiple subdomains
   *----------------------------------------------------------------*/
  else if (np_dh == 1) {
    HYPRE_Int i, j, k, idx = 1;

    fp=openFile_dh(filename, "w"); CHECK_V_ERROR;

    for (i=0; i<sg->blocks; ++i) {
      HYPRE_Int oldBlock = sg->n2o_sub[i];
      HYPRE_Int beg_row = sg->beg_rowP[oldBlock];
      HYPRE_Int end_row = beg_row + sg->row_count[oldBlock];

      for (j=beg_row; j<end_row; ++j) {
        HYPRE_Int len = 0, *cval;
        HYPRE_Real *aval;
        HYPRE_Int oldRow = sg->n2o_row[j];

        Mat_dhGetRow(A, oldRow, &len, &cval, &aval); CHECK_V_ERROR;

        if (noValues) {
          for (k=0; k<len; ++k) {
            hypre_fprintf(fp, "%i %i\n", idx, 1+sg->o2n_col[cval[k]]);
          }
          ++idx;
        }

        else {
          for (k=0; k<len; ++k) {
            HYPRE_Real val = aval[k];
            if (val == 0.0 && matlab) val = _MATLAB_ZERO_;
            hypre_fprintf(fp, TRIPLES_FORMAT, idx, 1+sg->o2n_col[cval[k]], val);
          }
          ++idx;
        }
        Mat_dhRestoreRow(A, oldRow, &len, &cval, &aval); CHECK_V_ERROR;
      }
    }
  }

  /*----------------------------------------------------------------
   * case 3: multiple mpi tasks, one subdomain per task
   *----------------------------------------------------------------*/
  else {
    Hash_i_dh hash = sg->o2n_ext;
    HYPRE_Int *o2n_col = sg->o2n_col, *n2o_row = sg->n2o_row;
    HYPRE_Int beg_row = sg->beg_row[myid_dh];
    HYPRE_Int beg_rowP = sg->beg_rowP[myid_dh];
    HYPRE_Int i, j, pe;
    HYPRE_Int id = sg->o2n_sub[myid_dh];

    for (pe=0; pe<np_dh; ++pe) {
      hypre_MPI_Barrier(comm_dh);
      if (id == pe) {
        if (pe == 0) {
          fp=openFile_dh(filename, "w"); CHECK_V_ERROR;
        }
        else {
          fp=openFile_dh(filename, "a"); CHECK_V_ERROR;
        }

        for (i=0; i<m; ++i) {
          HYPRE_Int row = n2o_row[i];
          for (j=rp[row]; j<rp[row+1]; ++j) {
            HYPRE_Int col = cval[j];
            HYPRE_Real val = 0.0;

            if (aval != NULL) val = aval[j];
            if (val == 0.0 && matlab) val = _MATLAB_ZERO_;

            /* find permuted (old-to-new) value for the column */
            /* case i: column is locally owned */
            if (col >= beg_row && col < beg_row+m) {
              col = o2n_col[col-beg_row] + beg_rowP;
            }

            /* case ii: column is external */
            else {
              HYPRE_Int tmp = col;
              tmp = Hash_i_dhLookup(hash, col); CHECK_V_ERROR;
              if (tmp == -1) {
                hypre_sprintf(msgBuf_dh, "nonlocal column= %i not in hash table", 1+col);
                SET_V_ERROR(msgBuf_dh);
              } else {
                col = tmp;
              }
            }

            if (noValues) {
              hypre_fprintf(fp, "%i %i\n", 1+i+beg_rowP, 1+col);
            } else {
              hypre_fprintf(fp, TRIPLES_FORMAT, 1+i+beg_rowP, 1+col, val);
            }
          }
        }
        closeFile_dh(fp); CHECK_V_ERROR;
      }
    }
  }
  END_FUNC_DH
}


/* seq only */
#undef __FUNC__
#define __FUNC__ "Mat_dhPrintCSR"
void Mat_dhPrintCSR(Mat_dh A, SubdomainGraph_dh sg, char *filename)
{
  START_FUNC_DH
  FILE *fp;

  if (np_dh > 1) {
    SET_V_ERROR("only implemented for a single mpi task");
  }
  if (sg != NULL) {
    SET_V_ERROR("not implemented for reordered matrix (SubdomainGraph_dh should be NULL)");
  }

  fp=openFile_dh(filename, "w"); CHECK_V_ERROR;

  if (sg == NULL) {
    mat_dh_print_csr_private(A->m, A->rp, A->cval, A->aval, fp); CHECK_V_ERROR;
  } else {
    mat_dh_print_csr_private(A->m, A->rp, A->cval, A->aval, fp); CHECK_V_ERROR;
  }
  closeFile_dh(fp); CHECK_V_ERROR;
  END_FUNC_DH
}

/* seq */
/* no reordering */
#undef __FUNC__
#define __FUNC__ "Mat_dhPrintBIN"
void Mat_dhPrintBIN(Mat_dh A, SubdomainGraph_dh sg, char *filename)
{
  START_FUNC_DH

  if (np_dh > 1) {
    SET_V_ERROR("only implemented for a single MPI task");
  }
/*  if (n2o != NULL || o2n != NULL || hash != NULL) {
*/
  if (sg != NULL) {
    SET_V_ERROR("not implemented for reordering; ensure sg=NULL");
  }

  io_dh_print_ebin_mat_private(A->m, A->beg_row, A->rp, A->cval, A->aval,
                              NULL, NULL, NULL, filename); CHECK_V_ERROR;
  END_FUNC_DH
}


/*----------------------------------------------------------------------
 * Read methods
 *----------------------------------------------------------------------*/
/* seq only */
#undef __FUNC__
#define __FUNC__ "Mat_dhReadCSR"
void Mat_dhReadCSR(Mat_dh *mat, char *filename)
{
  START_FUNC_DH
  Mat_dh A;
  FILE *fp;

  if (np_dh > 1) {
    SET_V_ERROR("only implemented for a single MPI task");
  }

  fp=openFile_dh(filename, "r"); CHECK_V_ERROR;

  Mat_dhCreate(&A); CHECK_V_ERROR;
  mat_dh_read_csr_private(&A->m, &A->rp, &A->cval, &A->aval, fp); CHECK_V_ERROR;
  A->n = A->m;
  *mat = A;

  closeFile_dh(fp); CHECK_V_ERROR;
  END_FUNC_DH
}

/* seq only */
#undef __FUNC__
#define __FUNC__ "Mat_dhReadTriples"
void Mat_dhReadTriples(Mat_dh *mat, HYPRE_Int ignore, char *filename)
{
  START_FUNC_DH
  FILE *fp = NULL;
  Mat_dh A = NULL;

  if (np_dh > 1) {
    SET_V_ERROR("only implemented for a single MPI task");
  }

  fp=openFile_dh(filename, "r"); CHECK_V_ERROR;

  Mat_dhCreate(&A); CHECK_V_ERROR;
  mat_dh_read_triples_private(ignore, &A->m, &A->rp, &A->cval, &A->aval, fp); CHECK_V_ERROR;
  A->n = A->m;
  *mat = A;

  closeFile_dh(fp); CHECK_V_ERROR;
  END_FUNC_DH
}

/* here we pass the private function a filename, instead of an open file,
   the reason being that Euclid's binary format is more complicated,
   i.e, the other "Read" methods are only for a single mpi task.
*/
#undef __FUNC__
#define __FUNC__ "Mat_dhReadBIN"
void Mat_dhReadBIN(Mat_dh *mat, char *filename)
{
  START_FUNC_DH
  Mat_dh A;

  if (np_dh > 1) {
    SET_V_ERROR("only implemented for a single MPI task");
  }

  Mat_dhCreate(&A); CHECK_V_ERROR;
  io_dh_read_ebin_mat_private(&A->m, &A->rp, &A->cval, &A->aval, filename); CHECK_V_ERROR;
  A->n = A->m;
  *mat = A;
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "Mat_dhTranspose"
void Mat_dhTranspose(Mat_dh A, Mat_dh *Bout)
{
  START_FUNC_DH
  Mat_dh B;

  if (np_dh > 1) { SET_V_ERROR("only for sequential"); }

  Mat_dhCreate(&B); CHECK_V_ERROR;
  *Bout = B;
  B->m = B->n = A->m;
  mat_dh_transpose_private(A->m, A->rp, &B->rp, A->cval, &B->cval,
                            A->aval, &B->aval); CHECK_V_ERROR;
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "Mat_dhMakeStructurallySymmetric"
void Mat_dhMakeStructurallySymmetric(Mat_dh A)
{
  START_FUNC_DH
  if (np_dh > 1) { SET_V_ERROR("only for sequential"); }
  make_symmetric_private(A->m, &A->rp, &A->cval, &A->aval); CHECK_V_ERROR;
  END_FUNC_DH
}

void insert_diags_private(Mat_dh A, HYPRE_Int ct);

/* inserts diagonal if not explicitly present;
   sets diagonal value in row i to sum of absolute
   values of all elts in row i.
*/
#undef __FUNC__
#define __FUNC__ "Mat_dhFixDiags"
void Mat_dhFixDiags(Mat_dh A)
{
  START_FUNC_DH
  HYPRE_Int i, j;
  HYPRE_Int *rp = A->rp, *cval = A->cval, m = A->m;
  HYPRE_Int ct = 0;  /* number of missing diagonals */
  HYPRE_Real *aval = A->aval;

  /* determine if any diagonals are missing */
  for (i=0; i<m; ++i) {
    bool flag = true;
    for (j=rp[i]; j<rp[i+1]; ++j) {
      HYPRE_Int col = cval[j];
      if (col == i) {
        flag = false;
        break;
      }
    }
    if (flag) ++ct;
  }

  /* insert any missing diagonal elements */
  if (ct) {
    hypre_printf("\nMat_dhFixDiags:: %i diags not explicitly present; inserting!\n", ct);
    insert_diags_private(A, ct); CHECK_V_ERROR;
    rp = A->rp;
    cval = A->cval;
    aval = A->aval;
  }

  /* set the value of all diagonal elements */
  for (i=0; i<m; ++i) {
    HYPRE_Real sum = 0.0;
    for (j=rp[i]; j<rp[i+1]; ++j) {
      sum += hypre_abs(aval[j]);
    }
    for (j=rp[i]; j<rp[i+1]; ++j) {
      if (cval[j] == i) {
        aval[j] = sum;
      }
    }
  }
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "insert_diags_private"
void insert_diags_private(Mat_dh A, HYPRE_Int ct)
{
  START_FUNC_DH
  HYPRE_Int *RP = A->rp, *CVAL = A->cval;
  HYPRE_Int *rp, *cval, m = A->m;
  HYPRE_Real *aval, *AVAL = A->aval;
  HYPRE_Int nz = RP[m] + ct;
  HYPRE_Int i, j, idx = 0;

  rp = A->rp = (HYPRE_Int*)MALLOC_DH((m+1)*sizeof(HYPRE_Int)); CHECK_V_ERROR;
  cval = A->cval = (HYPRE_Int*)MALLOC_DH(nz*sizeof(HYPRE_Int)); CHECK_V_ERROR;
  aval = A->aval = (HYPRE_Real*)MALLOC_DH(nz*sizeof(HYPRE_Real)); CHECK_V_ERROR;
  rp[0] = 0;

  for (i=0; i<m; ++i) {
    bool flag = true;
    for (j=RP[i]; j<RP[i+1]; ++j) {
      cval[idx] = CVAL[j];
      aval[idx] = AVAL[j];
      ++idx;
      if (CVAL[j] == i) flag = false;
    }

    if (flag) {
      cval[idx] = i;
      aval[idx] = 0.0;
      ++idx;
    }
    rp[i+1] = idx;
  }

  FREE_DH(RP); CHECK_V_ERROR;
  FREE_DH(CVAL); CHECK_V_ERROR;
  FREE_DH(AVAL); CHECK_V_ERROR;

  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "Mat_dhPrintDiags"
void Mat_dhPrintDiags(Mat_dh A, FILE *fp)
{
  START_FUNC_DH
  HYPRE_Int i, j, m = A->m;
  HYPRE_Int *rp = A->rp, *cval = A->cval;
  HYPRE_Real *aval = A->aval;

  hypre_fprintf(fp, "=================== diagonal elements ====================\n");
  for (i=0; i<m; ++i) {
    bool flag = true;
    for (j=rp[i]; j<rp[i+1]; ++j) {
      if (cval[j] == i) {
        hypre_fprintf(fp, "%i  %g\n", i+1, aval[j]);
        flag = false;
        break;
      }
    }
    if (flag) {
      hypre_fprintf(fp, "%i  ---------- missing\n", i+1);
    }
  }
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "Mat_dhGetRow"
void Mat_dhGetRow(Mat_dh B, HYPRE_Int globalRow, HYPRE_Int *len, HYPRE_Int **ind, HYPRE_Real **val)
{
  START_FUNC_DH
  HYPRE_Int row = globalRow - B->beg_row;
  if (row > B->m) {
    hypre_sprintf(msgBuf_dh, "requested globalRow= %i, which is local row= %i, but only have %i rows!",
                                globalRow, row, B->m);
    SET_V_ERROR(msgBuf_dh);
  }
  *len = B->rp[row+1] - B->rp[row];
  if (ind != NULL) *ind = B->cval + B->rp[row];
  if (val != NULL) *val = B->aval + B->rp[row];
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "Mat_dhRestoreRow"
void Mat_dhRestoreRow(Mat_dh B, HYPRE_Int row, HYPRE_Int *len, HYPRE_Int **ind, HYPRE_Real **val)
{
  HYPRE_UNUSED_VAR(B);
  HYPRE_UNUSED_VAR(row);
  HYPRE_UNUSED_VAR(len);
  HYPRE_UNUSED_VAR(ind);
  HYPRE_UNUSED_VAR(val);

  START_FUNC_DH
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "Mat_dhRowPermute"
void Mat_dhRowPermute(Mat_dh mat)
{
  HYPRE_UNUSED_VAR(mat);

  START_FUNC_DH
  if (ignoreMe) SET_V_ERROR("turned off; compilation problem on blue");

#if 0
  HYPRE_Int i, j, m = mat->m, nz = mat->rp[m];
  HYPRE_Int *o2n, *cval;
  HYPRE_Int algo = 1;
  HYPRE_Real *r1, *c1;
  bool debug = mat->debug;
  bool isNatural;
  Mat_dh B;

#if 0
 *        = 1 : Compute a row permutation of the matrix so that the
 *              permuted matrix has as many entries on its diagonal as
 *              possible. The values on the diagonal are of arbitrary size.
 *              HSL subroutine MC21A/AD is used for this.
 *        = 2 : Compute a row permutation of the matrix so that the smallest
 *              value on the diagonal of the permuted matrix is maximized.
 *        = 3 : Compute a row permutation of the matrix so that the smallest
 *              value on the diagonal of the permuted matrix is maximized.
 *              The algorithm differs from the one used for JOB = 2 and may
 *              have quite a different performance.
 *        = 4 : Compute a row permutation of the matrix so that the sum
 *              of the diagonal entries of the permuted matrix is maximized.
 *        = 5 : Compute a row permutation of the matrix so that the product
 *              of the diagonal entries of the permuted matrix is maximized
 *              and vectors to scale the matrix so that the nonzero diagonal
 *              entries of the permuted matrix are one in absolute value and
 *              all the off-diagonal entries are less than or equal to one in
 *              absolute value.
#endif

  Parser_dhReadInt(parser_dh, "-rowPermute", &algo); CHECK_V_ERROR;
  if (algo < 1) algo = 1;
  if (algo > 5) algo = 1;
  hypre_sprintf(msgBuf_dh, "calling row permutation with algo= %i", algo);
  SET_INFO(msgBuf_dh);

  r1 = (HYPRE_Real*)MALLOC_DH(m*sizeof(HYPRE_Real)); CHECK_V_ERROR;
  c1 = (HYPRE_Real*)MALLOC_DH(m*sizeof(HYPRE_Real)); CHECK_V_ERROR;
  if (mat->row_perm == NULL) {
    mat->row_perm = o2n = (HYPRE_Int*)MALLOC_DH(m*sizeof(HYPRE_Int)); CHECK_V_ERROR;
  } else {
    o2n = mat->row_perm;
  }

  Mat_dhTranspose(mat, &B); CHECK_V_ERROR;

  /* get row permutation and scaling vectors */
  dldperm(algo, m, nz, B->rp, B->cval, B->aval, o2n, r1, c1);

  /* permute column indices, then turn the matrix rightside up */
  cval = B->cval;
  for (i=0; i<nz; ++i) cval[i] = o2n[cval[i]];

  /* debug block */
  if (debug && logFile != NULL) {
    hypre_fprintf(logFile, "\n-------- row permutation vector --------\n");
    for (i=0; i<m; ++i) hypre_fprintf(logFile, "%i ", 1+o2n[i]);
    hypre_fprintf(logFile, "\n");

    if (myid_dh == 0) {
      hypre_printf("\n-------- row permutation vector --------\n");
      for (i=0; i<m; ++i) hypre_printf("%i ", 1+o2n[i]);
      hypre_printf("\n");
    }
  }

  /* check to see if permutation is non-natural */
  isNatural = true;
  for (i=0; i<m; ++i) {
    if (o2n[i] != i) {
      isNatural = false;
      break;
    }
  }

  if (isNatural) {
    hypre_printf("@@@ [%i] Mat_dhRowPermute :: got natural ordering!\n", myid_dh);
  } else {
    HYPRE_Int *rp = B->rp, *cval = B->cval;
    HYPRE_Real *aval = B->aval;

    if (algo == 5) {
      hypre_printf("@@@ [%i] Mat_dhRowPermute :: scaling matrix rows and columns!\n", myid_dh);

      /* scale matrix */
      for (i=0; i<m; i++) {
        r1[i] = exp(r1[i]);
        c1[i] = exp(c1[i]);
      }
      for (i=0; i<m; i++)
        for (j=rp[i]; j<rp[i+1]; j++)
          aval[j] *= r1[cval[j]] * c1[i];
    }

    mat_dh_transpose_reuse_private(B->m, B->rp, B->cval, B->aval,
                              mat->rp, mat->cval, mat->aval); CHECK_V_ERROR;
  }


  Mat_dhDestroy(B); CHECK_V_ERROR;
  FREE_DH(r1); CHECK_V_ERROR;
  FREE_DH(c1); CHECK_V_ERROR;

#endif
  END_FUNC_DH
}


/*==============================================================================*/
#undef __FUNC__
#define __FUNC__ "Mat_dhPartition"
void build_adj_lists_private(Mat_dh mat, HYPRE_Int **rpOUT, HYPRE_Int **cvalOUT)
{
  START_FUNC_DH
  HYPRE_Int m = mat->m;
  HYPRE_Int *RP = mat->rp, *CVAL = mat->cval;
  HYPRE_Int nz = RP[m];
  HYPRE_Int i, j, *rp, *cval, idx = 0;

  rp = *rpOUT = (HYPRE_Int *)MALLOC_DH((m+1)*sizeof(HYPRE_Int)); CHECK_V_ERROR;
  cval = *cvalOUT = (HYPRE_Int *)MALLOC_DH(nz*sizeof(HYPRE_Int)); CHECK_V_ERROR;
  rp[0] = 0;

  /* assume symmetry for now! */
  for (i=0; i<m; ++i)  {
    for (j=RP[i]; j<RP[i+1]; ++j) {
      HYPRE_Int col = CVAL[j];
      if (col != i) {
        cval[idx++] = col;
      }
    }
    rp[i+1] = idx;
  }
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "Mat_dhPartition"
void Mat_dhPartition(Mat_dh mat, HYPRE_Int blocks,
                     HYPRE_Int **beg_rowOUT, HYPRE_Int **row_countOUT,  HYPRE_Int **n2oOUT, HYPRE_Int **o2nOUT)
{
  HYPRE_UNUSED_VAR(mat);
  HYPRE_UNUSED_VAR(blocks);
  HYPRE_UNUSED_VAR(beg_rowOUT);
  HYPRE_UNUSED_VAR(row_countOUT);
  HYPRE_UNUSED_VAR(n2oOUT);
  HYPRE_UNUSED_VAR(o2nOUT);

  START_FUNC_DH
#ifndef HAVE_METIS_DH

  if (ignoreMe) SET_V_ERROR("not compiled for metis!");

#else

  HYPRE_Int *beg_row, *row_count, *n2o, *o2n, bk, new, *part;
  HYPRE_Int m = mat->m;
  HYPRE_Int i, cutEdgeCount;
  HYPRE_Real zero = 0.0;
  HYPRE_Int metisOpts[5] = {0, 0, 0, 0, 0};
  HYPRE_Int *rp, *cval;

  /* allocate storage for returned arrays */
  beg_row = *beg_rowOUT = (HYPRE_Int *)MALLOC_DH(blocks*sizeof(HYPRE_Int)); CHECK_V_ERROR;
  row_count = *row_countOUT = (HYPRE_Int *)MALLOC_DH(blocks*sizeof(HYPRE_Int)); CHECK_V_ERROR;
  *n2oOUT = n2o = (HYPRE_Int *)MALLOC_DH(m*sizeof(HYPRE_Int)); CHECK_V_ERROR;
  *o2nOUT = o2n = (HYPRE_Int *)MALLOC_DH(m*sizeof(HYPRE_Int)); CHECK_V_ERROR;

#if 0
=============================================================
Metis arguments:

n - number of nodes
rp[], cval[]
NULL, NULL,
0   /*no edge or vertex weights*/
0  /*use zero-based numbering*/
blocksIN,
options[5] =
  0 :: 0/1 use defauls; use uptions 1..4
  1 ::
edgecutOUT,
part[]
=============================================================
#endif

  /* form the graph representation that metis wants */
  build_adj_lists_private(mat, &rp, &cval); CHECK_V_ERROR;
  part = (HYPRE_Int *)MALLOC_DH(m*sizeof(HYPRE_Int)); CHECK_V_ERROR;

  /* get parition vector from metis */
  METIS_PartGraphKway(&m, rp, cval, NULL, NULL,
                          &zero, &zero, &blocks, metisOpts,
                          &cutEdgeCount, part);

  FREE_DH(rp); CHECK_V_ERROR;
  FREE_DH(cval); CHECK_V_ERROR;

  if (mat->debug) {
    printf_dh("\nmetis partitioning vector; blocks= %i\n", blocks);
    for (i=0; i<m; ++i) printf_dh("  %i %i\n", i+1, part[i]);
  }

  /* compute beg_row, row_count arrays from partition vector */
  for (i=0; i<blocks; ++i) row_count[i] = 0;
  for (i=0; i<m; ++i) {
    bk = part[i];  /* block to which row i belongs */
    row_count[bk] += 1;
  }
  beg_row[0] = 0;
  for (i=1; i<blocks; ++i) beg_row[i] = beg_row[i-1] + row_count[i-1];

  if (mat->debug) {
    printf_dh("\nrow_counts: ");
    for (i=0; i<blocks; ++i) printf_dh(" %i", row_count[i]);
    printf_dh("\nbeg_row: ");
    for (i=0; i<blocks; ++i) printf_dh(" %i", beg_row[i]+1);
    printf_dh("\n");
  }

  /* compute permutation vector */
  {
	 HYPRE_Int *tmp = (HYPRE_Int*)MALLOC_DH(blocks*sizeof(HYPRE_Int)); CHECK_V_ERROR;
	 hypre_TMemcpy(tmp,  beg_row, HYPRE_Int, blocks, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
	 for (i=0; i<m; ++i)
	 {
		bk = part[i];  /* block to which row i belongs */
		new = tmp[bk];
		tmp[bk] += 1;
		o2n[i] = new;
		n2o[new] = i;
	 }
	 FREE_DH(tmp);
  }

  FREE_DH(part); CHECK_V_ERROR;

#endif

  END_FUNC_DH
}
