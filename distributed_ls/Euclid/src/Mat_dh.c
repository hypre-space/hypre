#include "Mat_dh.h"
#include "Mem_dh.h"
#include "Hash_dh.h"
#include "Numbering_dh.h"

static void setupSends_private(Mat_dh mat, int *inlist);
static void setupReceives_private(Mat_dh mat, int *beg_rows, int *end_rows,
                           int reqlen, int *reqind, int *outlist);
static int findOwner_private(int *beg_rows, int *end_rows, int index);

#undef __FUNC__
#define __FUNC__ "Mat_dhCreate"
void Mat_dhCreate(Mat_dh *mat)
{
  START_FUNC_DH
  struct _mat_dh* tmp = (struct _mat_dh*)MALLOC_DH(sizeof(struct _mat_dh)); CHECK_V_ERROR;
  *mat = tmp;

  tmp->m = 0;
  tmp->n = 0;
  tmp->beg_row = 0; 

  tmp->rp = NULL;
  tmp->len = NULL;
  tmp->cval = NULL;
  tmp->aval = NULL;
  tmp->diag = NULL;
  tmp->fill = NULL;
  tmp->owner = true;

  tmp->num_recv = 0;
  tmp->num_send = 0;
  tmp->recv_req = NULL;
  tmp->send_req = NULL;
  tmp->status = NULL;
  tmp->recvbuf = NULL;
  tmp->sendbuf = NULL;
  tmp->sendind = NULL;
  tmp->sendlen = 0;
  tmp->numb = NULL;
  tmp->matvecIsSetup = false;
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "Mat_dhDestroy"
void Mat_dhDestroy(Mat_dh mat)
{
  START_FUNC_DH
  if (mat->owner) {
    if (mat->rp != NULL) { FREE_DH(mat->rp); CHECK_V_ERROR; }
    if (mat->len != NULL) { FREE_DH(mat->len); CHECK_V_ERROR; }
    if (mat->cval != NULL) { FREE_DH(mat->cval); CHECK_V_ERROR; }
    if (mat->aval != NULL) { FREE_DH(mat->aval); CHECK_V_ERROR; }
    if (mat->diag != NULL) { FREE_DH(mat->diag); CHECK_V_ERROR; }
    if (mat->fill != NULL) { FREE_DH(mat->fill); CHECK_V_ERROR; }
  }

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
#define __FUNC__ "Mat_dhMatVecSetup"
void Mat_dhMatVecSetdown(Mat_dh mat)
{
  START_FUNC_DH
  END_FUNC_DH
}


/* adopted from Edmond Chow's ParaSails */
#undef __FUNC__
#define __FUNC__ "Mat_dhMatVecSetup"
void Mat_dhMatVecSetup(Mat_dh mat)
{
  START_FUNC_DH
  int *outlist, *inlist;
  int i, row, *rp = mat->rp, *cval = mat->cval;
  Numbering_dh numb;
  int m = mat->m;
  int firstLocal = mat->beg_row;
  int lastLocal = firstLocal+m-1;
  int *beg_rows, *end_rows;

  mat->recv_req = (MPI_Request *)MALLOC_DH(np_dh * sizeof(MPI_Request)); CHECK_V_ERROR;
  mat->send_req = (MPI_Request *)MALLOC_DH(np_dh * sizeof(MPI_Request)); CHECK_V_ERROR;
  mat->status = (MPI_Status *)MALLOC_DH(np_dh * sizeof(MPI_Status)); CHECK_V_ERROR;
  beg_rows = (int*)MALLOC_DH(np_dh*sizeof(int)); CHECK_V_ERROR;
  end_rows = (int*)MALLOC_DH(np_dh*sizeof(int)); CHECK_V_ERROR;

  MPI_Allgather(&firstLocal, 1, MPI_INT, beg_rows, 1, MPI_INT, comm_dh); CHECK_V_ERROR;
  MPI_Allgather(&lastLocal, 1, MPI_INT, end_rows, 1, MPI_INT, comm_dh); CHECK_V_ERROR;

  outlist = (int *)MALLOC_DH(np_dh*sizeof(int)); CHECK_V_ERROR;
  inlist  = (int *)MALLOC_DH(np_dh*sizeof(int)); CHECK_V_ERROR;
  for (i=0; i<np_dh; ++i) {
    outlist[i] = 0;
    inlist[i] = 0;
  }

  /* Create Numbering object */
  Numbering_dhCreate(&(mat->numb)); CHECK_V_ERROR;
  numb = mat->numb;
  Numbering_dhSetup(numb, mat); CHECK_V_ERROR;

  setupReceives_private(mat, beg_rows, end_rows, numb->num_ext, 
         &(numb->local_to_global[numb->num_loc]), outlist); CHECK_V_ERROR;

  MPI_Alltoall(outlist, 1, MPI_INT, inlist, 1, MPI_INT, comm_dh);

  setupSends_private(mat, inlist); CHECK_V_ERROR;

  /* Convert to local indices */
  for (row=0; row<m; row++) {
    int len = rp[row+1]-rp[row];
    int *ind = cval+rp[row];
    Numbering_dhGlobalToLocal(numb, len, ind, ind); CHECK_V_ERROR;
  }

  FREE_DH(outlist); CHECK_V_ERROR;
  FREE_DH(inlist); CHECK_V_ERROR;
  FREE_DH(beg_rows); CHECK_V_ERROR;
  FREE_DH(end_rows); CHECK_V_ERROR;
  END_FUNC_DH
}

/* adopted from Edmond Chow's ParaSails */
#undef __FUNC__
#define __FUNC__ "setupReceives_private"
void setupReceives_private(Mat_dh mat, int *beg_rows, int *end_rows,
                           int reqlen, int *reqind, int *outlist)
{
  START_FUNC_DH
  int i, j, this_pe;
  MPI_Request request;
  int m = mat->m;

  mat->num_recv = 0;

  /* Allocate recvbuf */
  /* recvbuf has numlocal entries saved for local part of x, used in matvec */
  mat->recvbuf = (double*)MALLOC_DH((reqlen+m) * sizeof(double));

  for (i=0; i<reqlen; i=j) { /* j is set below */ 
    /* The processor that owns the row with index reqind[i] */
    this_pe = findOwner_private(beg_rows, end_rows, reqind[i]); CHECK_V_ERROR;

    /* Figure out other rows we need from this_pe */
    for (j=i+1; j<reqlen; j++) {
      /* if row is on different pe */
      if (reqind[j] < beg_rows[this_pe] ||
             reqind[j] > end_rows[this_pe])
        break;
    }

    /* Request rows in reqind[i..j-1] */
    MPI_Isend(&reqind[i], j-i, MPI_INT, this_pe, 444, comm_dh, &request);
    MPI_Request_free(&request);

    /* Count of number of number of indices needed from this_pe */
    outlist[this_pe] = j-i;

    MPI_Recv_init(&mat->recvbuf[i+m], j-i, MPI_DOUBLE, this_pe, 555,
            comm_dh, &mat->recv_req[mat->num_recv]);

    mat->num_recv++;
  }
  END_FUNC_DH
}


/* adopted from Edmond Chow's ParaSails */
#undef __FUNC__
#define __FUNC__ "setupSends_private"
void setupSends_private(Mat_dh mat, int *inlist)
{
  START_FUNC_DH
  int i, j, sendlen, first = mat->beg_row;
  MPI_Request *requests;
  MPI_Status  *statuses;

  requests = (MPI_Request *) MALLOC_DH(np_dh * sizeof(MPI_Request)); CHECK_V_ERROR;
  statuses = (MPI_Status *)  MALLOC_DH(np_dh * sizeof(MPI_Status)); CHECK_V_ERROR;

  /* Determine size of and allocate sendbuf and sendind */
  sendlen = 0;
  for (i=0; i<np_dh; i++) sendlen += inlist[i];
  mat->sendlen = sendlen;
  mat->sendbuf = (double *)MALLOC_DH(sendlen * sizeof(double)); CHECK_V_ERROR;
  mat->sendind = (int *)MALLOC_DH(sendlen * sizeof(int)); CHECK_V_ERROR;

  j = 0;
  mat->num_send = 0;
  for (i=0; i<np_dh; i++) {
    if (inlist[i] != 0) {
      /* Post receive for the actual indices */
      MPI_Irecv(&mat->sendind[j], inlist[i], MPI_INT, i, 444, comm_dh,
                                           &requests[mat->num_send]);
      /* Set up the send */
      MPI_Send_init(&mat->sendbuf[j], inlist[i], MPI_DOUBLE, i, 555, comm_dh,
                                               &mat->send_req[mat->num_send]);

      mat->num_send++;
      j += inlist[i];
    }
  }

  MPI_Waitall(mat->num_send, requests, statuses);
  /* convert global indices to local indices */
  /* these are all indices on this processor */
  for (i=0; i<mat->sendlen; i++) mat->sendind[i] -= first;

  FREE_DH(requests);
  FREE_DH(statuses);
  END_FUNC_DH
}



#undef __FUNC__
#define __FUNC__ "Mat_dhMatVec"
void Mat_dhMatVec(Mat_dh mat, double *x, double *b)
{
  START_FUNC_DH
  int    i, row, m = mat->m;
  int    *rp = mat->rp, *cval = mat->cval;
  double *aval = mat->aval;
  int    *sendind = mat->sendind;
  int    sendlen = mat->sendlen;
  double *sendbuf = mat->sendbuf; 
  double *recvbuf = mat->recvbuf;

  /* Put components of x into the right outgoing buffers */
  #pragma omp parallel  for schedule(runtime) 
  for (i=0; i<sendlen; i++) sendbuf[i] = x[sendind[i]]; 

  MPI_Startall(mat->num_recv, mat->recv_req);
  MPI_Startall(mat->num_send, mat->send_req);

  /* Copy local part of x into top part of recvbuf */
  #pragma omp parallel  for schedule(runtime) 
  for (i=0; i<m; i++) recvbuf[i] = x[i];

  MPI_Waitall(mat->num_recv, mat->recv_req, mat->status);

  /* do the multiply */
  #pragma omp parallel  for schedule(runtime) 
    for (row=0; row<m; row++) {
        int len = rp[row+1] - rp[row];
        int * ind = cval+rp[row];
        double * val = aval+rp[row];
        double temp = 0.0;
        for (i=0; i<len; i++) {
          temp += (val[i] * recvbuf[ind[i]]);
        }
        b[row] = temp;
    }

  MPI_Waitall(mat->num_send, mat->send_req, mat->status);
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "findOwner_private"
int findOwner_private(int *beg_rows, int *end_rows, int index)
{
  START_FUNC_DH
  int pe, owner = -1;
  for (pe=0; pe<np_dh; ++pe) {
    if (index>= beg_rows[pe] && index <= end_rows[pe]) {
      owner = pe;
      break;
    }
  }

  if (owner == -1) {
    sprintf(msgBuf_dh, "failed to find owner for index= %i", index);
    SET_ERROR(-1, msgBuf_dh);
  }

  END_FUNC_VAL(owner)
}


#undef __FUNC__
#define __FUNC__ "Mat_dhReadNz"
int Mat_dhReadNz(Mat_dh mat)
{
  START_FUNC_DH
  int retval = mat->rp[mat->m];
  int nz = retval;
  MPI_Allreduce(&nz, &retval, 1, MPI_INT, MPI_SUM, comm_dh);
  END_FUNC_VAL(retval)
}

#undef __FUNC__
#define __FUNC__ "Mat_dhReadNz"
void Mat_dhPrintTriples(Mat_dh mat, char *filename)
{
  START_FUNC_DH
  int pe, i, j, beg_row = mat->beg_row;
  int *rp = mat->rp, *cval = mat->cval;
  double *aval = mat->aval;
  FILE *fp;

  for (pe=0; pe<np_dh; ++pe) {
    MPI_Barrier(comm_dh);
    if (myid_dh == pe) {
      if (pe == 0) { fp=fopen(filename, "w"); } 
      else { fp=fopen(filename, "a"); }
      for (i=0; i<mat->m; ++i) {
        for (j=rp[i]; j<rp[i+1]; ++j) {
          fprintf(fp, "%i %i %g\n", 1+i, 1+beg_row+cval[j], aval[j]);
        }
      }
      fclose(fp);
    }
  }

  END_FUNC_DH
}



