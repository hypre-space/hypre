#include "Euclid_dh.h"
#include "ilu_dh.h"
#include "Mem_dh.h"
#include "Parser_dh.h"
#include "petsc_euclid.h"
#include "Hash_dh.h"

#undef __FUNC__
#define __FUNC__ "euclid_setup_private_mpi"
void euclid_setup_private_mpi(Euclid_dh ctx)
{
  START_FUNC_DH
  if (ctx->algo_par == PILU_PAR) {
    setup_pilu_private_mpi(ctx); CHECK_V_ERROR;
  }
  END_FUNC_DH
}


/* assumes no prior reordering */
#undef __FUNC__
#define __FUNC__ "order_bdry_nodes_private_mpi"
void order_bdry_nodes_private_mpi(Euclid_dh ctx)
{
  START_FUNC_DH
#if 0

fix

  int h, i, j, k, m = ctx->m, beg_row = ctx->beg_row;
  int end_row = beg_row + m;
  int *n2o_row = ctx->n2o_row, *n2o_col = ctx->n2o_col;
  int *rp = ctx->rp, *cval = ctx->cval;
  int *tmp, *tmp2, idxLO, idxHI, count;
  bool localFlag;

  /* order nodes within each subdomain so that boundary rows are ordered last */

  tmp = (int*)MALLOC_DH(m*sizeof(int));
  tmp2 = (int*)MALLOC_DH(m*sizeof(int));

  idxLO = 0;
  idxHI = m-1;

  for (j=0; j<m; ++j) { 
    int row = j;
    localFlag = true;
    for (k=rp[row]; k<rp[row+1]; ++k) {
      int col = cval[k];
      if (col  < beg_row || col >= end_row) {
        localFlag = false;
        break;
      }
    }
    if (localFlag) { tmp[idxLO++] = row; } 
    else           { tmp[idxHI--] = row; }
  }
  ++idxHI;
  ctx->first_bdry = idxHI;

  /* reverse ordering of bdry nodes; this puts them in same
     relative order as they were originally.   Why do I do
     this?  Well, it seems a good idea . . .
   */
  count = m-idxHI+1;
  if (count) {
    for (h=m-1, j=0; h>=idxHI; --h, ++j) tmp2[j] = tmp[h];
    memcpy(tmp+idxHI, tmp2, count*sizeof(int));
  }

  memcpy(ctx->n2o_row, tmp, m*sizeof(int));
  memcpy(ctx->n2o_col, ctx->n2o_row, m*sizeof(int));

  FREE_DH(tmp);  CHECK_V_ERROR;
  FREE_DH(tmp2); CHECK_V_ERROR;

fix

#endif 
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "find_nabors_private_mpi"
void find_nabors_private_mpi(Euclid_dh ctx)
{
  START_FUNC_DH
#if 0

fix


  int m = ctx->m, *rp = ctx->rp, *cval = ctx->cval;
  int ct, i, j, beg_row = ctx->beg_row, end_row = beg_row + m;
  int first_bdry = ctx->first_bdry;
  int *myNabors, *myNaborsIN, *n2o = ctx->n2o_row;

  myNabors = (int*)MALLOC_DH(np_dh*sizeof(int));
  myNaborsIN = (int*)MALLOC_DH(np_dh*sizeof(int));
  for (i=0; i<np_dh; ++i) myNabors[i] = 0;

  for (i=first_bdry; i<m; ++i) {
    int row = n2o[i];
    for (j=rp[row]; j<rp[row+1]; ++j) {
      int c = cval[j];
      if (c < beg_row || c >= end_row) {
        int nabor = find_owner_private(ctx, c);
        myNabors[nabor] = 1;
      }
    }
  }

  MPI_Alltoall(myNabors, 1, MPI_INT, myNaborsIN, 1, MPI_INT, comm_dh);

/*
fprintf(logFile, "[%i] myNabors: ", myid_dh);
for (i=0; i<np_dh; ++i) fprintf(logFile, "%i ", myNabors[i]);
fprintf(logFile, "[%i] myNaborsIN: ", myid_dh);
for (i=0; i<np_dh; ++i) fprintf(logFile, "%i ", myNaborsIN[i]);
fprintf(logFile, "\n");
*/

  ct = 0;
  for (i=0; i<np_dh; ++i) {
    if (i != myid_dh && myNaborsIN[i]) {
      myNabors[ct++] = i;
    }
  }

  ctx->nabors = myNabors;
  ctx->naborCount = ct;

/*
fprintf(logFile, "\nmynabors: ");
for (i=0; i<ctx->naborCount; ++i) {
  fprintf(stderr, "  %i\n", ctx->nabors[i]);
}
*/

  FREE_DH(myNaborsIN); CHECK_V_ERROR;

fix

#endif

  END_FUNC_DH
}

/* note: this duplicates code in Mat_dh; yuck! */
#undef __FUNC__
#define __FUNC__ "find_owner_private_mpi"
int find_owner_private_mpi(Euclid_dh ctx, int index)
{
  START_FUNC_DH
  int pe, owner = -1;
  int *beg_rows  = ctx->beg_rows;
  int *end_rows  = ctx->end_rows;

  for (pe=0; pe<np_dh; ++pe) {
    if (index>= beg_rows[pe] && index < end_rows[pe]) {
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
#define __FUNC__ "exchange_permutations_private_mpi"
void exchange_permutations_private_mpi(Euclid_dh ctx)
{
  START_FUNC_DH
  MPI_Request *recv_req, *send_req;
  MPI_Status *status;
  int *nabors = ctx->nabors, naborCount = ctx->naborCount;
  int i, j, *sendBuf, *recvBuf, *naborIdx, nz;
  int first_bdry = ctx->first_bdry, m = ctx->m; 
  int bdryCount = m-first_bdry, beg_row = ctx->beg_row;
  int *bdryNodeCounts = ctx->bdryNodeCounts;
  int *n2o_row = ctx->n2o_row;
  Hash_dh n2o_table, o2n_table;
  HashData record;

  /* allocate send buffer, and copy permutation info to buffer;
     each entry is a <old_value, new_value> pair.
   */
  sendBuf = (int*)MALLOC_DH(2*bdryCount*sizeof(int)); CHECK_V_ERROR;
  for (i=first_bdry, j=0; i<m; ++i, ++j) {
    sendBuf[2*j] = n2o_row[i]+beg_row;
    sendBuf[2*j+1] = i+beg_row;
  }

  /* allocate receive buffers, for each nabor in the subdomain graph,
     and set up index array for locating the beginning of each
     nabor's buffers.
   */
  naborIdx = (int*)MALLOC_DH((1+naborCount)*sizeof(int)); CHECK_V_ERROR;
  naborIdx[0] = 0;
  nz = 0;
  for (i=0; i<naborCount; ++i) {
    nz += (2*bdryNodeCounts[nabors[i]]);
    naborIdx[i+1] = nz;

  }
  recvBuf = (int*)MALLOC_DH(nz*sizeof(int)); CHECK_V_ERROR;

  /* perform sends and receives */
  recv_req = (MPI_Request*)MALLOC_DH(naborCount*sizeof(MPI_Request)); CHECK_V_ERROR;
  send_req = (MPI_Request*)MALLOC_DH(naborCount*sizeof(MPI_Request)); CHECK_V_ERROR;
  status = (MPI_Status*)MALLOC_DH(naborCount*sizeof(MPI_Status)); CHECK_V_ERROR;

  for (i=0; i<naborCount; ++i) {
    int nabr = nabors[i];
    int *buf = recvBuf + naborIdx[i];
    int ct = naborIdx[i+1] - naborIdx[i];
    MPI_Isend(sendBuf, 2*bdryCount, MPI_INT, nabr, 444, comm_dh, &(send_req[i]));
    MPI_Irecv(buf, ct, MPI_INT, nabr, 444, comm_dh, &(recv_req[i]));
  }
  MPI_Waitall(naborCount, send_req, status);
  MPI_Waitall(naborCount, recv_req, status);

  FREE_DH(naborIdx); CHECK_V_ERROR;
  FREE_DH(sendBuf); CHECK_V_ERROR;
  FREE_DH(recv_req); CHECK_V_ERROR;
  FREE_DH(send_req); CHECK_V_ERROR;
  FREE_DH(status); CHECK_V_ERROR;

  /* insert non-local boundary node permutations in lookup table */
  Hash_dhCreate(&n2o_table, nz/2); CHECK_V_ERROR;
  Hash_dhCreate(&o2n_table, nz/2); CHECK_V_ERROR;
  ctx->n2o_nonLocal = n2o_table;
  ctx->o2n_nonLocal = o2n_table;

  for (i=0; i<nz; i += 2) {
    int old = recvBuf[i];
    int new = recvBuf[i+1];
    record.iData = new;
    Hash_dhInsert(o2n_table, old, &record); CHECK_V_ERROR;
    record.iData = old;
    Hash_dhInsert(n2o_table, new, &record); CHECK_V_ERROR;
  }

  FREE_DH(recvBuf); CHECK_V_ERROR;
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "setup_pilu_private_mpi"
void setup_pilu_private_mpi(Euclid_dh ctx)
{
  START_FUNC_DH

  /* setup arrays for mapping rows to processors */
  int firstLocal = ctx->beg_row, lastLocal = firstLocal+ctx->m;
  ctx->beg_rows = (int*)MALLOC_DH(np_dh*sizeof(int)); CHECK_V_ERROR;
  ctx->end_rows = (int*)MALLOC_DH(np_dh*sizeof(int)); CHECK_V_ERROR;
  MPI_Allgather(&firstLocal, 1, MPI_INT, ctx->beg_rows, 1, MPI_INT, comm_dh); 
  MPI_Allgather(&lastLocal, 1, MPI_INT, ctx->end_rows, 1, MPI_INT, comm_dh); 

  /* locally order interior nodes, then boundary nodes in G(A) */
  order_bdry_nodes_private_mpi(ctx); CHECK_V_ERROR;

  /* find nearest-nabors in subdomain graph */
  find_nabors_private_mpi(ctx); CHECK_V_ERROR;

  /* exchange information for number of boundary nodes, and
     nonzero counts in boundary rows.
   */
  ctx->bdryNodeCounts = (int*)MALLOC_DH(np_dh*sizeof(int)); CHECK_V_ERROR;
  ctx->bdryRowNzCounts = (int*)MALLOC_DH(np_dh*sizeof(int)); CHECK_V_ERROR;

#if 0

fix

  { int nodeCount = ctx->m - ctx->first_bdry;
    int nzCount = ctx->rp[ctx->m] - ctx->rp[ctx->first_bdry];
    MPI_Allgather(&nodeCount, 1, MPI_INT, ctx->bdryNodeCounts, 1, MPI_INT, comm_dh); 
    MPI_Allgather(&nzCount, 1, MPI_INT, ctx->bdryRowNzCounts, 1, MPI_INT, comm_dh); 
  }

#endif

  /* exchange boundary node permutations with neighboring subdomains */ 
  exchange_permutations_private_mpi(ctx); CHECK_V_ERROR;

  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "exchange_bdry_rows_private_mpi"
void exchange_bdry_rows_private_mpi(Euclid_dh ctx)
{
#if 0
  START_FUNC_DH
  MPI_Request *recv_req, *send_req;
  MPI_Status *status;
  int *nabors = ctx->nabors, naborCount = ctx->naborCount;
  int i, nz, *sendBuf, *recvBuf, *naborIdx;
  int *n2o_row = ctx->n2o_row;
  int m = ctx->m, first_bdry = ctx->first_bdry;

  /* allocate storage for send buffer */
  nz = 0;
  for (i=first_bdry; i<m; ++i) nz += rp[n2o[i]] - rp[n2o[i]];
  

  /* copy boundary rows to send buffer */

  /* allocate storage for recv buffers, and set up index array for 
     locating the beginning of each nabor's buffer.
  */

  /* perform sends and receives */
?  recv_req = (MPI_Request*)MALLOC_DH(naborCount*sizeof(MPI_Request)); CHECK_V_ERROR;
  send_req = (MPI_Request*)MALLOC_DH(naborCount*sizeof(MPI_Request)); CHECK_V_ERROR;
  status = (MPI_Status*)MALLOC_DH(naborCount*sizeof(MPI_Status)); CHECK_V_ERROR;

  for (i=0; i<naborCount; ++i) {
    int nabr = nabors[i];
?    int *buf = recvBuf + naborIdx[i];
?    MPI_Isend(sendBuf, 2*bdryCount, MPI_INT, i, 444, comm_dh, &(send_req[i]));
?    MPI_Irecv(buf, ct, MPI_INT, i, 555, comm_dh, &(recv_req[i]));
  }
  MPI_Waitall(naborCount, send_req, status);
  MPI_Waitall(naborCount, recv_req, status);
#endif
}

