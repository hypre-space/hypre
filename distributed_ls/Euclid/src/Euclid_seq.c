#include "Euclid_dh.h"
#include "ilu_dh.h"
#include "Mem_dh.h"
#include "Parser_dh.h"
#include "petsc_euclid.h"

  /* private methods; called by Euclid_dhSetup() */


#undef __FUNC__
#define __FUNC__ "euclid_setup_private_seq"
void euclid_setup_private_seq(Euclid_dh ctx)
{
  START_FUNC_DH
  int m = ctx->m;

  /* partition matrix, for block jacobi or PILU parallelism */
  if (ctx->algo_par == BJILU_PAR  ||  ctx->algo_par == PILU_PAR) {
    partition_private_seq(ctx); CHECK_V_ERROR;
  }
  ctx->block[ctx->blockCount-1].end_row = m;

  /* order boundary nodes before interior nodes, for PILU */
  if (ctx->algo_par == PILU_PAR) {
    order_bdry_nodes_private_seq(ctx); CHECK_V_ERROR;
    ctx->isNaturallyOrdered = false;
  }
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "order_bdry_nodes_private_seq"
void order_bdry_nodes_private_seq(Euclid_dh ctx)
{
  START_FUNC_DH
  int h, i, j, n = ctx->n;
  int blockCount = ctx->blockCount;
  PartNode *part = ctx->block;
  int *n2o_row = ctx->n2o_row, *n2o_col = ctx->n2o_col, *o2n;
  int *cvalPtr, len;
  int *tmp, *tmp2, beg_row, end_row, idxLO, idxHI, count;
  bool localFlag;

  /* error checking */
  if (blockCount >= MAX_SUBDOMAINS) {
    sprintf(msgBuf_dh, "requested %i subdomains, but MAX_SUBDOMAINS = %i;\n", blockCount, MAX_SUBDOMAINS);
    SET_V_ERROR(msgBuf_dh);
  }

  if (blockCount == 1) goto NOTHING_TO_DO;

  ctx->isNaturallyOrdered = false;


  /* order nodes within each subdomain so that boundary rows are ordered last */

  /* first, must invert the column permutation */
  o2n = (int*)MALLOC_DH(n*sizeof(int));
  for (i=0; i<n; ++i) o2n[n2o_col[i]] = i;

  tmp = (int*)MALLOC_DH(n*sizeof(int));
  tmp2 = (int*)MALLOC_DH(n*sizeof(int));
  for (i=0; i<blockCount; ++i) {   /* loop over partition blocks */
    beg_row = idxLO = part[i].beg_row;
    end_row = idxHI = part[i].end_row-1;
    for (j=beg_row; j<=end_row; ++j) {  /* loop over rows within each block */
      int row = n2o_row[j];
      localFlag = true;

      EuclidGetRow(ctx->A, row, &len, &cvalPtr, NULL); CHECK_V_ERROR;
      while (len--) {
        int col = *cvalPtr++;
        col = o2n[col];
        if (col  < beg_row || col > end_row) {
          localFlag = false;
          break;
        }
      }
      EuclidRestoreRow(ctx->A, row, &len, &cvalPtr, NULL); CHECK_V_ERROR;

      if (localFlag) { tmp[idxLO++] = row; } 
      else           { tmp[idxHI--] = row; }
    }
    part[i].first_bdry = idxHI+1;

    /* reverse ordering of bdry nodes; this puts them in same
       relative order as they were originally.   Why do I do
       this?  Well, it seems a good idea . . .
     */
    count = end_row-idxHI;
    ++idxHI;
    for (h=end_row, j=0; h>=idxHI; --h, ++j) {
      tmp2[j] = tmp[h];
    }
    if (count) memcpy(tmp+idxHI, tmp2, count*sizeof(int));
  }  /* loop over partition block(i) */

  memcpy(ctx->n2o_row, tmp, n*sizeof(int));
  memcpy(ctx->n2o_col, ctx->n2o_row, n*sizeof(int));

NOTHING_TO_DO: ;
 
  FREE_DH(tmp);
  FREE_DH(tmp2);
  FREE_DH(o2n);
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "partition_private_seq"
void partition_private_seq(Euclid_dh ctx)
{
  START_FUNC_DH
  int i, m = ctx->m, blockCount = ctx->blockCount;
  PartNode *part = ctx->block;

  switch (ctx->partMethod) {
    case SIMPLE_PART:
          { int c = 0, rowsPerBlock = m/blockCount;
            SET_INFO("using simple partitioning strategy");
            for (i=0; i<blockCount; ++i) {
              part[i].beg_row = c;
              c += rowsPerBlock;
              part[i].end_row = c;
            }
            part[blockCount-1].end_row = m;
          }
          break;

#ifdef USING_METIS
    case METIS_PART:
           SET_INFO("using metis for partitioning");
           metis_order_private_seq(ctx);
           ctx->isNaturallyOrdered = false;
           break;
#endif

    default: sprintf(msgBuf_dh, "Unknown partitioning method");
             SET_INFO(msgBuf_dh);
  }
  END_FUNC_DH
}


#ifdef USING_METIS
#undef __FUNC__
#define __FUNC__ "metis_order_private_seq"
void metis_order_private_seq(Euclid_dh ctx)
{
  START_FUNC_DH
  int edgecut, options[5], *metisPart, zero = 0, m = ctx->m;
  int *rp = ctx->rp, *cval = ctx->cval, n2o = ctx->n2o_row;
  int c, i, blockCount = ctx->blockCount, *part;
  PartNode *pn = ctx->block;

  metisPart = (int*)MALLOC_DH(m*sizeof(int)); CHECK_V_ERROR;
  part = (int*)MALLOC_DH((1+blockCount)*sizeof(int)); CHECK_V_ERROR;

  options[0] = 0;  /* use default values for metis. */
  METIS_PartGraphRecursive(&m, rp, cval, NULL, NULL, &zero, &zero,
                                     &blockCount, options, &edgecut, metisPart);

  /* compute first, last row in each partition */
  for (i=0; i<=blockCount; ++i) part[i] = 0;
  for (i=0; i<m; ++i) part[1+metisPart[i]] += 1;
  for (i=1; i<=blockCount; ++i) part[i] += part[i-1];

  c = 0;
  for (i=0; i<blockCount; ++i) {
    pn[i].beg_row = c;
    c += part[i+1];
    pn[i].end_row = c;
  }

  /* form ordering vectors */
  for (i=0; i<m; ++i) {
    int p = metisPart[i];
    int newOrdering = part[p];
    n2o[newOrdering] = i;
    part[p] += 1;
  }

  FREE_DH(metisPart); CHECK_V_ERROR;
  FREE_DH(part); CHECK_V_ERROR;
  END_FUNC_DH
}

#endif /* #ifdef USING_METIS */


