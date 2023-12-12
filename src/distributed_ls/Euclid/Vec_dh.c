/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include <stdlib.h>
#include "_hypre_Euclid.h"
/* #include "Vec_dh.h" */
/* #include "Mem_dh.h" */
/* #include "SubdomainGraph_dh.h" */
/* #include "io_dh.h" */

#undef __FUNC__
#define __FUNC__ "Vec_dhCreate"
void Vec_dhCreate(Vec_dh *v)
{
  START_FUNC_DH
  struct _vec_dh* tmp = (struct _vec_dh*)MALLOC_DH(sizeof(struct _vec_dh)); CHECK_V_ERROR;
  *v = tmp;
  tmp->n = 0;
  tmp->vals = NULL;
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "Vec_dhDestroy"
void Vec_dhDestroy(Vec_dh v)
{
  START_FUNC_DH
  if (v->vals != NULL) {
    FREE_DH(v->vals); CHECK_V_ERROR;
  }
  FREE_DH(v); CHECK_V_ERROR;
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "Vec_dhInit"
void Vec_dhInit(Vec_dh v, HYPRE_Int size)
{
  START_FUNC_DH
  v->n = size;
  v->vals = (HYPRE_Real*)MALLOC_DH(size*sizeof(HYPRE_Real)); CHECK_V_ERROR;
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "Vec_dhCopy"
void Vec_dhCopy(Vec_dh x, Vec_dh y)
{
  START_FUNC_DH
  if (x->vals == NULL) SET_V_ERROR("x->vals is NULL");
  if (y->vals == NULL) SET_V_ERROR("y->vals is NULL");
  if (x->n != y->n) SET_V_ERROR("x and y are different lengths");
  hypre_TMemcpy(y->vals,  x->vals, HYPRE_Real, x->n, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "Vec_dhDuplicate"
void Vec_dhDuplicate(Vec_dh v, Vec_dh *out)
{
  START_FUNC_DH
  Vec_dh tmp;
  HYPRE_Int size = v->n;
  if (v->vals == NULL) SET_V_ERROR("v->vals is NULL");
  Vec_dhCreate(out); CHECK_V_ERROR;
  tmp = *out;
  tmp->n = size;
  tmp->vals = (HYPRE_Real*)MALLOC_DH(size*sizeof(HYPRE_Real)); CHECK_V_ERROR;
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "Vec_dhSet"
void Vec_dhSet(Vec_dh v, HYPRE_Real value)
{
  START_FUNC_DH
  HYPRE_Int i, m = v->n;
  HYPRE_Real *vals = v->vals;
  if (v->vals == NULL) SET_V_ERROR("v->vals is NULL");
  for (i=0; i<m; ++i) vals[i] = value;
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "Vec_dhSetRand"
void Vec_dhSetRand(Vec_dh v)
{
  START_FUNC_DH
  HYPRE_Int i, m = v->n;
  HYPRE_Real max = 0.0;
  HYPRE_Real *vals = v->vals;

  if (v->vals == NULL) SET_V_ERROR("v->vals is NULL");

  for (i=0; i<m; ++i) vals[i] = rand();

  /* find largest value in vector, and scale vector,
   * so all values are in [0.0,1.0]
   */
  for (i=0; i<m; ++i) max = MAX(max, vals[i]);
  for (i=0; i<m; ++i) vals[i] = vals[i]/max;
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "Vec_dhPrint"
void Vec_dhPrint(Vec_dh v, SubdomainGraph_dh sg, char *filename)
{
  START_FUNC_DH
  HYPRE_Real *vals = v->vals;
  HYPRE_Int pe, i, m = v->n;
  FILE *fp;

  if (v->vals == NULL) SET_V_ERROR("v->vals is NULL");

  /*--------------------------------------------------------
   * case 1: no permutation information
   *--------------------------------------------------------*/
  if (sg == NULL) {
    for (pe=0; pe<np_dh; ++pe) {
      hypre_MPI_Barrier(comm_dh);
      if (pe == myid_dh) {
        if (pe == 0) {
          fp=openFile_dh(filename, "w"); CHECK_V_ERROR;
        } else {
          fp=openFile_dh(filename, "a"); CHECK_V_ERROR;
        }

        for (i=0; i<m; ++i) hypre_fprintf(fp, "%g\n", vals[i]);

        closeFile_dh(fp); CHECK_V_ERROR;
      }
    }
  }

  /*--------------------------------------------------------
   * case 2: single mpi task, multiple subdomains
   *--------------------------------------------------------*/
  else if (np_dh == 1) {
    HYPRE_Int i, j;

    fp=openFile_dh(filename, "w"); CHECK_V_ERROR;

    for (i=0; i<sg->blocks; ++i) {
      HYPRE_Int oldBlock = sg->n2o_sub[i];
      HYPRE_Int beg_row = sg->beg_rowP[oldBlock];
      HYPRE_Int end_row = beg_row + sg->row_count[oldBlock];

hypre_printf("seq: block= %i  beg= %i  end= %i\n", oldBlock, beg_row, end_row);


      for (j=beg_row; j<end_row; ++j) {
        hypre_fprintf(fp, "%g\n", vals[j]);
      }
    }
  }

  /*--------------------------------------------------------
   * case 3: multiple mpi tasks, one subdomain per task
   *--------------------------------------------------------*/
  else {
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

hypre_fprintf(stderr, "par: block= %i\n", id);

        for (i=0; i<m; ++i) {
          hypre_fprintf(fp, "%g\n", vals[i]);
        }

        closeFile_dh(fp); CHECK_V_ERROR;
      }
    }
  }
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "Vec_dhPrintBIN"
void Vec_dhPrintBIN(Vec_dh v, SubdomainGraph_dh sg, char *filename)
{
  START_FUNC_DH
  if (np_dh > 1) {
    SET_V_ERROR("only implemented for a single MPI task");
  }
  if (sg != NULL) {
    SET_V_ERROR("not implemented for reordered vector; ensure sg=NULL");
  }

  io_dh_print_ebin_vec_private(v->n, 0, v->vals,
                               NULL, NULL, NULL, filename); CHECK_V_ERROR;
  END_FUNC_DH
}

#define MAX_JUNK 200

#undef __FUNC__
#define __FUNC__ "Vec_dhRead"
void Vec_dhRead(Vec_dh *vout, HYPRE_Int ignore, char *filename)
{
  START_FUNC_DH
  Vec_dh tmp = 0;
  FILE *fp;
  HYPRE_Int items, n, i;
  HYPRE_Real *v, w;
  char junk[MAX_JUNK];

  Vec_dhCreate(&tmp); CHECK_V_ERROR;
  *vout = tmp;

  if (np_dh > 1) {
    SET_V_ERROR("only implemented for a single MPI task");
  }

  fp=openFile_dh(filename, "w"); CHECK_V_ERROR;

  /* skip over file lines */
  if (ignore) {
    hypre_printf("Vec_dhRead:: ignoring following header lines:\n");
    hypre_printf("--------------------------------------------------------------\n");
    for (i=0; i<ignore; ++i) {
      if (fgets(junk, MAX_JUNK, fp) != NULL) {
        hypre_printf("%s", junk);
      }
    }
    hypre_printf("--------------------------------------------------------------\n");
  }

  /* count floating point entries in file */
  n = 0;
  while (!feof(fp)) {
    items = hypre_fscanf(fp,"%lg", &w);
    if (items != 1) {
      break;
    }
    ++n;
  }

  hypre_printf("Vec_dhRead:: n= %i\n", n);

  /* allocate storage */
  tmp->n = n;
  v = tmp->vals =  (HYPRE_Real*)MALLOC_DH(n*sizeof(HYPRE_Real)); CHECK_V_ERROR;

  /* reset file, and skip over header again */
  rewind(fp);
  rewind(fp);
  for (i=0; i<ignore; ++i) {
    if (fgets(junk, MAX_JUNK, fp) != NULL) {
      hypre_printf("%s", junk);
    }
  }

  /* read values */
  for (i=0; i<n;  ++i) {
    items = hypre_fscanf(fp,"%lg", v+i);
    if (items != 1) {
      hypre_sprintf(msgBuf_dh, "failed to read value %i of %i", i+1, n);
    }
  }

  closeFile_dh(fp); CHECK_V_ERROR;
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "Vec_dhReadBIN"
extern void Vec_dhReadBIN(Vec_dh *vout, char *filename)

{
  START_FUNC_DH
  Vec_dh tmp = 0;

  Vec_dhCreate(&tmp); CHECK_V_ERROR;
  *vout = tmp;
  io_dh_read_ebin_vec_private(&tmp->n, &tmp->vals, filename); CHECK_V_ERROR;
  END_FUNC_DH
}
