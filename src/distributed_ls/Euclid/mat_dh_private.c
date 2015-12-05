/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.8 $
 ***********************************************************************EHEADER*/

#include "mat_dh_private.h"
#include "Parser_dh.h"
#include "Hash_i_dh.h"
#include "Mat_dh.h"
#include "Mem_dh.h"
#include "Vec_dh.h"

#ifdef PETSC_MODE
#include "euclid_petsc.h"
#endif

#define IS_UPPER_TRI 97
#define IS_LOWER_TRI 98
#define IS_FULL      99
static HYPRE_Int isTriangular(HYPRE_Int m, HYPRE_Int *rp, HYPRE_Int *cval);

/* Instantiates Aout; allocates storage for rp, cval, and aval arrays;
   uses rowLengths[] and rowToBlock[] data to fill in rp[].
*/
static void mat_par_read_allocate_private(Mat_dh *Aout, HYPRE_Int n, 
                                    HYPRE_Int *rowLengths, HYPRE_Int *rowToBlock);

/* Currently, divides (partitions)matrix by contiguous sections of rows.
   For future expansion: use metis.
*/
void mat_partition_private(Mat_dh A, HYPRE_Int blocks, HYPRE_Int *o2n_row, HYPRE_Int *rowToBlock);


static void convert_triples_to_scr_private(HYPRE_Int m, HYPRE_Int nz, 
                                           HYPRE_Int *I, HYPRE_Int *J, double *A, 
                                           HYPRE_Int *rp, HYPRE_Int *cval, double *aval);

#if 0
#undef __FUNC__
#define __FUNC__ "mat_dh_print_graph_private"
void mat_dh_print_graph_private(HYPRE_Int m, HYPRE_Int beg_row, HYPRE_Int *rp, HYPRE_Int *cval, 
                    double *aval, HYPRE_Int *n2o, HYPRE_Int *o2n, Hash_i_dh hash, FILE* fp)
{
  START_FUNC_DH
  HYPRE_Int i, j, row, col;
  double val;
  bool private_n2o = false;
  bool private_hash = false;

  if (n2o == NULL) {
    private_n2o = true;
    create_nat_ordering_private(m, &n2o); CHECK_V_ERROR;
    create_nat_ordering_private(m, &o2n); CHECK_V_ERROR;
  }
 
  if (hash == NULL) {
    private_hash = true;
    Hash_i_dhCreate(&hash, -1); CHECK_V_ERROR;
  }

  for (i=0; i<m; ++i) {
    row = n2o[i];
    for (j=rp[row]; j<rp[row+1]; ++j) {
      col = cval[j];
      if (col < beg_row || col >= beg_row+m) {
        HYPRE_Int tmp = col;

        /* nonlocal column: get permutation from hash table */
        tmp = Hash_i_dhLookup(hash, col); CHECK_V_ERROR;
        if (tmp == -1) { 
          hypre_sprintf(msgBuf_dh, "beg_row= %i  m= %i; nonlocal column= %i not in hash table",
                                beg_row, m, col); 
          SET_V_ERROR(msgBuf_dh);
        } else {
          col = tmp;
        }
      } else {
        col = o2n[col];
      }

      if (aval == NULL) { 
        val = _MATLAB_ZERO_;
      } else {
        val = aval[j];
      }
      hypre_fprintf(fp, "%i %i %g\n", 1+row+beg_row, 1+col, val);
    }
  }

  if (private_n2o) {
    destroy_nat_ordering_private(n2o); CHECK_V_ERROR;
    destroy_nat_ordering_private(o2n); CHECK_V_ERROR;
  }

  if (private_hash) {
    Hash_i_dhDestroy(hash); CHECK_V_ERROR;
  }
  END_FUNC_DH
}

#endif


/* currently only for unpermuted */
#undef __FUNC__
#define __FUNC__ "mat_dh_print_graph_private"
void mat_dh_print_graph_private(HYPRE_Int m, HYPRE_Int beg_row, HYPRE_Int *rp, HYPRE_Int *cval, 
                    double *aval, HYPRE_Int *n2o, HYPRE_Int *o2n, Hash_i_dh hash, FILE* fp)
{
  START_FUNC_DH
  HYPRE_Int i, j, row, col;
  bool private_n2o = false;
  bool private_hash = false;
  HYPRE_Int *work = NULL;

  work = (HYPRE_Int*)MALLOC_DH(m*sizeof(HYPRE_Int)); CHECK_V_ERROR;

  if (n2o == NULL) {
    private_n2o = true;
    create_nat_ordering_private(m, &n2o); CHECK_V_ERROR;
    create_nat_ordering_private(m, &o2n); CHECK_V_ERROR;
  }
 
  if (hash == NULL) {
    private_hash = true;
    Hash_i_dhCreate(&hash, -1); CHECK_V_ERROR;
  }

  for (i=0; i<m; ++i) {
    for (j=0; j<m; ++j) work[j] = 0;
    row = n2o[i];
    for (j=rp[row]; j<rp[row+1]; ++j) {
      col = cval[j];

      /* local column */
      if (col >= beg_row || col < beg_row+m) {
        col = o2n[col];
      } 

      /* nonlocal column: get permutation from hash table */
      else {
        HYPRE_Int tmp = col;

        tmp = Hash_i_dhLookup(hash, col); CHECK_V_ERROR;
        if (tmp == -1) { 
          hypre_sprintf(msgBuf_dh, "beg_row= %i  m= %i; nonlocal column= %i not in hash table",
                                beg_row, m, col); 
          SET_V_ERROR(msgBuf_dh);
        } else {
          col = tmp;
        }
      } 

      work[col] = 1;
    }

    for (j=0; j<m; ++j) {
      if (work[j]) {
        hypre_fprintf(fp, " x ");
      } else {
        hypre_fprintf(fp, "   ");
      }
    }
    hypre_fprintf(fp, "\n");
  }

  if (private_n2o) {
    destroy_nat_ordering_private(n2o); CHECK_V_ERROR;
    destroy_nat_ordering_private(o2n); CHECK_V_ERROR;
  }

  if (private_hash) {
    Hash_i_dhDestroy(hash); CHECK_V_ERROR;
  }

  if (work != NULL) { FREE_DH(work); CHECK_V_ERROR; }
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "create_nat_ordering_private"
void create_nat_ordering_private(HYPRE_Int m, HYPRE_Int **p)
{
  START_FUNC_DH
  HYPRE_Int *tmp, i;

  tmp = *p = (HYPRE_Int*)MALLOC_DH(m*sizeof(HYPRE_Int)); CHECK_V_ERROR;
  for (i=0; i<m; ++i) {
    tmp[i] = i;
  }
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "destroy_nat_ordering_private"
void destroy_nat_ordering_private(HYPRE_Int *p)
{
  START_FUNC_DH
  FREE_DH(p); CHECK_V_ERROR;
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "invert_perm"
void invert_perm(HYPRE_Int m, HYPRE_Int *pIN, HYPRE_Int *pOUT)
{
  START_FUNC_DH
  HYPRE_Int i;

  for (i=0; i<m; ++i) pOUT[pIN[i]] = i;
  END_FUNC_DH
}



/* only implemented for a single cpu! */
#undef __FUNC__
#define __FUNC__ "mat_dh_print_csr_private"
void mat_dh_print_csr_private(HYPRE_Int m, HYPRE_Int *rp, HYPRE_Int *cval, double *aval, FILE* fp)
{
  START_FUNC_DH
  HYPRE_Int i, nz = rp[m];

  /* print header line */
  hypre_fprintf(fp, "%i %i\n", m, rp[m]);

  /* print rp[] */
  for (i=0; i<=m; ++i) hypre_fprintf(fp, "%i ", rp[i]);
  hypre_fprintf(fp, "\n");

  /* print cval[] */
  for (i=0; i<nz; ++i) hypre_fprintf(fp, "%i ", cval[i]);
  hypre_fprintf(fp, "\n");

  /* print aval[] */
  for (i=0; i<nz; ++i) hypre_fprintf(fp, "%1.19e ", aval[i]);
  hypre_fprintf(fp, "\n");

  END_FUNC_DH
}


/* only implemented for a single cpu! */
#undef __FUNC__
#define __FUNC__ "mat_dh_read_csr_private"
void mat_dh_read_csr_private(HYPRE_Int *mOUT, HYPRE_Int **rpOUT, HYPRE_Int **cvalOUT, 
                                            double **avalOUT, FILE* fp)
{
  START_FUNC_DH
  HYPRE_Int i, m, nz, items;
  HYPRE_Int *rp, *cval;
  double *aval;

  /* read header line */
  items = hypre_fscanf(fp,"%d %d",&m, &nz);
  if (items != 2) {
    SET_V_ERROR("failed to read header");
  } else {
    hypre_printf("mat_dh_read_csr_private:: m= %i  nz= %i\n", m, nz);
  }

  *mOUT = m;
  rp = *rpOUT = (HYPRE_Int*)MALLOC_DH((m+1)*sizeof(HYPRE_Int)); CHECK_V_ERROR;
  cval = *cvalOUT = (HYPRE_Int*)MALLOC_DH(nz*sizeof(HYPRE_Int)); CHECK_V_ERROR;
  aval = *avalOUT = (double*)MALLOC_DH(nz*sizeof(double)); CHECK_V_ERROR;

  /* read rp[] block */
  for (i=0; i<=m; ++i) {
    items = hypre_fscanf(fp,"%d", &(rp[i]));
    if (items != 1) {
      hypre_sprintf(msgBuf_dh, "failed item %i of %i in rp block", i, m+1);
      SET_V_ERROR(msgBuf_dh);
    }
  }

  /* read cval[] block */
  for (i=0; i<nz; ++i) {
    items = hypre_fscanf(fp,"%d", &(cval[i]));
    if (items != 1) {
      hypre_sprintf(msgBuf_dh, "failed item %i of %i in cval block", i, m+1);
      SET_V_ERROR(msgBuf_dh);
    }
  }

  /* read aval[] block */
  for (i=0; i<nz; ++i) {
    items = hypre_fscanf(fp,"%lg", &(aval[i]));
    if (items != 1) {
      hypre_sprintf(msgBuf_dh, "failed item %i of %i in aval block", i, m+1);
      SET_V_ERROR(msgBuf_dh);
    }
  }
  END_FUNC_DH
}

/*============================================*/
#define MAX_JUNK 200

#undef __FUNC__
#define __FUNC__ "mat_dh_read_triples_private"
void mat_dh_read_triples_private(HYPRE_Int ignore, HYPRE_Int *mOUT, HYPRE_Int **rpOUT, 
                                   HYPRE_Int **cvalOUT, double **avalOUT, FILE* fp)
{
  START_FUNC_DH
  HYPRE_Int m, n, nz, items, i, j;
  HYPRE_Int idx = 0;
  HYPRE_Int *cval, *rp, *I, *J;
  double *aval, *A, v;
  char junk[MAX_JUNK];
  fpos_t fpos;

  /* skip over header */
  if (ignore && myid_dh == 0) {
    hypre_printf("mat_dh_read_triples_private:: ignoring following header lines:\n");
    hypre_printf("--------------------------------------------------------------\n");
    for (i=0; i<ignore; ++i) {
      fgets(junk, MAX_JUNK, fp);
      hypre_printf("%s", junk);
    }
    hypre_printf("--------------------------------------------------------------\n");
    if (fgetpos(fp, &fpos)) SET_V_ERROR("fgetpos failed!");
    hypre_printf("\nmat_dh_read_triples_private::1st two non-ignored lines:\n");
    hypre_printf("--------------------------------------------------------------\n");
    for (i=0; i<2; ++i) {
      fgets(junk, MAX_JUNK, fp);
      hypre_printf("%s", junk);
    }
    hypre_printf("--------------------------------------------------------------\n");
    if (fsetpos(fp, &fpos)) SET_V_ERROR("fsetpos failed!");
  }


if (feof(fp)) hypre_printf("trouble!");

  /* determine matrix dimensions */
  m=n=nz=0;
  while (!feof(fp)) {
    items = hypre_fscanf(fp,"%d %d %lg",&i,&j,&v);
    if (items != 3) {
      break;
    }
    ++nz;
    if (i > m) m = i;
    if (j > n) n = j;
  }

  if (myid_dh == 0) {
    hypre_printf("mat_dh_read_triples_private: m= %i  nz= %i\n", m, nz);
  }


  /* reset file, and skip over header again */
  rewind(fp);
  for (i=0; i<ignore; ++i) {
    fgets(junk, MAX_JUNK, fp);
  }

  /* error check for squareness */
  if (m != n) {
    hypre_sprintf(msgBuf_dh, "matrix is not square; row= %i, cols= %i", m, n);
    SET_V_ERROR(msgBuf_dh);
  }

  *mOUT = m;

  /* allocate storage */
  rp = *rpOUT = (HYPRE_Int*)MALLOC_DH((m+1)*sizeof(HYPRE_Int)); CHECK_V_ERROR;
  cval = *cvalOUT = (HYPRE_Int*)MALLOC_DH(nz*sizeof(HYPRE_Int)); CHECK_V_ERROR;
  aval = *avalOUT = (double*)MALLOC_DH(nz*sizeof(double)); CHECK_V_ERROR;

  I = (HYPRE_Int*)MALLOC_DH(nz*sizeof(HYPRE_Int)); CHECK_V_ERROR;
  J = (HYPRE_Int*)MALLOC_DH(nz*sizeof(HYPRE_Int)); CHECK_V_ERROR;
  A = (double*)MALLOC_DH(nz*sizeof(double)); CHECK_V_ERROR;

  /* read <row, col, value> triples into arrays */
  while (!feof(fp)) {
    items = hypre_fscanf(fp,"%d %d %lg",&i,&j,&v);
    if (items < 3) break;
    j--;
    i--;
    I[idx] = i;
    J[idx] = j;
    A[idx] = v;
    ++idx;
  }

  /* convert from triples to sparse-compressed-row storage */
  convert_triples_to_scr_private(m, nz, I, J, A, rp, cval, aval); CHECK_V_ERROR;

  /* if matrix is triangular */
  { HYPRE_Int type;
    type = isTriangular(m, rp, cval); CHECK_V_ERROR;
    if (type == IS_UPPER_TRI) {
      hypre_printf("CAUTION: matrix is upper triangular; converting to full\n");
    } else if (type == IS_LOWER_TRI) {
      hypre_printf("CAUTION: matrix is lower triangular; converting to full\n");
    }

    if (type == IS_UPPER_TRI || type == IS_LOWER_TRI) {
      make_full_private(m, &rp, &cval, &aval); CHECK_V_ERROR;
    }
  }

  *rpOUT = rp;
  *cvalOUT = cval;
  *avalOUT = aval;

  FREE_DH(I); CHECK_V_ERROR;
  FREE_DH(J); CHECK_V_ERROR;
  FREE_DH(A); CHECK_V_ERROR;

  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "convert_triples_to_scr_private"
void convert_triples_to_scr_private(HYPRE_Int m, HYPRE_Int nz, HYPRE_Int *I, HYPRE_Int *J, double *A, 
                                      HYPRE_Int *rp, HYPRE_Int *cval, double *aval)
{
  START_FUNC_DH
  HYPRE_Int i;
  HYPRE_Int *rowCounts;

  rowCounts = (HYPRE_Int*)MALLOC_DH((m+1)*sizeof(HYPRE_Int)); CHECK_V_ERROR;
  for (i=0; i<m; ++i) rowCounts[i] =   0;

  /* count number of entries in each row */
  for (i=0; i<nz; ++i) {
    HYPRE_Int row = I[i];
    rowCounts[row] += 1;
  }

  /* prefix-sum to form rp[] */
  rp[0] = 0;
  for (i=1; i<=m; ++i) {
    rp[i] = rp[i-1] + rowCounts[i-1];
  }
  memcpy(rowCounts, rp, (m+1)*sizeof(HYPRE_Int));

  /* write SCR arrays */
  for (i=0; i<nz; ++i) {
    HYPRE_Int row = I[i];
    HYPRE_Int col = J[i];
    double val = A[i];
    HYPRE_Int idx = rowCounts[row];
    rowCounts[row] += 1;

    cval[idx] = col;
    aval[idx] = val;
  }


  FREE_DH(rowCounts); CHECK_V_ERROR;
  END_FUNC_DH
}


/*======================================================================
 * utilities for use in drivers that read, write, convert, and/or
 * compare different file types
 *======================================================================*/

void fix_diags_private(Mat_dh A);
void insert_missing_diags_private(Mat_dh A);

#undef __FUNC__
#define __FUNC__ "readMat"
void readMat(Mat_dh *Aout, char *ft, char *fn, HYPRE_Int ignore)
{
  START_FUNC_DH
  bool makeStructurallySymmetric;
  bool fixDiags;
  *Aout = NULL;

  makeStructurallySymmetric = 
      Parser_dhHasSwitch(parser_dh, "-makeSymmetric");
  fixDiags = 
      Parser_dhHasSwitch(parser_dh, "-fixDiags");

  if (fn == NULL) {
    SET_V_ERROR("passed NULL filename; can't open for reading!");
  }

  if (!strcmp(ft, "csr")) 
  {
    Mat_dhReadCSR(Aout, fn); CHECK_V_ERROR;
  } 

  else if (!strcmp(ft, "trip")) 
  {
    Mat_dhReadTriples(Aout, ignore, fn); CHECK_V_ERROR;
  } 

  else if (!strcmp(ft, "ebin"))
  {
    Mat_dhReadBIN(Aout, fn); CHECK_V_ERROR;
  } 

#ifdef PETSC_MODE
  else if (!strcmp(ft, "petsc")) {
    Viewer_DH viewer;
    Mat Apetsc;
    HYPRE_Int ierr;

    ierr = ViewerBinaryOpen_DH(comm_dh, fn, BINARY_RDONLY_DH, &viewer);
    if (ierr) { SET_V_ERROR("ViewerBinaryOpen failed! [PETSc lib]"); }
    ierr = MatLoad(viewer, MATSEQAIJ, &Apetsc);
    if (ierr) { SET_V_ERROR("MatLoad failed! [PETSc lib]"); }
    ierr = ViewerDestroy_DH(viewer);
    if (ierr) { SET_V_ERROR("ViewerDestroy failed! [PETSc lib]"); }
    ierr = convertPetscToEuclidMat(Apetsc, Aout);
    if (ierr) { SET_V_ERROR("convertPetscToEuclidMat failed!"); }
    ierr = MatDestroy(Apetsc);
    if (ierr) { SET_V_ERROR("MatDestroy failed! [PETSc lib]"); }
  } 
#else 
  else if (!strcmp(ft, "petsc")) {
    hypre_sprintf(msgBuf_dh, "must recompile Euclid using petsc mode!");
    SET_V_ERROR(msgBuf_dh);
  }
#endif

  else 
  {
    hypre_sprintf(msgBuf_dh, "unknown filetype: -ftin %s", ft);
    SET_V_ERROR(msgBuf_dh);
  }

  if (makeStructurallySymmetric) {
    hypre_printf("\npadding with zeros to make structurally symmetric\n");
    Mat_dhMakeStructurallySymmetric(*Aout); CHECK_V_ERROR;
  }

  if ( (*Aout)->m == 0) {
    SET_V_ERROR("row count = 0; something's wrong!");
  }

  if (fixDiags) {
    fix_diags_private(*Aout); CHECK_V_ERROR;
  }

  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "fix_diags_private"
void fix_diags_private(Mat_dh A)
{
  START_FUNC_DH
  HYPRE_Int i, j, m = A->m, *rp = A->rp, *cval = A->cval;
  double *aval = A->aval;
  bool insertDiags = false;

  /* verify that all diagonals are present */
  for (i=0; i<m; ++i) {
    bool isMissing = true;
    for (j=rp[i]; j<rp[i+1]; ++j) {
      if (cval[j] == i) {
        isMissing = false;
        break;
      }
    }
    if (isMissing) {
      insertDiags = true;
      break;
    }
  }

  if (insertDiags) {
    insert_missing_diags_private(A); CHECK_V_ERROR;
    rp = A->rp; 
    cval = A->cval;
    aval = A->aval;
  }

  /* set value of all diags to largest absolute value in each row */
  for (i=0; i<m; ++i) {
    double sum = 0;
    for (j=rp[i]; j<rp[i+1]; ++j) {
      sum = MAX(sum, fabs(aval[j]));
    }
    for (j=rp[i]; j<rp[i+1]; ++j) {
      if (cval[j] == i) {
        aval[j] = sum;
        break;
      }
    }
  }

  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "insert_missing_diags_private"
void insert_missing_diags_private(Mat_dh A)
{
  START_FUNC_DH
  HYPRE_Int *RP = A->rp, *CVAL = A->cval, m = A->m;
  HYPRE_Int *rp, *cval;
  double *AVAL = A->aval, *aval;
  HYPRE_Int i, j, nz = RP[m]+m;
  HYPRE_Int idx = 0;

  rp = A->rp = (HYPRE_Int *)MALLOC_DH((1+m)*sizeof(HYPRE_Int)); CHECK_V_ERROR;
  cval = A->cval = (HYPRE_Int *)MALLOC_DH(nz*sizeof(HYPRE_Int)); CHECK_V_ERROR;
  aval = A->aval = (double *)MALLOC_DH(nz*sizeof(double)); CHECK_V_ERROR;
  rp[0] = 0;

  for (i=0; i<m; ++i) {
    bool isMissing = true;
    for (j=RP[i]; j<RP[i+1]; ++j) {
      cval[idx] = CVAL[j];
      aval[idx] = AVAL[j];
      ++idx;
      if (CVAL[j] == i) isMissing = false;
    }
    if (isMissing) {
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
#define __FUNC__ "readVec"
void readVec(Vec_dh *bout, char *ft, char *fn, HYPRE_Int ignore)
{
  START_FUNC_DH
  *bout = NULL;

  if (fn == NULL) {
    SET_V_ERROR("passed NULL filename; can't open for reading!");
  }

  if (!strcmp(ft, "csr")  ||  !strcmp(ft, "trip")) 
  {
    Vec_dhRead(bout, ignore, fn); CHECK_V_ERROR;
  } 

  else if (!strcmp(ft, "ebin"))
  {
    Vec_dhReadBIN(bout, fn); CHECK_V_ERROR;
  } 

#ifdef PETSC_MODE
  else if (!strcmp(ft, "petsc")) {
    Viewer_DH viewer;
    HYPRE_Int ierr;
    Vec bb;

    ierr = ViewerBinaryOpen_DH(comm_dh, fn, BINARY_WRONLY_DH, &viewer);
    if (ierr) { SET_V_ERROR("ViewerBinaryOpen failed! [PETSc lib]"); }
    ierr = VecLoad(viewer, &bb);
    if (ierr) { SET_V_ERROR("VecLoad failed! [PETSc lib]"); }
    ierr = ViewerDestroy_DH(viewer);
    if (ierr) { SET_V_ERROR("ViewerDestroy failed! [PETSc lib]"); }
    ierr = convertPetscToEuclidVec(bb, bout);
    if (ierr) { SET_V_ERROR("convertPetscToEuclidVec failed!"); }
    ierr = VecDestroy(bb);
    if (ierr) { SET_V_ERROR("VecDestroy failed! [PETSc lib]"); }
  } 
#else
  else if (!strcmp(ft, "petsc")) {
    hypre_sprintf(msgBuf_dh, "must recompile Euclid using petsc mode!");
    SET_V_ERROR(msgBuf_dh);
  }
#endif

  else 
  {
    hypre_sprintf(msgBuf_dh, "unknown filetype: -ftin %s", ft);
    SET_V_ERROR(msgBuf_dh);
  }
  
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "writeMat"
void writeMat(Mat_dh Ain, char *ft, char *fn)
{
  START_FUNC_DH
  if (fn == NULL) {
    SET_V_ERROR("passed NULL filename; can't open for writing!");
  }

  if (!strcmp(ft, "csr")) 
  {
    Mat_dhPrintCSR(Ain, NULL, fn); CHECK_V_ERROR;
  } 

  else if (!strcmp(ft, "trip")) 
  {
    Mat_dhPrintTriples(Ain, NULL, fn); CHECK_V_ERROR;
  } 

  else if (!strcmp(ft, "ebin"))
  {
    Mat_dhPrintBIN(Ain, NULL, fn); CHECK_V_ERROR;
  } 

#ifdef PETSC_MODE
  else if (!strcmp(ft, "petsc")) 
  {
    Viewer_DH viewer;
    Mat Apetsc;
    HYPRE_Int ierr;

    ierr = buildPetscMat(Ain->m, Ain->n, Ain->beg_row, 
                         Ain->rp, Ain->cval, Ain->aval, &Apetsc);
    if (ierr) { SET_V_ERROR("buildPetscMat failed!"); }

    ierr = ViewerBinaryOpen_DH(comm_dh, fn, BINARY_CREATE_DH, &viewer);

    if (ierr) { SET_V_ERROR("ViewerBinaryOpen failed! [PETSc lib]"); }
    ierr = MatView(Apetsc, viewer);
    if (ierr) { SET_V_ERROR("MatView failed! [PETSc lib]"); }
    ierr = ViewerDestroy_DH(viewer);
    if (ierr) { SET_V_ERROR("ViewerDestroy failed! [PETSc lib]"); }
    ierr = MatDestroy(Apetsc);
    if (ierr) { SET_V_ERROR("MatDestroy failed! [PETSc lib]"); }
  }
#else

  else if (!strcmp(ft, "petsc")) {
    hypre_sprintf(msgBuf_dh, "must recompile Euclid using petsc mode!");
    SET_V_ERROR(msgBuf_dh);
  }
#endif

  else 
  {
    hypre_sprintf(msgBuf_dh, "unknown filetype: -ftout %s", ft);
    SET_V_ERROR(msgBuf_dh);
  }

  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "writeVec"
void writeVec(Vec_dh bin, char *ft, char *fn)
{
  START_FUNC_DH
  if (fn == NULL) {
    SET_V_ERROR("passed NULL filename; can't open for writing!");
  }

  if (!strcmp(ft, "csr")  ||  !strcmp(ft, "trip")) 
  {
    Vec_dhPrint(bin, NULL, fn); CHECK_V_ERROR;
  } 

  else if (!strcmp(ft, "ebin"))
  {
    Vec_dhPrintBIN(bin, NULL, fn); CHECK_V_ERROR;
  } 

#ifdef PETSC_MODE
  else if (!strcmp(ft, "petsc")) 
  {
    Viewer_DH viewer;
    HYPRE_Int ierr;
    Vec bb;

    ierr = buildPetscVec(bin->n, bin->n, 0, bin->vals, &bb);
    if (ierr) { SET_V_ERROR("buildPetscVec failed!");  }
    ierr = ViewerBinaryOpen_DH(comm_dh, fn, BINARY_CREATE_DH, &viewer);
    if (ierr) { SET_V_ERROR("ViewerBinaryOpen failed! [PETSc lib]"); }
    ierr = VecView(bb, viewer);
    if (ierr) { SET_V_ERROR("VecView failed! [PETSc lib]"); }
    ierr = ViewerDestroy_DH(viewer);
    if (ierr) { SET_V_ERROR("ViewerDestroy failed! [PETSc lib]"); }
    ierr = VecDestroy(bb);
    if (ierr) { SET_V_ERROR("VecDestroy failed! [PETSc lib]"); }
  } 
#else
  else if (!strcmp(ft, "petsc")) {
    hypre_sprintf(msgBuf_dh, "must recompile Euclid using petsc mode!");
    SET_V_ERROR(msgBuf_dh);
  }
#endif

  else 
  {
    hypre_sprintf(msgBuf_dh, "unknown filetype: -ftout %s", ft);
    SET_V_ERROR(msgBuf_dh);
  }

  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "isTriangular"
HYPRE_Int isTriangular(HYPRE_Int m, HYPRE_Int *rp, HYPRE_Int *cval)
{
  START_FUNC_DH
  HYPRE_Int row, j;
  HYPRE_Int type;
  bool type_lower = false, type_upper = false;

  if (np_dh > 1) {
    SET_ERROR(-1, "only implemented for a single cpu");
  }

  for (row=0; row<m; ++row) {
    for (j=rp[row]; j<rp[row+1]; ++j) {
      HYPRE_Int col = cval[j];
      if (col < row) type_lower = true;
      if (col > row) type_upper = true;
    }
    if (type_lower && type_upper) break;
  }

  if (type_lower && type_upper) {
    type = IS_FULL;
  } else if (type_lower) {
    type = IS_LOWER_TRI;
  } else {
    type = IS_UPPER_TRI;
  }
  END_FUNC_VAL(type)
}

/*-----------------------------------------------------------------------------------*/

static void mat_dh_transpose_reuse_private_private(
                              bool allocateMem, HYPRE_Int m, 
                              HYPRE_Int *rpIN, HYPRE_Int *cvalIN, double *avalIN,
                              HYPRE_Int **rpOUT, HYPRE_Int **cvalOUT, double **avalOUT);


#undef __FUNC__
#define __FUNC__ "mat_dh_transpose_reuse_private"
void mat_dh_transpose_reuse_private(HYPRE_Int m, 
                              HYPRE_Int *rpIN, HYPRE_Int *cvalIN, double *avalIN,
                              HYPRE_Int *rpOUT, HYPRE_Int *cvalOUT, double *avalOUT)
{
  START_FUNC_DH
  mat_dh_transpose_reuse_private_private(false, m, rpIN, cvalIN, avalIN,
                                       &rpOUT, &cvalOUT, &avalOUT); CHECK_V_ERROR;
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "mat_dh_transpose_private"
void mat_dh_transpose_private(HYPRE_Int m, HYPRE_Int *RP, HYPRE_Int **rpOUT,
                              HYPRE_Int *CVAL, HYPRE_Int **cvalOUT,
                              double *AVAL, double **avalOUT)
{
  START_FUNC_DH
  mat_dh_transpose_reuse_private_private(true, m, RP, CVAL, AVAL, 
                                       rpOUT, cvalOUT, avalOUT); CHECK_V_ERROR;
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "mat_dh_transpose_private_private"
void mat_dh_transpose_reuse_private_private(bool allocateMem, HYPRE_Int m, 
                              HYPRE_Int *RP, HYPRE_Int *CVAL, double *AVAL,
                              HYPRE_Int **rpOUT, HYPRE_Int **cvalOUT, double **avalOUT)
{
  START_FUNC_DH
  HYPRE_Int *rp, *cval, *tmp;
  HYPRE_Int i, j, nz = RP[m];
  double *aval;

  if (allocateMem) {
    rp = *rpOUT = (HYPRE_Int *)MALLOC_DH((1+m)*sizeof(HYPRE_Int)); CHECK_V_ERROR;
    cval = *cvalOUT = (HYPRE_Int *)MALLOC_DH(nz*sizeof(HYPRE_Int)); CHECK_V_ERROR;
    if (avalOUT != NULL) {
      aval = *avalOUT = (double*)MALLOC_DH(nz*sizeof(double)); CHECK_V_ERROR;
    }
  } else {
    rp = *rpOUT;
    cval = *cvalOUT;
    if (avalOUT != NULL) aval = *avalOUT;
  }


  tmp = (HYPRE_Int *)MALLOC_DH((1+m)*sizeof(HYPRE_Int)); CHECK_V_ERROR;
  for (i=0; i<=m; ++i) tmp[i] = 0;

  for (i=0; i<m; ++i) {
    for (j=RP[i]; j<RP[i+1]; ++j) {
      HYPRE_Int col = CVAL[j];
      tmp[col+1] += 1;
    }
  }
  for (i=1; i<=m; ++i) tmp[i] += tmp[i-1];
  memcpy(rp, tmp, (m+1)*sizeof(HYPRE_Int));

  if (avalOUT != NULL) {
    for (i=0; i<m; ++i) {
      for (j=RP[i]; j<RP[i+1]; ++j) {
        HYPRE_Int col = CVAL[j];
        HYPRE_Int idx = tmp[col];
        cval[idx] = i;
        aval[idx] = AVAL[j];
        tmp[col] += 1;
      }
    }
  }

  else {
    for (i=0; i<m; ++i) {
      for (j=RP[i]; j<RP[i+1]; ++j) {
        HYPRE_Int col = CVAL[j];
        HYPRE_Int idx = tmp[col];
        cval[idx] = i;
        tmp[col] += 1;
      }
    }
  }

  FREE_DH(tmp); CHECK_V_ERROR;
  END_FUNC_DH
}

/*-----------------------------------------------------------------------------------*/

#undef __FUNC__
#define __FUNC__ "mat_find_owner"
HYPRE_Int mat_find_owner(HYPRE_Int *beg_rows, HYPRE_Int *end_rows, HYPRE_Int index)
{
  START_FUNC_DH
  HYPRE_Int pe, owner = -1;

  for (pe=0; pe<np_dh; ++pe) {
    if (index >= beg_rows[pe] && index < end_rows[pe]) {
      owner = pe;
      break;
    }
  }

  if (owner == -1) {
    hypre_sprintf(msgBuf_dh, "failed to find owner for index= %i", index);
    SET_ERROR(-1, msgBuf_dh);
  }

  END_FUNC_VAL(owner)
}


#define AVAL_TAG 2
#define CVAL_TAG 3
void partition_and_distribute_private(Mat_dh A, Mat_dh *Bout);
void partition_and_distribute_metis_private(Mat_dh A, Mat_dh *Bout); 

#undef __FUNC__
#define __FUNC__ "readMat_par"
void readMat_par(Mat_dh *Aout, char *fileType, char *fileName, HYPRE_Int ignore)
{
  START_FUNC_DH
  Mat_dh A = NULL;

  if (myid_dh == 0) {
    HYPRE_Int tmp = np_dh;
    np_dh = 1;
    readMat(&A, fileType, fileName, ignore); CHECK_V_ERROR;
    np_dh = tmp;
  }

  if (np_dh == 1) {
    *Aout = A;
  } else {
    if (Parser_dhHasSwitch(parser_dh, "-metis")) {
      partition_and_distribute_metis_private(A, Aout); CHECK_V_ERROR;
    } else {
      partition_and_distribute_private(A, Aout); CHECK_V_ERROR;
    }
  }

  if (np_dh > 1 && A != NULL) {
    Mat_dhDestroy(A); CHECK_V_ERROR;
  }

 
  if (Parser_dhHasSwitch(parser_dh, "-printMAT")) {
    char xname[] = "A", *name = xname;
    Parser_dhReadString(parser_dh, "-printMat", &name);
    Mat_dhPrintTriples(*Aout, NULL, name); CHECK_V_ERROR;
    printf_dh("\n@@@ readMat_par: printed mat to %s\n\n", xname);
  }


  END_FUNC_DH
}

/* this is bad code! */
#undef __FUNC__
#define __FUNC__ "partition_and_distribute_metis_private"
void partition_and_distribute_metis_private(Mat_dh A, Mat_dh *Bout)
{
  START_FUNC_DH
  Mat_dh B = NULL;
  Mat_dh C = NULL;
  HYPRE_Int i, m;
  HYPRE_Int *rowLengths = NULL;
  HYPRE_Int *o2n_row = NULL, *n2o_col = NULL, *rowToBlock = NULL;
  HYPRE_Int *beg_row = NULL, *row_count = NULL;
  hypre_MPI_Request *send_req = NULL;
  hypre_MPI_Request *rcv_req = NULL;
  hypre_MPI_Status  *send_status = NULL;
  hypre_MPI_Status  *rcv_status = NULL;

  hypre_MPI_Barrier(comm_dh);
  printf_dh("@@@ partitioning with metis\n");

  /* broadcast number of rows to all processors */
  if (myid_dh == 0)  m = A->m;
  hypre_MPI_Bcast(&m, 1, HYPRE_MPI_INT, 0, hypre_MPI_COMM_WORLD);

  /* broadcast number of nonzeros in each row to all processors */
  rowLengths = (HYPRE_Int*)MALLOC_DH(m*sizeof(HYPRE_Int)); CHECK_V_ERROR;
  rowToBlock = (HYPRE_Int*)MALLOC_DH(m*sizeof(HYPRE_Int)); CHECK_V_ERROR;

  if (myid_dh == 0) {
    HYPRE_Int *tmp = A->rp;
    for (i=0; i<m; ++i) {
      rowLengths[i] = tmp[i+1] - tmp[i];
    }
  }
  hypre_MPI_Bcast(rowLengths, m, HYPRE_MPI_INT, 0, comm_dh);

  /* partition matrix */
  if (myid_dh == 0) {
    HYPRE_Int idx = 0;
    HYPRE_Int j;

    /* partition and permute matrix */
    Mat_dhPartition(A, np_dh, &beg_row, &row_count, &n2o_col, &o2n_row); ERRCHKA;
    Mat_dhPermute(A, n2o_col, &C); ERRCHKA;
 
    /* form rowToBlock array */
    for (i=0; i<np_dh; ++i) {
      for (j=beg_row[i]; j<beg_row[i]+row_count[i]; ++j) {
        rowToBlock[idx++] = i;
      }
    }
  }

  /* broadcast partitiioning information to all processors */
  hypre_MPI_Bcast(rowToBlock, m, HYPRE_MPI_INT, 0, comm_dh);

  /* allocate storage for local portion of matrix */
  mat_par_read_allocate_private(&B, m, rowLengths, rowToBlock); CHECK_V_ERROR;

  /* root sends each processor its portion of the matrix */
  if (myid_dh == 0) {
    HYPRE_Int *cval = C->cval, *rp = C->rp;
    double *aval = C->aval;
    send_req = (hypre_MPI_Request*)MALLOC_DH(2*m*sizeof(hypre_MPI_Request)); CHECK_V_ERROR;
    send_status = (hypre_MPI_Status*)MALLOC_DH(2*m*sizeof(hypre_MPI_Status)); CHECK_V_ERROR;
    for (i=0; i<m; ++i) {
      HYPRE_Int owner = rowToBlock[i];
      HYPRE_Int count = rp[i+1]-rp[i];

      /* error check for empty row */
      if (! count) {
        hypre_sprintf(msgBuf_dh, "row %i of %i is empty!", i+1, m);
        SET_V_ERROR(msgBuf_dh);
      }

      hypre_MPI_Isend(cval+rp[i], count, HYPRE_MPI_INT, owner, CVAL_TAG, comm_dh, send_req+2*i);
      hypre_MPI_Isend(aval+rp[i], count, hypre_MPI_DOUBLE, owner, AVAL_TAG, comm_dh, send_req+2*i+1);
    }
  } 

  /* all processors receive their local rows */
  { HYPRE_Int *cval = B->cval;
    HYPRE_Int *rp = B->rp;
    double *aval = B->aval;
    m = B->m;

    rcv_req = (hypre_MPI_Request*)MALLOC_DH(2*m*sizeof(hypre_MPI_Request)); CHECK_V_ERROR;
    rcv_status = (hypre_MPI_Status*)MALLOC_DH(2*m*sizeof(hypre_MPI_Status)); CHECK_V_ERROR;

    for (i=0; i<m; ++i) {

      /* error check for empty row */
      HYPRE_Int count = rp[i+1] - rp[i];
      if (! count) {
        hypre_sprintf(msgBuf_dh, "local row %i of %i is empty!", i+1, m);
        SET_V_ERROR(msgBuf_dh);
      }

      hypre_MPI_Irecv(cval+rp[i], count, HYPRE_MPI_INT, 0, CVAL_TAG, comm_dh, rcv_req+2*i);
      hypre_MPI_Irecv(aval+rp[i], count, hypre_MPI_DOUBLE, 0, AVAL_TAG, comm_dh, rcv_req+2*i+1);
    }
  }

  /* wait for all sends/receives to finish */
  if (myid_dh == 0) {
    hypre_MPI_Waitall(m*2, send_req, send_status);
  }
  hypre_MPI_Waitall(2*B->m, rcv_req, rcv_status);

  /* clean up */
  if (rowLengths != NULL) { FREE_DH(rowLengths); CHECK_V_ERROR; }
  if (o2n_row != NULL) { FREE_DH(o2n_row); CHECK_V_ERROR; }
  if (n2o_col != NULL) { FREE_DH(n2o_col); CHECK_V_ERROR; }
  if (rowToBlock != NULL) {FREE_DH(rowToBlock); CHECK_V_ERROR; }
  if (send_req != NULL) { FREE_DH(send_req); CHECK_V_ERROR; }
  if (rcv_req != NULL) { FREE_DH(rcv_req); CHECK_V_ERROR; }
  if (send_status != NULL) { FREE_DH(send_status); CHECK_V_ERROR; }
  if (rcv_status != NULL) { FREE_DH(rcv_status); CHECK_V_ERROR; }
  if (beg_row != NULL) { FREE_DH(beg_row); CHECK_V_ERROR; }
  if (row_count != NULL) { FREE_DH(row_count); CHECK_V_ERROR; }
  if (C != NULL) { Mat_dhDestroy(C); ERRCHKA; }

  *Bout = B;

  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "partition_and_distribute_private"
void partition_and_distribute_private(Mat_dh A, Mat_dh *Bout)
{
  START_FUNC_DH
  Mat_dh B = NULL;
  HYPRE_Int i, m;
  HYPRE_Int *rowLengths = NULL;
  HYPRE_Int *o2n_row = NULL, *n2o_col = NULL, *rowToBlock = NULL;
  hypre_MPI_Request *send_req = NULL;
  hypre_MPI_Request *rcv_req = NULL;
  hypre_MPI_Status  *send_status = NULL;
  hypre_MPI_Status  *rcv_status = NULL;

  hypre_MPI_Barrier(comm_dh);

  /* broadcast number of rows to all processors */
  if (myid_dh == 0)  m = A->m;
  hypre_MPI_Bcast(&m, 1, HYPRE_MPI_INT, 0, hypre_MPI_COMM_WORLD);

  /* broadcast number of nonzeros in each row to all processors */
  rowLengths = (HYPRE_Int*)MALLOC_DH(m*sizeof(HYPRE_Int)); CHECK_V_ERROR;
  if (myid_dh == 0) {
    HYPRE_Int *tmp = A->rp; 
    for (i=0; i<m; ++i) {
      rowLengths[i] = tmp[i+1] - tmp[i];
    }
  }
  hypre_MPI_Bcast(rowLengths, m, HYPRE_MPI_INT, 0, comm_dh);

  /* partition matrix */
  rowToBlock = (HYPRE_Int*)MALLOC_DH(m*sizeof(HYPRE_Int)); CHECK_V_ERROR;

  if (myid_dh == 0) {
    o2n_row = (HYPRE_Int*)MALLOC_DH(m*sizeof(HYPRE_Int)); CHECK_V_ERROR;
    mat_partition_private(A, np_dh, o2n_row, rowToBlock); CHECK_V_ERROR;
  }

  /* broadcast partitiioning information to all processors */
  hypre_MPI_Bcast(rowToBlock, m, HYPRE_MPI_INT, 0, comm_dh);

  /* allocate storage for local portion of matrix */
  mat_par_read_allocate_private(&B, m, rowLengths, rowToBlock); CHECK_V_ERROR;

  /* root sends each processor its portion of the matrix */
  if (myid_dh == 0) {
    HYPRE_Int *cval = A->cval, *rp = A->rp;
    double *aval = A->aval;
    send_req = (hypre_MPI_Request*)MALLOC_DH(2*m*sizeof(hypre_MPI_Request)); CHECK_V_ERROR;
    send_status = (hypre_MPI_Status*)MALLOC_DH(2*m*sizeof(hypre_MPI_Status)); CHECK_V_ERROR;
    for (i=0; i<m; ++i) {
      HYPRE_Int owner = rowToBlock[i];
      HYPRE_Int count = rp[i+1]-rp[i];

      /* error check for empty row */
      if (! count) {
        hypre_sprintf(msgBuf_dh, "row %i of %i is empty!", i+1, m);
        SET_V_ERROR(msgBuf_dh);
      }

      hypre_MPI_Isend(cval+rp[i], count, HYPRE_MPI_INT, owner, CVAL_TAG, comm_dh, send_req+2*i);
      hypre_MPI_Isend(aval+rp[i], count, hypre_MPI_DOUBLE, owner, AVAL_TAG, comm_dh, send_req+2*i+1);
    }
  } 

  /* all processors receive their local rows */
  { HYPRE_Int *cval = B->cval;
    HYPRE_Int *rp = B->rp;
    double *aval = B->aval;
    m = B->m;

    rcv_req = (hypre_MPI_Request*)MALLOC_DH(2*m*sizeof(hypre_MPI_Request)); CHECK_V_ERROR;
    rcv_status = (hypre_MPI_Status*)MALLOC_DH(2*m*sizeof(hypre_MPI_Status)); CHECK_V_ERROR;

    for (i=0; i<m; ++i) {

      /* error check for empty row */
      HYPRE_Int count = rp[i+1] - rp[i];
      if (! count) {
        hypre_sprintf(msgBuf_dh, "local row %i of %i is empty!", i+1, m);
        SET_V_ERROR(msgBuf_dh);
      }

      hypre_MPI_Irecv(cval+rp[i], count, HYPRE_MPI_INT, 0, CVAL_TAG, comm_dh, rcv_req+2*i);
      hypre_MPI_Irecv(aval+rp[i], count, hypre_MPI_DOUBLE, 0, AVAL_TAG, comm_dh, rcv_req+2*i+1);
    }
  }

  /* wait for all sends/receives to finish */
  if (myid_dh == 0) {
    hypre_MPI_Waitall(m*2, send_req, send_status);
  }
  hypre_MPI_Waitall(2*B->m, rcv_req, rcv_status);

  /* clean up */
  if (rowLengths != NULL) { FREE_DH(rowLengths); CHECK_V_ERROR; }
  if (o2n_row != NULL) { FREE_DH(o2n_row); CHECK_V_ERROR; }
  if (n2o_col != NULL) { FREE_DH(n2o_col); CHECK_V_ERROR; }
  if (rowToBlock != NULL) {FREE_DH(rowToBlock); CHECK_V_ERROR; }
  if (send_req != NULL) { FREE_DH(send_req); CHECK_V_ERROR; }
  if (rcv_req != NULL) { FREE_DH(rcv_req); CHECK_V_ERROR; }
  if (send_status != NULL) { FREE_DH(send_status); CHECK_V_ERROR; }
  if (rcv_status != NULL) { FREE_DH(rcv_status); CHECK_V_ERROR; }

  *Bout = B;

  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "mat_par_read_allocate_private"
void mat_par_read_allocate_private(Mat_dh *Aout, HYPRE_Int n, HYPRE_Int *rowLengths, HYPRE_Int *rowToBlock)
{
  START_FUNC_DH
  Mat_dh A;
  HYPRE_Int i, m, nz, beg_row, *rp, idx;

  Mat_dhCreate(&A); CHECK_V_ERROR;
  *Aout =  A;
  A->n = n;

  /* count number of rows owned by this processor */
  m = 0;
  for (i=0; i<n; ++i) {
    if (rowToBlock[i] == myid_dh) ++m;
  }
  A->m = m;

  /* compute global numbering of first  locally owned row */
  beg_row = 0;
  for (i=0; i<n; ++i) {
    if (rowToBlock[i] < myid_dh) ++beg_row;
  }
  A->beg_row = beg_row;

  /* allocate storage for row-pointer array */
  A->rp = rp = (HYPRE_Int*)MALLOC_DH((m+1)*sizeof(HYPRE_Int)); CHECK_V_ERROR;
  rp[0] = 0;

  /* count number of nonzeros owned by this processor, and form rp array */
  nz = 0;
  idx = 1;
  for (i=0; i<n; ++i) {
    if (rowToBlock[i] == myid_dh) {
      nz += rowLengths[i];
      rp[idx++] = nz;
    }
  }

  /* allocate storage for column indices and values arrays */
  A->cval = (HYPRE_Int*)MALLOC_DH(nz*sizeof(HYPRE_Int)); CHECK_V_ERROR;
  A->aval = (double*)MALLOC_DH(nz*sizeof(double)); CHECK_V_ERROR;
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "mat_partition_private"
void mat_partition_private(Mat_dh A, HYPRE_Int blocks, HYPRE_Int *o2n_row, HYPRE_Int *rowToBlock)
{
  START_FUNC_DH
  HYPRE_Int i, j, n = A->n;
  HYPRE_Int rpb = n/blocks;   /* rows per block (except possibly last block) */
  HYPRE_Int idx = 0;

  while (rpb*blocks < n) ++rpb;

  if (rpb*(blocks-1) == n) {
    --rpb;
    printf_dh("adjusted rpb to: %i\n", rpb);
  }

  for (i=0; i<n; ++i) o2n_row[i] = i;

  /* assign all rows to blocks, except for last block, which may
     contain less than "rpb" rows
   */
  for (i=0; i<blocks-1; ++i) {
    for (j=0; j<rpb; ++j) {
      rowToBlock[idx++] = i;
    }  
  }
 
  /* now deal with the last block in the partition */
  i = blocks - 1;
  while (idx < n) rowToBlock[idx++] = i;

  END_FUNC_DH
}


/* may produce incorrect result if input is not triangular! */
#undef __FUNC__
#define __FUNC__ "make_full_private"
void make_full_private(HYPRE_Int m, HYPRE_Int **rpIN, HYPRE_Int **cvalIN, double **avalIN)
{
  START_FUNC_DH
  HYPRE_Int i, j, *rpNew, *cvalNew, *rp = *rpIN, *cval = *cvalIN;
  double *avalNew, *aval = *avalIN;
  HYPRE_Int nz, *rowCounts = NULL;

  /* count the number of nonzeros in each row */
  rowCounts = (HYPRE_Int*)MALLOC_DH((m+1)*sizeof(HYPRE_Int)); CHECK_V_ERROR;
  for (i=0; i<=m; ++i) rowCounts[i] = 0;

  for (i=0; i<m; ++i) {
    for (j=rp[i]; j<rp[i+1]; ++j) {
      HYPRE_Int col = cval[j];
      rowCounts[i+1] += 1;
      if (col != i) rowCounts[col+1] += 1;
    }
  }

  /* prefix sum to form row pointers for full representation */
  rpNew = (HYPRE_Int*)MALLOC_DH((m+1)*sizeof(HYPRE_Int)); CHECK_V_ERROR;
  for (i=1; i<=m; ++i) rowCounts[i] += rowCounts[i-1];
  memcpy(rpNew, rowCounts, (m+1)*sizeof(HYPRE_Int));

  /* form full representation */
  nz = rpNew[m];

  cvalNew = (HYPRE_Int*)MALLOC_DH(nz*sizeof(HYPRE_Int)); CHECK_V_ERROR;
  avalNew = (double*)MALLOC_DH(nz*sizeof(double)); CHECK_V_ERROR;
  for (i=0; i<m; ++i) {
    for (j=rp[i]; j<rp[i+1]; ++j) {
      HYPRE_Int col = cval[j];
      double val  = aval[j];

      cvalNew[rowCounts[i]] = col;
      avalNew[rowCounts[i]] = val;
      rowCounts[i] += 1;
      if (col != i) {
        cvalNew[rowCounts[col]] = i;
        avalNew[rowCounts[col]] = val;
        rowCounts[col] += 1;
      }
    }
  }

  if (rowCounts != NULL) { FREE_DH(rowCounts); CHECK_V_ERROR; }
  FREE_DH(cval); CHECK_V_ERROR;
  FREE_DH(rp); CHECK_V_ERROR;
  FREE_DH(aval); CHECK_V_ERROR;
  *rpIN = rpNew;
  *cvalIN = cvalNew;
  *avalIN = avalNew;
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "make_symmetric_private"
void make_symmetric_private(HYPRE_Int m, HYPRE_Int **rpIN, HYPRE_Int **cvalIN, double **avalIN)
{
  START_FUNC_DH
  HYPRE_Int i, j, *rpNew, *cvalNew, *rp = *rpIN, *cval = *cvalIN;
  double *avalNew, *aval = *avalIN;
  HYPRE_Int nz, *rowCounts = NULL;
  HYPRE_Int *rpTrans, *cvalTrans;
  HYPRE_Int *work;
  double *avalTrans;
  HYPRE_Int nzCount = 0, transCount = 0;

  mat_dh_transpose_private(m, rp, &rpTrans,
                           cval, &cvalTrans, aval, &avalTrans); CHECK_V_ERROR;

  /* count the number of nonzeros in each row */
  work = (HYPRE_Int*)MALLOC_DH(m*sizeof(HYPRE_Int)); CHECK_V_ERROR;
  for (i=0; i<m; ++i) work[i] = -1;
  rowCounts = (HYPRE_Int*)MALLOC_DH((m+1)*sizeof(HYPRE_Int)); CHECK_V_ERROR;
  for (i=0; i<=m; ++i) rowCounts[i] = 0;

  for (i=0; i<m; ++i) {
    HYPRE_Int ct = 0;
    for (j=rp[i]; j<rp[i+1]; ++j) {
      HYPRE_Int col = cval[j];
      work[col] = i;
      ++ct;
      ++nzCount;
    }
    for (j=rpTrans[i]; j<rpTrans[i+1]; ++j) {
      HYPRE_Int col = cvalTrans[j];
      if (work[col] != i) {
        ++ct;
        ++transCount;
      }
    }
    rowCounts[i+1] = ct;
  }

  /*---------------------------------------------------------
   * if matrix is already symmetric, do nothing
   *---------------------------------------------------------*/
  if (transCount == 0) {
    hypre_printf("make_symmetric_private: matrix is already structurally symmetric!\n");
    FREE_DH(rpTrans); CHECK_V_ERROR;
    FREE_DH(cvalTrans); CHECK_V_ERROR;
    FREE_DH(avalTrans); CHECK_V_ERROR;
    FREE_DH(work); CHECK_V_ERROR;
    FREE_DH(rowCounts); CHECK_V_ERROR;
    goto END_OF_FUNCTION;
  } 

  /*---------------------------------------------------------
   * otherwise, finish symmetrizing
   *---------------------------------------------------------*/
    else {
    hypre_printf("original nz= %i\n", rp[m]);
    hypre_printf("zeros added= %i\n", transCount);
    hypre_printf("ratio of added zeros to nonzeros = %0.2f (assumes all original entries were nonzero!)\n", 
                 (double)transCount/(double)(nzCount) );
  }

  /* prefix sum to form row pointers for full representation */
  rpNew = (HYPRE_Int*)MALLOC_DH((m+1)*sizeof(HYPRE_Int)); CHECK_V_ERROR;
  for (i=1; i<=m; ++i) rowCounts[i] += rowCounts[i-1];
  memcpy(rpNew, rowCounts, (m+1)*sizeof(HYPRE_Int));
  for (i=0; i<m; ++i) work[i] = -1;

  /* form full representation */
  nz = rpNew[m];
  cvalNew = (HYPRE_Int*)MALLOC_DH(nz*sizeof(HYPRE_Int)); CHECK_V_ERROR;
  avalNew = (double*)MALLOC_DH(nz*sizeof(double)); CHECK_V_ERROR;
  for (i=0; i<m; ++i) work[i] = -1;

  for (i=0; i<m; ++i) {
    for (j=rp[i]; j<rp[i+1]; ++j) {
      HYPRE_Int col = cval[j];
      double val  = aval[j];
      work[col] = i;
      cvalNew[rowCounts[i]] = col;
      avalNew[rowCounts[i]] = val;
      rowCounts[i] += 1;
    }
    for (j=rpTrans[i]; j<rpTrans[i+1]; ++j) {
      HYPRE_Int col = cvalTrans[j];
      if (work[col] != i) {
        cvalNew[rowCounts[i]] = col;
        avalNew[rowCounts[i]] = 0.0;
        rowCounts[i] += 1;
      }
    }
  }

  if (rowCounts != NULL) { FREE_DH(rowCounts); CHECK_V_ERROR; }
  FREE_DH(work); CHECK_V_ERROR;
  FREE_DH(cval); CHECK_V_ERROR;
  FREE_DH(rp); CHECK_V_ERROR;
  FREE_DH(aval); CHECK_V_ERROR;
  FREE_DH(cvalTrans); CHECK_V_ERROR;
  FREE_DH(rpTrans); CHECK_V_ERROR;
  FREE_DH(avalTrans); CHECK_V_ERROR;
  *rpIN = rpNew;
  *cvalIN = cvalNew;
  *avalIN = avalNew;

END_OF_FUNCTION: ;

  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "profileMat"
void profileMat(Mat_dh A)
{
  START_FUNC_DH
  Mat_dh B = NULL;
  HYPRE_Int type;
  HYPRE_Int m;
  HYPRE_Int i, j;
  HYPRE_Int *work1;
  double *work2;
  bool isStructurallySymmetric = true;
  bool isNumericallySymmetric = true;
  bool is_Triangular = false;
  HYPRE_Int zeroCount = 0, nz;

  if (myid_dh > 0) {
    SET_V_ERROR("only for a single MPI task!");
  }

  m = A->m;

  hypre_printf("\nYY----------------------------------------------------\n");

  /* count number of explicit zeros */
  nz = A->rp[m];
  for (i=0; i<nz; ++i) {
    if (A->aval[i] == 0) ++zeroCount;
  }
  hypre_printf("YY  row count:      %i\n", m);
  hypre_printf("YY  nz count:       %i\n", nz);
  hypre_printf("YY  explicit zeros: %i (entire matrix)\n", zeroCount);

  /* count number of missing or zero diagonals */
  { HYPRE_Int m_diag = 0, z_diag = 0;
    for (i=0; i<m; ++i) {
      bool flag = true;
      for (j=A->rp[i]; j<A->rp[i+1]; ++j) {
        HYPRE_Int col = A->cval[j];

        /* row has an explicit diagonal element */
        if (col == i) {          
          double val = A->aval[j];
          flag = false;
          if (val == 0.0) ++z_diag;
          break;
        }
      }

      /* row has an implicit zero diagonal element */
      if (flag) ++m_diag;
    }
    hypre_printf("YY  missing diagonals:   %i\n", m_diag);
    hypre_printf("YY  explicit zero diags: %i\n", z_diag);
  }

  /* check to see if matrix is triangular */
  type = isTriangular(m, A->rp, A->cval); CHECK_V_ERROR;
  if (type == IS_UPPER_TRI) {
    hypre_printf("YY  matrix is upper triangular\n");
    is_Triangular = true;
    goto END_OF_FUNCTION;
  } else if (type == IS_LOWER_TRI) {
    hypre_printf("YY  matrix is lower triangular\n");
    is_Triangular = true;
    goto END_OF_FUNCTION;
  }

  /* if not triangular, count nz in each triangle */
  { HYPRE_Int unz = 0, lnz = 0;
    for (i=0; i<m; ++i) {
      for (j=A->rp[i]; j<A->rp[i+1]; ++j) {
        HYPRE_Int col = A->cval[j];
        if (col < i) ++lnz;
        if (col > i) ++unz;
      }
    }
    hypre_printf("YY  strict upper triangular nonzeros: %i\n", unz);
    hypre_printf("YY  strict lower triangular nonzeros: %i\n", lnz);
  }
 
   
  

  Mat_dhTranspose(A, &B); CHECK_V_ERROR;

  /* check for structural and numerical symmetry */

  work1 = (HYPRE_Int*)MALLOC_DH(m*sizeof(HYPRE_Int)); CHECK_V_ERROR;
  work2 = (double*)MALLOC_DH(m*sizeof(double)); CHECK_V_ERROR;
  for (i=0; i<m; ++i) work1[i] = -1;
  for (i=0; i<m; ++i) work2[i] = 0.0;

  for (i=0; i<m; ++i) {
    for (j=A->rp[i]; j<A->rp[i+1]; ++j) {
      HYPRE_Int col = A->cval[j];
      double val = A->aval[j];
      work1[col] = i;
      work2[col] = val;
    }
    for (j=B->rp[i]; j<B->rp[i+1]; ++j) {
      HYPRE_Int col = B->cval[j];
      double val = B->aval[j];

      if (work1[col] != i) {
        isStructurallySymmetric = false;
        isNumericallySymmetric = false;
        goto END_OF_FUNCTION;
      }
      if (work2[col] != val) {
        isNumericallySymmetric = false;
        work2[col] = 0.0;
      }
    }
  }


END_OF_FUNCTION: ;

  if (! is_Triangular) {
    hypre_printf("YY  matrix is NOT triangular\n");
    if (isStructurallySymmetric) {
      hypre_printf("YY  matrix IS structurally symmetric\n");
    } else {
      hypre_printf("YY  matrix is NOT structurally symmetric\n");
    }
    if (isNumericallySymmetric) {
      hypre_printf("YY  matrix IS numerically symmetric\n");
    } else {
      hypre_printf("YY  matrix is NOT numerically symmetric\n");
    }
  }

  if (work1 != NULL) { FREE_DH(work1); CHECK_V_ERROR; }
  if (work2 != NULL) { FREE_DH(work2); CHECK_V_ERROR; }
  if (B != NULL) { Mat_dhDestroy(B); CHECK_V_ERROR; }

  hypre_printf("YY----------------------------------------------------\n");

  END_FUNC_DH
}
