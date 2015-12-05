/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.11 $
 ***********************************************************************EHEADER*/

#include "io_dh.h"
#include "Mat_dh.h"
#include "Vec_dh.h"
#include "Mem_dh.h"
#include "Timer_dh.h"
#include "Parser_dh.h"
/* #include "euclid_petsc.h" */
#include "mat_dh_private.h"

#undef __FUNC__
#define __FUNC__ "openFile_dh"
FILE * openFile_dh(const char *filenameIN, const char *modeIN)
{
  START_FUNC_DH
  FILE *fp = NULL;

  if ((fp = fopen(filenameIN, modeIN)) == NULL) {
    hypre_sprintf(msgBuf_dh, "can't open file: %s for mode %s\n", filenameIN, modeIN);
    SET_ERROR(NULL, msgBuf_dh);
  }
  END_FUNC_VAL(fp)
}

#undef __FUNC__
#define __FUNC__ "closeFile_dh"
void closeFile_dh(FILE *fpIN)
{
  if (fclose(fpIN)) {
    SET_V_ERROR("attempt to close file failed");
  }
}

/*----------------------------------------------------------------*/
void io_dh_print_ebin_mat_private(HYPRE_Int m, HYPRE_Int beg_row,
                                HYPRE_Int *rp, HYPRE_Int *cval, double *aval, 
                           HYPRE_Int *n2o, HYPRE_Int *o2n, Hash_i_dh hash, char *filename)
{}

extern void io_dh_read_ebin_mat_private(HYPRE_Int *m, HYPRE_Int **rp, HYPRE_Int **cval,
                                     double **aval, char *filename)
{}

void io_dh_print_ebin_vec_private(HYPRE_Int n, HYPRE_Int beg_row, double *vals,
                           HYPRE_Int *n2o, HYPRE_Int *o2n, Hash_i_dh hash, char *filename)
{}

void io_dh_read_ebin_vec_private(HYPRE_Int *n, double **vals, char *filename)
{}
