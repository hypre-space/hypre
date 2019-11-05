/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_Euclid.h"
/* #include "io_dh.h" */
/* #include "Mat_dh.h" */
/* #include "Vec_dh.h" */
/* #include "Mem_dh.h" */
/* #include "Timer_dh.h" */
/* #include "Parser_dh.h" */
/* #include "euclid_petsc.h" */
/* #include "mat_dh_private.h" */

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
                                HYPRE_Int *rp, HYPRE_Int *cval, HYPRE_Real *aval, 
                           HYPRE_Int *n2o, HYPRE_Int *o2n, Hash_i_dh hash, char *filename)
{}

extern void io_dh_read_ebin_mat_private(HYPRE_Int *m, HYPRE_Int **rp, HYPRE_Int **cval,
                                     HYPRE_Real **aval, char *filename)
{}

void io_dh_print_ebin_vec_private(HYPRE_Int n, HYPRE_Int beg_row, HYPRE_Real *vals,
                           HYPRE_Int *n2o, HYPRE_Int *o2n, Hash_i_dh hash, char *filename)
{}

void io_dh_read_ebin_vec_private(HYPRE_Int *n, HYPRE_Real **vals, char *filename)
{}
