/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision: 2.8 $
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
    sprintf(msgBuf_dh, "can't open file: %s for mode %s\n", filenameIN, modeIN);
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
void io_dh_print_ebin_mat_private(int m, int beg_row, 
                                int *rp, int *cval, double *aval, 
                           int *n2o, int *o2n, Hash_i_dh hash, char *filename)
{}

extern void io_dh_read_ebin_mat_private(int *m, int **rp, int **cval, 
                                     double **aval, char *filename)
{}

void io_dh_print_ebin_vec_private(int n, int beg_row, double *vals, 
                           int *n2o, int *o2n, Hash_i_dh hash, char *filename)
{}

void io_dh_read_ebin_vec_private(int *n, double **vals, char *filename)
{}
