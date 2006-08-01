/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team <hypre-users@llnl.gov>, UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer and the GNU Lesser General Public License.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/

/*
   Note: this module contains functionality for reading/writing 
         Euclid's binary io format, and opening and closing files.
         Additional io can be found in in mat_dh_private, which contains
         private functions for reading/writing various matrix and
         vector formats; functions in that module are called by
         public class methods of the Mat_dh and Vec_dh classes.
*/

#ifndef IO_DH
#define IO_DH

#include "euclid_common.h"

/*--------------------------------------------------------------------------
 * open and close files, with error checking
 *--------------------------------------------------------------------------*/
extern FILE * openFile_dh(const char *filenameIN, const char *modeIN);
extern void closeFile_dh(FILE *fpIN);

/*---------------------------------------------------------------------------
 * binary io; these are called by functions in mat_dh_private
 *---------------------------------------------------------------------------*/

bool isSmallEndian();

/* seq only ?? */
extern void io_dh_print_ebin_mat_private(int m, int beg_row, 
                                int *rp, int *cval, double *aval, 
                           int *n2o, int *o2n, Hash_i_dh hash, char *filename);

/* seq only ?? */
extern void io_dh_read_ebin_mat_private(int *m, int **rp, int **cval, 
                                     double **aval, char *filename);

/* seq only */
extern void io_dh_print_ebin_vec_private(int n, int beg_row, double *vals, 
                           int *n2o, int *o2n, Hash_i_dh hash, char *filename);
/* seq only */
extern void io_dh_read_ebin_vec_private(int *n, double **vals, char *filename);


#endif

