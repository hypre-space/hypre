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
extern void io_dh_print_ebin_mat_private(HYPRE_Int m, HYPRE_Int beg_row,
                                HYPRE_Int *rp, HYPRE_Int *cval, double *aval, 
                           HYPRE_Int *n2o, HYPRE_Int *o2n, Hash_i_dh hash, char *filename);

/* seq only ?? */
extern void io_dh_read_ebin_mat_private(HYPRE_Int *m, HYPRE_Int **rp, HYPRE_Int **cval,
                                     double **aval, char *filename);

/* seq only */
extern void io_dh_print_ebin_vec_private(HYPRE_Int n, HYPRE_Int beg_row, double *vals,
                           HYPRE_Int *n2o, HYPRE_Int *o2n, Hash_i_dh hash, char *filename);
/* seq only */
extern void io_dh_read_ebin_vec_private(HYPRE_Int *n, double **vals, char *filename);


#endif

