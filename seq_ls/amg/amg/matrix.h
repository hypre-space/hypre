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


/******************************************************************************
 *
 * Header info for hypre_Matrix data structures
 *
 *****************************************************************************/

#ifndef HYPRE_MATRIX_HEADER
#define HYPRE_MATRIX_HEADER


/*--------------------------------------------------------------------------
 * hypre_Matrix
 *--------------------------------------------------------------------------*/

typedef struct
{
   double  *data;
   int     *ia;
   int     *ja;
   int      size;

} hypre_Matrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the hypre_Matrix structure
 *--------------------------------------------------------------------------*/

#define hypre_MatrixData(matrix)      ((matrix) -> data)
#define hypre_MatrixIA(matrix)        ((matrix) -> ia)
#define hypre_MatrixJA(matrix)        ((matrix) -> ja)
#define hypre_MatrixSize(matrix)      ((matrix) -> size)


#endif
