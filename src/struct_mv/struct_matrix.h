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
 * $Revision: 2.7 $
 ***********************************************************************EHEADER*/


/******************************************************************************
 *
 * Header info for the hypre_StructMatrix structures
 *
 *****************************************************************************/

#ifndef hypre_STRUCT_MATRIX_HEADER
#define hypre_STRUCT_MATRIX_HEADER

#include <assert.h>

/*--------------------------------------------------------------------------
 * hypre_StructMatrix:
 *--------------------------------------------------------------------------*/

typedef struct hypre_StructMatrix_struct
{
   MPI_Comm              comm;

   hypre_StructGrid     *grid;
   hypre_StructStencil  *user_stencil;
   hypre_StructStencil  *stencil;
   int                   num_values;   /* Number of "stored" coefficients */

   hypre_BoxArray       *data_space;

   double               *data;         /* Pointer to matrix data */
   int                   data_alloced; /* Boolean used for freeing data */
   int                   data_size;    /* Size of matrix data */
   int                 **data_indices; /* num-boxes by stencil-size array
                                          of indices into the data array.
                                          data_indices[b][s] is the starting
                                          index of matrix data corresponding
                                          to box b and stencil coefficient s */
   int                   constant_coefficient;  /* normally 0; set to 1 for
                                                   constant coefficient matrices
                                                   or 2 for constant coefficient
                                                   with variable diagonal */
                      
   int                   symmetric;    /* Is the matrix symmetric */
   int                  *symm_elements;/* Which elements are "symmetric" */
   int                   num_ghost[6]; /* Num ghost layers in each direction */
                      
   int                   global_size;  /* Total number of nonzero coeffs */

   int                   OffProcAdd;   /* offproc set values flag */

   int                   add_num_ghost[6]; /* ghostlayers to scan for offproc
                                              add values */

   hypre_CommPkg        *comm_pkg;     /* Info on how to update ghost data */

   int                   ref_count;

} hypre_StructMatrix;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_StructMatrix
 *--------------------------------------------------------------------------*/

#define hypre_StructMatrixComm(matrix)          ((matrix) -> comm)
#define hypre_StructMatrixGrid(matrix)          ((matrix) -> grid)
#define hypre_StructMatrixUserStencil(matrix)   ((matrix) -> user_stencil)
#define hypre_StructMatrixStencil(matrix)       ((matrix) -> stencil)
#define hypre_StructMatrixNumValues(matrix)     ((matrix) -> num_values)
#define hypre_StructMatrixDataSpace(matrix)     ((matrix) -> data_space)
#define hypre_StructMatrixData(matrix)          ((matrix) -> data)
#define hypre_StructMatrixDataAlloced(matrix)   ((matrix) -> data_alloced)
#define hypre_StructMatrixDataSize(matrix)      ((matrix) -> data_size)
#define hypre_StructMatrixDataIndices(matrix)   ((matrix) -> data_indices)
#define hypre_StructMatrixConstantCoefficient(matrix) ((matrix) -> constant_coefficient)
#define hypre_StructMatrixSymmetric(matrix)     ((matrix) -> symmetric)
#define hypre_StructMatrixSymmElements(matrix)  ((matrix) -> symm_elements)
#define hypre_StructMatrixNumGhost(matrix)      ((matrix) -> num_ghost)
#define hypre_StructMatrixGlobalSize(matrix)    ((matrix) -> global_size)
#define hypre_StructMatrixOffProcAdd(matrix)    ((matrix) -> OffProcAdd)
#define hypre_StructMatrixAddNumGhost(matrix)   ((matrix) -> add_num_ghost)
#define hypre_StructMatrixCommPkg(matrix)       ((matrix) -> comm_pkg)
#define hypre_StructMatrixRefCount(matrix)      ((matrix) -> ref_count)

#define hypre_StructMatrixBox(matrix, b) \
hypre_BoxArrayBox(hypre_StructMatrixDataSpace(matrix), b)

#define hypre_StructMatrixBoxData(matrix, b, s) \
(hypre_StructMatrixData(matrix) + hypre_StructMatrixDataIndices(matrix)[b][s])

#define hypre_StructMatrixBoxDataValue(matrix, b, s, index) \
(hypre_StructMatrixBoxData(matrix, b, s) + \
 hypre_BoxIndexRank(hypre_StructMatrixBox(matrix, b), index))

#define hypre_CCStructMatrixBoxDataValue(matrix, b, s, index) \
(hypre_StructMatrixBoxData(matrix, b, s) + \
 hypre_CCBoxIndexRank(hypre_StructMatrixBox(matrix, b), index))

#endif
