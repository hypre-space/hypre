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
 * Header info for the hypre_IJMatrix structures
 *
 *****************************************************************************/

#ifndef hypre_IJ_MATRIX_HEADER
#define hypre_IJ_MATRIX_HEADER

/*--------------------------------------------------------------------------
 * hypre_IJMatrix:
 *--------------------------------------------------------------------------*/

typedef struct hypre_IJMatrix_struct
{
   MPI_Comm    comm;

   int        *row_partitioning;    /* distribution of rows across processors */
   int        *col_partitioning;    /* distribution of columns */

   int         object_type;         /* Indicates the type of "object" */
   void       *object;              /* Structure for storing local portion */
   void       *translator;          /* optional storage_type specfic structure
                                       for holding additional local info */
   int         assemble_flag;       /* indicates whether matrix has been 
				       assembled */

   int         global_first_row;    /* these for data items are necessary */
   int         global_first_col;    /*   to be able to avoind using the global */
   int         global_num_rows;     /*   global partition */ 
   int         global_num_cols;
   


} hypre_IJMatrix;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_IJMatrix
 *--------------------------------------------------------------------------*/

#define hypre_IJMatrixComm(matrix)              ((matrix) -> comm)

#define hypre_IJMatrixRowPartitioning(matrix)   ((matrix) -> row_partitioning)
#define hypre_IJMatrixColPartitioning(matrix)   ((matrix) -> col_partitioning)

#define hypre_IJMatrixObjectType(matrix)        ((matrix) -> object_type)
#define hypre_IJMatrixObject(matrix)            ((matrix) -> object)
#define hypre_IJMatrixTranslator(matrix)        ((matrix) -> translator)

#define hypre_IJMatrixAssembleFlag(matrix)      ((matrix) -> assemble_flag)


#define hypre_IJMatrixGlobalFirstRow(matrix)      ((matrix) -> global_first_row)
#define hypre_IJMatrixGlobalFirstCol(matrix)      ((matrix) -> global_first_col)
#define hypre_IJMatrixGlobalNumRows(matrix)       ((matrix) -> global_num_rows)
#define hypre_IJMatrixGlobalNumCols(matrix)       ((matrix) -> global_num_cols)

/*--------------------------------------------------------------------------
 * prototypes for operations on local objects
 *--------------------------------------------------------------------------*/

#ifdef PETSC_AVAILABLE
/* IJMatrix_petsc.c */
int
hypre_GetIJMatrixParCSRMatrix( HYPRE_IJMatrix IJmatrix, Mat *reference )
#endif
  
#ifdef ISIS_AVAILABLE
/* IJMatrix_isis.c */
int
hypre_GetIJMatrixISISMatrix( HYPRE_IJMatrix IJmatrix, RowMatrix *reference )
#endif

#endif
