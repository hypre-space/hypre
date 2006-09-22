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
 * $Revision$
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 * Constructors and destructors for matrix structure.
 *
 *****************************************************************************/

#include "general.h"
#include "matrix.h"


/*--------------------------------------------------------------------------
 * hypre_NewMatrix
 *--------------------------------------------------------------------------*/

hypre_Matrix  *hypre_NewMatrix(data, ia, ja, size)
double  *data;
int     *ia;
int     *ja;
int      size;
{
   hypre_Matrix     *new;


   new = hypre_TAlloc(hypre_Matrix, 1);

   hypre_MatrixData(new) = data;
   hypre_MatrixIA(new)   = ia;
   hypre_MatrixJA(new)   = ja;
   hypre_MatrixSize(new) = size;

   return new;
}

/*--------------------------------------------------------------------------
 * hypre_FreeMatrix
 *--------------------------------------------------------------------------*/

void     hypre_FreeMatrix(matrix)
hypre_Matrix  *matrix;
{
   if (matrix)
   {
      hypre_TFree(hypre_MatrixData(matrix));
      hypre_TFree(hypre_MatrixIA(matrix));
      hypre_TFree(hypre_MatrixJA(matrix));
      hypre_TFree(matrix);
   }
}

