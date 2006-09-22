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
 * Write routines
 *
 *****************************************************************************/

#include "io.h"


/*--------------------------------------------------------------------------
 * hypre_WriteYSMP
 *--------------------------------------------------------------------------*/

void     hypre_WriteYSMP(file_name, matrix)
char    *file_name;
hypre_Matrix  *matrix;
{
   FILE    *fp;

   double  *data;
   int     *ia;
   int     *ja;
   int      size;
   
   int      j;


   /*----------------------------------------------------------
    * Write the matrix data
    *----------------------------------------------------------*/

   data = hypre_MatrixData(matrix);
   ia   = hypre_MatrixIA(matrix);
   ja   = hypre_MatrixJA(matrix);
   size = hypre_MatrixSize(matrix);

   fp = fopen(file_name, "w");

   fprintf(fp, "%d\n", size);

   for (j = 0; j < size+1; j++)
      fprintf(fp, "%d\n", ia[j]);

   for (j = 0; j < ia[size]-1; j++)
      fprintf(fp, "%d\n", ja[j]);

   for (j = 0; j < ia[size]-1; j++)
      fprintf(fp, "%le\n", data[j]);

   fclose(fp);

   return;
}

/*--------------------------------------------------------------------------
 * hypre_WriteVec
 *--------------------------------------------------------------------------*/

void     hypre_WriteVec(file_name, vector)
char    *file_name;
hypre_Vector  *vector;
{
   FILE    *fp;

   double  *data;
   int      size;
   
   int      j;


   /*----------------------------------------------------------
    * Write in the data
    *----------------------------------------------------------*/

   data = hypre_VectorData(vector);
   size = hypre_VectorSize(vector);

   fp = fopen(file_name, "w");

   fprintf(fp, "%d\n", size);

   for (j = 0; j < size; j++)
      fprintf(fp, "%le\n", data[j]);

   fclose(fp);

   return;
}

