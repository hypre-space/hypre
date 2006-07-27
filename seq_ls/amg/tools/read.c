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
 * Read routines
 *
 *****************************************************************************/

#include "io.h"


/*--------------------------------------------------------------------------
 * ReadYSMP
 *--------------------------------------------------------------------------*/

hypre_Matrix   *ReadYSMP(file_name)
char     *file_name;
{
   hypre_Matrix  *matrix;

   FILE    *fp;

   double  *data;
   int     *ia;
   int     *ja;
   int      size;
   
   int      j;


   /*----------------------------------------------------------
    * Read in the data
    *----------------------------------------------------------*/

   fp = fopen(file_name, "r");

   fscanf(fp, "%d", &size);

   ia = hypre_TAlloc(int, hypre_NDIMU(size+1));
   for (j = 0; j < size+1; j++)
      fscanf(fp, "%d", &ia[j]);

   ja = hypre_TAlloc(int, hypre_NDIMA(ia[size]-1));
   for (j = 0; j < ia[size]-1; j++)
      fscanf(fp, "%d", &ja[j]);

   data = hypre_TAlloc(double, hypre_NDIMA(ia[size]-1));
   for (j = 0; j < ia[size]-1; j++)
      fscanf(fp, "%le", &data[j]);

   fclose(fp);

   /*----------------------------------------------------------
    * Create the matrix structure
    *----------------------------------------------------------*/

   matrix = hypre_NewMatrix(data, ia, ja, size);

   return matrix;
}

/*--------------------------------------------------------------------------
 * ReadVec
 *--------------------------------------------------------------------------*/

hypre_Vector   *ReadVec(file_name)
char     *file_name;
{
   hypre_Vector  *vector;

   FILE    *fp;

   double  *data;
   int      size;
   
   int      j;


   /*----------------------------------------------------------
    * Read in the data
    *----------------------------------------------------------*/

   fp = fopen(file_name, "r");

   fscanf(fp, "%d", &size);

   data = hypre_TAlloc(double, hypre_NDIMU(size));
   for (j = 0; j < size; j++)
      fscanf(fp, "%le", &data[j]);

   fclose(fp);

   /*----------------------------------------------------------
    * Create the vector structure
    *----------------------------------------------------------*/

   vector = hypre_NewVector(data, size);

   return vector;
}

