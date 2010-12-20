/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
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
   HYPRE_Int     *ia;
   HYPRE_Int     *ja;
   HYPRE_Int      size;
   
   HYPRE_Int      j;


   /*----------------------------------------------------------
    * Write the matrix data
    *----------------------------------------------------------*/

   data = hypre_MatrixData(matrix);
   ia   = hypre_MatrixIA(matrix);
   ja   = hypre_MatrixJA(matrix);
   size = hypre_MatrixSize(matrix);

   fp = fopen(file_name, "w");

   hypre_fprintf(fp, "%d\n", size);

   for (j = 0; j < size+1; j++)
      hypre_fprintf(fp, "%d\n", ia[j]);

   for (j = 0; j < ia[size]-1; j++)
      hypre_fprintf(fp, "%d\n", ja[j]);

   for (j = 0; j < ia[size]-1; j++)
      hypre_fprintf(fp, "%le\n", data[j]);

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
   HYPRE_Int      size;
   
   HYPRE_Int      j;


   /*----------------------------------------------------------
    * Write in the data
    *----------------------------------------------------------*/

   data = hypre_VectorData(vector);
   size = hypre_VectorSize(vector);

   fp = fopen(file_name, "w");

   hypre_fprintf(fp, "%d\n", size);

   for (j = 0; j < size; j++)
      hypre_fprintf(fp, "%le\n", data[j]);

   fclose(fp);

   return;
}

