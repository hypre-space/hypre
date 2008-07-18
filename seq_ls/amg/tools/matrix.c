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

