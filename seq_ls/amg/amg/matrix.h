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
   HYPRE_Real  *data;
   HYPRE_Int     *ia;
   HYPRE_Int     *ja;
   HYPRE_Int      size;

} hypre_Matrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the hypre_Matrix structure
 *--------------------------------------------------------------------------*/

#define hypre_MatrixData(matrix)      ((matrix) -> data)
#define hypre_MatrixIA(matrix)        ((matrix) -> ia)
#define hypre_MatrixJA(matrix)        ((matrix) -> ja)
#define hypre_MatrixSize(matrix)      ((matrix) -> size)


#endif
