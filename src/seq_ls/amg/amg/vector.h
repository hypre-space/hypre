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
 * Header info for hypre_Vector data structures
 *
 *****************************************************************************/

#ifndef HYPRE_VECTOR_HEADER
#define HYPRE_VECTOR_HEADER


/*--------------------------------------------------------------------------
 * hypre_Vector
 *--------------------------------------------------------------------------*/

typedef struct
{
   HYPRE_Real  *data;
   HYPRE_Int      size;

} hypre_Vector;

/*--------------------------------------------------------------------------
 * Accessor functions for the hypre_Vector structure
 *--------------------------------------------------------------------------*/

#define hypre_VectorData(vector)      ((vector) -> data)
#define hypre_VectorSize(vector)      ((vector) -> size)


typedef struct
{
   HYPRE_Int     *data;
   HYPRE_Int      size;

} hypre_VectorInt;

/*--------------------------------------------------------------------------
 * Accessor functions for the hypre_VectorInt structure
 *--------------------------------------------------------------------------*/

#define hypre_VectorIntData(vector)      ((vector) -> data)
#define hypre_VectorIntSize(vector)      ((vector) -> size)

#endif
