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
   double  *data;
   int      size;

} hypre_Vector;

/*--------------------------------------------------------------------------
 * Accessor functions for the hypre_Vector structure
 *--------------------------------------------------------------------------*/

#define hypre_VectorData(vector)      ((vector) -> data)
#define hypre_VectorSize(vector)      ((vector) -> size)

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

#ifdef __STDC__
# define        P(s) s
#else
# define P(s) ()
#endif


/* vector.c */
hypre_Vector *hypre_NewVector P((double *data , int size ));
void hypre_FreeVector P((hypre_Vector *vector ));

#undef P


#endif
