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
 * Header info for Auxiliary Parallel Vector data structures
 *
 * Note: this vector currently uses 0-based indexing.
 *
 *****************************************************************************/

#ifndef hypre_AUX_PAR_VECTOR_HEADER
#define hypre_AUX_PAR_VECTOR_HEADER

/*--------------------------------------------------------------------------
 * Auxiliary Parallel Vector
 *--------------------------------------------------------------------------*/

typedef struct
{
   int	    max_off_proc_elmts; /* length of off processor stash for
					SetValues and AddToValues*/
   int	    current_num_elmts; /* current no. of elements stored in stash */
   int     *off_proc_i; /* contains column indices */
   double  *off_proc_data; /* contains corresponding data */
} hypre_AuxParVector;

/*--------------------------------------------------------------------------
 * Accessor functions for the Parallel Vector structure
 *--------------------------------------------------------------------------*/

#define hypre_AuxParVectorMaxOffProcElmts(matrix)  ((matrix) -> max_off_proc_elmts)
#define hypre_AuxParVectorCurrentNumElmts(matrix)  ((matrix) -> current_num_elmts)
#define hypre_AuxParVectorOffProcI(matrix)  ((matrix) -> off_proc_i)
#define hypre_AuxParVectorOffProcData(matrix)  ((matrix) -> off_proc_data)

#endif
