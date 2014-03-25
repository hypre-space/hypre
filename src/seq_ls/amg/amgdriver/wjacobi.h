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
 * Header info for WJacobi solver
 *
 *****************************************************************************/

#ifndef _WJACOBI_HEADER
#define _WJACOBI_HEADER


/*--------------------------------------------------------------------------
 * WJacobiData
 *--------------------------------------------------------------------------*/

typedef struct
{
   HYPRE_Real   weight;
   HYPRE_Int      max_iter;

   hypre_Matrix  *A;
   hypre_Vector  *t;

   char    *log_file_name;

} WJacobiData;

/*--------------------------------------------------------------------------
 * Accessor functions for the WJacobiData structure
 *--------------------------------------------------------------------------*/

#define WJacobiDataWeight(wjacobi_data)      ((wjacobi_data) -> weight)
#define WJacobiDataMaxIter(wjacobi_data)     ((wjacobi_data) -> max_iter)

#define WJacobiDataA(wjacobi_data)           ((wjacobi_data) -> A)
#define WJacobiDataT(wjacobi_data)           ((wjacobi_data) -> t)

#define WJacobiDataLogFileName(wjacobi_data) ((wjacobi_data) -> log_file_name)


#endif
