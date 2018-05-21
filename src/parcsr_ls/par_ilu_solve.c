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
 * ILU solve routine
 *
 *****************************************************************************/
#include "_hypre_parcsr_ls.h"
#include "par_ilu.h"

/*--------------------------------------------------------------------
 * hypre_ILUSolve
 *--------------------------------------------------------------------*/
HYPRE_Int
hypre_ILUSolve( void               *ilu_vdata,
                  hypre_ParCSRMatrix *A,
                  hypre_ParVector    *f,
                  hypre_ParVector    *u )
{

   return hypre_error_flag;
}
