/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.5 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 * Member functions for hypre_SStructStencil class.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_SStructStencilRef
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructStencilRef( hypre_SStructStencil  *stencil,
                         hypre_SStructStencil **stencil_ref )
{
   hypre_SStructStencilRefCount(stencil) ++;
   *stencil_ref = stencil;

   return 0;
}

