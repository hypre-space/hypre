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

#include "_hypre_struct_ls.h"
#include "smg.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

hypre_StructMatrix *
hypre_SMGCreateRestrictOp( hypre_StructMatrix *A,
                           hypre_StructGrid   *cgrid,
                           HYPRE_Int           cdir  )
{
   hypre_StructMatrix *R = NULL;

   return R;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SMGSetupRestrictOp( hypre_StructMatrix *A,
                          hypre_StructMatrix *R,
                          hypre_StructVector *temp_vec,
                          HYPRE_Int           cdir,
                          hypre_Index         cindex,
                          hypre_Index         cstride  )
{
   return hypre_error_flag;
}
