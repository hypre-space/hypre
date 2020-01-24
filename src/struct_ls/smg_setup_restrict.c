/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

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
