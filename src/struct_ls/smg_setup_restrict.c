/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
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
   HYPRE_UNUSED_VAR(A);
   HYPRE_UNUSED_VAR(cgrid);
   HYPRE_UNUSED_VAR(cdir);

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
   HYPRE_UNUSED_VAR(A);
   HYPRE_UNUSED_VAR(R);
   HYPRE_UNUSED_VAR(temp_vec);
   HYPRE_UNUSED_VAR(cdir);
   HYPRE_UNUSED_VAR(cindex);
   HYPRE_UNUSED_VAR(cstride);

   return hypre_error_flag;
}
