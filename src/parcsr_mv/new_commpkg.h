/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/
#ifndef hypre_NEW_COMMPKG
#define hypre_NEW_COMMPKG

typedef struct
{
   HYPRE_Int       length;
   HYPRE_Int       storage_length;
   HYPRE_Int      *id;
   HYPRE_Int      *vec_starts;
   HYPRE_Int       element_storage_length;
   HYPRE_BigInt   *elements;
   HYPRE_Real     *d_elements; /* Is this used anywhere? */
   void           *v_elements;
}  hypre_ProcListElements;

#endif /* hypre_NEW_COMMPKG */

