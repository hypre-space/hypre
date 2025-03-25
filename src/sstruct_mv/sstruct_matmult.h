/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for the semi-structured matrix/matrix multiplication structures
 *
 *****************************************************************************/

#ifndef hypre_SSTRUCT_MATMULT_HEADER
#define hypre_SSTRUCT_MATMULT_HEADER

/*--------------------------------------------------------------------------
 * SStructPMatmult data structure
 *--------------------------------------------------------------------------*/

typedef struct hypre_SStructPMatmultData_struct
{
   hypre_StructMatmultData    *smmdata;      /* struct matmult data object */
   HYPRE_Int                ***smmid;        /* (nvars x nvars) array of matmult id-arrays */
   HYPRE_Int                 **smmsz;        /* (nvars x nvars) array of matmult id-array sizes */
   HYPRE_Int                   nvars;

   HYPRE_Int                   nmatrices;
   hypre_SStructPMatrix      **pmatrices;  /* matrices we are multiplying */
   HYPRE_Int                   nterms;
   HYPRE_Int                  *terms;
   HYPRE_Int                  *transposes;

   hypre_CommPkg              *comm_pkg;        /* agglomerated communication package */
   HYPRE_Complex             **comm_data;       /* agglomerated communication data */

} hypre_SStructPMatmultData;

/*--------------------------------------------------------------------------
 * SStructMatmult data structure
 *--------------------------------------------------------------------------*/

typedef struct hypre_SStructMatmultData_struct
{
   HYPRE_Int                    nparts;
   hypre_SStructPMatmultData  **pmmdata;   /* pointer to nparts array */

   HYPRE_Int                    nmatrices;
   hypre_SStructMatrix        **matrices;  /* matrices we are multiplying */
   HYPRE_Int                    nterms;
   HYPRE_Int                   *terms;
   HYPRE_Int                   *transposes;

   hypre_CommPkg               *comm_pkg;        /* agglomerated communication package */
   HYPRE_Complex              **comm_data;       /* agglomerated communication data */

} hypre_SStructMatmultData;

#endif
