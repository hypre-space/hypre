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
   hypre_StructMatmultData ****smmdata;    /* (nvars x nvars) array of smmdata arrays */
   HYPRE_Int                 **smmdasz;    /* (nvars x nvars) array of smmdata-array sizes */
   HYPRE_Int                   nvars;

   HYPRE_Int                   nmatrices;
   hypre_SStructPMatrix      **pmatrices;  /* matrices we are multiplying */
   HYPRE_Int                   nterms;
   HYPRE_Int                  *terms;
   HYPRE_Int                  *transposes;

   hypre_CommPkg              *comm_pkg;        /* agglomerated communication package */
   HYPRE_Complex             **comm_data;       /* agglomerated communication data */
   hypre_CommPkg             **comm_pkg_a;      /* array of communication packages */
   HYPRE_Complex            ***comm_data_a;     /* array of communication data */
   HYPRE_Int                   num_comm_pkgs;   /* array size of comm_pkg_a and comm_data_a */
   HYPRE_Int                   num_comm_blocks; /* total number of communication blocks */

} hypre_SStructPMatmultData;

/*--------------------------------------------------------------------------
 * SStructMatmult data structure
 *--------------------------------------------------------------------------*/

typedef struct hypre_SStructMatmultData_struct
{
   hypre_SStructPMatmultData  **pmmdata;   /* pointer to nparts array */
   HYPRE_Int                    nparts;

   HYPRE_Int                    nmatrices;
   hypre_SStructMatrix        **matrices;  /* matrices we are multiplying */
   HYPRE_Int                    nterms;
   HYPRE_Int                   *terms;
   HYPRE_Int                   *transposes;

   hypre_CommPkg               *comm_pkg;        /* agglomerated communication package */
   HYPRE_Complex              **comm_data;       /* agglomerated communication data */
   hypre_CommPkg              **comm_pkg_a;      /* array of communication packages */
   HYPRE_Complex             ***comm_data_a;     /* array of communication data */
   HYPRE_Int                    num_comm_pkgs;   /* array size of comm_pkg_a and comm_data_a */
   HYPRE_Int                    num_comm_blocks; /* total number of communication blocks */

} hypre_SStructMatmultData;

#endif
