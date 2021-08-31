/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
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
 * SStructPMatrixMult data structure
 *--------------------------------------------------------------------------*/

typedef struct hypre_SStructPMatrixMultData_struct
{
   hypre_StructMMData    ***smmdata;   /* pointer to (nvars x nvars) array */
   HYPRE_Int                nvars;

   HYPRE_Int                nmatrices;
   hypre_SStructPMatrix   **pmatrices;  /* matrices we are multiplying */
   HYPRE_Int                nterms;
   HYPRE_Int               *terms;
   HYPRE_Int               *transposes;
} hypre_SStructPMMData;

/*--------------------------------------------------------------------------
 * SStructMatrixMult data structure
 *--------------------------------------------------------------------------*/

typedef struct hypre_SStructMatrixMultData_struct
{
   hypre_SStructPMMData   **pmmdata;   /* pointer to nparts array */
   HYPRE_Int                nparts;

   HYPRE_Int                nmatrices;
   hypre_SStructMatrix    **matrices;  /* matrices we are multiplying */
   HYPRE_Int                nterms;
   HYPRE_Int               *terms;
   HYPRE_Int               *transposes;

   hypre_CommPkg           *comm_pkg;        /* pointer to agglomerated communication package */
   hypre_CommPkg          **comm_pkg_a;      /* pointer to communication packages */
   HYPRE_Complex          **comm_data;       /* pointer to agglomerated communication data */
   HYPRE_Complex         ***comm_data_a;     /* pointer to communication data */
   HYPRE_Int                num_comm_pkgs;   /* number of communication packages to agglomerate */
   HYPRE_Int                num_comm_blocks; /* total number of communication blocks */
} hypre_SStructMMData;

#endif
