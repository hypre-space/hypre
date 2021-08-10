/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for the struct matrix-matrix multiplication structures
 *
 *****************************************************************************/

#ifndef hypre_STRUCT_MATMULT_HEADER
#define hypre_STRUCT_MATMULT_HEADER

#ifdef MAXTERMS
#undef MAXTERMS
#endif
#define MAXTERMS 3

/*--------------------------------------------------------------------------
 * StructMMHelper data structure
 *--------------------------------------------------------------------------*/

/* product term used to compute the variable stencil entries in M */
typedef struct hypre_StructMMhelper_struct
{
   hypre_StTerm    terms[MAXTERMS]; /* stencil info for each term */
   HYPRE_Int       mentry;          /* stencil entry for M */
   HYPRE_Complex   cprod;           /* product of the constant terms */
   HYPRE_Int       types[MAXTERMS]; /* types of computations to do for each term */
   HYPRE_Complex  *tptrs[MAXTERMS]; /* pointers to matrix data for each term */
   HYPRE_Complex  *mptr;            /* pointer to matrix data for M */
} hypre_StructMMHelper;

/*--------------------------------------------------------------------------
 * StructMMData data structure
 *--------------------------------------------------------------------------*/

typedef struct hypre_StructMMData_struct
{
   hypre_StructMatrix  **matrices;       /* matrices we are multiplying */
   HYPRE_Int             nmatrices;      /* number of matrices */
   HYPRE_Int             nterms;         /* number of terms involved in the multiplication */
   HYPRE_Int            *terms;          /* pointers to matrices involved in the multiplication */
   HYPRE_Int            *transposes;     /* transpose flag for each term */
   HYPRE_Int            *mtypes;         /* data-map types for each matrix (fine or coarse) */

   hypre_IndexRef        fstride;        /* fine data-map stride */
   hypre_IndexRef        cstride;        /* coarse data-map stride */
   hypre_IndexRef        coarsen_stride; /* coarsening factor for M's grid */
   hypre_BoxArray       *cdata_space;    /* coarse data space */
   hypre_BoxArray       *fdata_space;    /* fine data space */

   hypre_StructVector   *mask;           /* bit mask vector for cte. coefs multiplication */

   hypre_StMatrix       *st_M;           /* stencil matrix for M */

   hypre_StructMMHelper *a;              /* helper for running multiplication */
   HYPRE_Int             na;             /* size of hypre_StructMMhelper object */
   hypre_CommPkg        *comm_pkg;       /* pointer to agglomerated communication package */
   HYPRE_Complex       **comm_data;      /* pointer to agglomerated communication data */
} hypre_StructMMData;

#endif
