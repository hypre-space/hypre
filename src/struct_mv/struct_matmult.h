/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
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
 * StructMatmultDataMH data structure
 *--------------------------------------------------------------------------*/

/* product term used to compute the variable stencil entries in M */
typedef struct hypre_StructMatmultDataMH_struct
{
   hypre_StTerm    terms[MAXTERMS];   /* stencil info for each term */
   HYPRE_Int       mentry;            /* stencil entry for M */
   HYPRE_Complex   cprod;             /* product of the constant terms */
   HYPRE_Int       types[MAXTERMS];   /* types of computations to do for each term */
   HYPRE_Complex  *tptrs[MAXTERMS];   /* pointers to matrix data for each term */
   //HYPRE_Int       offsets[MAXTERMS]; /* (RDF: Needed? Similar to tptrs and not used.) */
   HYPRE_Complex  *mptr;              /* pointer to matrix data for M */

} hypre_StructMatmultDataMH;

/*--------------------------------------------------------------------------
 * StructMatmultDataM data structure
 *--------------------------------------------------------------------------*/

typedef struct hypre_StructMatmultDataM_struct
{
   HYPRE_Int                   nterms;         /* number of terms in the matmult */
   HYPRE_Int                  *terms;          /* matrix reference for each term */
   HYPRE_Int                  *transposes;     /* transpose flag for each term */

   hypre_StructMatrix         *M;              /* matmult matrix being computed */
   hypre_StMatrix             *st_M;           /* stencil matrix for M */

   HYPRE_Int                   nc;             /* size of array c */
   hypre_StructMatmultDataMH  *c;              /* helper for computing constant entries */
   HYPRE_Int                   na;             /* size of array a */
   hypre_StructMatmultDataMH  *a;              /* helper for computing variable entries */

} hypre_StructMatmultDataM;

/*--------------------------------------------------------------------------
 * StructMatmultData data structure
 *--------------------------------------------------------------------------*/

typedef struct hypre_StructMatmultData_struct
{
   HYPRE_Int                  nmatmults;  /* number of matmults */
   hypre_StructMatmultDataM  *matmults;   /* data for each matmult */

   HYPRE_Int             nmatrices;       /* number of matrices */
   hypre_StructMatrix  **matrices;        /* matrices we are multiplying */
   HYPRE_Int            *mtypes;          /* data-map type for each matrix (fine or coarse) */

   hypre_IndexRef        coarsen_stride;  /* coarsening factor for M's grid */
   HYPRE_Int             coarsen;         /* indicates if M's grid is obtained by coarsening */
   hypre_IndexRef        fstride;         /* fine data-map stride (base index space) */
   hypre_IndexRef        cstride;         /* coarse data-map stride (base index space) */
   hypre_BoxArray       *fdata_space;     /* fine data space */
   hypre_BoxArray       *cdata_space;     /* coarse data space */

   hypre_StructVector   *mask;            /* bit mask for mixed constant-variable coeff multiplies */
   hypre_CommPkg        *comm_pkg;        /* pointer to agglomerated communication package */
   HYPRE_Complex       **comm_data;       /* pointer to agglomerated communication data */
   hypre_CommStencil   **comm_stencils;   /* comm stencils used to define communication */

} hypre_StructMatmultData;

#endif
