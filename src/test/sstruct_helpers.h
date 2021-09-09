/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef hypre_SSTRUCT_HELPERS_HEADER
#define hypre_SSTRUCT_HELPERS_HEADER

#include "HYPRE_utilities.h"
#include "_hypre_utilities.h"
#include "HYPRE_sstruct_mv.h"

typedef HYPRE_Int Index[3];
typedef HYPRE_Int ProblemIndex[9];

/*------------------------------------------------------------
 * ProblemIndex:
 *
 * The index has extra information stored in entries 3-8 that
 * determine how the index gets "mapped" to finer index spaces.
 *
 * NOTE: For implementation convenience, the index is "pre-shifted"
 * according to the values in entries 6,7,8.  The following discussion
 * describes how "un-shifted" indexes are mapped, because that is a
 * more natural way to think about this mapping problem, and because
 * that is the convention used in the input file for this code.  The
 * reason that pre-shifting is convenient is because it makes the true
 * value of the index on the unrefined index space readily available
 * in entries 0-2, hence, all operations on that unrefined space are
 * straightforward.  Also, the only time that the extra mapping
 * information is needed is when an index is mapped to a new refined
 * index space, allowing us to isolate the mapping details to the
 * routine MapProblemIndex.  The only other effected routine is
 * SScanProblemIndex, which takes the user input and pre-shifts it.
 *
 * - Entries 3,4,5 have values of either 0 or 1 that indicate
 *   whether to map an index "to the left" or "to the right".
 *   Here is a 1D diagram:
 *
 *    --  |     *     |    unrefined index space
 *   |
 *    --> | * | . | * |    refined index space (factor = 3)
 *          0       1
 *
 *   The '*' index on the unrefined index space gets mapped to one of
 *   the '*' indexes on the refined space based on the value (0 or 1)
 *   of the relevent entry (3,4, or 5).  The actual mapping formula is
 *   as follows (with refinement factor, r):
 *
 *   mapped_index[i] = r*index[i] + (r-1)*index[i+3]
 *
 * - Entries 6,7,8 contain "shift" information.  The shift is
 *   simply added to the mapped index just described.  So, the
 *   complete mapping formula is as follows:
 *
 *   mapped_index[i] = r*index[i] + (r-1)*index[i+3] + index[i+6]
 *
 *------------------------------------------------------------*/

typedef struct
{
   /* for GridSetExtents */
   HYPRE_Int              nboxes;
   ProblemIndex          *ilowers;
   ProblemIndex          *iuppers;
   HYPRE_Int             *boxsizes;
   HYPRE_Int              max_boxsize;

   /* for GridSetVariables */
   HYPRE_Int              nvars;
   HYPRE_SStructVariable *vartypes;

   /* for GridAddVariables */
   HYPRE_Int              add_nvars;
   ProblemIndex          *add_indexes;
   HYPRE_SStructVariable *add_vartypes;

   /* for GridSetNeighborPart and GridSetSharedPart */
   HYPRE_Int              glue_nboxes;
   HYPRE_Int             *glue_shared;
   ProblemIndex          *glue_ilowers;
   ProblemIndex          *glue_iuppers;
   Index                 *glue_offsets;
   HYPRE_Int             *glue_nbor_parts;
   ProblemIndex          *glue_nbor_ilowers;
   ProblemIndex          *glue_nbor_iuppers;
   Index                 *glue_nbor_offsets;
   Index                 *glue_index_maps;
   Index                 *glue_index_dirs;
   HYPRE_Int             *glue_primaries;

   /* for GraphSetStencil */
   HYPRE_Int             *stencil_num;

   /* for GraphAddEntries */
   HYPRE_Int              graph_nboxes;
   ProblemIndex          *graph_ilowers;
   ProblemIndex          *graph_iuppers;
   Index                 *graph_strides;
   HYPRE_Int             *graph_vars;
   HYPRE_Int             *graph_to_parts;
   ProblemIndex          *graph_to_ilowers;
   ProblemIndex          *graph_to_iuppers;
   Index                 *graph_to_strides;
   HYPRE_Int             *graph_to_vars;
   Index                 *graph_index_maps;
   Index                 *graph_index_signs;
   HYPRE_Int             *graph_entries;
   HYPRE_Real            *graph_values;
   HYPRE_Int             *graph_boxsizes;

   /* MatrixSetValues */
   HYPRE_Int              matset_nboxes;
   ProblemIndex          *matset_ilowers;
   ProblemIndex          *matset_iuppers;
   Index                 *matset_strides;
   HYPRE_Int             *matset_vars;
   HYPRE_Int             *matset_entries;
   HYPRE_Real            *matset_values;

   /* MatrixAddToValues */
   HYPRE_Int              matadd_nboxes;
   ProblemIndex          *matadd_ilowers;
   ProblemIndex          *matadd_iuppers;
   HYPRE_Int             *matadd_vars;
   HYPRE_Int             *matadd_nentries;
   HYPRE_Int            **matadd_entries;
   HYPRE_Real           **matadd_values;

   /* FEMMatrixAddToValues */
   HYPRE_Int              fem_matadd_nboxes;
   ProblemIndex          *fem_matadd_ilowers;
   ProblemIndex          *fem_matadd_iuppers;
   HYPRE_Int             *fem_matadd_nrows;
   HYPRE_Int            **fem_matadd_rows;
   HYPRE_Int             *fem_matadd_ncols;
   HYPRE_Int            **fem_matadd_cols;
   HYPRE_Real           **fem_matadd_values;

   /* RhsAddToValues */
   HYPRE_Int              rhsadd_nboxes;
   ProblemIndex          *rhsadd_ilowers;
   ProblemIndex          *rhsadd_iuppers;
   HYPRE_Int             *rhsadd_vars;
   HYPRE_Real            *rhsadd_values;

   /* FEMRhsAddToValues */
   HYPRE_Int              fem_rhsadd_nboxes;
   ProblemIndex          *fem_rhsadd_ilowers;
   ProblemIndex          *fem_rhsadd_iuppers;
   HYPRE_Real           **fem_rhsadd_values;

   Index                  periodic;

} ProblemPartData;

typedef struct
{
   HYPRE_Int        ndim;
   HYPRE_Int        nparts;
   ProblemPartData *pdata;
   HYPRE_Int        max_boxsize;

   /* for GridSetNumGhost */
   HYPRE_Int       *numghost;

   HYPRE_Int        nstencils;
   HYPRE_Int       *stencil_sizes;
   Index          **stencil_offsets;
   HYPRE_Int      **stencil_vars;
   HYPRE_Real     **stencil_values;

   HYPRE_Int        rhs_true;
   HYPRE_Real       rhs_value;

   HYPRE_Int        fem_nvars;
   Index           *fem_offsets;
   HYPRE_Int       *fem_vars;
   HYPRE_Real     **fem_values_full;
   HYPRE_Int      **fem_ivalues_full;
   HYPRE_Int       *fem_ordering; /* same info as vars/offsets */
   HYPRE_Int        fem_nsparse;  /* number of nonzeros in values_full */
   HYPRE_Int       *fem_sparsity; /* nonzeros in values_full */
   HYPRE_Real      *fem_values;   /* nonzero values in values_full */

   HYPRE_Int        fem_rhs_true;
   HYPRE_Real      *fem_rhs_values;
   HYPRE_Real      *d_fem_rhs_values;

   HYPRE_Int        symmetric_num;
   HYPRE_Int       *symmetric_parts;
   HYPRE_Int       *symmetric_vars;
   HYPRE_Int       *symmetric_to_vars;
   HYPRE_Int       *symmetric_booleans;

   HYPRE_Int        ns_symmetric;

   HYPRE_Int        npools;
   HYPRE_Int       *pools;   /* array of size nparts */
   HYPRE_Int        ndists;  /* number of (pool) distributions */
   HYPRE_Int       *dist_npools;
   HYPRE_Int      **dist_pools;

} ProblemData;

/* Function prototypes */
HYPRE_Int GetVariableBox( Index cell_ilower , Index cell_iupper , HYPRE_Int vartype , Index var_ilower, Index  var_iupper );
HYPRE_Int SScanIntArray ( char *sdata_ptr , char **sdata_ptr_ptr , HYPRE_Int size , HYPRE_Int *array );
HYPRE_Int SScanDblArray ( char *sdata_ptr , char **sdata_ptr_ptr , HYPRE_Int size , HYPRE_Real *array );
HYPRE_Int SScanProblemIndex ( char *sdata_ptr , char **sdata_ptr_ptr , HYPRE_Int ndim , ProblemIndex index );
HYPRE_Int ReadData ( char *filename , ProblemData *data_ptr );
HYPRE_Int MapProblemIndex ( ProblemIndex index , Index m );
HYPRE_Int IntersectBoxes ( ProblemIndex ilower1 , ProblemIndex iupper1 , ProblemIndex ilower2 , ProblemIndex iupper2 , ProblemIndex int_ilower , ProblemIndex int_iupper );
HYPRE_Int DistributeData ( ProblemData global_data , HYPRE_Int pooldist , Index *refine , Index *distribute , Index *block , HYPRE_Int num_procs , HYPRE_Int myid , ProblemData *data_ptr );
HYPRE_Int DestroyData ( ProblemData data );
HYPRE_Int SetCosineVector ( HYPRE_Real scale , Index ilower , Index iupper , HYPRE_Real *values );

#endif
