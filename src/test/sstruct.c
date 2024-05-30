/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "_hypre_utilities.h"

#include "HYPRE_sstruct_ls.h"
#include "HYPRE_struct_ls.h"
#include "HYPRE_krylov.h"
#include "_hypre_sstruct_mv.h"
//#include "_hypre_struct_mv.hpp"

/* begin lobpcg */

#include <time.h>

#include "HYPRE_lobpcg.h"

#define NO_SOLVER -9198

/* end lobpcg */

#define DEBUG 0

#define SECOND_TIME 0

/*--------------------------------------------------------------------------
 * Data structures
 *--------------------------------------------------------------------------*/

char infile_default[50] = "sstruct.in.default";

typedef HYPRE_Int Index[3];

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

typedef HYPRE_Int ProblemIndex[9];

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
   HYPRE_Int              graph_values_size;
   HYPRE_Real            *graph_values;
   HYPRE_Real            *d_graph_values;
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

   HYPRE_MemoryLocation memory_location;

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
   HYPRE_Real      *d_fem_values;

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

/*--------------------------------------------------------------------------
 * Compute new box based on variable type
 *--------------------------------------------------------------------------*/

HYPRE_Int
GetVariableBox( Index  cell_ilower,
                Index  cell_iupper,
                HYPRE_Int    vartype,
                Index  var_ilower,
                Index  var_iupper )
{
   HYPRE_Int ierr = 0;

   var_ilower[0] = cell_ilower[0];
   var_ilower[1] = cell_ilower[1];
   var_ilower[2] = cell_ilower[2];
   var_iupper[0] = cell_iupper[0];
   var_iupper[1] = cell_iupper[1];
   var_iupper[2] = cell_iupper[2];

   switch (vartype)
   {
      case HYPRE_SSTRUCT_VARIABLE_CELL:
         var_ilower[0] -= 0; var_ilower[1] -= 0; var_ilower[2] -= 0;
         break;
      case HYPRE_SSTRUCT_VARIABLE_NODE:
         var_ilower[0] -= 1; var_ilower[1] -= 1; var_ilower[2] -= 1;
         break;
      case HYPRE_SSTRUCT_VARIABLE_XFACE:
         var_ilower[0] -= 1; var_ilower[1] -= 0; var_ilower[2] -= 0;
         break;
      case HYPRE_SSTRUCT_VARIABLE_YFACE:
         var_ilower[0] -= 0; var_ilower[1] -= 1; var_ilower[2] -= 0;
         break;
      case HYPRE_SSTRUCT_VARIABLE_ZFACE:
         var_ilower[0] -= 0; var_ilower[1] -= 0; var_ilower[2] -= 1;
         break;
      case HYPRE_SSTRUCT_VARIABLE_XEDGE:
         var_ilower[0] -= 0; var_ilower[1] -= 1; var_ilower[2] -= 1;
         break;
      case HYPRE_SSTRUCT_VARIABLE_YEDGE:
         var_ilower[0] -= 1; var_ilower[1] -= 0; var_ilower[2] -= 1;
         break;
      case HYPRE_SSTRUCT_VARIABLE_ZEDGE:
         var_ilower[0] -= 1; var_ilower[1] -= 1; var_ilower[2] -= 0;
         break;
      case HYPRE_SSTRUCT_VARIABLE_UNDEFINED:
         break;
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * Read routines
 *--------------------------------------------------------------------------*/

HYPRE_Int
SScanIntArray( char  *sdata_ptr,
               char **sdata_ptr_ptr,
               HYPRE_Int    size,
               HYPRE_Int   *array )
{
   HYPRE_Int i;

   sdata_ptr += strspn(sdata_ptr, " \t\n[");
   for (i = 0; i < size; i++)
   {
      array[i] = strtol(sdata_ptr, &sdata_ptr, 10);
   }
   sdata_ptr += strcspn(sdata_ptr, "]") + 1;

   *sdata_ptr_ptr = sdata_ptr;
   return 0;
}

HYPRE_Int
SScanDblArray( char   *sdata_ptr,
               char  **sdata_ptr_ptr,
               HYPRE_Int     size,
               HYPRE_Real *array )
{
   HYPRE_Int i;

   sdata_ptr += strspn(sdata_ptr, " \t\n[");
   for (i = 0; i < size; i++)
   {
      array[i] = (HYPRE_Real)strtod(sdata_ptr, &sdata_ptr);
   }
   sdata_ptr += strcspn(sdata_ptr, "]") + 1;

   *sdata_ptr_ptr = sdata_ptr;
   return 0;
}

HYPRE_Int
SScanProblemIndex( char          *sdata_ptr,
                   char         **sdata_ptr_ptr,
                   HYPRE_Int      ndim,
                   ProblemIndex   index )
{
   HYPRE_Int  i;
   char sign[3];

   /* initialize index array */
   for (i = 0; i < 9; i++)
   {
      index[i]   = 0;
   }

   sdata_ptr += strspn(sdata_ptr, " \t\n(");
   switch (ndim)
   {
      case 1:
         hypre_sscanf(sdata_ptr, "%d%c",
                      &index[0], &sign[0]);
         break;

      case 2:
         hypre_sscanf(sdata_ptr, "%d%c%d%c",
                      &index[0], &sign[0], &index[1], &sign[1]);
         break;

      case 3:
         hypre_sscanf(sdata_ptr, "%d%c%d%c%d%c",
                      &index[0], &sign[0], &index[1], &sign[1], &index[2], &sign[2]);
         break;
   }
   sdata_ptr += strcspn(sdata_ptr, ":)");
   if ( *sdata_ptr == ':' )
   {
      /* read in optional shift */
      sdata_ptr += 1;
      switch (ndim)
      {
         case 1:
            hypre_sscanf(sdata_ptr, "%d", &index[6]);
            break;

         case 2:
            hypre_sscanf(sdata_ptr, "%d%d", &index[6], &index[7]);
            break;

         case 3:
            hypre_sscanf(sdata_ptr, "%d%d%d", &index[6], &index[7], &index[8]);
            break;
      }
      /* pre-shift the index */
      for (i = 0; i < ndim; i++)
      {
         index[i] += index[i + 6];
      }
   }
   sdata_ptr += strcspn(sdata_ptr, ")") + 1;

   for (i = 0; i < ndim; i++)
   {
      if (sign[i] == '+')
      {
         index[i + 3] = 1;
      }
   }

   *sdata_ptr_ptr = sdata_ptr;
   return 0;
}

HYPRE_Int
ReadData( char         *filename,
          ProblemData  *data_ptr )
{
   ProblemData        data;
   ProblemPartData    pdata;

   HYPRE_Int          myid;
   FILE              *file;

   char              *sdata = NULL;
   char              *sdata_line;
   char              *sdata_ptr;
   HYPRE_Int          sdata_size;
   HYPRE_Int          size;
   HYPRE_Int          memchunk = 10000;
   HYPRE_Int          maxline  = 250;

   char               key[250];
   HYPRE_Int          part, var, s, entry, i, j, k, il, iu;

   HYPRE_MemoryLocation memory_location = data_ptr -> memory_location;

   /*-----------------------------------------------------------
    * Read data file from process 0, then broadcast
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);

   if (myid == 0)
   {
      if ((file = fopen(filename, "r")) == NULL)
      {
         hypre_printf("Error: can't open input file %s\n", filename);
         exit(1);
      }

      /* allocate initial space, and read first input line */
      sdata_size = 0;
      sdata = hypre_TAlloc(char,  memchunk, HYPRE_MEMORY_HOST);
      sdata_line = fgets(sdata, maxline, file);

      s = 0;
      while (sdata_line != NULL)
      {
         sdata_size += strlen(sdata_line) + 1;

         /* allocate more space, if necessary */
         if ((sdata_size + maxline) > s)
         {
            sdata = hypre_TReAlloc(sdata,  char,  (sdata_size + memchunk), HYPRE_MEMORY_HOST);
            s = sdata_size + memchunk;
         }

         /* read the next input line */
         sdata_line = fgets((sdata + sdata_size), maxline, file);
      }
   }
   /* broadcast the data size */
   hypre_MPI_Bcast(&sdata_size, 1, HYPRE_MPI_INT, 0, hypre_MPI_COMM_WORLD);

   /* broadcast the data */
   sdata = hypre_TReAlloc(sdata,  char,  sdata_size, HYPRE_MEMORY_HOST);
   hypre_MPI_Bcast(sdata, sdata_size, hypre_MPI_CHAR, 0, hypre_MPI_COMM_WORLD);

   /*-----------------------------------------------------------
    * Parse the data and fill ProblemData structure
    *-----------------------------------------------------------*/

   data.memory_location = memory_location;
   data.max_boxsize = 0;
   data.numghost = NULL;
   data.nstencils = 0;
   data.rhs_true = 0;
   data.fem_nvars = 0;
   data.fem_nsparse = 0;
   data.fem_rhs_true = 0;
   data.symmetric_num = 0;
   data.symmetric_parts    = NULL;
   data.symmetric_vars     = NULL;
   data.symmetric_to_vars  = NULL;
   data.symmetric_booleans = NULL;
   data.ns_symmetric = 0;
   data.ndists = 0;
   data.dist_npools = NULL;
   data.dist_pools  = NULL;

   sdata_line = sdata;
   while (sdata_line < (sdata + sdata_size))
   {
      sdata_ptr = sdata_line;

      if ( ( hypre_sscanf(sdata_ptr, "%s", key) > 0 ) && ( sdata_ptr[0] != '#' ) )
      {
         sdata_ptr += strcspn(sdata_ptr, " \t\n");

         if ( strcmp(key, "GridCreate:") == 0 )
         {
            data.ndim = strtol(sdata_ptr, &sdata_ptr, 10);
            data.nparts = strtol(sdata_ptr, &sdata_ptr, 10);
            data.pdata = hypre_CTAlloc(ProblemPartData,  data.nparts, HYPRE_MEMORY_HOST);
         }
         else if ( strcmp(key, "GridSetNumGhost:") == 0 )
         {
            // # GridSetNumGhost: numghost[2*ndim]
            // GridSetNumGhost: [3 3 3 3]
            data.numghost = hypre_CTAlloc(HYPRE_Int,  2 * data.ndim, HYPRE_MEMORY_HOST);
            SScanIntArray(sdata_ptr, &sdata_ptr, 2 * data.ndim, data.numghost);
         }
         else if ( strcmp(key, "GridSetExtents:") == 0 )
         {
            part = strtol(sdata_ptr, &sdata_ptr, 10);
            pdata = data.pdata[part];
            if ((pdata.nboxes % 10) == 0)
            {
               size = pdata.nboxes + 10;
               pdata.ilowers =
                  hypre_TReAlloc(pdata.ilowers,  ProblemIndex,  size, HYPRE_MEMORY_HOST);
               pdata.iuppers =
                  hypre_TReAlloc(pdata.iuppers,  ProblemIndex,  size, HYPRE_MEMORY_HOST);
               pdata.boxsizes =
                  hypre_TReAlloc(pdata.boxsizes,  HYPRE_Int,  size, HYPRE_MEMORY_HOST);
            }
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.ilowers[pdata.nboxes]);
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.iuppers[pdata.nboxes]);
            /* check use of +- in GridSetExtents */
            il = 1;
            iu = 1;
            for (i = 0; i < data.ndim; i++)
            {
               il *= pdata.ilowers[pdata.nboxes][i + 3];
               iu *= pdata.iuppers[pdata.nboxes][i + 3];
            }
            if ( (il != 0) || (iu != 1) )
            {
               hypre_printf("Error: Invalid use of `+-' in GridSetExtents\n");
               exit(1);
            }
            pdata.boxsizes[pdata.nboxes] = 1;
            for (i = 0; i < 3; i++)
            {
               pdata.boxsizes[pdata.nboxes] *=
                  (pdata.iuppers[pdata.nboxes][i] -
                   pdata.ilowers[pdata.nboxes][i] + 2);
            }
            pdata.max_boxsize =
               hypre_max(pdata.max_boxsize, pdata.boxsizes[pdata.nboxes]);
            pdata.nboxes++;
            data.pdata[part] = pdata;
         }
         else if ( strcmp(key, "GridSetVariables:") == 0 )
         {
            part = strtol(sdata_ptr, &sdata_ptr, 10);
            pdata = data.pdata[part];
            pdata.nvars = strtol(sdata_ptr, &sdata_ptr, 10);
            pdata.vartypes = hypre_CTAlloc(HYPRE_SStructVariable,  pdata.nvars, HYPRE_MEMORY_HOST);
            SScanIntArray(sdata_ptr, &sdata_ptr, pdata.nvars, pdata.vartypes);
            data.pdata[part] = pdata;
         }
         else if ( strcmp(key, "GridAddVariables:") == 0 )
         {
            /* TODO */
            hypre_printf("GridAddVariables not yet implemented!\n");
            exit(1);
         }
         else if ( strcmp(key, "GridSetNeighborPart:") == 0 ||
                   strcmp(key, "GridSetSharedPart:") == 0 )
         {
            part = strtol(sdata_ptr, &sdata_ptr, 10);
            pdata = data.pdata[part];
            if ((pdata.glue_nboxes % 10) == 0)
            {
               size = pdata.glue_nboxes + 10;
               pdata.glue_shared =
                  hypre_TReAlloc(pdata.glue_shared,  HYPRE_Int,  size, HYPRE_MEMORY_HOST);
               pdata.glue_ilowers =
                  hypre_TReAlloc(pdata.glue_ilowers,  ProblemIndex,  size, HYPRE_MEMORY_HOST);
               pdata.glue_iuppers =
                  hypre_TReAlloc(pdata.glue_iuppers,  ProblemIndex,  size, HYPRE_MEMORY_HOST);
               pdata.glue_offsets =
                  hypre_TReAlloc(pdata.glue_offsets,  Index,  size, HYPRE_MEMORY_HOST);
               pdata.glue_nbor_parts =
                  hypre_TReAlloc(pdata.glue_nbor_parts,  HYPRE_Int,  size, HYPRE_MEMORY_HOST);
               pdata.glue_nbor_ilowers =
                  hypre_TReAlloc(pdata.glue_nbor_ilowers,  ProblemIndex,  size, HYPRE_MEMORY_HOST);
               pdata.glue_nbor_iuppers =
                  hypre_TReAlloc(pdata.glue_nbor_iuppers,  ProblemIndex,  size, HYPRE_MEMORY_HOST);
               pdata.glue_nbor_offsets =
                  hypre_TReAlloc(pdata.glue_nbor_offsets,  Index,  size, HYPRE_MEMORY_HOST);
               pdata.glue_index_maps =
                  hypre_TReAlloc(pdata.glue_index_maps,  Index,  size, HYPRE_MEMORY_HOST);
               pdata.glue_index_dirs =
                  hypre_TReAlloc(pdata.glue_index_dirs,  Index,  size, HYPRE_MEMORY_HOST);
               pdata.glue_primaries =
                  hypre_TReAlloc(pdata.glue_primaries,  HYPRE_Int,  size, HYPRE_MEMORY_HOST);
            }
            pdata.glue_shared[pdata.glue_nboxes] = 0;
            if ( strcmp(key, "GridSetSharedPart:") == 0 )
            {
               pdata.glue_shared[pdata.glue_nboxes] = 1;
            }
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.glue_ilowers[pdata.glue_nboxes]);
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.glue_iuppers[pdata.glue_nboxes]);
            if (pdata.glue_shared[pdata.glue_nboxes])
            {
               SScanIntArray(sdata_ptr, &sdata_ptr, data.ndim,
                             pdata.glue_offsets[pdata.glue_nboxes]);
            }
            pdata.glue_nbor_parts[pdata.glue_nboxes] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.glue_nbor_ilowers[pdata.glue_nboxes]);
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.glue_nbor_iuppers[pdata.glue_nboxes]);
            if (pdata.glue_shared[pdata.glue_nboxes])
            {
               SScanIntArray(sdata_ptr, &sdata_ptr, data.ndim,
                             pdata.glue_nbor_offsets[pdata.glue_nboxes]);
            }
            SScanIntArray(sdata_ptr, &sdata_ptr, data.ndim,
                          pdata.glue_index_maps[pdata.glue_nboxes]);
            for (i = data.ndim; i < 3; i++)
            {
               pdata.glue_index_maps[pdata.glue_nboxes][i] = i;
            }
            SScanIntArray(sdata_ptr, &sdata_ptr, data.ndim,
                          pdata.glue_index_dirs[pdata.glue_nboxes]);
            for (i = data.ndim; i < 3; i++)
            {
               pdata.glue_index_dirs[pdata.glue_nboxes][i] = 1;
            }
            sdata_ptr += strcspn(sdata_ptr, ":\t\n");
            if ( *sdata_ptr == ':' )
            {
               /* read in optional primary indicator */
               sdata_ptr += 1;
               pdata.glue_primaries[pdata.glue_nboxes] =
                  strtol(sdata_ptr, &sdata_ptr, 10);
            }
            else
            {
               pdata.glue_primaries[pdata.glue_nboxes] = -1;
               sdata_ptr -= 1;
            }
            pdata.glue_nboxes++;
            data.pdata[part] = pdata;
         }
         else if ( strcmp(key, "GridSetPeriodic:") == 0 )
         {
            part = strtol(sdata_ptr, &sdata_ptr, 10);
            pdata = data.pdata[part];
            SScanIntArray(sdata_ptr, &sdata_ptr, data.ndim, pdata.periodic);
            for (i = data.ndim; i < 3; i++)
            {
               pdata.periodic[i] = 0;
            }
            data.pdata[part] = pdata;
         }
         else if ( strcmp(key, "StencilCreate:") == 0 )
         {
            if (data.fem_nvars > 0)
            {
               hypre_printf("Stencil and FEMStencil cannot be used together\n");
               exit(1);
            }
            data.nstencils = strtol(sdata_ptr, &sdata_ptr, 10);
            data.stencil_sizes   = hypre_CTAlloc(HYPRE_Int,  data.nstencils, HYPRE_MEMORY_HOST);
            data.stencil_offsets = hypre_CTAlloc(Index *,  data.nstencils, HYPRE_MEMORY_HOST);
            data.stencil_vars    = hypre_CTAlloc(HYPRE_Int *,  data.nstencils, HYPRE_MEMORY_HOST);
            data.stencil_values  = hypre_CTAlloc(HYPRE_Real *,  data.nstencils, HYPRE_MEMORY_HOST);
            SScanIntArray(sdata_ptr, &sdata_ptr,
                          data.nstencils, data.stencil_sizes);
            for (s = 0; s < data.nstencils; s++)
            {
               data.stencil_offsets[s] =
                  hypre_CTAlloc(Index,  data.stencil_sizes[s], HYPRE_MEMORY_HOST);
               data.stencil_vars[s] =
                  hypre_CTAlloc(HYPRE_Int,  data.stencil_sizes[s], HYPRE_MEMORY_HOST);
               data.stencil_values[s] =
                  hypre_CTAlloc(HYPRE_Real,  data.stencil_sizes[s], HYPRE_MEMORY_HOST);
            }
         }
         else if ( strcmp(key, "StencilSetEntry:") == 0 )
         {
            s = strtol(sdata_ptr, &sdata_ptr, 10);
            entry = strtol(sdata_ptr, &sdata_ptr, 10);
            SScanIntArray(sdata_ptr, &sdata_ptr,
                          data.ndim, data.stencil_offsets[s][entry]);
            for (i = data.ndim; i < 3; i++)
            {
               data.stencil_offsets[s][entry][i] = 0;
            }
            data.stencil_vars[s][entry] = strtol(sdata_ptr, &sdata_ptr, 10);
            data.stencil_values[s][entry] = (HYPRE_Real)strtod(sdata_ptr, &sdata_ptr);
         }
         else if ( strcmp(key, "RhsSet:") == 0 )
         {
            if (data.rhs_true == 0)
            {
               data.rhs_true = 1;
            }
            data.rhs_value = (HYPRE_Real)strtod(sdata_ptr, &sdata_ptr);
         }
         else if ( strcmp(key, "FEMStencilCreate:") == 0 )
         {
            if (data.nstencils > 0)
            {
               hypre_printf("Stencil and FEMStencil cannot be used together\n");
               exit(1);
            }
            data.fem_nvars = strtol(sdata_ptr, &sdata_ptr, 10);
            data.fem_offsets = hypre_CTAlloc(Index,  data.fem_nvars, HYPRE_MEMORY_HOST);
            data.fem_vars = hypre_CTAlloc(HYPRE_Int,  data.fem_nvars, HYPRE_MEMORY_HOST);
            data.fem_values_full = hypre_CTAlloc(HYPRE_Real *,  data.fem_nvars, HYPRE_MEMORY_HOST);
            for (i = 0; i < data.fem_nvars; i++)
            {
               data.fem_values_full[i] = hypre_CTAlloc(HYPRE_Real,  data.fem_nvars, HYPRE_MEMORY_HOST);
            }
         }
         else if ( strcmp(key, "FEMStencilSetRow:") == 0 )
         {
            i = strtol(sdata_ptr, &sdata_ptr, 10);
            SScanIntArray(sdata_ptr, &sdata_ptr,
                          data.ndim, data.fem_offsets[i]);
            for (k = data.ndim; k < 3; k++)
            {
               data.fem_offsets[i][k] = 0;
            }
            data.fem_vars[i] = strtol(sdata_ptr, &sdata_ptr, 10);
            SScanDblArray(sdata_ptr, &sdata_ptr,
                          data.fem_nvars, data.fem_values_full[i]);
         }
         else if ( strcmp(key, "FEMRhsSet:") == 0 )
         {
            if (data.fem_rhs_true == 0)
            {
               data.fem_rhs_true = 1;
               data.fem_rhs_values = hypre_CTAlloc(HYPRE_Real, data.fem_nvars, HYPRE_MEMORY_HOST);
               data.d_fem_rhs_values = hypre_CTAlloc(HYPRE_Real, data.fem_nvars, memory_location);
            }
            SScanDblArray(sdata_ptr, &sdata_ptr,
                          data.fem_nvars, data.fem_rhs_values);
         }
         else if ( strcmp(key, "GraphSetStencil:") == 0 )
         {
            part = strtol(sdata_ptr, &sdata_ptr, 10);
            var = strtol(sdata_ptr, &sdata_ptr, 10);
            s = strtol(sdata_ptr, &sdata_ptr, 10);
            pdata = data.pdata[part];
            if (pdata.stencil_num == NULL)
            {
               pdata.stencil_num = hypre_CTAlloc(HYPRE_Int,  pdata.nvars, HYPRE_MEMORY_HOST);
            }
            pdata.stencil_num[var] = s;
            data.pdata[part] = pdata;
         }
         else if ( strcmp(key, "GraphAddEntries:") == 0 )
         {
            part = strtol(sdata_ptr, &sdata_ptr, 10);
            pdata = data.pdata[part];
            if ((pdata.graph_nboxes % 10) == 0)
            {
               size = pdata.graph_nboxes + 10;
               pdata.graph_ilowers =
                  hypre_TReAlloc(pdata.graph_ilowers,  ProblemIndex,  size, HYPRE_MEMORY_HOST);
               pdata.graph_iuppers =
                  hypre_TReAlloc(pdata.graph_iuppers,  ProblemIndex,  size, HYPRE_MEMORY_HOST);
               pdata.graph_strides =
                  hypre_TReAlloc(pdata.graph_strides,  Index,  size, HYPRE_MEMORY_HOST);
               pdata.graph_vars =
                  hypre_TReAlloc(pdata.graph_vars,  HYPRE_Int,  size, HYPRE_MEMORY_HOST);
               pdata.graph_to_parts =
                  hypre_TReAlloc(pdata.graph_to_parts,  HYPRE_Int,  size, HYPRE_MEMORY_HOST);
               pdata.graph_to_ilowers =
                  hypre_TReAlloc(pdata.graph_to_ilowers,  ProblemIndex,  size, HYPRE_MEMORY_HOST);
               pdata.graph_to_iuppers =
                  hypre_TReAlloc(pdata.graph_to_iuppers,  ProblemIndex,  size, HYPRE_MEMORY_HOST);
               pdata.graph_to_strides =
                  hypre_TReAlloc(pdata.graph_to_strides,  Index,  size, HYPRE_MEMORY_HOST);
               pdata.graph_to_vars =
                  hypre_TReAlloc(pdata.graph_to_vars,  HYPRE_Int,  size, HYPRE_MEMORY_HOST);
               pdata.graph_index_maps =
                  hypre_TReAlloc(pdata.graph_index_maps,  Index,  size, HYPRE_MEMORY_HOST);
               pdata.graph_index_signs =
                  hypre_TReAlloc(pdata.graph_index_signs,  Index,  size, HYPRE_MEMORY_HOST);
               pdata.graph_entries =
                  hypre_TReAlloc(pdata.graph_entries,  HYPRE_Int,  size, HYPRE_MEMORY_HOST);
               pdata.graph_values =
                  hypre_TReAlloc(pdata.graph_values,  HYPRE_Real,  size, HYPRE_MEMORY_HOST);
               pdata.d_graph_values =
                  hypre_TReAlloc_v2(pdata.d_graph_values, HYPRE_Real, pdata.graph_values_size,
                                    HYPRE_Real, size, memory_location);
               pdata.graph_values_size = size;
               pdata.graph_boxsizes =
                  hypre_TReAlloc(pdata.graph_boxsizes,  HYPRE_Int,  size, HYPRE_MEMORY_HOST);
            }
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.graph_ilowers[pdata.graph_nboxes]);
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.graph_iuppers[pdata.graph_nboxes]);
            SScanIntArray(sdata_ptr, &sdata_ptr, data.ndim,
                          pdata.graph_strides[pdata.graph_nboxes]);
            for (i = data.ndim; i < 3; i++)
            {
               pdata.graph_strides[pdata.graph_nboxes][i] = 1;
            }
            pdata.graph_vars[pdata.graph_nboxes] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            pdata.graph_to_parts[pdata.graph_nboxes] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.graph_to_ilowers[pdata.graph_nboxes]);
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.graph_to_iuppers[pdata.graph_nboxes]);
            SScanIntArray(sdata_ptr, &sdata_ptr, data.ndim,
                          pdata.graph_to_strides[pdata.graph_nboxes]);
            for (i = data.ndim; i < 3; i++)
            {
               pdata.graph_to_strides[pdata.graph_nboxes][i] = 1;
            }
            pdata.graph_to_vars[pdata.graph_nboxes] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            SScanIntArray(sdata_ptr, &sdata_ptr, data.ndim,
                          pdata.graph_index_maps[pdata.graph_nboxes]);
            for (i = data.ndim; i < 3; i++)
            {
               pdata.graph_index_maps[pdata.graph_nboxes][i] = i;
            }
            for (i = 0; i < 3; i++)
            {
               pdata.graph_index_signs[pdata.graph_nboxes][i] = 1;
               if ( pdata.graph_to_iuppers[pdata.graph_nboxes][i] <
                    pdata.graph_to_ilowers[pdata.graph_nboxes][i] )
               {
                  pdata.graph_index_signs[pdata.graph_nboxes][i] = -1;
               }
            }
            pdata.graph_entries[pdata.graph_nboxes] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            pdata.graph_values[pdata.graph_nboxes] =
               (HYPRE_Real)strtod(sdata_ptr, &sdata_ptr);
            pdata.graph_boxsizes[pdata.graph_nboxes] = 1;
            for (i = 0; i < 3; i++)
            {
               pdata.graph_boxsizes[pdata.graph_nboxes] *=
                  (pdata.graph_iuppers[pdata.graph_nboxes][i] -
                   pdata.graph_ilowers[pdata.graph_nboxes][i] + 1);
            }
            pdata.graph_nboxes++;
            data.pdata[part] = pdata;
         }
         else if ( strcmp(key, "MatrixSetSymmetric:") == 0 )
         {
            if ((data.symmetric_num % 10) == 0)
            {
               size = data.symmetric_num + 10;
               data.symmetric_parts =
                  hypre_TReAlloc(data.symmetric_parts,  HYPRE_Int,  size, HYPRE_MEMORY_HOST);
               data.symmetric_vars =
                  hypre_TReAlloc(data.symmetric_vars,  HYPRE_Int,  size, HYPRE_MEMORY_HOST);
               data.symmetric_to_vars =
                  hypre_TReAlloc(data.symmetric_to_vars,  HYPRE_Int,  size, HYPRE_MEMORY_HOST);
               data.symmetric_booleans =
                  hypre_TReAlloc(data.symmetric_booleans,  HYPRE_Int,  size, HYPRE_MEMORY_HOST);
            }
            data.symmetric_parts[data.symmetric_num] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            data.symmetric_vars[data.symmetric_num] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            data.symmetric_to_vars[data.symmetric_num] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            data.symmetric_booleans[data.symmetric_num] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            data.symmetric_num++;
         }
         else if ( strcmp(key, "MatrixSetNSSymmetric:") == 0 )
         {
            data.ns_symmetric = strtol(sdata_ptr, &sdata_ptr, 10);
         }
         else if ( strcmp(key, "MatrixSetValues:") == 0 )
         {
            part = strtol(sdata_ptr, &sdata_ptr, 10);
            pdata = data.pdata[part];
            if ((pdata.matset_nboxes % 10) == 0)
            {
               size = pdata.matset_nboxes + 10;
               pdata.matset_ilowers =
                  hypre_TReAlloc(pdata.matset_ilowers,  ProblemIndex,  size, HYPRE_MEMORY_HOST);
               pdata.matset_iuppers =
                  hypre_TReAlloc(pdata.matset_iuppers,  ProblemIndex,  size, HYPRE_MEMORY_HOST);
               pdata.matset_strides =
                  hypre_TReAlloc(pdata.matset_strides,  Index,  size, HYPRE_MEMORY_HOST);
               pdata.matset_vars =
                  hypre_TReAlloc(pdata.matset_vars,  HYPRE_Int,  size, HYPRE_MEMORY_HOST);
               pdata.matset_entries =
                  hypre_TReAlloc(pdata.matset_entries,  HYPRE_Int,  size, HYPRE_MEMORY_HOST);
               pdata.matset_values =
                  hypre_TReAlloc(pdata.matset_values,  HYPRE_Real,  size, HYPRE_MEMORY_HOST);
            }
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.matset_ilowers[pdata.matset_nboxes]);
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.matset_iuppers[pdata.matset_nboxes]);
            SScanIntArray(sdata_ptr, &sdata_ptr, data.ndim,
                          pdata.matset_strides[pdata.matset_nboxes]);
            for (i = data.ndim; i < 3; i++)
            {
               pdata.matset_strides[pdata.matset_nboxes][i] = 1;
            }
            pdata.matset_vars[pdata.matset_nboxes] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            pdata.matset_entries[pdata.matset_nboxes] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            pdata.matset_values[pdata.matset_nboxes] =
               (HYPRE_Real)strtod(sdata_ptr, &sdata_ptr);
            pdata.matset_nboxes++;
            data.pdata[part] = pdata;
         }
         else if ( strcmp(key, "MatrixAddToValues:") == 0 )
         {
            part = strtol(sdata_ptr, &sdata_ptr, 10);
            pdata = data.pdata[part];
            if ((pdata.matadd_nboxes % 10) == 0)
            {
               size = pdata.matadd_nboxes + 10;
               pdata.matadd_ilowers =
                  hypre_TReAlloc(pdata.matadd_ilowers,  ProblemIndex,  size, HYPRE_MEMORY_HOST);
               pdata.matadd_iuppers =
                  hypre_TReAlloc(pdata.matadd_iuppers,  ProblemIndex,  size, HYPRE_MEMORY_HOST);
               pdata.matadd_vars =
                  hypre_TReAlloc(pdata.matadd_vars,  HYPRE_Int,  size, HYPRE_MEMORY_HOST);
               pdata.matadd_nentries =
                  hypre_TReAlloc(pdata.matadd_nentries,  HYPRE_Int,  size, HYPRE_MEMORY_HOST);
               pdata.matadd_entries =
                  hypre_TReAlloc(pdata.matadd_entries,  HYPRE_Int *,  size, HYPRE_MEMORY_HOST);
               pdata.matadd_values =
                  hypre_TReAlloc(pdata.matadd_values,  HYPRE_Real *,  size, HYPRE_MEMORY_HOST);
            }
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.matadd_ilowers[pdata.matadd_nboxes]);
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.matadd_iuppers[pdata.matadd_nboxes]);
            pdata.matadd_vars[pdata.matadd_nboxes] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            i = strtol(sdata_ptr, &sdata_ptr, 10);
            pdata.matadd_nentries[pdata.matadd_nboxes] = i;
            pdata.matadd_entries[pdata.matadd_nboxes] =
               hypre_TAlloc(HYPRE_Int,  i, HYPRE_MEMORY_HOST);
            SScanIntArray(sdata_ptr, &sdata_ptr, i,
                          (HYPRE_Int*) pdata.matadd_entries[pdata.matadd_nboxes]);
            pdata.matadd_values[pdata.matadd_nboxes] =
               hypre_TAlloc(HYPRE_Real,  i, HYPRE_MEMORY_HOST);
            SScanDblArray(sdata_ptr, &sdata_ptr, i,
                          (HYPRE_Real *) pdata.matadd_values[pdata.matadd_nboxes]);
            pdata.matadd_nboxes++;
            data.pdata[part] = pdata;
         }
         else if ( strcmp(key, "FEMMatrixAddToValues:") == 0 )
         {
            part = strtol(sdata_ptr, &sdata_ptr, 10);
            pdata = data.pdata[part];
            if ((pdata.fem_matadd_nboxes % 10) == 0)
            {
               size = pdata.fem_matadd_nboxes + 10;
               pdata.fem_matadd_ilowers =
                  hypre_TReAlloc(pdata.fem_matadd_ilowers,  ProblemIndex,  size, HYPRE_MEMORY_HOST);
               pdata.fem_matadd_iuppers =
                  hypre_TReAlloc(pdata.fem_matadd_iuppers,  ProblemIndex,  size, HYPRE_MEMORY_HOST);
               pdata.fem_matadd_nrows =
                  hypre_TReAlloc(pdata.fem_matadd_nrows,  HYPRE_Int,  size, HYPRE_MEMORY_HOST);
               pdata.fem_matadd_rows =
                  hypre_TReAlloc(pdata.fem_matadd_rows,  HYPRE_Int *,  size, HYPRE_MEMORY_HOST);
               pdata.fem_matadd_ncols =
                  hypre_TReAlloc(pdata.fem_matadd_ncols,  HYPRE_Int,  size, HYPRE_MEMORY_HOST);
               pdata.fem_matadd_cols =
                  hypre_TReAlloc(pdata.fem_matadd_cols,  HYPRE_Int *,  size, HYPRE_MEMORY_HOST);
               pdata.fem_matadd_values =
                  hypre_TReAlloc(pdata.fem_matadd_values,  HYPRE_Real *,  size, HYPRE_MEMORY_HOST);
            }
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.fem_matadd_ilowers[pdata.fem_matadd_nboxes]);
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.fem_matadd_iuppers[pdata.fem_matadd_nboxes]);
            i = strtol(sdata_ptr, &sdata_ptr, 10);
            pdata.fem_matadd_nrows[pdata.fem_matadd_nboxes] = i;
            pdata.fem_matadd_rows[pdata.fem_matadd_nboxes] = hypre_TAlloc(HYPRE_Int,  i, HYPRE_MEMORY_HOST);
            SScanIntArray(sdata_ptr, &sdata_ptr, i,
                          (HYPRE_Int*) pdata.fem_matadd_rows[pdata.fem_matadd_nboxes]);
            j = strtol(sdata_ptr, &sdata_ptr, 10);
            pdata.fem_matadd_ncols[pdata.fem_matadd_nboxes] = j;
            pdata.fem_matadd_cols[pdata.fem_matadd_nboxes] = hypre_TAlloc(HYPRE_Int,  j, HYPRE_MEMORY_HOST);
            SScanIntArray(sdata_ptr, &sdata_ptr, j,
                          (HYPRE_Int*) pdata.fem_matadd_cols[pdata.fem_matadd_nboxes]);
            pdata.fem_matadd_values[pdata.fem_matadd_nboxes] =
               hypre_TAlloc(HYPRE_Real,  i * j, HYPRE_MEMORY_HOST);
            SScanDblArray(sdata_ptr, &sdata_ptr, i * j,
                          (HYPRE_Real *) pdata.fem_matadd_values[pdata.fem_matadd_nboxes]);
            pdata.fem_matadd_nboxes++;
            data.pdata[part] = pdata;
         }
         else if ( strcmp(key, "RhsAddToValues:") == 0 )
         {
            part = strtol(sdata_ptr, &sdata_ptr, 10);
            pdata = data.pdata[part];
            if ((pdata.rhsadd_nboxes % 10) == 0)
            {
               size = pdata.rhsadd_nboxes + 10;
               pdata.rhsadd_ilowers =
                  hypre_TReAlloc(pdata.rhsadd_ilowers,  ProblemIndex,  size, HYPRE_MEMORY_HOST);
               pdata.rhsadd_iuppers =
                  hypre_TReAlloc(pdata.rhsadd_iuppers,  ProblemIndex,  size, HYPRE_MEMORY_HOST);
               pdata.rhsadd_vars =
                  hypre_TReAlloc(pdata.rhsadd_vars,  HYPRE_Int,  size, HYPRE_MEMORY_HOST);
               pdata.rhsadd_values =
                  hypre_TReAlloc(pdata.rhsadd_values,  HYPRE_Real,  size, HYPRE_MEMORY_HOST);
            }
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.rhsadd_ilowers[pdata.rhsadd_nboxes]);
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.rhsadd_iuppers[pdata.rhsadd_nboxes]);
            pdata.rhsadd_vars[pdata.rhsadd_nboxes] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            pdata.rhsadd_values[pdata.rhsadd_nboxes] =
               (HYPRE_Real)strtod(sdata_ptr, &sdata_ptr);
            pdata.rhsadd_nboxes++;
            data.pdata[part] = pdata;
         }
         else if ( strcmp(key, "FEMRhsAddToValues:") == 0 )
         {
            part = strtol(sdata_ptr, &sdata_ptr, 10);
            pdata = data.pdata[part];
            if ((pdata.fem_rhsadd_nboxes % 10) == 0)
            {
               size = pdata.fem_rhsadd_nboxes + 10;
               pdata.fem_rhsadd_ilowers =
                  hypre_TReAlloc(pdata.fem_rhsadd_ilowers,  ProblemIndex,  size, HYPRE_MEMORY_HOST);
               pdata.fem_rhsadd_iuppers =
                  hypre_TReAlloc(pdata.fem_rhsadd_iuppers,  ProblemIndex,  size, HYPRE_MEMORY_HOST);
               pdata.fem_rhsadd_values =
                  hypre_TReAlloc(pdata.fem_rhsadd_values,  HYPRE_Real *,  size, HYPRE_MEMORY_HOST);
            }
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.fem_rhsadd_ilowers[pdata.fem_rhsadd_nboxes]);
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.fem_rhsadd_iuppers[pdata.fem_rhsadd_nboxes]);
            pdata.fem_rhsadd_values[pdata.fem_rhsadd_nboxes] =
               hypre_TAlloc(HYPRE_Real,  data.fem_nvars, HYPRE_MEMORY_HOST);
            SScanDblArray(sdata_ptr, &sdata_ptr, data.fem_nvars,
                          (HYPRE_Real *) pdata.fem_rhsadd_values[pdata.fem_rhsadd_nboxes]);
            pdata.fem_rhsadd_nboxes++;
            data.pdata[part] = pdata;
         }
         else if ( strcmp(key, "ProcessPoolCreate:") == 0 )
         {
            data.ndists++;
            data.dist_npools = hypre_TReAlloc(data.dist_npools,  HYPRE_Int,  data.ndists, HYPRE_MEMORY_HOST);
            data.dist_pools = hypre_TReAlloc(data.dist_pools,  HYPRE_Int *,  data.ndists, HYPRE_MEMORY_HOST);
            data.dist_npools[data.ndists - 1] = strtol(sdata_ptr, &sdata_ptr, 10);
            data.dist_pools[data.ndists - 1] = hypre_CTAlloc(HYPRE_Int,  data.nparts, HYPRE_MEMORY_HOST);
#if 0
            data.npools = strtol(sdata_ptr, &sdata_ptr, 10);
            data.pools = hypre_CTAlloc(HYPRE_Int,  data.nparts, HYPRE_MEMORY_HOST);
#endif
         }
         else if ( strcmp(key, "ProcessPoolSetPart:") == 0 )
         {
            i = strtol(sdata_ptr, &sdata_ptr, 10);
            part = strtol(sdata_ptr, &sdata_ptr, 10);
            data.dist_pools[data.ndists - 1][part] = i;
         }
         else if ( strcmp(key, "GridSetNeighborBox:") == 0 )
         {
            hypre_printf("Error: No longer supporting SetNeighborBox\n");
         }
      }

      sdata_line += strlen(sdata_line) + 1;
   }

   data.max_boxsize = 0;
   for (part = 0; part < data.nparts; part++)
   {
      data.max_boxsize =
         hypre_max(data.max_boxsize, data.pdata[part].max_boxsize);
   }

   /* build additional FEM information */
   if (data.fem_nvars > 0)
   {
      HYPRE_Int d;

      data.fem_ivalues_full = hypre_CTAlloc(HYPRE_Int *, data.fem_nvars, HYPRE_MEMORY_HOST);
      data.fem_ordering = hypre_CTAlloc(HYPRE_Int, (1 + data.ndim) * data.fem_nvars, HYPRE_MEMORY_HOST);
      data.fem_sparsity = hypre_CTAlloc(HYPRE_Int, 2 * data.fem_nvars * data.fem_nvars,
                                        HYPRE_MEMORY_HOST);
      data.fem_values   = hypre_CTAlloc(HYPRE_Real, data.fem_nvars * data.fem_nvars, HYPRE_MEMORY_HOST);
      data.d_fem_values = hypre_TAlloc(HYPRE_Real, data.fem_nvars * data.fem_nvars, memory_location);

      for (i = 0; i < data.fem_nvars; i++)
      {
         data.fem_ivalues_full[i] = hypre_CTAlloc(HYPRE_Int,  data.fem_nvars, HYPRE_MEMORY_HOST);
         k = (1 + data.ndim) * i;
         data.fem_ordering[k] = data.fem_vars[i];
         for (d = 0; d < data.ndim; d++)
         {
            data.fem_ordering[k + 1 + d] = data.fem_offsets[i][d];
         }
         for (j = 0; j < data.fem_nvars; j++)
         {
            if (data.fem_values_full[i][j] != 0.0)
            {
               k = 2 * data.fem_nsparse;
               data.fem_sparsity[k]   = i;
               data.fem_sparsity[k + 1] = j;
               data.fem_values[data.fem_nsparse] = data.fem_values_full[i][j];
               data.fem_ivalues_full[i][j] = data.fem_nsparse;
               data.fem_nsparse ++;
            }
         }
      }
   }

   hypre_TFree(sdata, HYPRE_MEMORY_HOST);

   *data_ptr = data;
   return 0;
}

/*--------------------------------------------------------------------------
 * Distribute routines
 *--------------------------------------------------------------------------*/

HYPRE_Int
MapProblemIndex( ProblemIndex index,
                 Index        m )
{
   /* un-shift the index */
   index[0] -= index[6];
   index[1] -= index[7];
   index[2] -= index[8];
   /* map the index */
   index[0] = m[0] * index[0] + (m[0] - 1) * index[3];
   index[1] = m[1] * index[1] + (m[1] - 1) * index[4];
   index[2] = m[2] * index[2] + (m[2] - 1) * index[5];
   /* pre-shift the new mapped index */
   index[0] += index[6];
   index[1] += index[7];
   index[2] += index[8];

   return 0;
}

HYPRE_Int
IntersectBoxes( ProblemIndex ilower1,
                ProblemIndex iupper1,
                ProblemIndex ilower2,
                ProblemIndex iupper2,
                ProblemIndex int_ilower,
                ProblemIndex int_iupper )
{
   HYPRE_Int d, size;

   size = 1;
   for (d = 0; d < 3; d++)
   {
      int_ilower[d] = hypre_max(ilower1[d], ilower2[d]);
      int_iupper[d] = hypre_min(iupper1[d], iupper2[d]);
      size *= hypre_max(0, (int_iupper[d] - int_ilower[d] + 1));
   }

   return size;
}

HYPRE_Int
DistributeData( ProblemData   global_data,
                HYPRE_Int     pooldist,
                Index        *refine,
                Index        *distribute,
                Index        *block,
                HYPRE_Int     num_procs,
                HYPRE_Int     myid,
                ProblemData  *data_ptr )
{
   HYPRE_MemoryLocation memory_location = global_data.memory_location;
   ProblemData      data = global_data;
   ProblemPartData  pdata;
   HYPRE_Int       *pool_procs;
   HYPRE_Int        np, pid;
   HYPRE_Int        pool, part, box, b, p, q, r, i, d;
   HYPRE_Int        dmap, sign, size;
   HYPRE_Int       *iptr;
   HYPRE_Real      *dptr;
   Index            m, mmap, n;
   ProblemIndex     ilower, iupper, int_ilower, int_iupper;

   /* set default pool distribution */
   data.npools = data.dist_npools[pooldist];
   data.pools  = data.dist_pools[pooldist];

   /* determine first process number in each pool */
   pool_procs = hypre_CTAlloc(HYPRE_Int,  (data.npools + 1), HYPRE_MEMORY_HOST);
   for (part = 0; part < data.nparts; part++)
   {
      pool = data.pools[part] + 1;
      np = distribute[part][0] * distribute[part][1] * distribute[part][2];
      pool_procs[pool] = hypre_max(pool_procs[pool], np);

   }
   pool_procs[0] = 0;
   for (pool = 1; pool < (data.npools + 1); pool++)
   {
      pool_procs[pool] = pool_procs[pool - 1] + pool_procs[pool];
   }

   /* check number of processes */
   if (pool_procs[data.npools] != num_procs)
   {
      hypre_printf("%d,  %d \n", pool_procs[data.npools], num_procs);
      hypre_printf("Error: Invalid number of processes or process topology \n");
      exit(1);
   }

   /* modify part data */
   for (part = 0; part < data.nparts; part++)
   {
      pdata = data.pdata[part];
      pool  = data.pools[part];
      np  = distribute[part][0] * distribute[part][1] * distribute[part][2];
      pid = myid - pool_procs[pool];

      if ( (pid < 0) || (pid >= np) )
      {
         /* none of this part data lives on this process */
         pdata.nboxes = 0;
#if 1 /* set this to 0 to make all of the SetSharedPart calls */
         pdata.glue_nboxes = 0;
#endif
         pdata.graph_nboxes = 0;
         pdata.matset_nboxes = 0;
         for (box = 0; box < pdata.matadd_nboxes; box++)
         {
            hypre_TFree(pdata.matadd_entries[box], HYPRE_MEMORY_HOST);
            hypre_TFree(pdata.matadd_values[box], HYPRE_MEMORY_HOST);
         }
         pdata.matadd_nboxes = 0;
         for (box = 0; box < pdata.fem_matadd_nboxes; box++)
         {
            hypre_TFree(pdata.fem_matadd_rows[box], HYPRE_MEMORY_HOST);
            hypre_TFree(pdata.fem_matadd_cols[box], HYPRE_MEMORY_HOST);
            hypre_TFree(pdata.fem_matadd_values[box], HYPRE_MEMORY_HOST);
         }
         pdata.fem_matadd_nboxes = 0;
         pdata.rhsadd_nboxes = 0;
         for (box = 0; box < pdata.fem_rhsadd_nboxes; box++)
         {
            hypre_TFree(pdata.fem_rhsadd_values[box], HYPRE_MEMORY_HOST);
         }
         pdata.fem_rhsadd_nboxes = 0;
      }
      else
      {
         /* refine boxes */
         m[0] = refine[part][0];
         m[1] = refine[part][1];
         m[2] = refine[part][2];
         if ( (m[0] * m[1] * m[2]) > 1)
         {
            for (box = 0; box < pdata.nboxes; box++)
            {
               MapProblemIndex(pdata.ilowers[box], m);
               MapProblemIndex(pdata.iuppers[box], m);
            }

            for (box = 0; box < pdata.graph_nboxes; box++)
            {
               MapProblemIndex(pdata.graph_ilowers[box], m);
               MapProblemIndex(pdata.graph_iuppers[box], m);
               mmap[0] = m[pdata.graph_index_maps[box][0]];
               mmap[1] = m[pdata.graph_index_maps[box][1]];
               mmap[2] = m[pdata.graph_index_maps[box][2]];
               MapProblemIndex(pdata.graph_to_ilowers[box], mmap);
               MapProblemIndex(pdata.graph_to_iuppers[box], mmap);
            }
            for (box = 0; box < pdata.matset_nboxes; box++)
            {
               MapProblemIndex(pdata.matset_ilowers[box], m);
               MapProblemIndex(pdata.matset_iuppers[box], m);
            }
            for (box = 0; box < pdata.matadd_nboxes; box++)
            {
               MapProblemIndex(pdata.matadd_ilowers[box], m);
               MapProblemIndex(pdata.matadd_iuppers[box], m);
            }
            for (box = 0; box < pdata.fem_matadd_nboxes; box++)
            {
               MapProblemIndex(pdata.fem_matadd_ilowers[box], m);
               MapProblemIndex(pdata.fem_matadd_iuppers[box], m);
            }
            for (box = 0; box < pdata.rhsadd_nboxes; box++)
            {
               MapProblemIndex(pdata.rhsadd_ilowers[box], m);
               MapProblemIndex(pdata.rhsadd_iuppers[box], m);
            }
            for (box = 0; box < pdata.fem_rhsadd_nboxes; box++)
            {
               MapProblemIndex(pdata.fem_rhsadd_ilowers[box], m);
               MapProblemIndex(pdata.fem_rhsadd_iuppers[box], m);
            }
         }

         /* refine and distribute boxes */
         m[0] = distribute[part][0];
         m[1] = distribute[part][1];
         m[2] = distribute[part][2];
         if ( (m[0] * m[1] * m[2]) > 1)
         {
            p = pid % m[0];
            q = ((pid - p) / m[0]) % m[1];
            r = (pid - p - q * m[0]) / (m[0] * m[1]);

            for (box = 0; box < pdata.nboxes; box++)
            {
               n[0] = pdata.iuppers[box][0] - pdata.ilowers[box][0] + 1;
               n[1] = pdata.iuppers[box][1] - pdata.ilowers[box][1] + 1;
               n[2] = pdata.iuppers[box][2] - pdata.ilowers[box][2] + 1;

               MapProblemIndex(pdata.ilowers[box], m);
               MapProblemIndex(pdata.iuppers[box], m);
               pdata.iuppers[box][0] = pdata.ilowers[box][0] + n[0] - 1;
               pdata.iuppers[box][1] = pdata.ilowers[box][1] + n[1] - 1;
               pdata.iuppers[box][2] = pdata.ilowers[box][2] + n[2] - 1;

               pdata.ilowers[box][0] = pdata.ilowers[box][0] + p * n[0];
               pdata.ilowers[box][1] = pdata.ilowers[box][1] + q * n[1];
               pdata.ilowers[box][2] = pdata.ilowers[box][2] + r * n[2];
               pdata.iuppers[box][0] = pdata.iuppers[box][0] + p * n[0];
               pdata.iuppers[box][1] = pdata.iuppers[box][1] + q * n[1];
               pdata.iuppers[box][2] = pdata.iuppers[box][2] + r * n[2];
            }

            i = 0;
            for (box = 0; box < pdata.graph_nboxes; box++)
            {
               MapProblemIndex(pdata.graph_ilowers[box], m);
               MapProblemIndex(pdata.graph_iuppers[box], m);
               mmap[0] = m[pdata.graph_index_maps[box][0]];
               mmap[1] = m[pdata.graph_index_maps[box][1]];
               mmap[2] = m[pdata.graph_index_maps[box][2]];
               MapProblemIndex(pdata.graph_to_ilowers[box], mmap);
               MapProblemIndex(pdata.graph_to_iuppers[box], mmap);

               for (b = 0; b < pdata.nboxes; b++)
               {
                  /* first convert the box extents based on vartype */
                  GetVariableBox(pdata.ilowers[b], pdata.iuppers[b],
                                 pdata.vartypes[pdata.graph_vars[box]],
                                 ilower, iupper);
                  size = IntersectBoxes(pdata.graph_ilowers[box],
                                        pdata.graph_iuppers[box],
                                        ilower, iupper,
                                        int_ilower, int_iupper);
                  if (size > 0)
                  {
                     /* if there is an intersection, it is the only one */
                     for (d = 0; d < 3; d++)
                     {
                        dmap = pdata.graph_index_maps[box][d];
                        sign = pdata.graph_index_signs[box][d];
                        pdata.graph_to_ilowers[i][dmap] =
                           pdata.graph_to_ilowers[box][dmap] +
                           sign * pdata.graph_to_strides[box][d] *
                           ((int_ilower[d] - pdata.graph_ilowers[box][d]) /
                            pdata.graph_strides[box][d]);
                        pdata.graph_to_iuppers[i][dmap] =
                           pdata.graph_to_iuppers[box][dmap] +
                           sign * pdata.graph_to_strides[box][d] *
                           ((int_iupper[d] - pdata.graph_iuppers[box][d]) /
                            pdata.graph_strides[box][d]);
                        pdata.graph_ilowers[i][d] = int_ilower[d];
                        pdata.graph_iuppers[i][d] = int_iupper[d];
                        pdata.graph_strides[i][d] =
                           pdata.graph_strides[box][d];
                        pdata.graph_to_strides[i][d] =
                           pdata.graph_to_strides[box][d];
                        pdata.graph_index_maps[i][d]  = dmap;
                        pdata.graph_index_signs[i][d] = sign;
                     }
                     for (d = 3; d < 9; d++)
                     {
                        pdata.graph_ilowers[i][d] =
                           pdata.graph_ilowers[box][d];
                        pdata.graph_iuppers[i][d] =
                           pdata.graph_iuppers[box][d];
                        pdata.graph_to_ilowers[i][d] =
                           pdata.graph_to_ilowers[box][d];
                        pdata.graph_to_iuppers[i][d] =
                           pdata.graph_to_iuppers[box][d];
                     }
                     pdata.graph_vars[i]     = pdata.graph_vars[box];
                     pdata.graph_to_parts[i] = pdata.graph_to_parts[box];
                     pdata.graph_to_vars[i]  = pdata.graph_to_vars[box];
                     pdata.graph_entries[i]  = pdata.graph_entries[box];
                     pdata.graph_values[i]   = pdata.graph_values[box];
                     i++;
                     break;
                  }
               }
            }
            pdata.graph_nboxes = i;

            i = 0;
            for (box = 0; box < pdata.matset_nboxes; box++)
            {
               MapProblemIndex(pdata.matset_ilowers[box], m);
               MapProblemIndex(pdata.matset_iuppers[box], m);

               for (b = 0; b < pdata.nboxes; b++)
               {
                  /* first convert the box extents based on vartype */
                  GetVariableBox(pdata.ilowers[b], pdata.iuppers[b],
                                 pdata.vartypes[pdata.matset_vars[box]],
                                 ilower, iupper);
                  size = IntersectBoxes(pdata.matset_ilowers[box],
                                        pdata.matset_iuppers[box],
                                        ilower, iupper,
                                        int_ilower, int_iupper);
                  if (size > 0)
                  {
                     /* if there is an intersection, it is the only one */
                     for (d = 0; d < 3; d++)
                     {
                        pdata.matset_ilowers[i][d] = int_ilower[d];
                        pdata.matset_iuppers[i][d] = int_iupper[d];
                        pdata.matset_strides[i][d] =
                           pdata.matset_strides[box][d];
                     }
                     for (d = 3; d < 9; d++)
                     {
                        pdata.matset_ilowers[i][d] =
                           pdata.matset_ilowers[box][d];
                        pdata.matset_iuppers[i][d] =
                           pdata.matset_iuppers[box][d];
                     }
                     pdata.matset_vars[i]     = pdata.matset_vars[box];
                     pdata.matset_entries[i]  = pdata.matset_entries[box];
                     pdata.matset_values[i]   = pdata.matset_values[box];
                     i++;
                     break;
                  }
               }
            }
            pdata.matset_nboxes = i;

            i = 0;
            for (box = 0; box < pdata.matadd_nboxes; box++)
            {
               MapProblemIndex(pdata.matadd_ilowers[box], m);
               MapProblemIndex(pdata.matadd_iuppers[box], m);

               for (b = 0; b < pdata.nboxes; b++)
               {
                  /* first convert the box extents based on vartype */
                  GetVariableBox(pdata.ilowers[b], pdata.iuppers[b],
                                 pdata.vartypes[pdata.matadd_vars[box]],
                                 ilower, iupper);
                  size = IntersectBoxes(pdata.matadd_ilowers[box],
                                        pdata.matadd_iuppers[box],
                                        ilower, iupper,
                                        int_ilower, int_iupper);
                  if (size > 0)
                  {
                     /* if there is an intersection, it is the only one */
                     for (d = 0; d < 3; d++)
                     {
                        pdata.matadd_ilowers[i][d] = int_ilower[d];
                        pdata.matadd_iuppers[i][d] = int_iupper[d];
                     }
                     for (d = 3; d < 9; d++)
                     {
                        pdata.matadd_ilowers[i][d] =
                           pdata.matadd_ilowers[box][d];
                        pdata.matadd_iuppers[i][d] =
                           pdata.matadd_iuppers[box][d];
                     }
                     pdata.matadd_vars[i]     = pdata.matadd_vars[box];
                     pdata.matadd_nentries[i] = pdata.matadd_nentries[box];
                     iptr = pdata.matadd_entries[i];
                     pdata.matadd_entries[i] = pdata.matadd_entries[box];
                     pdata.matadd_entries[box] = iptr;
                     dptr = pdata.matadd_values[i];
                     pdata.matadd_values[i] = pdata.matadd_values[box];
                     pdata.matadd_values[box] = dptr;
                     i++;
                     break;
                  }
               }
            }
            for (box = i; box < pdata.matadd_nboxes; box++)
            {
               hypre_TFree(pdata.matadd_entries[box], HYPRE_MEMORY_HOST);
               hypre_TFree(pdata.matadd_values[box], HYPRE_MEMORY_HOST);
            }
            pdata.matadd_nboxes = i;

            i = 0;
            for (box = 0; box < pdata.fem_matadd_nboxes; box++)
            {
               MapProblemIndex(pdata.fem_matadd_ilowers[box], m);
               MapProblemIndex(pdata.fem_matadd_iuppers[box], m);

               for (b = 0; b < pdata.nboxes; b++)
               {
                  /* fe is cell-based, so no need to convert box extents */
                  size = IntersectBoxes(pdata.fem_matadd_ilowers[box],
                                        pdata.fem_matadd_iuppers[box],
                                        pdata.ilowers[b], pdata.iuppers[b],
                                        int_ilower, int_iupper);
                  if (size > 0)
                  {
                     /* if there is an intersection, it is the only one */
                     for (d = 0; d < 3; d++)
                     {
                        pdata.fem_matadd_ilowers[i][d] = int_ilower[d];
                        pdata.fem_matadd_iuppers[i][d] = int_iupper[d];
                     }
                     for (d = 3; d < 9; d++)
                     {
                        pdata.fem_matadd_ilowers[i][d] =
                           pdata.fem_matadd_ilowers[box][d];
                        pdata.fem_matadd_iuppers[i][d] =
                           pdata.fem_matadd_iuppers[box][d];
                     }
                     pdata.fem_matadd_nrows[i]  = pdata.fem_matadd_nrows[box];
                     iptr = pdata.fem_matadd_rows[box];
                     iptr = pdata.fem_matadd_rows[i];
                     pdata.fem_matadd_rows[i] = pdata.fem_matadd_rows[box];
                     pdata.fem_matadd_rows[box] = iptr;
                     pdata.fem_matadd_ncols[i]  = pdata.fem_matadd_ncols[box];
                     iptr = pdata.fem_matadd_cols[i];
                     pdata.fem_matadd_cols[i] = pdata.fem_matadd_cols[box];
                     pdata.fem_matadd_cols[box] = iptr;
                     dptr = pdata.fem_matadd_values[i];
                     pdata.fem_matadd_values[i] = pdata.fem_matadd_values[box];
                     pdata.fem_matadd_values[box] = dptr;
                     i++;
                     break;
                  }
               }
            }
            for (box = i; box < pdata.fem_matadd_nboxes; box++)
            {
               hypre_TFree(pdata.fem_matadd_rows[box], HYPRE_MEMORY_HOST);
               hypre_TFree(pdata.fem_matadd_cols[box], HYPRE_MEMORY_HOST);
               hypre_TFree(pdata.fem_matadd_values[box], HYPRE_MEMORY_HOST);
            }
            pdata.fem_matadd_nboxes = i;

            i = 0;
            for (box = 0; box < pdata.rhsadd_nboxes; box++)
            {
               MapProblemIndex(pdata.rhsadd_ilowers[box], m);
               MapProblemIndex(pdata.rhsadd_iuppers[box], m);

               for (b = 0; b < pdata.nboxes; b++)
               {
                  /* first convert the box extents based on vartype */
                  GetVariableBox(pdata.ilowers[b], pdata.iuppers[b],
                                 pdata.vartypes[pdata.rhsadd_vars[box]],
                                 ilower, iupper);
                  size = IntersectBoxes(pdata.rhsadd_ilowers[box],
                                        pdata.rhsadd_iuppers[box],
                                        ilower, iupper,
                                        int_ilower, int_iupper);
                  if (size > 0)
                  {
                     /* if there is an intersection, it is the only one */
                     for (d = 0; d < 3; d++)
                     {
                        pdata.rhsadd_ilowers[i][d] = int_ilower[d];
                        pdata.rhsadd_iuppers[i][d] = int_iupper[d];
                     }
                     for (d = 3; d < 9; d++)
                     {
                        pdata.rhsadd_ilowers[i][d] =
                           pdata.rhsadd_ilowers[box][d];
                        pdata.rhsadd_iuppers[i][d] =
                           pdata.rhsadd_iuppers[box][d];
                     }
                     pdata.rhsadd_vars[i]   = pdata.rhsadd_vars[box];
                     pdata.rhsadd_values[i] = pdata.rhsadd_values[box];
                     i++;
                     break;
                  }
               }
            }
            pdata.rhsadd_nboxes = i;

            i = 0;
            for (box = 0; box < pdata.fem_rhsadd_nboxes; box++)
            {
               MapProblemIndex(pdata.fem_rhsadd_ilowers[box], m);
               MapProblemIndex(pdata.fem_rhsadd_iuppers[box], m);

               for (b = 0; b < pdata.nboxes; b++)
               {
                  /* fe is cell-based, so no need to convert box extents */
                  size = IntersectBoxes(pdata.fem_rhsadd_ilowers[box],
                                        pdata.fem_rhsadd_iuppers[box],
                                        pdata.ilowers[b], pdata.iuppers[b],
                                        int_ilower, int_iupper);
                  if (size > 0)
                  {
                     /* if there is an intersection, it is the only one */
                     for (d = 0; d < 3; d++)
                     {
                        pdata.fem_rhsadd_ilowers[i][d] = int_ilower[d];
                        pdata.fem_rhsadd_iuppers[i][d] = int_iupper[d];
                     }
                     for (d = 3; d < 9; d++)
                     {
                        pdata.fem_rhsadd_ilowers[i][d] =
                           pdata.fem_rhsadd_ilowers[box][d];
                        pdata.fem_rhsadd_iuppers[i][d] =
                           pdata.fem_rhsadd_iuppers[box][d];
                     }
                     dptr = pdata.fem_rhsadd_values[i];
                     pdata.fem_rhsadd_values[i] = pdata.fem_rhsadd_values[box];
                     pdata.fem_rhsadd_values[box] = dptr;
                     i++;
                     break;
                  }
               }
            }
            for (box = i; box < pdata.fem_rhsadd_nboxes; box++)
            {
               hypre_TFree(pdata.fem_rhsadd_values[box], HYPRE_MEMORY_HOST);
            }
            pdata.fem_rhsadd_nboxes = i;
         }

         /* refine and block boxes */
         m[0] = block[part][0];
         m[1] = block[part][1];
         m[2] = block[part][2];
         if ( (m[0] * m[1] * m[2]) > 1)
         {
            pdata.ilowers = hypre_TReAlloc(pdata.ilowers,  ProblemIndex,
                                           m[0] * m[1] * m[2] * pdata.nboxes, HYPRE_MEMORY_HOST);
            pdata.iuppers = hypre_TReAlloc(pdata.iuppers,  ProblemIndex,
                                           m[0] * m[1] * m[2] * pdata.nboxes, HYPRE_MEMORY_HOST);
            pdata.boxsizes = hypre_TReAlloc(pdata.boxsizes,  HYPRE_Int,
                                            m[0] * m[1] * m[2] * pdata.nboxes, HYPRE_MEMORY_HOST);
            for (box = 0; box < pdata.nboxes; box++)
            {
               n[0] = pdata.iuppers[box][0] - pdata.ilowers[box][0] + 1;
               n[1] = pdata.iuppers[box][1] - pdata.ilowers[box][1] + 1;
               n[2] = pdata.iuppers[box][2] - pdata.ilowers[box][2] + 1;

               MapProblemIndex(pdata.ilowers[box], m);

               MapProblemIndex(pdata.iuppers[box], m);
               pdata.iuppers[box][0] = pdata.ilowers[box][0] + n[0] - 1;
               pdata.iuppers[box][1] = pdata.ilowers[box][1] + n[1] - 1;
               pdata.iuppers[box][2] = pdata.ilowers[box][2] + n[2] - 1;

               i = box;
               for (r = 0; r < m[2]; r++)
               {
                  for (q = 0; q < m[1]; q++)
                  {
                     for (p = 0; p < m[0]; p++)
                     {
                        pdata.ilowers[i][0] = pdata.ilowers[box][0] + p * n[0];
                        pdata.ilowers[i][1] = pdata.ilowers[box][1] + q * n[1];
                        pdata.ilowers[i][2] = pdata.ilowers[box][2] + r * n[2];
                        pdata.iuppers[i][0] = pdata.iuppers[box][0] + p * n[0];
                        pdata.iuppers[i][1] = pdata.iuppers[box][1] + q * n[1];
                        pdata.iuppers[i][2] = pdata.iuppers[box][2] + r * n[2];
                        for (d = 3; d < 9; d++)
                        {
                           pdata.ilowers[i][d] = pdata.ilowers[box][d];
                           pdata.iuppers[i][d] = pdata.iuppers[box][d];
                        }
                        i += pdata.nboxes;
                     }
                  }
               }
            }
            pdata.nboxes *= m[0] * m[1] * m[2];

            for (box = 0; box < pdata.graph_nboxes; box++)
            {
               MapProblemIndex(pdata.graph_ilowers[box], m);
               MapProblemIndex(pdata.graph_iuppers[box], m);
               mmap[0] = m[pdata.graph_index_maps[box][0]];
               mmap[1] = m[pdata.graph_index_maps[box][1]];
               mmap[2] = m[pdata.graph_index_maps[box][2]];
               MapProblemIndex(pdata.graph_to_ilowers[box], mmap);
               MapProblemIndex(pdata.graph_to_iuppers[box], mmap);
            }
            for (box = 0; box < pdata.matset_nboxes; box++)
            {
               MapProblemIndex(pdata.matset_ilowers[box], m);
               MapProblemIndex(pdata.matset_iuppers[box], m);
            }
            for (box = 0; box < pdata.matadd_nboxes; box++)
            {
               MapProblemIndex(pdata.matadd_ilowers[box], m);
               MapProblemIndex(pdata.matadd_iuppers[box], m);
            }
            for (box = 0; box < pdata.fem_matadd_nboxes; box++)
            {
               MapProblemIndex(pdata.fem_matadd_ilowers[box], m);
               MapProblemIndex(pdata.fem_matadd_iuppers[box], m);
            }
            for (box = 0; box < pdata.rhsadd_nboxes; box++)
            {
               MapProblemIndex(pdata.rhsadd_ilowers[box], m);
               MapProblemIndex(pdata.rhsadd_iuppers[box], m);
            }
            for (box = 0; box < pdata.fem_rhsadd_nboxes; box++)
            {
               MapProblemIndex(pdata.fem_rhsadd_ilowers[box], m);
               MapProblemIndex(pdata.fem_rhsadd_iuppers[box], m);
            }
         }

         /* map remaining ilowers & iuppers */
         m[0] = refine[part][0] * block[part][0] * distribute[part][0];
         m[1] = refine[part][1] * block[part][1] * distribute[part][1];
         m[2] = refine[part][2] * block[part][2] * distribute[part][2];
         if ( (m[0] * m[1] * m[2]) > 1)
         {
            for (box = 0; box < pdata.glue_nboxes; box++)
            {
               MapProblemIndex(pdata.glue_ilowers[box], m);
               MapProblemIndex(pdata.glue_iuppers[box], m);
               mmap[0] = m[pdata.glue_index_maps[box][0]];
               mmap[1] = m[pdata.glue_index_maps[box][1]];
               mmap[2] = m[pdata.glue_index_maps[box][2]];
               MapProblemIndex(pdata.glue_nbor_ilowers[box], mmap);
               MapProblemIndex(pdata.glue_nbor_iuppers[box], mmap);
            }
         }

         /* compute box sizes, etc. */
         pdata.max_boxsize = 0;
         for (box = 0; box < pdata.nboxes; box++)
         {
            pdata.boxsizes[box] = 1;
            for (i = 0; i < 3; i++)
            {
               pdata.boxsizes[box] *=
                  (pdata.iuppers[box][i] - pdata.ilowers[box][i] + 2);
            }
            pdata.max_boxsize =
               hypre_max(pdata.max_boxsize, pdata.boxsizes[box]);
         }
         for (box = 0; box < pdata.graph_nboxes; box++)
         {
            pdata.graph_boxsizes[box] = 1;
            for (i = 0; i < 3; i++)
            {
               pdata.graph_boxsizes[box] *=
                  (pdata.graph_iuppers[box][i] -
                   pdata.graph_ilowers[box][i] + 1);
            }
         }
         for (box = 0; box < pdata.matset_nboxes; box++)
         {
            size = 1;
            for (i = 0; i < 3; i++)
            {
               size *= (pdata.matset_iuppers[box][i] -
                        pdata.matset_ilowers[box][i] + 1);
            }
            pdata.max_boxsize = hypre_max(pdata.max_boxsize, size);
         }
         for (box = 0; box < pdata.matadd_nboxes; box++)
         {
            size = 1;
            for (i = 0; i < 3; i++)
            {
               size *= (pdata.matadd_iuppers[box][i] -
                        pdata.matadd_ilowers[box][i] + 1);
            }
            pdata.max_boxsize = hypre_max(pdata.max_boxsize, size);
         }
         for (box = 0; box < pdata.fem_matadd_nboxes; box++)
         {
            size = 1;
            for (i = 0; i < 3; i++)
            {
               size *= (pdata.fem_matadd_iuppers[box][i] -
                        pdata.fem_matadd_ilowers[box][i] + 1);
            }
            pdata.max_boxsize = hypre_max(pdata.max_boxsize, size);
         }
         for (box = 0; box < pdata.rhsadd_nboxes; box++)
         {
            size = 1;
            for (i = 0; i < 3; i++)
            {
               size *= (pdata.rhsadd_iuppers[box][i] -
                        pdata.rhsadd_ilowers[box][i] + 1);
            }
            pdata.max_boxsize = hypre_max(pdata.max_boxsize, size);
         }
         for (box = 0; box < pdata.fem_rhsadd_nboxes; box++)
         {
            size = 1;
            for (i = 0; i < 3; i++)
            {
               size *= (pdata.fem_rhsadd_iuppers[box][i] -
                        pdata.fem_rhsadd_ilowers[box][i] + 1);
            }
            pdata.max_boxsize = hypre_max(pdata.max_boxsize, size);
         }

         /* refine periodicity */
         pdata.periodic[0] *= refine[part][0] * block[part][0] * distribute[part][0];
         pdata.periodic[1] *= refine[part][1] * block[part][1] * distribute[part][1];
         pdata.periodic[2] *= refine[part][2] * block[part][2] * distribute[part][2];
      }

      if (pdata.nboxes == 0)
      {
         hypre_TFree(pdata.ilowers, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.iuppers, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.boxsizes, HYPRE_MEMORY_HOST);
         pdata.max_boxsize = 0;
      }

      if (pdata.glue_nboxes == 0)
      {
         hypre_TFree(pdata.glue_shared, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.glue_ilowers, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.glue_iuppers, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.glue_offsets, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.glue_nbor_parts, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.glue_nbor_ilowers, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.glue_nbor_iuppers, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.glue_nbor_offsets, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.glue_index_maps, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.glue_index_dirs, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.glue_primaries, HYPRE_MEMORY_HOST);
      }

      if (pdata.graph_nboxes == 0)
      {
         hypre_TFree(pdata.graph_ilowers, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.graph_iuppers, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.graph_strides, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.graph_vars, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.graph_to_parts, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.graph_to_ilowers, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.graph_to_iuppers, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.graph_to_strides, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.graph_to_vars, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.graph_index_maps, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.graph_index_signs, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.graph_entries, HYPRE_MEMORY_HOST);
         pdata.graph_values_size = 0;
         hypre_TFree(pdata.graph_values, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.d_graph_values, memory_location);
         hypre_TFree(pdata.graph_boxsizes, HYPRE_MEMORY_HOST);
      }

      if (pdata.matset_nboxes == 0)
      {
         hypre_TFree(pdata.matset_ilowers, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.matset_iuppers, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.matset_strides, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.matset_vars, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.matset_entries, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.matset_values, HYPRE_MEMORY_HOST);
      }

      if (pdata.matadd_nboxes == 0)
      {
         hypre_TFree(pdata.matadd_ilowers, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.matadd_iuppers, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.matadd_vars, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.matadd_nentries, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.matadd_entries, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.matadd_values, HYPRE_MEMORY_HOST);
      }

      if (pdata.fem_matadd_nboxes == 0)
      {
         hypre_TFree(pdata.fem_matadd_ilowers, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.fem_matadd_iuppers, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.fem_matadd_nrows, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.fem_matadd_ncols, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.fem_matadd_rows, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.fem_matadd_cols, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.fem_matadd_values, HYPRE_MEMORY_HOST);
      }

      if (pdata.rhsadd_nboxes == 0)
      {
         hypre_TFree(pdata.rhsadd_ilowers, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.rhsadd_iuppers, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.rhsadd_vars, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.rhsadd_values, HYPRE_MEMORY_HOST);
      }

      if (pdata.fem_rhsadd_nboxes == 0)
      {
         hypre_TFree(pdata.fem_rhsadd_ilowers, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.fem_rhsadd_iuppers, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.fem_rhsadd_values, HYPRE_MEMORY_HOST);
      }

      data.pdata[part] = pdata;
   }

   data.max_boxsize = 0;
   for (part = 0; part < data.nparts; part++)
   {
      data.max_boxsize =
         hypre_max(data.max_boxsize, data.pdata[part].max_boxsize);
   }

   hypre_TFree(pool_procs, HYPRE_MEMORY_HOST);

   *data_ptr = data;
   return 0;
}

/*--------------------------------------------------------------------------
 * Destroy data
 *--------------------------------------------------------------------------*/

HYPRE_Int
DestroyData( ProblemData   data )
{
   HYPRE_MemoryLocation memory_location = data.memory_location;
   ProblemPartData  pdata;
   HYPRE_Int        part, box, s, i;

   for (part = 0; part < data.nparts; part++)
   {
      pdata = data.pdata[part];

      if (pdata.nboxes > 0)
      {
         hypre_TFree(pdata.ilowers, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.iuppers, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.boxsizes, HYPRE_MEMORY_HOST);
      }

      if (pdata.nvars > 0)
      {
         hypre_TFree(pdata.vartypes, HYPRE_MEMORY_HOST);
      }

      if (pdata.add_nvars > 0)
      {
         hypre_TFree(pdata.add_indexes, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.add_vartypes, HYPRE_MEMORY_HOST);
      }

      if (pdata.glue_nboxes > 0)
      {
         hypre_TFree(pdata.glue_shared, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.glue_ilowers, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.glue_iuppers, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.glue_offsets, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.glue_nbor_parts, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.glue_nbor_ilowers, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.glue_nbor_iuppers, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.glue_nbor_offsets, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.glue_index_maps, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.glue_index_dirs, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.glue_primaries, HYPRE_MEMORY_HOST);
      }

      if (pdata.nvars > 0)
      {
         hypre_TFree(pdata.stencil_num, HYPRE_MEMORY_HOST);
      }

      if (pdata.graph_nboxes > 0)
      {
         hypre_TFree(pdata.graph_ilowers, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.graph_iuppers, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.graph_strides, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.graph_vars, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.graph_to_parts, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.graph_to_ilowers, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.graph_to_iuppers, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.graph_to_strides, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.graph_to_vars, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.graph_index_maps, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.graph_index_signs, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.graph_entries, HYPRE_MEMORY_HOST);
         pdata.graph_values_size = 0;
         hypre_TFree(pdata.graph_values, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.d_graph_values, memory_location);
         hypre_TFree(pdata.graph_boxsizes, HYPRE_MEMORY_HOST);
      }

      if (pdata.matset_nboxes > 0)
      {
         hypre_TFree(pdata.matset_ilowers, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.matset_iuppers, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.matset_strides, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.matset_vars, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.matset_entries, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.matset_values, HYPRE_MEMORY_HOST);
      }

      if (pdata.matadd_nboxes > 0)
      {
         hypre_TFree(pdata.matadd_ilowers, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.matadd_iuppers, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.matadd_vars, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.matadd_nentries, HYPRE_MEMORY_HOST);
         for (box = 0; box < pdata.matadd_nboxes; box++)
         {
            hypre_TFree(pdata.matadd_entries[box], HYPRE_MEMORY_HOST);
            hypre_TFree(pdata.matadd_values[box], HYPRE_MEMORY_HOST);
         }
         hypre_TFree(pdata.matadd_entries, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.matadd_values, HYPRE_MEMORY_HOST);
      }

      if (pdata.fem_matadd_nboxes > 0)
      {
         hypre_TFree(pdata.fem_matadd_ilowers, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.fem_matadd_iuppers, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.fem_matadd_nrows, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.fem_matadd_ncols, HYPRE_MEMORY_HOST);
         for (box = 0; box < pdata.fem_matadd_nboxes; box++)
         {
            hypre_TFree(pdata.fem_matadd_rows[box], HYPRE_MEMORY_HOST);
            hypre_TFree(pdata.fem_matadd_cols[box], HYPRE_MEMORY_HOST);
            hypre_TFree(pdata.fem_matadd_values[box], HYPRE_MEMORY_HOST);
         }
         hypre_TFree(pdata.fem_matadd_rows, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.fem_matadd_cols, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.fem_matadd_values, HYPRE_MEMORY_HOST);
      }

      if (pdata.rhsadd_nboxes > 0)
      {
         hypre_TFree(pdata.rhsadd_ilowers, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.rhsadd_iuppers, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.rhsadd_vars, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.rhsadd_values, HYPRE_MEMORY_HOST);
      }

      if (pdata.fem_rhsadd_nboxes > 0)
      {
         hypre_TFree(pdata.fem_rhsadd_ilowers, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.fem_rhsadd_iuppers, HYPRE_MEMORY_HOST);
         for (box = 0; box < pdata.fem_rhsadd_nboxes; box++)
         {
            hypre_TFree(pdata.fem_rhsadd_values[box], HYPRE_MEMORY_HOST);
         }
         hypre_TFree(pdata.fem_rhsadd_values, HYPRE_MEMORY_HOST);
      }
   }
   hypre_TFree(data.pdata, HYPRE_MEMORY_HOST);

   hypre_TFree(data.numghost, HYPRE_MEMORY_HOST);

   if (data.nstencils > 0)
   {
      for (s = 0; s < data.nstencils; s++)
      {
         hypre_TFree(data.stencil_offsets[s], HYPRE_MEMORY_HOST);
         hypre_TFree(data.stencil_vars[s], HYPRE_MEMORY_HOST);
         hypre_TFree(data.stencil_values[s], HYPRE_MEMORY_HOST);
      }
      hypre_TFree(data.stencil_sizes, HYPRE_MEMORY_HOST);
      hypre_TFree(data.stencil_offsets, HYPRE_MEMORY_HOST);
      hypre_TFree(data.stencil_vars, HYPRE_MEMORY_HOST);
      hypre_TFree(data.stencil_values, HYPRE_MEMORY_HOST);
   }

   if (data.fem_nvars > 0)
   {
      for (s = 0; s < data.fem_nvars; s++)
      {
         hypre_TFree(data.fem_values_full[s], HYPRE_MEMORY_HOST);
         hypre_TFree(data.fem_ivalues_full[s], HYPRE_MEMORY_HOST);
      }
      hypre_TFree(data.fem_offsets, HYPRE_MEMORY_HOST);
      hypre_TFree(data.fem_vars, HYPRE_MEMORY_HOST);
      hypre_TFree(data.fem_values_full, HYPRE_MEMORY_HOST);
      hypre_TFree(data.fem_ivalues_full, HYPRE_MEMORY_HOST);
      hypre_TFree(data.fem_ordering, HYPRE_MEMORY_HOST);
      hypre_TFree(data.fem_sparsity, HYPRE_MEMORY_HOST);
      hypre_TFree(data.fem_values, HYPRE_MEMORY_HOST);
      hypre_TFree(data.d_fem_values, memory_location);
   }

   if (data.fem_rhs_true > 0)
   {
      hypre_TFree(data.fem_rhs_values, HYPRE_MEMORY_HOST);
      hypre_TFree(data.d_fem_rhs_values, memory_location);
   }

   if (data.symmetric_num > 0)
   {
      hypre_TFree(data.symmetric_parts, HYPRE_MEMORY_HOST);
      hypre_TFree(data.symmetric_vars, HYPRE_MEMORY_HOST);
      hypre_TFree(data.symmetric_to_vars, HYPRE_MEMORY_HOST);
      hypre_TFree(data.symmetric_booleans, HYPRE_MEMORY_HOST);
   }

   for (i = 0; i < data.ndists; i++)
   {
      hypre_TFree(data.dist_pools[i], HYPRE_MEMORY_HOST);
   }
   hypre_TFree(data.dist_pools, HYPRE_MEMORY_HOST);
   hypre_TFree(data.dist_npools, HYPRE_MEMORY_HOST);

   return 0;
}

/*--------------------------------------------------------------------------
 * Routine to load cosine function
 *--------------------------------------------------------------------------*/

HYPRE_Int
SetCosineVector(HYPRE_Real  scale,
                Index       ilower,
                Index       iupper,
                HYPRE_Real *values)
{
   HYPRE_Int  i, j, k;
   HYPRE_Int  count = 0;

   for (k = ilower[2]; k <= iupper[2]; k++)
   {
      for (j = ilower[1]; j <= iupper[1]; j++)
      {
         for (i = ilower[0]; i <= iupper[0]; i++)
         {
            values[count] = scale * hypre_cos((i + j + k) / 10.0);
            count++;
         }
      }
   }

   return (0);
}

/*--------------------------------------------------------------------------
 * Print usage info
 *--------------------------------------------------------------------------*/

HYPRE_Int
PrintUsage( char *progname,
            HYPRE_Int   myid )
{
   if ( myid == 0 )
   {
      hypre_printf("\n");
      hypre_printf("Usage: %s [-in <filename>] [<options>]\n", progname);
      hypre_printf("       %s -help | -version | -vernum \n", progname);
      hypre_printf("\n");
      hypre_printf("  -in <filename> : input file (default is `%s')\n",
                   infile_default);
      hypre_printf("  -fromfile <filename> : read SStructMatrix from file\n");
      hypre_printf("\n");
      hypre_printf("  -pt <pt1> <pt2> ... : set part(s) for subsequent options\n");
      hypre_printf("  -pooldist <p>       : pool distribution to use\n");
      hypre_printf("  -r <rx> <ry> <rz>   : refine part(s)\n");
      hypre_printf("  -P <Px> <Py> <Pz>   : refine and distribute part(s)\n");
      hypre_printf("  -b <bx> <by> <bz>   : refine and block part(s)\n");
      hypre_printf("  -solver <ID>        : solver ID (default = 39)\n");
      hypre_printf("                         0 - SMG split solver\n");
      hypre_printf("                         1 - PFMG split solver\n");
      hypre_printf("                         3 - SysPFMG\n");
      hypre_printf("                         8 - 1-step Jacobi split solver\n");
      hypre_printf("                        10 - PCG with SMG split precond\n");
      hypre_printf("                        11 - PCG with PFMG split precond\n");
      hypre_printf("                        13 - PCG with SysPFMG precond\n");
      hypre_printf("                        18 - PCG with diagonal scaling\n");
      hypre_printf("                        19 - PCG\n");
      hypre_printf("                        20 - PCG with BoomerAMG precond\n");
      hypre_printf("                        21 - PCG with EUCLID precond\n");
      hypre_printf("                        22 - PCG with ParaSails precond\n");
      hypre_printf("                        28 - PCG with diagonal scaling\n");
      hypre_printf("                        30 - GMRES with SMG split precond\n");
      hypre_printf("                        31 - GMRES with PFMG split precond\n");
      hypre_printf("                        38 - GMRES with diagonal scaling\n");
      hypre_printf("                        39 - GMRES\n");
      hypre_printf("                        40 - GMRES with BoomerAMG precond\n");
      hypre_printf("                        41 - GMRES with EUCLID precond\n");
      hypre_printf("                        42 - GMRES with ParaSails precond\n");
      hypre_printf("                        50 - BiCGSTAB with SMG split precond\n");
      hypre_printf("                        51 - BiCGSTAB with PFMG split precond\n");
      hypre_printf("                        58 - BiCGSTAB with diagonal scaling\n");
      hypre_printf("                        59 - BiCGSTAB\n");
      hypre_printf("                        60 - BiCGSTAB with BoomerAMG precond\n");
      hypre_printf("                        61 - BiCGSTAB with EUCLID precond\n");
      hypre_printf("                        62 - BiCGSTAB with ParaSails precond\n");
      hypre_printf("                        70 - Flexible GMRES with SMG split precond\n");
      hypre_printf("                        71 - Flexible GMRES with PFMG split precond\n");
      hypre_printf("                        78 - Flexible GMRES with diagonal scaling\n");
      hypre_printf("                        80 - Flexible GMRES with BoomerAMG precond\n");
      hypre_printf("                        90 - LGMRES with BoomerAMG precond\n");
      hypre_printf("                        120- ParCSRHybrid with DSCG/BoomerAMG precond\n");
      hypre_printf("                        150- AMS solver\n");
      hypre_printf("                        200- Struct SMG\n");
      hypre_printf("                        201- Struct PFMG\n");
      hypre_printf("                        202- Struct SparseMSG\n");
      hypre_printf("                        203- Struct PFMG constant coefficients\n");
      hypre_printf("                        204- Struct PFMG constant coefficients variable diagonal\n");
      hypre_printf("                        205- Struct Cyclic Reduction\n");
      hypre_printf("                        208- Struct Jacobi\n");
      hypre_printf("                        210- Struct CG with SMG precond\n");
      hypre_printf("                        211- Struct CG with PFMG precond\n");
      hypre_printf("                        212- Struct CG with SparseMSG precond\n");
      hypre_printf("                        217- Struct CG with 2-step Jacobi\n");
      hypre_printf("                        218- Struct CG with diagonal scaling\n");
      hypre_printf("                        219- Struct CG\n");
      hypre_printf("                        220- Struct Hybrid with SMG precond\n");
      hypre_printf("                        221- Struct Hybrid with PFMG precond\n");
      hypre_printf("                        222- Struct Hybrid with SparseMSG precond\n");
      hypre_printf("                        230- Struct GMRES with SMG precond\n");
      hypre_printf("                        231- Struct GMRES with PFMG precond\n");
      hypre_printf("                        232- Struct GMRES with SparseMSG precond\n");
      hypre_printf("                        237- Struct GMRES with 2-step Jacobi\n");
      hypre_printf("                        238- Struct GMRES with diagonal scaling\n");
      hypre_printf("                        239- Struct GMRES\n");
      hypre_printf("                        240- Struct BiCGSTAB with SMG precond\n");
      hypre_printf("                        241- Struct BiCGSTAB with PFMG precond\n");
      hypre_printf("                        242- Struct BiCGSTAB with SparseMSG precond\n");
      hypre_printf("                        247- Struct BiCGSTAB with 2-step Jacobi\n");
      hypre_printf("                        248- Struct BiCGSTAB with diagonal scaling\n");
      hypre_printf("                        249- Struct BiCGSTAB\n");
      hypre_printf("  -print             : print out the system\n");
      hypre_printf("  -rhsfromcosine     : solution is cosine function (default)\n");
      hypre_printf("  -rhsone            : rhs is vector with unit components\n");
      hypre_printf("  -tol <val>         : convergence tolerance (default 1e-6)\n");
      hypre_printf("  -solver_type <ID>  : Solver type for Hybrid\n");
      hypre_printf("                        1 - PCG (default)\n");
      hypre_printf("                        2 - GMRES\n");
      hypre_printf("                        3 - BiCGSTAB (only ParCSRHybrid)\n");
      hypre_printf("  -recompute <bool>  : Recompute residual in PCG?\n");
      hypre_printf("  -v <n_pre> <n_post>: SysPFMG and Struct- # of pre and post relax\n");
      hypre_printf("  -skip <s>          : SysPFMG and Struct- skip relaxation (0 or 1)\n");
      hypre_printf("  -rap <r>           : Struct- coarse grid operator type\n");
      hypre_printf("                        0 - Galerkin (default)\n");
      hypre_printf("                        1 - non-Galerkin ParFlow operators\n");
      hypre_printf("                        2 - Galerkin, general operators\n");
      hypre_printf("  -relax <r>         : Struct- relaxation type\n");
      hypre_printf("                        0 - Jacobi\n");
      hypre_printf("                        1 - Weighted Jacobi (default)\n");
      hypre_printf("                        2 - R/B Gauss-Seidel\n");
      hypre_printf("                        3 - R/B Gauss-Seidel (nonsymmetric)\n");
      hypre_printf("  -w <jacobi_weight> : jacobi weight\n");
      hypre_printf("  -jump <num>        : Struct- num levels to jump in SparseMSG\n");
      hypre_printf("  -cf <cf>           : Struct- convergence factor for Hybrid\n");
      hypre_printf("  -crtdim <tdim>     : Struct- cyclic reduction tdim\n");
      hypre_printf("  -cri <ix> <iy> <iz>: Struct- cyclic reduction base_index\n");
      hypre_printf("  -crs <sx> <sy> <sz>: Struct- cyclic reduction base_stride\n");
      hypre_printf("  -old_default: sets old BoomerAMG defaults, possibly better for 2D problems\n");

      /* begin lobpcg */

      hypre_printf("\nLOBPCG options:\n");
      hypre_printf("\n");
      hypre_printf("  -lobpcg            : run LOBPCG instead of PCG\n");
      hypre_printf("\n");
      hypre_printf("  -solver none       : no HYPRE preconditioner is used\n");
      hypre_printf("\n");
      hypre_printf("  -itr <val>         : maximal number of LOBPCG iterations (default 100);\n");
      hypre_printf("\n");
      hypre_printf("  -vrand <val>       : compute <val> eigenpairs using random initial vectors (default 1)\n");
      hypre_printf("\n");
      hypre_printf("  -seed <val>        : use <val> as the seed for the pseudo-random number generator\n");
      hypre_printf("                       (default seed is based on the time of the run)\n");
      hypre_printf("\n");
      hypre_printf("  -orthchk           : check eigenvectors for orthonormality\n");
      hypre_printf("\n");
      hypre_printf("  -verb <val>        : verbosity level\n");
      hypre_printf("  -verb 0            : no print\n");
      hypre_printf("  -verb 1            : print initial eigenvalues and residuals, iteration number, number of\n");
      hypre_printf("                       non-convergent eigenpairs and final eigenvalues and residuals (default)\n");
      hypre_printf("  -verb 2            : print eigenvalues and residuals on each iteration\n");
      hypre_printf("\n");
      hypre_printf("  -pcgitr <val>      : maximal number of inner PCG iterations for preconditioning (default 1);\n");
      hypre_printf("                       if <val> = 0 then the preconditioner is applied directly\n");
      hypre_printf("\n");
      hypre_printf("  -pcgtol <val>      : residual tolerance for inner iterations (default 0.01)\n");
      hypre_printf("\n");
      hypre_printf("  -vout <val>        : file output level\n");
      hypre_printf("  -vout 0            : no files created (default)\n");
      hypre_printf("  -vout 1            : write eigenvalues to values.txt and residuals to residuals.txt\n");
      hypre_printf("  -vout 2            : in addition to the above, write the eigenvalues history (the matrix whose\n");
      hypre_printf("                       i-th column contains eigenvalues at (i+1)-th iteration) to val_hist.txt and\n");
      hypre_printf("                       residuals history to res_hist.txt\n");
      hypre_printf("\nNOTE: in this test driver LOBPCG only works with solvers 10, 11, 13, and 18\n");
      hypre_printf("\ndefault solver is 10\n");

      /* end lobpcg */

      hypre_printf("\n");
   }

   return 0;
}

/*--------------------------------------------------------------------------
 * Test driver for semi-structured matrix interface
 *--------------------------------------------------------------------------*/

hypre_int
main( hypre_int argc,
      char *argv[] )
{
   MPI_Comm              comm = hypre_MPI_COMM_WORLD;

   char                 *infile;
   ProblemData           global_data;
   ProblemData           data;
   ProblemPartData       pdata;
   HYPRE_Int             nparts = 0;
   HYPRE_Int             pooldist;
   HYPRE_Int            *parts = NULL;
   Index                *refine = NULL;
   Index                *distribute = NULL;
   Index                *block = NULL;
   HYPRE_Int             solver_id, object_type;
   HYPRE_Int             print_system;
   HYPRE_Int             cosine;
   HYPRE_Real            scale;
   HYPRE_Int             read_fromfile_flag = 0;
   HYPRE_Int             read_fromfile_index[3] = {-1, -1, -1};

   HYPRE_SStructGrid     grid = NULL;
   HYPRE_SStructGrid     G_grid = NULL;
   HYPRE_SStructStencil *stencils = NULL;
   HYPRE_SStructStencil *G_stencils = NULL;
   HYPRE_SStructGraph    graph = NULL;
   HYPRE_SStructGraph    G_graph = NULL;
   HYPRE_SStructMatrix   A = NULL;
   HYPRE_SStructMatrix   G = NULL;
   HYPRE_SStructVector   b = NULL;
   HYPRE_SStructVector   x = NULL;
   HYPRE_SStructSolver   solver;
   HYPRE_SStructSolver   precond;

   HYPRE_ParCSRMatrix    par_A;
   HYPRE_ParVector       par_b;
   HYPRE_ParVector       par_x;
   HYPRE_Solver          par_solver;
   HYPRE_Solver          par_precond;

   HYPRE_StructMatrix    sA;
   HYPRE_StructVector    sb;
   HYPRE_StructVector    sx;
   HYPRE_StructSolver    struct_solver;
   HYPRE_StructSolver    struct_precond;

   Index                 ilower, iupper;
   Index                 index, to_index;

   HYPRE_Int             values_size = 0;
   HYPRE_Real           *values = NULL;
   HYPRE_Real           *d_values = NULL;

   HYPRE_Int             num_iterations;
   HYPRE_Real            final_res_norm;

   HYPRE_Int             num_procs, myid;
   HYPRE_Int             device_id = -1;
   HYPRE_Int             lazy_device_init = 0;
   HYPRE_Int             time_index;

   HYPRE_Int             n_pre, n_post;
   HYPRE_Int             skip;
   HYPRE_Int             rap;
   HYPRE_Int             relax;
   HYPRE_Real            jacobi_weight;
   HYPRE_Int             usr_jacobi_weight;
   HYPRE_Int             jump;
   HYPRE_Int             solver_type;
   HYPRE_Int             recompute_res;

   HYPRE_Real            cf_tol;

   HYPRE_Int             cycred_tdim;
   Index                 cycred_index, cycred_stride;

   HYPRE_Int             arg_index, part, var, box, s, entry, i, j, k, size;
   HYPRE_Int             row, col;
   HYPRE_Int             gradient_matrix;
   HYPRE_Int             old_default;

   /* begin lobpcg */

   HYPRE_SStructSolver   lobpcg_solver;

   HYPRE_Int lobpcgFlag = 0;
   HYPRE_Int lobpcgSeed = 0;
   HYPRE_Int blockSize = 1;
   HYPRE_Int verbosity = 1;
   HYPRE_Int iterations;
   HYPRE_Int maxIterations = 100;
   HYPRE_Int checkOrtho = 0;
   HYPRE_Int printLevel = 0;
   HYPRE_Int pcgIterations = 0;
   HYPRE_Int pcgMode = 0;
   HYPRE_Real tol = 1e-6;
   HYPRE_Real pcgTol = 1e-2;
   HYPRE_Real nonOrthF;

   FILE* filePtr;

   mv_MultiVectorPtr eigenvectors = NULL;
   mv_MultiVectorPtr constrains = NULL;
   HYPRE_Real* eigenvalues = NULL;

   HYPRE_Real* residuals;
   utilities_FortranMatrix* residualNorms;
   utilities_FortranMatrix* residualNormsHistory;
   utilities_FortranMatrix* eigenvaluesHistory;
   utilities_FortranMatrix* printBuffer;
   utilities_FortranMatrix* gramXX;
   utilities_FortranMatrix* identity;

   mv_InterfaceInterpreter* interpreter;
   HYPRE_MatvecFunctions matvec_fn;

   /* end lobpcg */

#if defined(HYPRE_USING_MEMORY_TRACKER)
   HYPRE_Int print_mem_tracker = 0;
   char mem_tracker_name[HYPRE_MAX_FILE_NAME_LEN] = {0};
#endif

#if defined(HYPRE_USING_GPU)
   HYPRE_Int spgemm_use_vendor = 0;
#endif

#if defined(HYPRE_TEST_USING_HOST)
   HYPRE_MemoryLocation memory_location = HYPRE_MEMORY_HOST;
   HYPRE_ExecutionPolicy default_exec_policy = HYPRE_EXEC_HOST;
#else
   HYPRE_MemoryLocation memory_location = HYPRE_MEMORY_DEVICE;
   HYPRE_ExecutionPolicy default_exec_policy = HYPRE_EXEC_DEVICE;
#endif

   global_data.memory_location = memory_location;

   HYPRE_Int gpu_aware_mpi = 0;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   /* Initialize MPI */
   hypre_MPI_Init(&argc, &argv);
   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &myid);

   /*-----------------------------------------------------------------
    * GPU Device binding
    * Must be done before HYPRE_Initialize() and should not be changed after
    *-----------------------------------------------------------------*/
   for (arg_index = 1; arg_index < argc; arg_index ++)
   {
      if (strcmp(argv[arg_index], "-lazy_device_init") == 0)
      {
         lazy_device_init = atoi(argv[++arg_index]);
      }
      else if (strcmp(argv[arg_index], "-device_id") == 0)
      {
         device_id = atoi(argv[++arg_index]);
      }
   }

   hypre_bind_device_id(device_id, myid, num_procs, comm);

   /*-----------------------------------------------------------
    * Initialize : must be the first HYPRE function to call
    *-----------------------------------------------------------*/
   HYPRE_Initialize();

   if (!lazy_device_init)
   {
      HYPRE_DeviceInitialize();
   }

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/

   skip  = 0;
   rap   = 0;
   relax = 1;
   jacobi_weight = 1.0;
   usr_jacobi_weight = 0;
   jump  = 0;
   gradient_matrix = 0;
   object_type = HYPRE_SSTRUCT;
   solver_type = 1;
   recompute_res = 0;   /* What should be the default here? */
   cf_tol = 0.90;
   pooldist = 0;
   cycred_tdim = 0;
   for (i = 0; i < 3; i++)
   {
      cycred_index[i]  = 0;
      cycred_stride[i] = 1;
   }

   solver_id = 39;
   print_system = 0;
   cosine = 1;
   skip = 0;
   n_pre  = 1;
   n_post = 1;

   old_default = 0;

   /*-----------------------------------------------------------
    * Read input file
    *-----------------------------------------------------------*/
   arg_index = 1;

   /* parse command line for input file name */
   infile = infile_default;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-in") == 0 )
      {
         arg_index++;
         infile = argv[arg_index++];
      }
      else if (strcmp(argv[arg_index], "-fromfile") == 0 )
      {
         arg_index++;
         read_fromfile_flag += 1;
         read_fromfile_index[0] = arg_index++;
      }
      else if (strcmp(argv[arg_index], "-rhsfromfile") == 0 )
      {
         arg_index++;
         read_fromfile_flag += 2;
         read_fromfile_index[1] = arg_index++;
      }
      else if (strcmp(argv[arg_index], "-x0fromfile") == 0 )
      {
         arg_index++;
         read_fromfile_flag += 4;
         read_fromfile_index[2] = arg_index++;
      }
      else if ( strcmp(argv[arg_index], "-help") == 0 )
      {
         PrintUsage(argv[0], myid);
         exit(1);
      }
      else if ( strcmp(argv[arg_index], "-version") == 0 )
      {
         char *version_string;
         HYPRE_Version(&version_string);
         hypre_printf("%s\n", version_string);
         hypre_TFree(version_string, HYPRE_MEMORY_HOST);
         exit(1);
      }
      else if ( strcmp(argv[arg_index], "-vernum") == 0 )
      {
         HYPRE_Int major, minor, patch, single;
         HYPRE_VersionNumber(&major, &minor, &patch, &single);
         hypre_printf("HYPRE Version %d.%d.%d\n", major, minor, patch);
         hypre_printf("HYPRE Single = %d\n", single);
         exit(1);
      }
      else
      {
         break;
      }
   }

   /*-----------------------------------------------------------
    * Are we reading matrices/vectors directly from file?
    *-----------------------------------------------------------*/

   if (read_fromfile_index[0] == -1 &&
       read_fromfile_index[1] == -1 &&
       read_fromfile_index[2] == -1)
   {
      ReadData(infile, &global_data);

      nparts = global_data.nparts;
      parts      = hypre_TAlloc(HYPRE_Int,  nparts, HYPRE_MEMORY_HOST);
      refine     = hypre_TAlloc(Index,  nparts, HYPRE_MEMORY_HOST);
      distribute = hypre_TAlloc(Index,  nparts, HYPRE_MEMORY_HOST);
      block      = hypre_TAlloc(Index,  nparts, HYPRE_MEMORY_HOST);
      for (part = 0; part < nparts; part++)
      {
         parts[part] = part;
         for (j = 0; j < 3; j++)
         {
            refine[part][j]     = 1;
            distribute[part][j] = 1;
            block[part][j]      = 1;
         }
      }

      if (global_data.rhs_true || global_data.fem_rhs_true)
      {
         cosine = 0;
      }
   }
   else
   {
      if (read_fromfile_flag < 7)
      {
         if (!myid)
         {
            hypre_printf("Error: Must read A, b, and x from file! \n");
         }
         exit(1);
      }
   }

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/

   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-pt") == 0 )
      {
         arg_index++;
         nparts = 0;
         while ( strncmp(argv[arg_index], "-", 1) != 0 )
         {
            parts[nparts++] = atoi(argv[arg_index++]);
         }
      }
      else if ( strcmp(argv[arg_index], "-pooldist") == 0 )
      {
         arg_index++;
         pooldist = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-r") == 0 )
      {
         arg_index++;
         for (i = 0; i < nparts; i++)
         {
            part = parts[i];
            k = arg_index;
            for (j = 0; j < 3; j++)
            {
               refine[part][j] = atoi(argv[k++]);
            }
         }
         arg_index += 3;
      }
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         for (i = 0; i < nparts; i++)
         {
            part = parts[i];
            k = arg_index;
            for (j = 0; j < 3; j++)
            {
               distribute[part][j] = atoi(argv[k++]);
            }
         }
         arg_index += 3;
      }
      else if ( strcmp(argv[arg_index], "-b") == 0 )
      {
         arg_index++;
         for (i = 0; i < nparts; i++)
         {
            part = parts[i];
            k = arg_index;
            for (j = 0; j < 3; j++)
            {
               block[part][j] = atoi(argv[k++]);
            }
         }
         arg_index += 3;
      }
      else if ( strcmp(argv[arg_index], "-solver") == 0 )
      {
         arg_index++;

         /* begin lobpcg */
         if ( strcmp(argv[arg_index], "none") == 0 )
         {
            solver_id = NO_SOLVER;
            arg_index++;
         }
         else /* end lobpcg */
         {
            solver_id = atoi(argv[arg_index++]);
         }
      }
      else if ( strcmp(argv[arg_index], "-print") == 0 )
      {
         arg_index++;
         print_system = 1;
      }
      else if ( strcmp(argv[arg_index], "-rhsfromcosine") == 0 )
      {
         arg_index++;
         cosine = 1;
      }
      else if ( strcmp(argv[arg_index], "-rhsone") == 0 )
      {
         arg_index++;
         cosine = 0;
      }
      else if ( strcmp(argv[arg_index], "-tol") == 0 )
      {
         arg_index++;
         tol = (HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-v") == 0 )
      {
         arg_index++;
         n_pre = atoi(argv[arg_index++]);
         n_post = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-skip") == 0 )
      {
         arg_index++;
         skip = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-rap") == 0 )
      {
         arg_index++;
         rap = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-relax") == 0 )
      {
         arg_index++;
         relax = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-w") == 0 )
      {
         arg_index++;
         jacobi_weight = (HYPRE_Real)atof(argv[arg_index++]);
         usr_jacobi_weight = 1; /* flag user weight */
      }
      else if ( strcmp(argv[arg_index], "-jump") == 0 )
      {
         arg_index++;
         jump = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-solver_type") == 0 )
      {
         arg_index++;
         solver_type = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-recompute") == 0 )
      {
         arg_index++;
         recompute_res = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-cf") == 0 )
      {
         arg_index++;
         cf_tol = (HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-crtdim") == 0 )
      {
         arg_index++;
         cycred_tdim = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-cri") == 0 )
      {
         arg_index++;
         for (i = 0; i < 3; i++)
         {
            cycred_index[i] = atoi(argv[arg_index++]);
         }
      }
      else if ( strcmp(argv[arg_index], "-crs") == 0 )
      {
         arg_index++;
         for (i = 0; i < 3; i++)
         {
            cycred_stride[i] = atoi(argv[arg_index++]);
         }
      }
      else if ( strcmp(argv[arg_index], "-old_default") == 0 )
      {
         /* uses old BoomerAMG defaults */
         arg_index++;
         old_default = 1;
      }
      /* begin lobpcg */
      else if ( strcmp(argv[arg_index], "-lobpcg") == 0 )
      {
         /* use lobpcg */
         arg_index++;
         lobpcgFlag = 1;
      }
      else if ( strcmp(argv[arg_index], "-orthchk") == 0 )
      {
         /* lobpcg: check orthonormality */
         arg_index++;
         checkOrtho = 1;
      }
      else if ( strcmp(argv[arg_index], "-verb") == 0 )
      {
         /* lobpcg: verbosity level */
         arg_index++;
         verbosity = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-vrand") == 0 )
      {
         /* lobpcg: block size */
         arg_index++;
         blockSize = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-seed") == 0 )
      {
         /* lobpcg: seed for srand */
         arg_index++;
         lobpcgSeed = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-itr") == 0 )
      {
         /* lobpcg: max # of iterations */
         arg_index++;
         maxIterations = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-pcgitr") == 0 )
      {
         /* lobpcg: max inner pcg iterations */
         arg_index++;
         pcgIterations = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-pcgtol") == 0 )
      {
         /* lobpcg: inner pcg iterations tolerance */
         arg_index++;
         pcgTol = (HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-pcgmode") == 0 )
      {
         /* lobpcg: initial guess for inner pcg */
         arg_index++;
         /* 0: zero, otherwise rhs */
         pcgMode = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-vout") == 0 )
      {
         /* lobpcg: print level */
         arg_index++;
         printLevel = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-memory_host") == 0 )
      {
         arg_index++;
         memory_location = HYPRE_MEMORY_HOST;
      }
      else if ( strcmp(argv[arg_index], "-memory_device") == 0 )
      {
         arg_index++;
         memory_location = HYPRE_MEMORY_DEVICE;
      }
      else if ( strcmp(argv[arg_index], "-exec_host") == 0 )
      {
         arg_index++;
         default_exec_policy = HYPRE_EXEC_HOST;
      }
      else if ( strcmp(argv[arg_index], "-exec_device") == 0 )
      {
         arg_index++;
         default_exec_policy = HYPRE_EXEC_DEVICE;
      }
#if defined(HYPRE_USING_GPU)
      else if ( strcmp(argv[arg_index], "-mm_vendor") == 0 )
      {
         arg_index++;
         spgemm_use_vendor = atoi(argv[arg_index++]);
      }
#endif
#if defined(HYPRE_USING_MEMORY_TRACKER)
      else if ( strcmp(argv[arg_index], "-print_mem_tracker") == 0 )
      {
         arg_index++;
         print_mem_tracker = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-mem_tracker_filename") == 0 )
      {
         arg_index++;
         snprintf(mem_tracker_name, HYPRE_MAX_FILE_NAME_LEN, "%s", argv[arg_index++]);
      }
#endif
      else if ( strcmp(argv[arg_index], "-gpu_mpi") == 0 )
      {
         arg_index++;
         gpu_aware_mpi = atoi(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

#if defined(HYPRE_USING_MEMORY_TRACKER)
   hypre_MemoryTrackerSetPrint(print_mem_tracker);
   if (mem_tracker_name[0]) { hypre_MemoryTrackerSetFileName(mem_tracker_name); }
#endif

   /* default memory location */
   HYPRE_SetMemoryLocation(memory_location);

   /* default execution policy */
   HYPRE_SetExecutionPolicy(default_exec_policy);

#if defined(HYPRE_USING_GPU)
   HYPRE_SetSpGemmUseVendor(spgemm_use_vendor);
#endif

   HYPRE_SetGpuAwareMPI(gpu_aware_mpi);

   if ( solver_id == 39 && lobpcgFlag )
   {
      solver_id = 10;
   }

   /* end lobpcg */

   /*-----------------------------------------------------------
    * Print driver parameters TODO
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
#if defined(HYPRE_DEVELOP_STRING) && defined(HYPRE_DEVELOP_BRANCH)
      hypre_printf("\nUsing HYPRE_DEVELOP_STRING: %s (branch %s; the develop branch)\n\n",
                   HYPRE_DEVELOP_STRING, HYPRE_DEVELOP_BRANCH);

#elif defined(HYPRE_DEVELOP_STRING) && !defined(HYPRE_DEVELOP_BRANCH)
      hypre_printf("\nUsing HYPRE_DEVELOP_STRING: %s (branch %s; not the develop branch)\n\n",
                   HYPRE_DEVELOP_STRING, HYPRE_BRANCH_NAME);

#elif defined(HYPRE_RELEASE_VERSION)
      hypre_printf("\nUsing HYPRE_RELEASE_VERSION: %s\n\n",
                   HYPRE_RELEASE_VERSION);
#endif
   }

   /*-----------------------------------------------------------
    * Synchronize so that timings make sense
    *-----------------------------------------------------------*/

   hypre_MPI_Barrier(comm);

   /*-----------------------------------------------------------
    * Set up the grid
    *-----------------------------------------------------------*/

   time_index = hypre_InitializeTiming("SStruct Interface");
   hypre_BeginTiming(time_index);

   if (read_fromfile_flag & 0x1)
   {
      if (!myid)
      {
         hypre_printf("Reading SStructMatrix A from file: %s\n", argv[read_fromfile_index[0]]);
      }

      HYPRE_SStructMatrixRead(comm, argv[read_fromfile_index[0]], &A);
   }
   else
   {
      /*-----------------------------------------------------------
       * Distribute data
       *-----------------------------------------------------------*/

      DistributeData(global_data, pooldist, refine, distribute, block,
                     num_procs, myid, &data);

      /*-----------------------------------------------------------
       * Check a few things
       *-----------------------------------------------------------*/
      if (solver_id >= 200)
      {
         pdata = data.pdata[0];
         if (nparts > 1)
         {
            if (!myid)
            {
               hypre_printf("Warning: Invalid number of parts for Struct Solver. Part 0 taken.\n");
            }
         }

         if (pdata.nvars > 1)
         {
            if (!myid)
            {
               hypre_printf("Error: Invalid number of nvars for Struct Solver \n");
            }
            exit(1);
         }
      }

      HYPRE_SStructGridCreate(comm, data.ndim, data.nparts, &grid);
      if (data.numghost != NULL)
      {
         HYPRE_SStructGridSetNumGhost(grid, data.numghost);
      }
      for (part = 0; part < data.nparts; part++)
      {
         pdata = data.pdata[part];
         for (box = 0; box < pdata.nboxes; box++)
         {
            HYPRE_SStructGridSetExtents(grid, part,
                                        pdata.ilowers[box], pdata.iuppers[box]);
         }

         HYPRE_SStructGridSetVariables(grid, part, pdata.nvars, pdata.vartypes);

         /* GridAddVariabes */

         if (data.fem_nvars > 0)
         {
            HYPRE_SStructGridSetFEMOrdering(grid, part, data.fem_ordering);
         }

         /* GridSetNeighborPart and GridSetSharedPart */
         for (box = 0; box < pdata.glue_nboxes; box++)
         {
            if (pdata.glue_shared[box])
            {
               HYPRE_SStructGridSetSharedPart(grid, part,
                                              pdata.glue_ilowers[box],
                                              pdata.glue_iuppers[box],
                                              pdata.glue_offsets[box],
                                              pdata.glue_nbor_parts[box],
                                              pdata.glue_nbor_ilowers[box],
                                              pdata.glue_nbor_iuppers[box],
                                              pdata.glue_nbor_offsets[box],
                                              pdata.glue_index_maps[box],
                                              pdata.glue_index_dirs[box]);
            }
            else
            {
               HYPRE_SStructGridSetNeighborPart(grid, part,
                                                pdata.glue_ilowers[box],
                                                pdata.glue_iuppers[box],
                                                pdata.glue_nbor_parts[box],
                                                pdata.glue_nbor_ilowers[box],
                                                pdata.glue_nbor_iuppers[box],
                                                pdata.glue_index_maps[box],
                                                pdata.glue_index_dirs[box]);
            }
         }

         HYPRE_SStructGridSetPeriodic(grid, part, pdata.periodic);
      }
      HYPRE_SStructGridAssemble(grid);

      /*-----------------------------------------------------------
       * Set up the stencils
       *-----------------------------------------------------------*/

      stencils = hypre_CTAlloc(HYPRE_SStructStencil,  data.nstencils, HYPRE_MEMORY_HOST);
      for (s = 0; s < data.nstencils; s++)
      {
         HYPRE_SStructStencilCreate(data.ndim, data.stencil_sizes[s],
                                    &stencils[s]);
         for (entry = 0; entry < data.stencil_sizes[s]; entry++)
         {
            HYPRE_SStructStencilSetEntry(stencils[s], entry,
                                         data.stencil_offsets[s][entry],
                                         data.stencil_vars[s][entry]);
         }
      }

      /*-----------------------------------------------------------
       * Set object type
       *-----------------------------------------------------------*/
      /* determine if we build a gradient matrix */
      if (solver_id == 150)
      {
         gradient_matrix = 1;
         /* for now, change solver 150 to solver 28 */
         solver_id = 28;
      }

      if ( ((solver_id >= 20) && (solver_id < 30)) ||
           ((solver_id >= 40) && (solver_id < 50)) ||
           ((solver_id >= 60) && (solver_id < 70)) ||
           ((solver_id >= 80) && (solver_id < 90)) ||
           ((solver_id >= 90) && (solver_id < 100)) ||
           (solver_id == 120) )
      {
         object_type = HYPRE_PARCSR;
      }

      if (solver_id >= 200)
      {
         object_type = HYPRE_STRUCT;
      }

      /*-----------------------------------------------------------
       * Set up the graph
       *-----------------------------------------------------------*/

      HYPRE_SStructGraphCreate(comm, grid, &graph);

      /* HYPRE_SSTRUCT is the default, so we don't have to call SetObjectType */
      if ( object_type != HYPRE_SSTRUCT )
      {
         HYPRE_SStructGraphSetObjectType(graph, object_type);
      }

      for (part = 0; part < data.nparts; part++)
      {
         pdata = data.pdata[part];

         if (data.nstencils > 0)
         {
            /* set stencils */
            for (var = 0; var < pdata.nvars; var++)
            {
               HYPRE_SStructGraphSetStencil(graph, part, var,
                                            stencils[pdata.stencil_num[var]]);
            }
         }
         else if (data.fem_nvars > 0)
         {
            /* indicate FEM approach */
            HYPRE_SStructGraphSetFEM(graph, part);

            /* set sparsity */
            HYPRE_SStructGraphSetFEMSparsity(graph, part,
                                             data.fem_nsparse, data.fem_sparsity);
         }

         /* add entries */
         for (box = 0; box < pdata.graph_nboxes; box++)
         {
            for (index[2] = pdata.graph_ilowers[box][2];
                 index[2] <= pdata.graph_iuppers[box][2];
                 index[2] += pdata.graph_strides[box][2])
            {
               for (index[1] = pdata.graph_ilowers[box][1];
                    index[1] <= pdata.graph_iuppers[box][1];
                    index[1] += pdata.graph_strides[box][1])
               {
                  for (index[0] = pdata.graph_ilowers[box][0];
                       index[0] <= pdata.graph_iuppers[box][0];
                       index[0] += pdata.graph_strides[box][0])
                  {
                     for (i = 0; i < 3; i++)
                     {
                        j = pdata.graph_index_maps[box][i];
                        k = index[i] - pdata.graph_ilowers[box][i];
                        k /= pdata.graph_strides[box][i];
                        k *= pdata.graph_index_signs[box][i];
#if 0 /* the following does not work with some Intel compilers with -O2 */
                        to_index[j] = pdata.graph_to_ilowers[box][j] +
                                      k * pdata.graph_to_strides[box][j];
#else
                        to_index[j] = pdata.graph_to_ilowers[box][j];
                        to_index[j] += k * pdata.graph_to_strides[box][j];
#endif
                     }
                     HYPRE_SStructGraphAddEntries(graph, part, index,
                                                  pdata.graph_vars[box],
                                                  pdata.graph_to_parts[box],
                                                  to_index,
                                                  pdata.graph_to_vars[box]);
                  }
               }
            }
         }
      }

      HYPRE_SStructGraphAssemble(graph);

      /*-----------------------------------------------------------
       * Set up the matrix
       *-----------------------------------------------------------*/

      values_size = hypre_max(data.max_boxsize, data.max_boxsize * data.fem_nsparse);

      values   = hypre_TAlloc(HYPRE_Real, values_size, HYPRE_MEMORY_HOST);
      d_values = hypre_TAlloc(HYPRE_Real, values_size, memory_location);

      HYPRE_SStructMatrixCreate(comm, graph, &A);

      /* TODO HYPRE_SStructMatrixSetSymmetric(A, 1); */
      for (i = 0; i < data.symmetric_num; i++)
      {
         HYPRE_SStructMatrixSetSymmetric(A, data.symmetric_parts[i],
                                         data.symmetric_vars[i],
                                         data.symmetric_to_vars[i],
                                         data.symmetric_booleans[i]);
      }
      HYPRE_SStructMatrixSetNSSymmetric(A, data.ns_symmetric);

      /* HYPRE_SSTRUCT is the default, so we don't have to call SetObjectType */
      if ( object_type != HYPRE_SSTRUCT )
      {
         HYPRE_SStructMatrixSetObjectType(A, object_type);
      }

      HYPRE_SStructMatrixInitialize(A);

      if (data.nstencils > 0)
      {
         /* StencilSetEntry: set stencil values */
         for (part = 0; part < data.nparts; part++)
         {
            pdata = data.pdata[part];
            for (var = 0; var < pdata.nvars; var++)
            {
               s = pdata.stencil_num[var];
               for (i = 0; i < data.stencil_sizes[s]; i++)
               {
                  for (j = 0; j < pdata.max_boxsize; j++)
                  {
                     values[j] = data.stencil_values[s][i];
                  }

                  hypre_TMemcpy(d_values, values, HYPRE_Real, values_size,
                                memory_location, HYPRE_MEMORY_HOST);

                  for (box = 0; box < pdata.nboxes; box++)
                  {
                     GetVariableBox(pdata.ilowers[box], pdata.iuppers[box],
                                    pdata.vartypes[var], ilower, iupper);

                     HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
                                                     var, 1, &i, d_values);
                  }
               }
            }
         }
      }
      else if (data.fem_nvars > 0)
      {
         /* FEMStencilSetRow: add to stencil values */
#if 0    // Use AddFEMValues
         hypre_TMemcpy(data.d_fem_values, data.fem_values, HYPRE_Real,
                       data.fem_nsparse, memory_location, HYPRE_MEMORY_HOST);

         for (part = 0; part < data.nparts; part++)
         {
            pdata = data.pdata[part];
            for (box = 0; box < pdata.nboxes; box++)
            {
               for (index[2] = pdata.ilowers[box][2];
                    index[2] <= pdata.iuppers[box][2]; index[2]++)
               {
                  for (index[1] = pdata.ilowers[box][1];
                       index[1] <= pdata.iuppers[box][1]; index[1]++)
                  {
                     for (index[0] = pdata.ilowers[box][0];
                          index[0] <= pdata.iuppers[box][0]; index[0]++)
                     {
                        HYPRE_SStructMatrixAddFEMValues(A, part, index,
                                                        data.d_fem_values);
                     }
                  }
               }
            }
         }
#else    // Use AddFEMBoxValues
         /* TODO: There is probably a smarter way to do this copy */
         for (i = 0; i < data.max_boxsize; i++)
         {
            j = i * data.fem_nsparse;
            hypre_TMemcpy(&d_values[j], data.fem_values, HYPRE_Real,
                          data.fem_nsparse, memory_location, HYPRE_MEMORY_HOST);
         }
         for (part = 0; part < data.nparts; part++)
         {
            pdata = data.pdata[part];
            for (box = 0; box < pdata.nboxes; box++)
            {
               HYPRE_SStructMatrixAddFEMBoxValues(
                  A, part, pdata.ilowers[box], pdata.iuppers[box], d_values);
            }
         }
#endif
      }

      /* GraphAddEntries: set non-stencil entries */
      for (part = 0; part < data.nparts; part++)
      {
         pdata = data.pdata[part];

         hypre_TMemcpy(pdata.d_graph_values, pdata.graph_values,
                       HYPRE_Real, pdata.graph_values_size,
                       memory_location, HYPRE_MEMORY_HOST);

         for (box = 0; box < pdata.graph_nboxes; box++)
         {
            /*
             * RDF NOTE: Add a separate interface routine for setting non-stencil
             * entries.  It would be more efficient to set boundary values a box
             * at a time, but AMR may require striding, and some codes may already
             * have a natural values array to pass in, but can't because it uses
             * ghost values.
             *
             * Example new interface routine:
             *   SetNSBoxValues(matrix, part, ilower, iupper, stride, entry
             *                  values_ilower, values_iupper, values);
             */

            /* since we have already tested SetBoxValues above, use SetValues here */
#if 0
            for (j = 0; j < pdata.graph_boxsizes[box]; j++)
            {
               values[j] = pdata.graph_values[box];
            }
            HYPRE_SStructMatrixSetBoxValues(A, part,
                                            pdata.graph_ilowers[box],
                                            pdata.graph_iuppers[box],
                                            pdata.graph_vars[box],
                                            1, &pdata.graph_entries[box],
                                            values);
#else
            for (index[2] = pdata.graph_ilowers[box][2];
                 index[2] <= pdata.graph_iuppers[box][2];
                 index[2] += pdata.graph_strides[box][2])
            {
               for (index[1] = pdata.graph_ilowers[box][1];
                    index[1] <= pdata.graph_iuppers[box][1];
                    index[1] += pdata.graph_strides[box][1])
               {
                  for (index[0] = pdata.graph_ilowers[box][0];
                       index[0] <= pdata.graph_iuppers[box][0];
                       index[0] += pdata.graph_strides[box][0])
                  {
                     HYPRE_SStructMatrixSetValues(A, part, index,
                                                  pdata.graph_vars[box],
                                                  1, &pdata.graph_entries[box],
                                                  &pdata.d_graph_values[box]);
                  }
               }
            }
#endif
         }
      }

      /* MatrixSetValues: reset some matrix values */
      for (part = 0; part < data.nparts; part++)
      {
         pdata = data.pdata[part];
         for (box = 0; box < pdata.matset_nboxes; box++)
         {
            size = 1;
            for (j = 0; j < 3; j++)
            {
               size *= (pdata.matset_iuppers[box][j] -
                        pdata.matset_ilowers[box][j] + 1);
            }
            for (j = 0; j < size; j++)
            {
               values[j] = pdata.matset_values[box];
            }

            hypre_TMemcpy(d_values, values, HYPRE_Real, values_size,
                          memory_location, HYPRE_MEMORY_HOST);

            HYPRE_SStructMatrixSetBoxValues(A, part,
                                            pdata.matset_ilowers[box],
                                            pdata.matset_iuppers[box],
                                            pdata.matset_vars[box],
                                            1, &pdata.matset_entries[box],
                                            d_values);
         }
      }

      /* MatrixAddToValues: add to some matrix values */
      for (part = 0; part < data.nparts; part++)
      {
         pdata = data.pdata[part];
         for (box = 0; box < pdata.matadd_nboxes; box++)
         {
            size = 1;
            for (j = 0; j < 3; j++)
            {
               size *= (pdata.matadd_iuppers[box][j] -
                        pdata.matadd_ilowers[box][j] + 1);
            }

            for (entry = 0; entry < pdata.matadd_nentries[box]; entry++)
            {
               for (j = 0; j < size; j++)
               {
                  values[j] = pdata.matadd_values[box][entry];
               }

               hypre_TMemcpy(d_values, values, HYPRE_Real, values_size,
                             memory_location, HYPRE_MEMORY_HOST);

               HYPRE_SStructMatrixAddToBoxValues(A, part,
                                                 pdata.matadd_ilowers[box],
                                                 pdata.matadd_iuppers[box],
                                                 pdata.matadd_vars[box],
                                                 1, &pdata.matadd_entries[box][entry],
                                                 d_values);
            }
         }
      }

      /* FEMMatrixAddToValues: add to some matrix values */
      for (part = 0; part < data.nparts; part++)
      {
         pdata = data.pdata[part];
         for (box = 0; box < pdata.fem_matadd_nboxes; box++)
         {
            for (i = 0; i < data.fem_nsparse; i++)
            {
               values[i] = 0.0;
            }
            s = 0;
            for (i = 0; i < pdata.fem_matadd_nrows[box]; i++)
            {
               row = pdata.fem_matadd_rows[box][i];
               for (j = 0; j < pdata.fem_matadd_ncols[box]; j++)
               {
                  col = pdata.fem_matadd_cols[box][j];
                  values[data.fem_ivalues_full[row][col]] =
                     pdata.fem_matadd_values[box][s];
                  s++;
               }
            }

            hypre_TMemcpy(d_values, values, HYPRE_Real, values_size,
                          memory_location, HYPRE_MEMORY_HOST);

            for (index[2] = pdata.fem_matadd_ilowers[box][2];
                 index[2] <= pdata.fem_matadd_iuppers[box][2]; index[2]++)
            {
               for (index[1] = pdata.fem_matadd_ilowers[box][1];
                    index[1] <= pdata.fem_matadd_iuppers[box][1]; index[1]++)
               {
                  for (index[0] = pdata.fem_matadd_ilowers[box][0];
                       index[0] <= pdata.fem_matadd_iuppers[box][0]; index[0]++)
                  {
                     HYPRE_SStructMatrixAddFEMValues(A, part, index, d_values);
                  }
               }
            }
         }
      }

      HYPRE_SStructMatrixAssemble(A);
   }

   /*-----------------------------------------------------------
    * Set up the RHS vector
    *-----------------------------------------------------------*/

   if (read_fromfile_flag & 0x2)
   {
      if (!myid)
      {
         hypre_printf("Reading SStructVector b from file: %s\n", argv[read_fromfile_index[1]]);
      }
      cosine = 0;

      HYPRE_SStructVectorRead(comm, argv[read_fromfile_index[1]], &b);
   }
   else
   {
      HYPRE_SStructVectorCreate(comm, grid, &b);

      /* HYPRE_SSTRUCT is the default, so we don't have to call SetObjectType */
      if ( object_type != HYPRE_SSTRUCT )
      {
         HYPRE_SStructVectorSetObjectType(b, object_type);
      }

      HYPRE_SStructVectorInitialize(b);

      /* Initialize the rhs values */
      if (data.rhs_true)
      {
         for (j = 0; j < data.max_boxsize; j++)
         {
            values[j] = data.rhs_value;
         }
      }
      else if (data.fem_rhs_true)
      {
         for (j = 0; j < data.max_boxsize; j++)
         {
            values[j] = 0.0;
         }
      }
      else /* rhs=1 is the default */
      {
         for (j = 0; j < data.max_boxsize; j++)
         {
            values[j] = 1.0;
         }
      }

      hypre_TMemcpy(d_values, values, HYPRE_Real, values_size,
                    memory_location, HYPRE_MEMORY_HOST);

      for (part = 0; part < data.nparts; part++)
      {
         pdata = data.pdata[part];
         for (var = 0; var < pdata.nvars; var++)
         {
            for (box = 0; box < pdata.nboxes; box++)
            {
               GetVariableBox(pdata.ilowers[box], pdata.iuppers[box],
                              pdata.vartypes[var], ilower, iupper);
               HYPRE_SStructVectorSetBoxValues(b, part, ilower, iupper,
                                               var, d_values);
            }
         }
      }

      /* Add values for FEMRhsSet */
      if (data.fem_rhs_true)
      {
#if 0    // Use AddFEMValues
         hypre_TMemcpy(data.d_fem_rhs_values, data.fem_rhs_values, HYPRE_Real,
                       data.fem_nvars, memory_location, HYPRE_MEMORY_HOST);

         for (part = 0; part < data.nparts; part++)
         {
            pdata = data.pdata[part];
            for (box = 0; box < pdata.nboxes; box++)
            {
               for (index[2] = pdata.ilowers[box][2];
                    index[2] <= pdata.iuppers[box][2]; index[2]++)
               {
                  for (index[1] = pdata.ilowers[box][1];
                       index[1] <= pdata.iuppers[box][1]; index[1]++)
                  {
                     for (index[0] = pdata.ilowers[box][0];
                          index[0] <= pdata.iuppers[box][0]; index[0]++)
                     {
                        HYPRE_SStructVectorAddFEMValues(b, part, index,
                                                        data.d_fem_rhs_values);
                     }
                  }
               }
            }
         }
#else    // Use AddFEMBoxValues
         /* TODO: There is probably a smarter way to do this copy */
         for (i = 0; i < data.max_boxsize; i++)
         {
            j = i * data.fem_nvars;
            hypre_TMemcpy(&d_values[j], data.fem_rhs_values, HYPRE_Real,
                          data.fem_nvars, memory_location, HYPRE_MEMORY_HOST);
         }
         for (part = 0; part < data.nparts; part++)
         {
            pdata = data.pdata[part];
            for (box = 0; box < pdata.nboxes; box++)
            {
               HYPRE_SStructVectorAddFEMBoxValues(
                  b, part, pdata.ilowers[box], pdata.iuppers[box], d_values);
            }
         }
#endif
      }

      /* RhsAddToValues: add to some RHS values */
      for (part = 0; part < data.nparts; part++)
      {
         pdata = data.pdata[part];
         for (box = 0; box < pdata.rhsadd_nboxes; box++)
         {
            size = 1;
            for (j = 0; j < 3; j++)
            {
               size *= (pdata.rhsadd_iuppers[box][j] -
                        pdata.rhsadd_ilowers[box][j] + 1);
            }

            for (j = 0; j < size; j++)
            {
               values[j] = pdata.rhsadd_values[box];
            }

            hypre_TMemcpy(d_values, values, HYPRE_Real, values_size,
                          memory_location, HYPRE_MEMORY_HOST);

            HYPRE_SStructVectorAddToBoxValues(b, part,
                                              pdata.rhsadd_ilowers[box],
                                              pdata.rhsadd_iuppers[box],
                                              pdata.rhsadd_vars[box], d_values);
         }
      }

      /* FEMRhsAddToValues: add to some RHS values */
      for (part = 0; part < data.nparts; part++)
      {
         pdata = data.pdata[part];
         for (box = 0; box < pdata.fem_rhsadd_nboxes; box++)
         {
            for (index[2] = pdata.fem_rhsadd_ilowers[box][2];
                 index[2] <= pdata.fem_rhsadd_iuppers[box][2]; index[2]++)
            {
               for (index[1] = pdata.fem_rhsadd_ilowers[box][1];
                    index[1] <= pdata.fem_rhsadd_iuppers[box][1]; index[1]++)
               {
                  for (index[0] = pdata.fem_rhsadd_ilowers[box][0];
                       index[0] <= pdata.fem_rhsadd_iuppers[box][0]; index[0]++)
                  {
                     HYPRE_SStructVectorAddFEMValues(b, part, index,
                                                     pdata.fem_rhsadd_values[box]);
                  }
               }
            }
         }
      }

      HYPRE_SStructVectorAssemble(b);
   }

   /*-----------------------------------------------------------
    * Set up the initial solution vector
    *-----------------------------------------------------------*/

   if (read_fromfile_flag & 0x4)
   {
      if (!myid)
      {
         hypre_printf("Reading SStructVector x0 from file: %s\n", argv[read_fromfile_index[2]]);
      }

      HYPRE_SStructVectorRead(comm, argv[read_fromfile_index[2]], &x);
   }
   else
   {
      HYPRE_SStructVectorCreate(comm, grid, &x);

      /* HYPRE_SSTRUCT is the default, so we don't have to call SetObjectType */
      if ( object_type != HYPRE_SSTRUCT )
      {
         HYPRE_SStructVectorSetObjectType(x, object_type);
      }

      HYPRE_SStructVectorInitialize(x);

      /*-----------------------------------------------------------
       * If requested, reset linear system so that it has
       * exact solution:
       *
       *   u(part,var,i,j,k) = (part+1)*(var+1)*cosine[(i+j+k)/10]
       *
       *-----------------------------------------------------------*/

      if (cosine)
      {
         for (part = 0; part < data.nparts; part++)
         {
            pdata = data.pdata[part];
            for (var = 0; var < pdata.nvars; var++)
            {
               scale = (part + 1.0) * (var + 1.0);
               for (box = 0; box < pdata.nboxes; box++)
               {
                  /*
                     GetVariableBox(pdata.ilowers[box], pdata.iuppers[box],
                     pdata.vartypes[var], ilower, iupper);
                  */
                  GetVariableBox(pdata.ilowers[box], pdata.iuppers[box],
                                 var, ilower, iupper);
                  SetCosineVector(scale, ilower, iupper, values);

                  hypre_TMemcpy(d_values, values, HYPRE_Real, values_size,
                                memory_location, HYPRE_MEMORY_HOST);

                  HYPRE_SStructVectorSetBoxValues(x, part, ilower, iupper,
                                                  var, d_values);
               }
            }
         }
      }

      HYPRE_SStructVectorAssemble(x);
   }

   hypre_EndTiming(time_index);
   hypre_PrintTiming("SStruct Interface", comm);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();

   /*-----------------------------------------------------------
    * Get the objects out
    * NOTE: This should go after the cosine part, but for the bug
    *-----------------------------------------------------------*/

   if (object_type == HYPRE_PARCSR)
   {
      HYPRE_SStructMatrixGetObject(A, (void **) &par_A);
      HYPRE_SStructVectorGetObject(b, (void **) &par_b);
      HYPRE_SStructVectorGetObject(x, (void **) &par_x);
   }
   else if (object_type == HYPRE_STRUCT)
   {
      HYPRE_SStructMatrixGetObject(A, (void **) &sA);
      HYPRE_SStructVectorGetObject(b, (void **) &sb);
      HYPRE_SStructVectorGetObject(x, (void **) &sx);
   }

   /*-----------------------------------------------------------
    * Finish resetting the linear system
    *-----------------------------------------------------------*/

   if (cosine)
   {
      /* This if/else is due to a bug in SStructMatvec */
      if (object_type == HYPRE_SSTRUCT)
      {
         /* Apply A to cosine vector to yield righthand side */
         hypre_SStructMatvec(1.0, A, x, 0.0, b);
         /* Reset initial guess to zero */
         hypre_SStructMatvec(0.0, A, b, 0.0, x);
      }
      else if (object_type == HYPRE_PARCSR)
      {
         /* Apply A to cosine vector to yield righthand side */
         HYPRE_ParCSRMatrixMatvec(1.0, par_A, par_x, 0.0, par_b );
         /* Reset initial guess to zero */
         HYPRE_ParCSRMatrixMatvec(0.0, par_A, par_b, 0.0, par_x );
      }
      else if (object_type == HYPRE_STRUCT)
      {
         /* Apply A to cosine vector to yield righthand side */
         hypre_StructMatvec(1.0, sA, sx, 0.0, sb);
         /* Reset initial guess to zero */
         hypre_StructMatvec(0.0, sA, sb, 0.0, sx);
      }
   }

   /*-----------------------------------------------------------
    * Set up a gradient matrix G
    *-----------------------------------------------------------*/

   if (gradient_matrix)
   {
      HYPRE_SStructVariable vartypes[1] = {HYPRE_SSTRUCT_VARIABLE_NODE};
      HYPRE_Int offsets[3][2][3] = { {{0, 0, 0}, {-1, 0, 0}},
         {{0, 0, 0}, {0, -1, 0}},
         {{0, 0, 0}, {0, 0, -1}}
      };
      HYPRE_Real stencil_values[2] = {1.0, -1.0};

      /* Set up the domain grid */

      HYPRE_SStructGridCreate(comm, data.ndim, data.nparts, &G_grid);
      for (part = 0; part < data.nparts; part++)
      {
         pdata = data.pdata[part];
         for (box = 0; box < pdata.nboxes; box++)
         {
            HYPRE_SStructGridSetExtents(G_grid, part,
                                        pdata.ilowers[box], pdata.iuppers[box]);
         }
         HYPRE_SStructGridSetVariables(G_grid, part, 1, vartypes);
         for (box = 0; box < pdata.glue_nboxes; box++)
         {
            if (pdata.glue_shared[box])
            {
               HYPRE_SStructGridSetSharedPart(G_grid, part,
                                              pdata.glue_ilowers[box],
                                              pdata.glue_iuppers[box],
                                              pdata.glue_offsets[box],
                                              pdata.glue_nbor_parts[box],
                                              pdata.glue_nbor_ilowers[box],
                                              pdata.glue_nbor_iuppers[box],
                                              pdata.glue_nbor_offsets[box],
                                              pdata.glue_index_maps[box],
                                              pdata.glue_index_dirs[box]);
            }
            else
            {
               HYPRE_SStructGridSetNeighborPart(G_grid, part,
                                                pdata.glue_ilowers[box],
                                                pdata.glue_iuppers[box],
                                                pdata.glue_nbor_parts[box],
                                                pdata.glue_nbor_ilowers[box],
                                                pdata.glue_nbor_iuppers[box],
                                                pdata.glue_index_maps[box],
                                                pdata.glue_index_dirs[box]);
            }
         }
      }
      HYPRE_SStructGridAssemble(G_grid);

      /* Set up the gradient stencils */

      G_stencils = hypre_CTAlloc(HYPRE_SStructStencil,  data.ndim, HYPRE_MEMORY_HOST);
      for (s = 0; s < data.ndim; s++)
      {
         HYPRE_SStructStencilCreate(data.ndim, 2, &G_stencils[s]);
         for (entry = 0; entry < 2; entry++)
         {
            HYPRE_SStructStencilSetEntry(
               G_stencils[s], entry, offsets[s][entry], 0);
         }
      }

      /* Set up the gradient graph */

      HYPRE_SStructGraphCreate(comm, grid, &G_graph);
      HYPRE_SStructGraphSetDomainGrid(G_graph, G_grid);
      HYPRE_SStructGraphSetObjectType(G_graph, HYPRE_PARCSR);
      for (part = 0; part < data.nparts; part++)
      {
         pdata = data.pdata[part];
         for (var = 0; var < data.ndim; var++)
         {
            HYPRE_SStructGraphSetStencil(G_graph, part, var, G_stencils[var]);
         }
      }
      HYPRE_SStructGraphAssemble(G_graph);

      /* Set up the matrix */

      HYPRE_SStructMatrixCreate(comm, G_graph, &G);
      HYPRE_SStructMatrixSetObjectType(G, HYPRE_PARCSR);
      HYPRE_SStructMatrixInitialize(G);
      for (part = 0; part < data.nparts; part++)
      {
         pdata = data.pdata[part];
         for (var = 0; var < data.ndim; var++)
         {
            for (i = 0; i < 2; i++)
            {
               for (j = 0; j < pdata.max_boxsize; j++)
               {
                  values[j] = stencil_values[i];
               }

               hypre_TMemcpy(d_values, values, HYPRE_Real, values_size, memory_location, HYPRE_MEMORY_HOST);

               for (box = 0; box < pdata.nboxes; box++)
               {
                  GetVariableBox(pdata.ilowers[box], pdata.iuppers[box],
                                 pdata.vartypes[var], ilower, iupper);
                  HYPRE_SStructMatrixSetBoxValues(G, part, ilower, iupper,
                                                  var, 1, &i, d_values);
               }
            }
         }
      }

      HYPRE_SStructMatrixAssemble(G);
   }

   /*-----------------------------------------------------------
    * Print out the system and initial guess
    *-----------------------------------------------------------*/

   if (print_system)
   {
      HYPRE_SStructVectorGather(b);
      HYPRE_SStructVectorGather(x);
      HYPRE_SStructMatrixPrint("sstruct.out.A",  A, 0);
      HYPRE_SStructVectorPrint("sstruct.out.b",  b, 0);
      HYPRE_SStructVectorPrint("sstruct.out.x0", x, 0);

      if (gradient_matrix)
      {
         HYPRE_SStructMatrixPrint("sstruct.out.G",  G, 0);
      }
   }

   /*-----------------------------------------------------------
    * Debugging code
    *-----------------------------------------------------------*/

#if DEBUG
   {
      FILE *file;
      char  filename[255];

      /* result is 1's on the interior of the grid */
      hypre_SStructMatvec(1.0, A, b, 0.0, x);
      HYPRE_SStructVectorPrint("sstruct.out.matvec", x, 0);

      /* result is all 1's */
      hypre_SStructCopy(b, x);
      HYPRE_SStructVectorPrint("sstruct.out.copy", x, 0);

      /* result is all 2's */
      hypre_SStructScale(2.0, x);
      HYPRE_SStructVectorPrint("sstruct.out.scale", x, 0);

      /* result is all 0's */
      hypre_SStructAxpy(-2.0, b, x);
      HYPRE_SStructVectorPrint("sstruct.out.axpy", x, 0);

      /* result is 1's with 0's on some boundaries */
      hypre_SStructCopy(b, x);
      hypre_sprintf(filename, "sstruct.out.gatherpre.%05d", myid);
      file = fopen(filename, "w");
      for (part = 0; part < data.nparts; part++)
      {
         pdata = data.pdata[part];
         for (var = 0; var < pdata.nvars; var++)
         {
            for (box = 0; box < pdata.nboxes; box++)
            {
               GetVariableBox(pdata.ilowers[box], pdata.iuppers[box],
                              pdata.vartypes[var], ilower, iupper);
               HYPRE_SStructVectorGetBoxValues(x, part, ilower, iupper,
                                               var, values);
               hypre_fprintf(file, "\nPart %d, var %d, box %d:\n", part, var, box);
               for (i = 0; i < pdata.boxsizes[box]; i++)
               {
                  hypre_fprintf(file, "%e\n", values[i]);
               }
            }
         }
      }
      fclose(file);

      /* result is all 1's */
      HYPRE_SStructVectorGather(x);
      hypre_sprintf(filename, "sstruct.out.gatherpost.%05d", myid);
      file = fopen(filename, "w");
      for (part = 0; part < data.nparts; part++)
      {
         pdata = data.pdata[part];
         for (var = 0; var < pdata.nvars; var++)
         {
            for (box = 0; box < pdata.nboxes; box++)
            {
               GetVariableBox(pdata.ilowers[box], pdata.iuppers[box],
                              pdata.vartypes[var], ilower, iupper);
               HYPRE_SStructVectorGetBoxValues(x, part, ilower, iupper,
                                               var, values);
               hypre_fprintf(file, "\nPart %d, var %d, box %d:\n", part, var, box);
               for (i = 0; i < pdata.boxsizes[box]; i++)
               {
                  hypre_fprintf(file, "%e\n", values[i]);
               }
            }
         }
      }

      /* re-initializes x to 0 */
      hypre_SStructAxpy(-1.0, b, x);
   }
#endif

   hypre_TFree(values, HYPRE_MEMORY_HOST);
   hypre_TFree(d_values, memory_location);

   /*-----------------------------------------------------------
    * Solve the system using SysPFMG or Split
    *-----------------------------------------------------------*/

   if (solver_id == 3)
   {
      time_index = hypre_InitializeTiming("SysPFMG Setup");
      hypre_BeginTiming(time_index);

      HYPRE_SStructSysPFMGCreate(comm, &solver);
      HYPRE_SStructSysPFMGSetMaxIter(solver, 100);
      HYPRE_SStructSysPFMGSetTol(solver, tol);
      HYPRE_SStructSysPFMGSetRelChange(solver, 0);
      /* weighted Jacobi = 1; red-black GS = 2 */
      HYPRE_SStructSysPFMGSetRelaxType(solver, relax);
      if (usr_jacobi_weight)
      {
         HYPRE_SStructSysPFMGSetJacobiWeight(solver, jacobi_weight);
      }
      HYPRE_SStructSysPFMGSetNumPreRelax(solver, n_pre);
      HYPRE_SStructSysPFMGSetNumPostRelax(solver, n_post);
      HYPRE_SStructSysPFMGSetSkipRelax(solver, skip);
      /*HYPRE_StructPFMGSetDxyz(solver, dxyz);*/
      HYPRE_SStructSysPFMGSetPrintLevel(solver, 1);
      HYPRE_SStructSysPFMGSetLogging(solver, 1);
      HYPRE_SStructSysPFMGSetup(solver, A, b, x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("SysPFMG Solve");
      hypre_BeginTiming(time_index);

      HYPRE_SStructSysPFMGSolve(solver, A, b, x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_SStructSysPFMGGetNumIterations(solver, &num_iterations);
      HYPRE_SStructSysPFMGGetFinalRelativeResidualNorm(solver, &final_res_norm);

      HYPRE_SStructSysPFMGDestroy(solver);
   }

   else if ((solver_id >= 0) && (solver_id < 10) && (solver_id != 3))
   {
      time_index = hypre_InitializeTiming("Split Setup");
      hypre_BeginTiming(time_index);

      HYPRE_SStructSplitCreate(comm, &solver);
      HYPRE_SStructSplitSetMaxIter(solver, 100);
      HYPRE_SStructSplitSetTol(solver, tol);
      if (solver_id == 0)
      {
         HYPRE_SStructSplitSetStructSolver(solver, HYPRE_SMG);
      }
      else if (solver_id == 1)
      {
         HYPRE_SStructSplitSetStructSolver(solver, HYPRE_PFMG);
      }
      else if (solver_id == 8)
      {
         HYPRE_SStructSplitSetStructSolver(solver, HYPRE_Jacobi);
      }
      HYPRE_SStructSplitSetup(solver, A, b, x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("Split Solve");
      hypre_BeginTiming(time_index);

      HYPRE_SStructSplitSolve(solver, A, b, x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_SStructSplitGetNumIterations(solver, &num_iterations);
      HYPRE_SStructSplitGetFinalRelativeResidualNorm(solver, &final_res_norm);

      HYPRE_SStructSplitDestroy(solver);
   }

   /*-----------------------------------------------------------
    * Solve the system using PCG
    *-----------------------------------------------------------*/

   if ((solver_id >= 10) && (solver_id < 20))
   {
      time_index = hypre_InitializeTiming("PCG Setup");
      hypre_BeginTiming(time_index);

      HYPRE_SStructPCGCreate(comm, &solver);
      HYPRE_PCGSetMaxIter( (HYPRE_Solver) solver, 100 );
      HYPRE_PCGSetTol( (HYPRE_Solver) solver, tol );
      HYPRE_PCGSetTwoNorm( (HYPRE_Solver) solver, 1 );
      HYPRE_PCGSetRelChange( (HYPRE_Solver) solver, 0 );
      HYPRE_PCGSetPrintLevel( (HYPRE_Solver) solver, 1 );
      HYPRE_PCGSetRecomputeResidual( (HYPRE_Solver) solver, recompute_res);

      if ((solver_id == 10) || (solver_id == 11))
      {
         /* use Split solver as preconditioner */
         HYPRE_SStructSplitCreate(comm, &precond);
         HYPRE_SStructSplitSetMaxIter(precond, 1);
         HYPRE_SStructSplitSetTol(precond, 0.0);
         HYPRE_SStructSplitSetZeroGuess(precond);
         if (solver_id == 10)
         {
            HYPRE_SStructSplitSetStructSolver(precond, HYPRE_SMG);
         }
         else if (solver_id == 11)
         {
            HYPRE_SStructSplitSetStructSolver(precond, HYPRE_PFMG);
         }
         HYPRE_PCGSetPrecond( (HYPRE_Solver) solver,
                              (HYPRE_PtrToSolverFcn) HYPRE_SStructSplitSolve,
                              (HYPRE_PtrToSolverFcn) HYPRE_SStructSplitSetup,
                              (HYPRE_Solver) precond);
      }

      else if (solver_id == 13)
      {
         /* use SysPFMG solver as preconditioner */
         HYPRE_SStructSysPFMGCreate(comm, &precond);
         HYPRE_SStructSysPFMGSetMaxIter(precond, 1);
         HYPRE_SStructSysPFMGSetTol(precond, 0.0);
         HYPRE_SStructSysPFMGSetZeroGuess(precond);
         /* weighted Jacobi = 1; red-black GS = 2 */
         HYPRE_SStructSysPFMGSetRelaxType(precond, relax);
         if (usr_jacobi_weight)
         {
            HYPRE_SStructSysPFMGSetJacobiWeight(precond, jacobi_weight);
         }
         HYPRE_SStructSysPFMGSetNumPreRelax(precond, n_pre);
         HYPRE_SStructSysPFMGSetNumPostRelax(precond, n_post);
         HYPRE_SStructSysPFMGSetSkipRelax(precond, skip);
         /*HYPRE_StructPFMGSetDxyz(precond, dxyz);*/
         HYPRE_PCGSetPrecond( (HYPRE_Solver) solver,
                              (HYPRE_PtrToSolverFcn) HYPRE_SStructSysPFMGSolve,
                              (HYPRE_PtrToSolverFcn) HYPRE_SStructSysPFMGSetup,
                              (HYPRE_Solver) precond);

      }
      else if (solver_id == 18)
      {
         /* use diagonal scaling as preconditioner */
         precond = NULL;
         HYPRE_PCGSetPrecond( (HYPRE_Solver) solver,
                              (HYPRE_PtrToSolverFcn) HYPRE_SStructDiagScale,
                              (HYPRE_PtrToSolverFcn) HYPRE_SStructDiagScaleSetup,
                              (HYPRE_Solver) precond);
      }

      HYPRE_PCGSetup( (HYPRE_Solver) solver, (HYPRE_Matrix) A,
                      (HYPRE_Vector) b, (HYPRE_Vector) x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("PCG Solve");
      hypre_BeginTiming(time_index);

      HYPRE_PCGSolve( (HYPRE_Solver) solver, (HYPRE_Matrix) A,
                      (HYPRE_Vector) b, (HYPRE_Vector) x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_PCGGetNumIterations( (HYPRE_Solver) solver, &num_iterations );
      HYPRE_PCGGetFinalRelativeResidualNorm( (HYPRE_Solver) solver, &final_res_norm );
      HYPRE_SStructPCGDestroy(solver);

      if ((solver_id == 10) || (solver_id == 11))
      {
         HYPRE_SStructSplitDestroy(precond);
      }
      else if (solver_id == 13)
      {
         HYPRE_SStructSysPFMGDestroy(precond);
      }
   }

   /* begin lobpcg */

   /*-----------------------------------------------------------
    * Solve the eigenvalue problem using LOBPCG
    *-----------------------------------------------------------*/

   if ( lobpcgFlag && ( solver_id < 10 || solver_id >= 20 ) && verbosity )
   {
      hypre_printf("\nLOBPCG works with solvers 10, 11, 13 and 18 only\n");
   }

   if ( lobpcgFlag && (solver_id >= 10) && (solver_id < 20) )
   {

      interpreter = hypre_CTAlloc(mv_InterfaceInterpreter, 1, HYPRE_MEMORY_HOST);

      HYPRE_SStructSetupInterpreter( interpreter );
      HYPRE_SStructSetupMatvec(&matvec_fn);

      if (myid != 0)
      {
         verbosity = 0;
      }

      if ( pcgIterations > 0 )
      {

         time_index = hypre_InitializeTiming("PCG Setup");
         hypre_BeginTiming(time_index);

         HYPRE_SStructPCGCreate(comm, &solver);
         HYPRE_PCGSetMaxIter( (HYPRE_Solver) solver, pcgIterations );
         HYPRE_PCGSetTol( (HYPRE_Solver) solver, pcgTol );
         HYPRE_PCGSetTwoNorm( (HYPRE_Solver) solver, 1 );
         HYPRE_PCGSetRelChange( (HYPRE_Solver) solver, 0 );
         HYPRE_PCGSetPrintLevel( (HYPRE_Solver) solver, 0 );

         if ((solver_id == 10) || (solver_id == 11))
         {
            /* use Split solver as preconditioner */
            HYPRE_SStructSplitCreate(comm, &precond);
            HYPRE_SStructSplitSetMaxIter(precond, 1);
            HYPRE_SStructSplitSetTol(precond, 0.0);
            HYPRE_SStructSplitSetZeroGuess(precond);
            if (solver_id == 10)
            {
               HYPRE_SStructSplitSetStructSolver(precond, HYPRE_SMG);
            }
            else if (solver_id == 11)
            {
               HYPRE_SStructSplitSetStructSolver(precond, HYPRE_PFMG);
            }
            HYPRE_PCGSetPrecond( (HYPRE_Solver) solver,
                                 (HYPRE_PtrToSolverFcn) HYPRE_SStructSplitSolve,
                                 (HYPRE_PtrToSolverFcn) HYPRE_SStructSplitSetup,
                                 (HYPRE_Solver) precond);
         }

         else if (solver_id == 13)
         {
            /* use SysPFMG solver as preconditioner */
            HYPRE_SStructSysPFMGCreate(comm, &precond);
            HYPRE_SStructSysPFMGSetMaxIter(precond, 1);
            HYPRE_SStructSysPFMGSetTol(precond, 0.0);
            HYPRE_SStructSysPFMGSetZeroGuess(precond);
            /* weighted Jacobi = 1; red-black GS = 2 */
            HYPRE_SStructSysPFMGSetRelaxType(precond, 1);
            HYPRE_SStructSysPFMGSetNumPreRelax(precond, n_pre);
            HYPRE_SStructSysPFMGSetNumPostRelax(precond, n_post);
            HYPRE_SStructSysPFMGSetSkipRelax(precond, skip);
            /*HYPRE_StructPFMGSetDxyz(precond, dxyz);*/
            HYPRE_PCGSetPrecond( (HYPRE_Solver) solver,
                                 (HYPRE_PtrToSolverFcn) HYPRE_SStructSysPFMGSolve,
                                 (HYPRE_PtrToSolverFcn) HYPRE_SStructSysPFMGSetup,
                                 (HYPRE_Solver) precond);

         }
         else if (solver_id == 18)
         {
            /* use diagonal scaling as preconditioner */
            precond = NULL;
            HYPRE_PCGSetPrecond( (HYPRE_Solver) solver,
                                 (HYPRE_PtrToSolverFcn) HYPRE_SStructDiagScale,
                                 (HYPRE_PtrToSolverFcn) HYPRE_SStructDiagScaleSetup,
                                 (HYPRE_Solver) precond);
         }
         else if (solver_id != NO_SOLVER )
         {
            if ( verbosity )
            {
               hypre_printf("Solver ID not recognized - running inner PCG iterations without preconditioner\n\n");
            }
         }


         hypre_EndTiming(time_index);
         hypre_PrintTiming("Setup phase times", comm);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();

         HYPRE_LOBPCGCreate(interpreter, &matvec_fn, (HYPRE_Solver*)&lobpcg_solver);
         HYPRE_LOBPCGSetMaxIter((HYPRE_Solver)lobpcg_solver, maxIterations);
         HYPRE_LOBPCGSetPrecondUsageMode((HYPRE_Solver)lobpcg_solver, pcgMode);
         HYPRE_LOBPCGSetTol((HYPRE_Solver)lobpcg_solver, tol);
         HYPRE_LOBPCGSetPrintLevel((HYPRE_Solver)lobpcg_solver, verbosity);

         HYPRE_LOBPCGSetPrecond((HYPRE_Solver)lobpcg_solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_PCGSolve,
                                (HYPRE_PtrToSolverFcn) HYPRE_PCGSetup,
                                (HYPRE_Solver)solver);

         HYPRE_LOBPCGSetup((HYPRE_Solver)lobpcg_solver, (HYPRE_Matrix)A,
                           (HYPRE_Vector)b, (HYPRE_Vector)x);

         eigenvectors = mv_MultiVectorCreateFromSampleVector( interpreter,
                                                              blockSize,
                                                              x );
         eigenvalues = hypre_CTAlloc(HYPRE_Real,  blockSize, HYPRE_MEMORY_HOST);

         if ( lobpcgSeed )
         {
            mv_MultiVectorSetRandom( eigenvectors, lobpcgSeed );
         }
         else
         {
            mv_MultiVectorSetRandom( eigenvectors, (HYPRE_Int)time(0) );
         }

         time_index = hypre_InitializeTiming("PCG Solve");
         hypre_BeginTiming(time_index);

         HYPRE_LOBPCGSolve((HYPRE_Solver)lobpcg_solver, constrains,
                           eigenvectors, eigenvalues );

         hypre_EndTiming(time_index);
         hypre_PrintTiming("Solve phase times", comm);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();

         if ( checkOrtho )
         {

            gramXX = utilities_FortranMatrixCreate();
            identity = utilities_FortranMatrixCreate();

            utilities_FortranMatrixAllocateData( blockSize, blockSize, gramXX );
            utilities_FortranMatrixAllocateData( blockSize, blockSize, identity );

            lobpcg_MultiVectorByMultiVector( eigenvectors, eigenvectors, gramXX );
            utilities_FortranMatrixSetToIdentity( identity );
            utilities_FortranMatrixAdd( -1, identity, gramXX, gramXX );
            nonOrthF = utilities_FortranMatrixFNorm( gramXX );
            if ( myid == 0 )
            {
               hypre_printf("Non-orthonormality of eigenvectors: %12.5e\n", nonOrthF);
            }

            utilities_FortranMatrixDestroy( gramXX );
            utilities_FortranMatrixDestroy( identity );

         }

         if ( printLevel )
         {

            if ( myid == 0 )
            {
               if ( (filePtr = fopen("values.txt", "w")) )
               {
                  hypre_fprintf(filePtr, "%d\n", blockSize);
                  for ( i = 0; i < blockSize; i++ )
                  {
                     hypre_fprintf(filePtr, "%22.14e\n", eigenvalues[i]);
                  }
                  fclose(filePtr);
               }

               if ( (filePtr = fopen("residuals.txt", "w")) )
               {
                  residualNorms = HYPRE_LOBPCGResidualNorms( (HYPRE_Solver)lobpcg_solver );
                  residuals = utilities_FortranMatrixValues( residualNorms );
                  hypre_fprintf(filePtr, "%d\n", blockSize);
                  for ( i = 0; i < blockSize; i++ )
                  {
                     hypre_fprintf(filePtr, "%22.14e\n", residuals[i]);
                  }
                  fclose(filePtr);
               }

               if ( printLevel > 1 )
               {

                  printBuffer = utilities_FortranMatrixCreate();

                  iterations = HYPRE_LOBPCGIterations( (HYPRE_Solver)lobpcg_solver );

                  eigenvaluesHistory = HYPRE_LOBPCGEigenvaluesHistory( (HYPRE_Solver)lobpcg_solver );
                  utilities_FortranMatrixSelectBlock( eigenvaluesHistory,
                                                      1, blockSize, 1, iterations + 1, printBuffer );
                  utilities_FortranMatrixPrint( printBuffer, "val_hist.txt" );

                  residualNormsHistory = HYPRE_LOBPCGResidualNormsHistory( (HYPRE_Solver)lobpcg_solver );
                  utilities_FortranMatrixSelectBlock(residualNormsHistory,
                                                     1, blockSize, 1, iterations + 1, printBuffer );
                  utilities_FortranMatrixPrint( printBuffer, "res_hist.txt" );

                  utilities_FortranMatrixDestroy( printBuffer );
               }
            }
         }

         HYPRE_SStructPCGDestroy(solver);

         if ((solver_id == 10) || (solver_id == 11))
         {
            HYPRE_SStructSplitDestroy(precond);
         }
         else if (solver_id == 13)
         {
            HYPRE_SStructSysPFMGDestroy(precond);
         }

         HYPRE_LOBPCGDestroy((HYPRE_Solver)lobpcg_solver);
         mv_MultiVectorDestroy( eigenvectors );
         hypre_TFree(eigenvalues, HYPRE_MEMORY_HOST);
      }
      else
      {

         time_index = hypre_InitializeTiming("LOBPCG Setup");
         hypre_BeginTiming(time_index);

         HYPRE_LOBPCGCreate(interpreter, &matvec_fn, (HYPRE_Solver*)&solver);
         HYPRE_LOBPCGSetMaxIter( (HYPRE_Solver) solver, maxIterations );
         HYPRE_LOBPCGSetTol( (HYPRE_Solver) solver, tol );
         HYPRE_LOBPCGSetPrintLevel( (HYPRE_Solver) solver, verbosity );

         if ((solver_id == 10) || (solver_id == 11))
         {
            /* use Split solver as preconditioner */
            HYPRE_SStructSplitCreate(comm, &precond);
            HYPRE_SStructSplitSetMaxIter(precond, 1);
            HYPRE_SStructSplitSetTol(precond, 0.0);
            HYPRE_SStructSplitSetZeroGuess(precond);
            if (solver_id == 10)
            {
               HYPRE_SStructSplitSetStructSolver(precond, HYPRE_SMG);
            }
            else if (solver_id == 11)
            {
               HYPRE_SStructSplitSetStructSolver(precond, HYPRE_PFMG);
            }
            HYPRE_LOBPCGSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_SStructSplitSolve,
                                    (HYPRE_PtrToSolverFcn) HYPRE_SStructSplitSetup,
                                    (HYPRE_Solver) precond);
         }

         else if (solver_id == 13)
         {
            /* use SysPFMG solver as preconditioner */
            HYPRE_SStructSysPFMGCreate(comm, &precond);
            HYPRE_SStructSysPFMGSetMaxIter(precond, 1);
            HYPRE_SStructSysPFMGSetTol(precond, 0.0);
            HYPRE_SStructSysPFMGSetZeroGuess(precond);
            /* weighted Jacobi = 1; red-black GS = 2 */
            HYPRE_SStructSysPFMGSetRelaxType(precond, 1);
            HYPRE_SStructSysPFMGSetNumPreRelax(precond, n_pre);
            HYPRE_SStructSysPFMGSetNumPostRelax(precond, n_post);
            HYPRE_SStructSysPFMGSetSkipRelax(precond, skip);
            /*HYPRE_StructPFMGSetDxyz(precond, dxyz);*/
            HYPRE_LOBPCGSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_SStructSysPFMGSolve,
                                    (HYPRE_PtrToSolverFcn) HYPRE_SStructSysPFMGSetup,
                                    (HYPRE_Solver) precond);

         }
         else if (solver_id == 18)
         {
            /* use diagonal scaling as preconditioner */
            precond = NULL;
            HYPRE_LOBPCGSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_SStructDiagScale,
                                    (HYPRE_PtrToSolverFcn) HYPRE_SStructDiagScaleSetup,
                                    (HYPRE_Solver) precond);
         }
         else if (solver_id != NO_SOLVER )
         {
            if ( verbosity )
            {
               hypre_printf("Solver ID not recognized - running LOBPCG without preconditioner\n\n");
            }
         }

         HYPRE_LOBPCGSetup( (HYPRE_Solver) solver, (HYPRE_Matrix) A,
                            (HYPRE_Vector) b, (HYPRE_Vector) x);

         hypre_EndTiming(time_index);
         hypre_PrintTiming("Setup phase times", comm);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();

         eigenvectors = mv_MultiVectorCreateFromSampleVector( interpreter,
                                                              blockSize,
                                                              x );
         eigenvalues = hypre_CTAlloc(HYPRE_Real,  blockSize, HYPRE_MEMORY_HOST);

         if ( lobpcgSeed )
         {
            mv_MultiVectorSetRandom( eigenvectors, lobpcgSeed );
         }
         else
         {
            mv_MultiVectorSetRandom( eigenvectors, (HYPRE_Int)time(0) );
         }

         time_index = hypre_InitializeTiming("LOBPCG Solve");
         hypre_BeginTiming(time_index);

         HYPRE_LOBPCGSolve
         ( (HYPRE_Solver) solver, constrains, eigenvectors, eigenvalues );

         hypre_EndTiming(time_index);
         hypre_PrintTiming("Solve phase times", comm);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();

         if ( checkOrtho )
         {

            gramXX = utilities_FortranMatrixCreate();
            identity = utilities_FortranMatrixCreate();

            utilities_FortranMatrixAllocateData( blockSize, blockSize, gramXX );
            utilities_FortranMatrixAllocateData( blockSize, blockSize, identity );

            lobpcg_MultiVectorByMultiVector( eigenvectors, eigenvectors, gramXX );
            utilities_FortranMatrixSetToIdentity( identity );
            utilities_FortranMatrixAdd( -1, identity, gramXX, gramXX );
            nonOrthF = utilities_FortranMatrixFNorm( gramXX );
            if ( myid == 0 )
            {
               hypre_printf("Non-orthonormality of eigenvectors: %12.5e\n", nonOrthF);
            }

            utilities_FortranMatrixDestroy( gramXX );
            utilities_FortranMatrixDestroy( identity );

         }

         if ( printLevel )
         {

            if ( myid == 0 )
            {
               if ( (filePtr = fopen("values.txt", "w")) )
               {
                  hypre_fprintf(filePtr, "%d\n", blockSize);
                  for ( i = 0; i < blockSize; i++ )
                  {
                     hypre_fprintf(filePtr, "%22.14e\n", eigenvalues[i]);
                  }
                  fclose(filePtr);
               }

               if ( (filePtr = fopen("residuals.txt", "w")) )
               {
                  residualNorms = HYPRE_LOBPCGResidualNorms( (HYPRE_Solver)solver );
                  residuals = utilities_FortranMatrixValues( residualNorms );
                  hypre_fprintf(filePtr, "%d\n", blockSize);
                  for ( i = 0; i < blockSize; i++ )
                  {
                     hypre_fprintf(filePtr, "%22.14e\n", residuals[i]);
                  }
                  fclose(filePtr);
               }

               if ( printLevel > 1 )
               {

                  printBuffer = utilities_FortranMatrixCreate();

                  iterations = HYPRE_LOBPCGIterations( (HYPRE_Solver)solver );

                  eigenvaluesHistory = HYPRE_LOBPCGEigenvaluesHistory( (HYPRE_Solver)solver );
                  utilities_FortranMatrixSelectBlock( eigenvaluesHistory,
                                                      1, blockSize, 1, iterations + 1, printBuffer );
                  utilities_FortranMatrixPrint( printBuffer, "val_hist.txt" );

                  residualNormsHistory = HYPRE_LOBPCGResidualNormsHistory( (HYPRE_Solver)solver );
                  utilities_FortranMatrixSelectBlock(residualNormsHistory,
                                                     1, blockSize, 1, iterations + 1, printBuffer );
                  utilities_FortranMatrixPrint( printBuffer, "res_hist.txt" );

                  utilities_FortranMatrixDestroy( printBuffer );
               }
            }
         }

         HYPRE_LOBPCGDestroy((HYPRE_Solver)solver);

         if ((solver_id == 10) || (solver_id == 11))
         {
            HYPRE_SStructSplitDestroy(precond);
         }
         else if (solver_id == 13)
         {
            HYPRE_SStructSysPFMGDestroy(precond);
         }

         mv_MultiVectorDestroy( eigenvectors );
         hypre_TFree(eigenvalues, HYPRE_MEMORY_HOST);
      }

      hypre_TFree( interpreter, HYPRE_MEMORY_HOST);

   }

   /* end lobpcg */

   /*-----------------------------------------------------------
    * Solve the system using ParCSR version of PCG
    *-----------------------------------------------------------*/

   if ((solver_id >= 20) && (solver_id < 30))
   {
      time_index = hypre_InitializeTiming("PCG Setup");
      hypre_BeginTiming(time_index);

      HYPRE_ParCSRPCGCreate(comm, &par_solver);
      HYPRE_PCGSetMaxIter( par_solver, 100 );
      HYPRE_PCGSetTol( par_solver, tol );
      HYPRE_PCGSetTwoNorm( par_solver, 1 );
      HYPRE_PCGSetRelChange( par_solver, 0 );
      HYPRE_PCGSetPrintLevel( par_solver, 1 );
      HYPRE_PCGSetRecomputeResidual( (HYPRE_Solver) par_solver, recompute_res);

      if (solver_id == 20)
      {
         /* use BoomerAMG as preconditioner */
         HYPRE_BoomerAMGCreate(&par_precond);
         if (old_default) { HYPRE_BoomerAMGSetOldDefault(par_precond); }
         HYPRE_BoomerAMGSetStrongThreshold(par_precond, 0.25);
         HYPRE_BoomerAMGSetTol(par_precond, 0.0);
         HYPRE_BoomerAMGSetPrintLevel(par_precond, 1);
         HYPRE_BoomerAMGSetPrintFileName(par_precond, "sstruct.out.log");
         HYPRE_BoomerAMGSetMaxIter(par_precond, 1);
         HYPRE_PCGSetPrecond( par_solver,
                              (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                              (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
                              par_precond );
      }
      else if (solver_id == 21)
      {
         /* use Euclid as preconditioner */
         HYPRE_EuclidCreate(comm, &par_precond);
         HYPRE_EuclidSetParams(par_precond, argc, argv);
         HYPRE_PCGSetPrecond(par_solver,
                             (HYPRE_PtrToSolverFcn) HYPRE_EuclidSolve,
                             (HYPRE_PtrToSolverFcn) HYPRE_EuclidSetup,
                             par_precond);
      }
      else if (solver_id == 22)
      {
         /* use ParaSails as preconditioner */
         HYPRE_ParCSRParaSailsCreate(comm, &par_precond );
         HYPRE_ParCSRParaSailsSetParams(par_precond, 0.1, 1);
         HYPRE_PCGSetPrecond( par_solver,
                              (HYPRE_PtrToSolverFcn) HYPRE_ParCSRParaSailsSolve,
                              (HYPRE_PtrToSolverFcn) HYPRE_ParCSRParaSailsSetup,
                              par_precond );
      }

      else if (solver_id == 28)
      {
         /* use diagonal scaling as preconditioner */
         par_precond = NULL;
         HYPRE_PCGSetPrecond(  par_solver,
                               (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale,
                               (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScaleSetup,
                               par_precond );
      }

      HYPRE_PCGSetup( par_solver, (HYPRE_Matrix) par_A,
                      (HYPRE_Vector) par_b, (HYPRE_Vector) par_x );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("PCG Solve");
      hypre_BeginTiming(time_index);

      HYPRE_PCGSolve( par_solver, (HYPRE_Matrix) par_A,
                      (HYPRE_Vector) par_b, (HYPRE_Vector) par_x );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_PCGGetNumIterations( par_solver, &num_iterations );
      HYPRE_PCGGetFinalRelativeResidualNorm( par_solver, &final_res_norm );
      HYPRE_ParCSRPCGDestroy(par_solver);

      if (solver_id == 20)
      {
         HYPRE_BoomerAMGDestroy(par_precond);
      }
      else if (solver_id == 21)
      {
         HYPRE_EuclidDestroy(par_precond);
      }
      else if (solver_id == 22)
      {
         HYPRE_ParCSRParaSailsDestroy(par_precond);
      }
   }

   /*-----------------------------------------------------------
    * Solve the system using GMRES
    *-----------------------------------------------------------*/

   if ((solver_id >= 30) && (solver_id < 40))
   {
      time_index = hypre_InitializeTiming("GMRES Setup");
      hypre_BeginTiming(time_index);

      HYPRE_SStructGMRESCreate(comm, &solver);
      HYPRE_GMRESSetKDim( (HYPRE_Solver) solver, 5 );
      HYPRE_GMRESSetMaxIter( (HYPRE_Solver) solver, 100 );
      HYPRE_GMRESSetTol( (HYPRE_Solver) solver, tol );
      HYPRE_GMRESSetPrintLevel( (HYPRE_Solver) solver, 1 );
      HYPRE_GMRESSetLogging( (HYPRE_Solver) solver, 1 );

      if ((solver_id == 30) || (solver_id == 31))
      {
         /* use Split solver as preconditioner */
         HYPRE_SStructSplitCreate(comm, &precond);
         HYPRE_SStructSplitSetMaxIter(precond, 1);
         HYPRE_SStructSplitSetTol(precond, 0.0);
         HYPRE_SStructSplitSetZeroGuess(precond);
         if (solver_id == 30)
         {
            HYPRE_SStructSplitSetStructSolver(precond, HYPRE_SMG);
         }
         else if (solver_id == 31)
         {
            HYPRE_SStructSplitSetStructSolver(precond, HYPRE_PFMG);
         }
         HYPRE_GMRESSetPrecond( (HYPRE_Solver) solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_SStructSplitSolve,
                                (HYPRE_PtrToSolverFcn) HYPRE_SStructSplitSetup,
                                (HYPRE_Solver) precond );
      }

      else if (solver_id == 38)
      {
         /* use diagonal scaling as preconditioner */
         precond = NULL;
         HYPRE_GMRESSetPrecond( (HYPRE_Solver) solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_SStructDiagScale,
                                (HYPRE_PtrToSolverFcn) HYPRE_SStructDiagScaleSetup,
                                (HYPRE_Solver) precond );
      }

      HYPRE_GMRESSetup( (HYPRE_Solver) solver, (HYPRE_Matrix) A,
                        (HYPRE_Vector) b, (HYPRE_Vector) x );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("GMRES Solve");
      hypre_BeginTiming(time_index);

      HYPRE_GMRESSolve( (HYPRE_Solver) solver, (HYPRE_Matrix) A,
                        (HYPRE_Vector) b, (HYPRE_Vector) x );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_GMRESGetNumIterations( (HYPRE_Solver) solver, &num_iterations );
      HYPRE_GMRESGetFinalRelativeResidualNorm( (HYPRE_Solver) solver, &final_res_norm );
      HYPRE_SStructGMRESDestroy(solver);

      if ((solver_id == 30) || (solver_id == 31))
      {
         HYPRE_SStructSplitDestroy(precond);
      }
   }

   /*-----------------------------------------------------------
    * Solve the system using ParCSR version of GMRES
    *-----------------------------------------------------------*/

   if ((solver_id >= 40) && (solver_id < 50))
   {
      time_index = hypre_InitializeTiming("GMRES Setup");
      hypre_BeginTiming(time_index);

      HYPRE_ParCSRGMRESCreate(comm, &par_solver);
      HYPRE_GMRESSetKDim(par_solver, 5);
      HYPRE_GMRESSetMaxIter(par_solver, 100);
      HYPRE_GMRESSetTol(par_solver, tol);
      HYPRE_GMRESSetPrintLevel(par_solver, 1);
      HYPRE_GMRESSetLogging(par_solver, 1);

      if (solver_id == 40)
      {
         /* use BoomerAMG as preconditioner */
         HYPRE_BoomerAMGCreate(&par_precond);
         if (old_default) { HYPRE_BoomerAMGSetOldDefault(par_precond); }
         HYPRE_BoomerAMGSetStrongThreshold(par_precond, 0.25);
         HYPRE_BoomerAMGSetTol(par_precond, 0.0);
         HYPRE_BoomerAMGSetPrintLevel(par_precond, 1);
         HYPRE_BoomerAMGSetPrintFileName(par_precond, "sstruct.out.log");
         HYPRE_BoomerAMGSetMaxIter(par_precond, 1);
         HYPRE_GMRESSetPrecond( par_solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                                (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
                                par_precond);
      }
      else if (solver_id == 41)
      {
         /* use Euclid as preconditioner */
         HYPRE_EuclidCreate(comm, &par_precond);
         HYPRE_EuclidSetParams(par_precond, argc, argv);
         HYPRE_GMRESSetPrecond(par_solver,
                               (HYPRE_PtrToSolverFcn) HYPRE_EuclidSolve,
                               (HYPRE_PtrToSolverFcn) HYPRE_EuclidSetup,
                               par_precond);
      }
      else if (solver_id == 42)
      {
         /* use ParaSails as preconditioner */
         HYPRE_ParCSRParaSailsCreate(comm, &par_precond );
         HYPRE_ParCSRParaSailsSetParams(par_precond, 0.1, 1);
         HYPRE_ParCSRParaSailsSetSym(par_precond, 0);
         HYPRE_GMRESSetPrecond( par_solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_ParCSRParaSailsSolve,
                                (HYPRE_PtrToSolverFcn) HYPRE_ParCSRParaSailsSetup,
                                par_precond);
      }

      HYPRE_GMRESSetup( par_solver, (HYPRE_Matrix) par_A,
                        (HYPRE_Vector) par_b, (HYPRE_Vector) par_x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("GMRES Solve");
      hypre_BeginTiming(time_index);

      HYPRE_GMRESSolve( par_solver, (HYPRE_Matrix) par_A,
                        (HYPRE_Vector) par_b, (HYPRE_Vector) par_x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_GMRESGetNumIterations( par_solver, &num_iterations);
      HYPRE_GMRESGetFinalRelativeResidualNorm( par_solver, &final_res_norm);
      HYPRE_ParCSRGMRESDestroy(par_solver);

      if (solver_id == 40)
      {
         HYPRE_BoomerAMGDestroy(par_precond);
      }
      else if (solver_id == 41)
      {
         HYPRE_EuclidDestroy(par_precond);
      }
      else if (solver_id == 42)
      {
         HYPRE_ParCSRParaSailsDestroy(par_precond);
      }
   }

   /*-----------------------------------------------------------
    * Solve the system using BiCGSTAB
    *-----------------------------------------------------------*/

   if ((solver_id >= 50) && (solver_id < 60))
   {
      time_index = hypre_InitializeTiming("BiCGSTAB Setup");
      hypre_BeginTiming(time_index);

      HYPRE_SStructBiCGSTABCreate(comm, &solver);
      HYPRE_BiCGSTABSetMaxIter( (HYPRE_Solver) solver, 100 );
      HYPRE_BiCGSTABSetTol( (HYPRE_Solver) solver, tol );
      HYPRE_BiCGSTABSetPrintLevel( (HYPRE_Solver) solver, 1 );
      HYPRE_BiCGSTABSetLogging( (HYPRE_Solver) solver, 1 );

      if ((solver_id == 50) || (solver_id == 51))
      {
         /* use Split solver as preconditioner */
         HYPRE_SStructSplitCreate(comm, &precond);
         HYPRE_SStructSplitSetMaxIter(precond, 1);
         HYPRE_SStructSplitSetTol(precond, 0.0);
         HYPRE_SStructSplitSetZeroGuess(precond);
         if (solver_id == 50)
         {
            HYPRE_SStructSplitSetStructSolver(precond, HYPRE_SMG);
         }
         else if (solver_id == 51)
         {
            HYPRE_SStructSplitSetStructSolver(precond, HYPRE_PFMG);
         }
         HYPRE_BiCGSTABSetPrecond( (HYPRE_Solver) solver,
                                   (HYPRE_PtrToSolverFcn) HYPRE_SStructSplitSolve,
                                   (HYPRE_PtrToSolverFcn) HYPRE_SStructSplitSetup,
                                   (HYPRE_Solver) precond );
      }

      else if (solver_id == 58)
      {
         /* use diagonal scaling as preconditioner */
         precond = NULL;
         HYPRE_BiCGSTABSetPrecond( (HYPRE_Solver) solver,
                                   (HYPRE_PtrToSolverFcn) HYPRE_SStructDiagScale,
                                   (HYPRE_PtrToSolverFcn) HYPRE_SStructDiagScaleSetup,
                                   (HYPRE_Solver) precond );
      }

      HYPRE_BiCGSTABSetup( (HYPRE_Solver) solver, (HYPRE_Matrix) A,
                           (HYPRE_Vector) b, (HYPRE_Vector) x );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("BiCGSTAB Solve");
      hypre_BeginTiming(time_index);

      HYPRE_BiCGSTABSolve( (HYPRE_Solver) solver, (HYPRE_Matrix) A,
                           (HYPRE_Vector) b, (HYPRE_Vector) x );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_BiCGSTABGetNumIterations( (HYPRE_Solver) solver, &num_iterations );
      HYPRE_BiCGSTABGetFinalRelativeResidualNorm( (HYPRE_Solver) solver, &final_res_norm );
      HYPRE_SStructBiCGSTABDestroy(solver);

      if ((solver_id == 50) || (solver_id == 51))
      {
         HYPRE_SStructSplitDestroy(precond);
      }
   }

   /*-----------------------------------------------------------
    * Solve the system using ParCSR version of BiCGSTAB
    *-----------------------------------------------------------*/

   if ((solver_id >= 60) && (solver_id < 70))
   {
      time_index = hypre_InitializeTiming("BiCGSTAB Setup");
      hypre_BeginTiming(time_index);

      HYPRE_ParCSRBiCGSTABCreate(comm, &par_solver);
      HYPRE_BiCGSTABSetMaxIter(par_solver, 100);
      HYPRE_BiCGSTABSetTol(par_solver, tol);
      HYPRE_BiCGSTABSetPrintLevel(par_solver, 1);
      HYPRE_BiCGSTABSetLogging(par_solver, 1);

      if (solver_id == 60)
      {
         /* use BoomerAMG as preconditioner */
         HYPRE_BoomerAMGCreate(&par_precond);
         if (old_default) { HYPRE_BoomerAMGSetOldDefault(par_precond); }
         HYPRE_BoomerAMGSetStrongThreshold(par_precond, 0.25);
         HYPRE_BoomerAMGSetTol(par_precond, 0.0);
         HYPRE_BoomerAMGSetPrintLevel(par_precond, 1);
         HYPRE_BoomerAMGSetPrintFileName(par_precond, "sstruct.out.log");
         HYPRE_BoomerAMGSetMaxIter(par_precond, 1);
         HYPRE_BiCGSTABSetPrecond( par_solver,
                                   (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                                   (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
                                   par_precond);
      }
      else if (solver_id == 61)
      {
         /* use Euclid as preconditioner */
         HYPRE_EuclidCreate(comm, &par_precond);
         HYPRE_EuclidSetParams(par_precond, argc, argv);
         HYPRE_BiCGSTABSetPrecond(par_solver,
                                  (HYPRE_PtrToSolverFcn) HYPRE_EuclidSolve,
                                  (HYPRE_PtrToSolverFcn) HYPRE_EuclidSetup,
                                  par_precond);
      }

      else if (solver_id == 62)
      {
         /* use ParaSails as preconditioner */
         HYPRE_ParCSRParaSailsCreate(comm, &par_precond );
         HYPRE_ParCSRParaSailsSetParams(par_precond, 0.1, 1);
         HYPRE_ParCSRParaSailsSetSym(par_precond, 0);
         HYPRE_BiCGSTABSetPrecond( par_solver,
                                   (HYPRE_PtrToSolverFcn) HYPRE_ParCSRParaSailsSolve,
                                   (HYPRE_PtrToSolverFcn) HYPRE_ParCSRParaSailsSetup,
                                   par_precond);
      }

      HYPRE_BiCGSTABSetup( par_solver, (HYPRE_Matrix) par_A,
                           (HYPRE_Vector) par_b, (HYPRE_Vector) par_x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("BiCGSTAB Solve");
      hypre_BeginTiming(time_index);

      HYPRE_BiCGSTABSolve( par_solver, (HYPRE_Matrix) par_A,
                           (HYPRE_Vector) par_b, (HYPRE_Vector) par_x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_BiCGSTABGetNumIterations( par_solver, &num_iterations);
      HYPRE_BiCGSTABGetFinalRelativeResidualNorm( par_solver, &final_res_norm);
      HYPRE_ParCSRBiCGSTABDestroy(par_solver);

      if (solver_id == 60)
      {
         HYPRE_BoomerAMGDestroy(par_precond);
      }
      else if (solver_id == 61)
      {
         HYPRE_EuclidDestroy(par_precond);
      }
      else if (solver_id == 62)
      {
         HYPRE_ParCSRParaSailsDestroy(par_precond);
      }
   }

   /*-----------------------------------------------------------
    * Solve the system using Flexible GMRES
    *-----------------------------------------------------------*/

   if ((solver_id >= 70) && (solver_id < 80))
   {
      time_index = hypre_InitializeTiming("FlexGMRES Setup");
      hypre_BeginTiming(time_index);

      HYPRE_SStructFlexGMRESCreate(comm, &solver);
      HYPRE_FlexGMRESSetKDim( (HYPRE_Solver) solver, 5 );
      HYPRE_FlexGMRESSetMaxIter( (HYPRE_Solver) solver, 100 );
      HYPRE_FlexGMRESSetTol( (HYPRE_Solver) solver, tol );
      HYPRE_FlexGMRESSetPrintLevel( (HYPRE_Solver) solver, 1 );
      HYPRE_FlexGMRESSetLogging( (HYPRE_Solver) solver, 1 );

      if ((solver_id == 70) || (solver_id == 71))
      {
         /* use Split solver as preconditioner */
         HYPRE_SStructSplitCreate(comm, &precond);
         HYPRE_SStructSplitSetMaxIter(precond, 1);
         HYPRE_SStructSplitSetTol(precond, 0.0);
         HYPRE_SStructSplitSetZeroGuess(precond);
         if (solver_id == 70)
         {
            HYPRE_SStructSplitSetStructSolver(precond, HYPRE_SMG);
         }
         else if (solver_id == 71)
         {
            HYPRE_SStructSplitSetStructSolver(precond, HYPRE_PFMG);
         }
         HYPRE_FlexGMRESSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_SStructSplitSolve,
                                    (HYPRE_PtrToSolverFcn) HYPRE_SStructSplitSetup,
                                    (HYPRE_Solver) precond );
      }

      else if (solver_id == 78)
      {
         /* use diagonal scaling as preconditioner */
         precond = NULL;
         HYPRE_FlexGMRESSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_SStructDiagScale,
                                    (HYPRE_PtrToSolverFcn) HYPRE_SStructDiagScaleSetup,
                                    (HYPRE_Solver) precond );
      }

      HYPRE_FlexGMRESSetup( (HYPRE_Solver) solver, (HYPRE_Matrix) A,
                            (HYPRE_Vector) b, (HYPRE_Vector) x );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("FlexGMRES Solve");
      hypre_BeginTiming(time_index);

      HYPRE_FlexGMRESSolve( (HYPRE_Solver) solver, (HYPRE_Matrix) A,
                            (HYPRE_Vector) b, (HYPRE_Vector) x );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_FlexGMRESGetNumIterations( (HYPRE_Solver) solver, &num_iterations );
      HYPRE_FlexGMRESGetFinalRelativeResidualNorm( (HYPRE_Solver) solver, &final_res_norm );
      HYPRE_SStructFlexGMRESDestroy(solver);

      if ((solver_id == 70) || (solver_id == 71))
      {
         HYPRE_SStructSplitDestroy(precond);
      }
   }

   /*-----------------------------------------------------------
    * Solve the system using ParCSR version of Flexible GMRES
    *-----------------------------------------------------------*/

   if ((solver_id >= 80) && (solver_id < 90))
   {
      time_index = hypre_InitializeTiming("FlexGMRES Setup");
      hypre_BeginTiming(time_index);

      HYPRE_ParCSRFlexGMRESCreate(comm, &par_solver);
      HYPRE_FlexGMRESSetKDim(par_solver, 5);
      HYPRE_FlexGMRESSetMaxIter(par_solver, 100);
      HYPRE_FlexGMRESSetTol(par_solver, tol);
      HYPRE_FlexGMRESSetPrintLevel(par_solver, 1);
      HYPRE_FlexGMRESSetLogging(par_solver, 1);

      if (solver_id == 80)
      {
         /* use BoomerAMG as preconditioner */
         HYPRE_BoomerAMGCreate(&par_precond);
         if (old_default) { HYPRE_BoomerAMGSetOldDefault(par_precond); }
         HYPRE_BoomerAMGSetStrongThreshold(par_precond, 0.25);
         HYPRE_BoomerAMGSetTol(par_precond, 0.0);
         HYPRE_BoomerAMGSetPrintLevel(par_precond, 1);
         HYPRE_BoomerAMGSetPrintFileName(par_precond, "sstruct.out.log");
         HYPRE_BoomerAMGSetMaxIter(par_precond, 1);
         HYPRE_FlexGMRESSetPrecond( par_solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                                    (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
                                    par_precond);
      }

      HYPRE_FlexGMRESSetup( par_solver, (HYPRE_Matrix) par_A,
                            (HYPRE_Vector) par_b, (HYPRE_Vector) par_x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("FlexGMRES Solve");
      hypre_BeginTiming(time_index);

      HYPRE_FlexGMRESSolve( par_solver, (HYPRE_Matrix) par_A,
                            (HYPRE_Vector) par_b, (HYPRE_Vector) par_x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_FlexGMRESGetNumIterations( par_solver, &num_iterations);
      HYPRE_FlexGMRESGetFinalRelativeResidualNorm( par_solver, &final_res_norm);
      HYPRE_ParCSRFlexGMRESDestroy(par_solver);

      if (solver_id == 80)
      {
         HYPRE_BoomerAMGDestroy(par_precond);
      }
   }

   /*-----------------------------------------------------------
    * Solve the system using ParCSR version of LGMRES
    *-----------------------------------------------------------*/

   if ((solver_id >= 90) && (solver_id < 100))
   {
      time_index = hypre_InitializeTiming("LGMRES Setup");
      hypre_BeginTiming(time_index);

      HYPRE_ParCSRLGMRESCreate(comm, &par_solver);
      HYPRE_LGMRESSetKDim(par_solver, 10);
      HYPRE_LGMRESSetAugDim(par_solver, 2);
      HYPRE_LGMRESSetMaxIter(par_solver, 100);
      HYPRE_LGMRESSetTol(par_solver, tol);
      HYPRE_LGMRESSetPrintLevel(par_solver, 1);
      HYPRE_LGMRESSetLogging(par_solver, 1);

      if (solver_id == 90)
      {
         /* use BoomerAMG as preconditioner */
         HYPRE_BoomerAMGCreate(&par_precond);
         if (old_default) { HYPRE_BoomerAMGSetOldDefault(par_precond); }
         HYPRE_BoomerAMGSetStrongThreshold(par_precond, 0.25);
         HYPRE_BoomerAMGSetTol(par_precond, 0.0);
         HYPRE_BoomerAMGSetPrintLevel(par_precond, 1);
         HYPRE_BoomerAMGSetPrintFileName(par_precond, "sstruct.out.log");
         HYPRE_BoomerAMGSetMaxIter(par_precond, 1);
         HYPRE_LGMRESSetPrecond( par_solver,
                                 (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                                 (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
                                 par_precond);
      }

      HYPRE_LGMRESSetup( par_solver, (HYPRE_Matrix) par_A,
                         (HYPRE_Vector) par_b, (HYPRE_Vector) par_x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("LGMRES Solve");
      hypre_BeginTiming(time_index);

      HYPRE_LGMRESSolve( par_solver, (HYPRE_Matrix) par_A,
                         (HYPRE_Vector) par_b, (HYPRE_Vector) par_x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_LGMRESGetNumIterations( par_solver, &num_iterations);
      HYPRE_LGMRESGetFinalRelativeResidualNorm( par_solver, &final_res_norm);
      HYPRE_ParCSRLGMRESDestroy(par_solver);

      if (solver_id == 90)
      {
         HYPRE_BoomerAMGDestroy(par_precond);
      }
   }

   /*-----------------------------------------------------------
    * Solve the system using ParCSR hybrid DSCG/BoomerAMG
    *-----------------------------------------------------------*/

   if (solver_id == 120)
   {
      time_index = hypre_InitializeTiming("Hybrid Setup");
      hypre_BeginTiming(time_index);

      HYPRE_ParCSRHybridCreate(&par_solver);
      HYPRE_ParCSRHybridSetTol(par_solver, tol);
      HYPRE_ParCSRHybridSetTwoNorm(par_solver, 1);
      HYPRE_ParCSRHybridSetRelChange(par_solver, 0);
      HYPRE_ParCSRHybridSetPrintLevel(par_solver, 1); //13
      HYPRE_ParCSRHybridSetLogging(par_solver, 1);
      HYPRE_ParCSRHybridSetSolverType(par_solver, solver_type);
      HYPRE_ParCSRHybridSetRecomputeResidual(par_solver, recompute_res);

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
      /*
      HYPRE_ParCSRHybridSetPMaxElmts(par_solver, 8);
      HYPRE_ParCSRHybridSetRelaxType(par_solver, 18);
      HYPRE_ParCSRHybridSetCycleRelaxType(par_solver, 9, 3);
      HYPRE_ParCSRHybridSetCoarsenType(par_solver, 8);
      HYPRE_ParCSRHybridSetInterpType(par_solver, 3);
      HYPRE_ParCSRHybridSetMaxCoarseSize(par_solver, 20);
      */
#endif

#if SECOND_TIME
      hypre_ParVector *par_x2 =
         hypre_ParVectorCreate(hypre_ParVectorComm(par_x), hypre_ParVectorGlobalSize(par_x),
                               hypre_ParVectorPartitioning(par_x));
      hypre_ParVectorInitialize(par_x2);
      hypre_ParVectorCopy(par_x, par_x2);

      HYPRE_ParCSRHybridSetup(par_solver, par_A, par_b, par_x);
      HYPRE_ParCSRHybridSolve(par_solver, par_A, par_b, par_x);

      hypre_ParVectorCopy(par_x2, par_x);
#endif

      hypre_GpuProfilingPushRange("HybridSolve");
      //cudaProfilerStart();

      HYPRE_ParCSRHybridSetup(par_solver, par_A, par_b, par_x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("Hybrid Solve");
      hypre_BeginTiming(time_index);

      HYPRE_ParCSRHybridSolve(par_solver, par_A, par_b, par_x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_ParCSRHybridGetNumIterations(par_solver, &num_iterations);
      HYPRE_ParCSRHybridGetFinalRelativeResidualNorm(par_solver, &final_res_norm);

      /*
      HYPRE_Real time[4];
      HYPRE_ParCSRHybridGetSetupSolveTime(par_solver, time);
      if (myid == 0)
      {
         printf("ParCSRHybrid: Setup-Time1 %f, Solve-Time1 %f, Setup-Time2 %f, Solve-Time2 %f\n",
                time[0], time[1], time[2], time[3]);
      }
      */

      HYPRE_ParCSRHybridDestroy(par_solver);

      hypre_GpuProfilingPopRange();
      //cudaProfilerStop();

#if SECOND_TIME
      hypre_ParVectorDestroy(par_x2);
#endif
   }

   /*-----------------------------------------------------------
    * Solve the system using Struct solvers
    *-----------------------------------------------------------*/

   if (solver_id == 200)
   {
      time_index = hypre_InitializeTiming("SMG Setup");
      hypre_BeginTiming(time_index);

      HYPRE_StructSMGCreate(comm, &struct_solver);
      HYPRE_StructSMGSetMemoryUse(struct_solver, 0);
      HYPRE_StructSMGSetMaxIter(struct_solver, 50);
      HYPRE_StructSMGSetTol(struct_solver, tol);
      HYPRE_StructSMGSetRelChange(struct_solver, 0);
      HYPRE_StructSMGSetNumPreRelax(struct_solver, n_pre);
      HYPRE_StructSMGSetNumPostRelax(struct_solver, n_post);
      HYPRE_StructSMGSetPrintLevel(struct_solver, 1);
      HYPRE_StructSMGSetLogging(struct_solver, 1);
      HYPRE_StructSMGSetup(struct_solver, sA, sb, sx);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("SMG Solve");
      hypre_BeginTiming(time_index);

      HYPRE_StructSMGSolve(struct_solver, sA, sb, sx);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_StructSMGGetNumIterations(struct_solver, &num_iterations);
      HYPRE_StructSMGGetFinalRelativeResidualNorm(struct_solver, &final_res_norm);
      HYPRE_StructSMGDestroy(struct_solver);
   }

   else if ( solver_id == 201 || solver_id == 203 || solver_id == 204 )
   {
      time_index = hypre_InitializeTiming("PFMG Setup");
      hypre_BeginTiming(time_index);

      HYPRE_StructPFMGCreate(comm, &struct_solver);
      HYPRE_StructPFMGSetMaxIter(struct_solver, 50);
      HYPRE_StructPFMGSetTol(struct_solver, tol);
      HYPRE_StructPFMGSetRelChange(struct_solver, 0);
      HYPRE_StructPFMGSetRAPType(struct_solver, rap);
      HYPRE_StructPFMGSetRelaxType(struct_solver, relax);
      if (usr_jacobi_weight)
      {
         HYPRE_StructPFMGSetJacobiWeight(struct_solver, jacobi_weight);
      }
      HYPRE_StructPFMGSetNumPreRelax(struct_solver, n_pre);
      HYPRE_StructPFMGSetNumPostRelax(struct_solver, n_post);
      HYPRE_StructPFMGSetSkipRelax(struct_solver, skip);
      /*HYPRE_StructPFMGSetDxyz(struct_solver, dxyz);*/
      HYPRE_StructPFMGSetPrintLevel(struct_solver, 1);
      HYPRE_StructPFMGSetLogging(struct_solver, 1);
      HYPRE_StructPFMGSetup(struct_solver, sA, sb, sx);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("PFMG Solve");
      hypre_BeginTiming(time_index);

      HYPRE_StructPFMGSolve(struct_solver, sA, sb, sx);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_StructPFMGGetNumIterations(struct_solver, &num_iterations);
      HYPRE_StructPFMGGetFinalRelativeResidualNorm(struct_solver, &final_res_norm);
      HYPRE_StructPFMGDestroy(struct_solver);
   }

   /*-----------------------------------------------------------
    * Solve the system using Cyclic Reduction
    *-----------------------------------------------------------*/

   else if ( solver_id == 205 )
   {
      HYPRE_StructVector  sr;

      time_index = hypre_InitializeTiming("CycRed Setup");
      hypre_BeginTiming(time_index);

      HYPRE_StructCycRedCreate(comm, &struct_solver);
      HYPRE_StructCycRedSetTDim(struct_solver, cycred_tdim);
      HYPRE_StructCycRedSetBase(struct_solver, data.ndim,
                                cycred_index, cycred_stride);
      HYPRE_StructCycRedSetup(struct_solver, sA, sb, sx);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("CycRed Solve");
      hypre_BeginTiming(time_index);

      HYPRE_StructCycRedSolve(struct_solver, sA, sb, sx);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      num_iterations = 1;
      HYPRE_StructVectorCreate(comm,
                               hypre_StructVectorGrid(sb), &sr);
      HYPRE_StructVectorInitialize(sr);
      HYPRE_StructVectorAssemble(sr);
      HYPRE_StructVectorCopy(sb, sr);
      hypre_StructMatvec(-1.0, sA, sx, 1.0, sr);
      /* Using an inner product instead of a norm to help with testing */
      final_res_norm = hypre_StructInnerProd(sr, sr);
      if (final_res_norm < 1.0e-20)
      {
         final_res_norm = 0.0;
      }
      HYPRE_StructVectorDestroy(sr);

      HYPRE_StructCycRedDestroy(struct_solver);
   }

   /*-----------------------------------------------------------
    * Solve the system using SparseMSG
    *-----------------------------------------------------------*/

   else if (solver_id == 202)
   {
      time_index = hypre_InitializeTiming("SparseMSG Setup");
      hypre_BeginTiming(time_index);

      HYPRE_StructSparseMSGCreate(comm, &struct_solver);
      HYPRE_StructSparseMSGSetMaxIter(struct_solver, 50);
      HYPRE_StructSparseMSGSetJump(struct_solver, jump);
      HYPRE_StructSparseMSGSetTol(struct_solver, tol);
      HYPRE_StructSparseMSGSetRelChange(struct_solver, 0);
      HYPRE_StructSparseMSGSetRelaxType(struct_solver, relax);
      if (usr_jacobi_weight)
      {
         HYPRE_StructSparseMSGSetJacobiWeight(struct_solver, jacobi_weight);
      }
      HYPRE_StructSparseMSGSetNumPreRelax(struct_solver, n_pre);
      HYPRE_StructSparseMSGSetNumPostRelax(struct_solver, n_post);
      HYPRE_StructSparseMSGSetPrintLevel(struct_solver, 1);
      HYPRE_StructSparseMSGSetLogging(struct_solver, 1);
      HYPRE_StructSparseMSGSetup(struct_solver, sA, sb, sx);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("SparseMSG Solve");
      hypre_BeginTiming(time_index);

      HYPRE_StructSparseMSGSolve(struct_solver, sA, sb, sx);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_StructSparseMSGGetNumIterations(struct_solver, &num_iterations);
      HYPRE_StructSparseMSGGetFinalRelativeResidualNorm(struct_solver,
                                                        &final_res_norm);
      HYPRE_StructSparseMSGDestroy(struct_solver);
   }

   /*-----------------------------------------------------------
    * Solve the system using Jacobi
    *-----------------------------------------------------------*/

   else if ( solver_id == 208 )
   {
      time_index = hypre_InitializeTiming("Jacobi Setup");
      hypre_BeginTiming(time_index);

      HYPRE_StructJacobiCreate(comm, &struct_solver);
      HYPRE_StructJacobiSetMaxIter(struct_solver, 100);
      HYPRE_StructJacobiSetTol(struct_solver, tol);
      HYPRE_StructJacobiSetup(struct_solver, sA, sb, sx);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("Jacobi Solve");
      hypre_BeginTiming(time_index);

      HYPRE_StructJacobiSolve(struct_solver, sA, sb, sx);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_StructJacobiGetNumIterations(struct_solver, &num_iterations);
      HYPRE_StructJacobiGetFinalRelativeResidualNorm(struct_solver,
                                                     &final_res_norm);
      HYPRE_StructJacobiDestroy(struct_solver);
   }

   /*-----------------------------------------------------------
    * Solve the system using CG
    *-----------------------------------------------------------*/

   if ((solver_id > 209) && (solver_id < 220))
   {
      time_index = hypre_InitializeTiming("PCG Setup");
      hypre_BeginTiming(time_index);

      HYPRE_StructPCGCreate(comm, &struct_solver);
      HYPRE_PCGSetMaxIter( (HYPRE_Solver)struct_solver, 100 );
      HYPRE_PCGSetTol( (HYPRE_Solver)struct_solver, tol );
      HYPRE_PCGSetTwoNorm( (HYPRE_Solver)struct_solver, 1 );
      HYPRE_PCGSetRelChange( (HYPRE_Solver)struct_solver, 0 );
      HYPRE_PCGSetPrintLevel( (HYPRE_Solver)struct_solver, 1 );
      HYPRE_PCGSetRecomputeResidual( (HYPRE_Solver)struct_solver, recompute_res);

      if (solver_id == 210)
      {
         /* use symmetric SMG as preconditioner */
         HYPRE_StructSMGCreate(comm, &struct_precond);
         HYPRE_StructSMGSetMemoryUse(struct_precond, 0);
         HYPRE_StructSMGSetMaxIter(struct_precond, 1);
         HYPRE_StructSMGSetTol(struct_precond, 0.0);
         HYPRE_StructSMGSetZeroGuess(struct_precond);
         HYPRE_StructSMGSetNumPreRelax(struct_precond, n_pre);
         HYPRE_StructSMGSetNumPostRelax(struct_precond, n_post);
         HYPRE_StructSMGSetPrintLevel(struct_precond, 0);
         HYPRE_StructSMGSetLogging(struct_precond, 0);
         HYPRE_PCGSetPrecond( (HYPRE_Solver) struct_solver,
                              (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSolve,
                              (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSetup,
                              (HYPRE_Solver) struct_precond);
      }

      else if (solver_id == 211)
      {
         /* use symmetric PFMG as preconditioner */
         HYPRE_StructPFMGCreate(comm, &struct_precond);
         HYPRE_StructPFMGSetMaxIter(struct_precond, 1);
         HYPRE_StructPFMGSetTol(struct_precond, 0.0);
         HYPRE_StructPFMGSetZeroGuess(struct_precond);
         HYPRE_StructPFMGSetRAPType(struct_precond, rap);
         HYPRE_StructPFMGSetRelaxType(struct_precond, relax);
         if (usr_jacobi_weight)
         {
            HYPRE_StructPFMGSetJacobiWeight(struct_precond, jacobi_weight);
         }
         HYPRE_StructPFMGSetNumPreRelax(struct_precond, n_pre);
         HYPRE_StructPFMGSetNumPostRelax(struct_precond, n_post);
         HYPRE_StructPFMGSetSkipRelax(struct_precond, skip);
         /*HYPRE_StructPFMGSetDxyz(struct_precond, dxyz);*/
         HYPRE_StructPFMGSetPrintLevel(struct_precond, 0);
         HYPRE_StructPFMGSetLogging(struct_precond, 0);
         HYPRE_PCGSetPrecond( (HYPRE_Solver) struct_solver,
                              (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSolve,
                              (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSetup,
                              (HYPRE_Solver) struct_precond);
      }

      else if (solver_id == 212)
      {
         /* use symmetric SparseMSG as preconditioner */
         HYPRE_StructSparseMSGCreate(comm, &struct_precond);
         HYPRE_StructSparseMSGSetMaxIter(struct_precond, 1);
         HYPRE_StructSparseMSGSetJump(struct_precond, jump);
         HYPRE_StructSparseMSGSetTol(struct_precond, 0.0);
         HYPRE_StructSparseMSGSetZeroGuess(struct_precond);
         HYPRE_StructSparseMSGSetRelaxType(struct_precond, relax);
         if (usr_jacobi_weight)
         {
            HYPRE_StructSparseMSGSetJacobiWeight(struct_precond, jacobi_weight);
         }
         HYPRE_StructSparseMSGSetNumPreRelax(struct_precond, n_pre);
         HYPRE_StructSparseMSGSetNumPostRelax(struct_precond, n_post);
         HYPRE_StructSparseMSGSetPrintLevel(struct_precond, 0);
         HYPRE_StructSparseMSGSetLogging(struct_precond, 0);
         HYPRE_PCGSetPrecond( (HYPRE_Solver) struct_solver,
                              (HYPRE_PtrToSolverFcn) HYPRE_StructSparseMSGSolve,
                              (HYPRE_PtrToSolverFcn) HYPRE_StructSparseMSGSetup,
                              (HYPRE_Solver) struct_precond);
      }

      else if (solver_id == 217)
      {
         /* use two-step Jacobi as preconditioner */
         HYPRE_StructJacobiCreate(comm, &struct_precond);
         HYPRE_StructJacobiSetMaxIter(struct_precond, 2);
         HYPRE_StructJacobiSetTol(struct_precond, 0.0);
         HYPRE_StructJacobiSetZeroGuess(struct_precond);
         HYPRE_PCGSetPrecond( (HYPRE_Solver) struct_solver,
                              (HYPRE_PtrToSolverFcn) HYPRE_StructJacobiSolve,
                              (HYPRE_PtrToSolverFcn) HYPRE_StructJacobiSetup,
                              (HYPRE_Solver) struct_precond);
      }

      else if (solver_id == 218)
      {
         /* use diagonal scaling as preconditioner */
         struct_precond = NULL;
         HYPRE_PCGSetPrecond( (HYPRE_Solver) struct_solver,
                              (HYPRE_PtrToSolverFcn) HYPRE_StructDiagScale,
                              (HYPRE_PtrToSolverFcn) HYPRE_StructDiagScaleSetup,
                              (HYPRE_Solver) struct_precond);
      }

      HYPRE_PCGSetup
      ( (HYPRE_Solver)struct_solver, (HYPRE_Matrix)sA, (HYPRE_Vector)sb,
        (HYPRE_Vector)sx );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("PCG Solve");
      hypre_BeginTiming(time_index);

      HYPRE_PCGSolve
      ( (HYPRE_Solver) struct_solver, (HYPRE_Matrix)sA, (HYPRE_Vector)sb,
        (HYPRE_Vector)sx);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_PCGGetNumIterations( (HYPRE_Solver)struct_solver, &num_iterations );
      HYPRE_PCGGetFinalRelativeResidualNorm( (HYPRE_Solver)struct_solver, &final_res_norm );
      HYPRE_StructPCGDestroy(struct_solver);

      if (solver_id == 210)
      {
         HYPRE_StructSMGDestroy(struct_precond);
      }
      else if (solver_id == 211)
      {
         HYPRE_StructPFMGDestroy(struct_precond);
      }
      else if (solver_id == 212)
      {
         HYPRE_StructSparseMSGDestroy(struct_precond);
      }
      else if (solver_id == 217)
      {
         HYPRE_StructJacobiDestroy(struct_precond);
      }
   }

   /*-----------------------------------------------------------
    * Solve the system using Hybrid
    *-----------------------------------------------------------*/

   if ((solver_id > 219) && (solver_id < 230))
   {
      time_index = hypre_InitializeTiming("Hybrid Setup");
      hypre_BeginTiming(time_index);

      HYPRE_StructHybridCreate(comm, &struct_solver);
      HYPRE_StructHybridSetDSCGMaxIter(struct_solver, 100);
      HYPRE_StructHybridSetPCGMaxIter(struct_solver, 100);
      HYPRE_StructHybridSetTol(struct_solver, tol);
      /*HYPRE_StructHybridSetPCGAbsoluteTolFactor(struct_solver, 1.0e-200);*/
      HYPRE_StructHybridSetConvergenceTol(struct_solver, cf_tol);
      HYPRE_StructHybridSetTwoNorm(struct_solver, 1);
      HYPRE_StructHybridSetRelChange(struct_solver, 0);
      if (solver_type == 2) /* for use with GMRES */
      {
         HYPRE_StructHybridSetStopCrit(struct_solver, 0);
         HYPRE_StructHybridSetKDim(struct_solver, 10);
      }
      HYPRE_StructHybridSetPrintLevel(struct_solver, 1);
      HYPRE_StructHybridSetLogging(struct_solver, 1);
      HYPRE_StructHybridSetSolverType(struct_solver, solver_type);
      HYPRE_StructHybridSetRecomputeResidual(struct_solver, recompute_res);

      if (solver_id == 220)
      {
         /* use symmetric SMG as preconditioner */
         HYPRE_StructSMGCreate(comm, &struct_precond);
         HYPRE_StructSMGSetMemoryUse(struct_precond, 0);
         HYPRE_StructSMGSetMaxIter(struct_precond, 1);
         HYPRE_StructSMGSetTol(struct_precond, 0.0);
         HYPRE_StructSMGSetZeroGuess(struct_precond);
         HYPRE_StructSMGSetNumPreRelax(struct_precond, n_pre);
         HYPRE_StructSMGSetNumPostRelax(struct_precond, n_post);
         HYPRE_StructSMGSetPrintLevel(struct_precond, 0);
         HYPRE_StructSMGSetLogging(struct_precond, 0);
         HYPRE_StructHybridSetPrecond(struct_solver,
                                      HYPRE_StructSMGSolve,
                                      HYPRE_StructSMGSetup,
                                      struct_precond);
      }

      else if (solver_id == 221)
      {
         /* use symmetric PFMG as preconditioner */
         HYPRE_StructPFMGCreate(comm, &struct_precond);
         HYPRE_StructPFMGSetMaxIter(struct_precond, 1);
         HYPRE_StructPFMGSetTol(struct_precond, 0.0);
         HYPRE_StructPFMGSetZeroGuess(struct_precond);
         HYPRE_StructPFMGSetRAPType(struct_precond, rap);
         HYPRE_StructPFMGSetRelaxType(struct_precond, relax);
         if (usr_jacobi_weight)
         {
            HYPRE_StructPFMGSetJacobiWeight(struct_precond, jacobi_weight);
         }
         HYPRE_StructPFMGSetNumPreRelax(struct_precond, n_pre);
         HYPRE_StructPFMGSetNumPostRelax(struct_precond, n_post);
         HYPRE_StructPFMGSetSkipRelax(struct_precond, skip);
         /*HYPRE_StructPFMGSetDxyz(struct_precond, dxyz);*/
         HYPRE_StructPFMGSetPrintLevel(struct_precond, 0);
         HYPRE_StructPFMGSetLogging(struct_precond, 0);
         HYPRE_StructHybridSetPrecond(struct_solver,
                                      HYPRE_StructPFMGSolve,
                                      HYPRE_StructPFMGSetup,
                                      struct_precond);
      }

      else if (solver_id == 222)
      {
         /* use symmetric SparseMSG as preconditioner */
         HYPRE_StructSparseMSGCreate(comm, &struct_precond);
         HYPRE_StructSparseMSGSetJump(struct_precond, jump);
         HYPRE_StructSparseMSGSetMaxIter(struct_precond, 1);
         HYPRE_StructSparseMSGSetTol(struct_precond, 0.0);
         HYPRE_StructSparseMSGSetZeroGuess(struct_precond);
         HYPRE_StructSparseMSGSetRelaxType(struct_precond, relax);
         if (usr_jacobi_weight)
         {
            HYPRE_StructSparseMSGSetJacobiWeight(struct_precond, jacobi_weight);
         }
         HYPRE_StructSparseMSGSetNumPreRelax(struct_precond, n_pre);
         HYPRE_StructSparseMSGSetNumPostRelax(struct_precond, n_post);
         HYPRE_StructSparseMSGSetPrintLevel(struct_precond, 0);
         HYPRE_StructSparseMSGSetLogging(struct_precond, 0);
         HYPRE_StructHybridSetPrecond(struct_solver,
                                      HYPRE_StructSparseMSGSolve,
                                      HYPRE_StructSparseMSGSetup,
                                      struct_precond);
      }

      HYPRE_StructHybridSetup(struct_solver, sA, sb, sx);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("Hybrid Solve");
      hypre_BeginTiming(time_index);

      HYPRE_StructHybridSolve(struct_solver, sA, sb, sx);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_StructHybridGetNumIterations(struct_solver, &num_iterations);
      HYPRE_StructHybridGetFinalRelativeResidualNorm(struct_solver, &final_res_norm);
      HYPRE_StructHybridDestroy(struct_solver);

      if (solver_id == 220)
      {
         HYPRE_StructSMGDestroy(struct_precond);
      }
      else if (solver_id == 221)
      {
         HYPRE_StructPFMGDestroy(struct_precond);
      }
      else if (solver_id == 222)
      {
         HYPRE_StructSparseMSGDestroy(struct_precond);
      }
   }

   /*-----------------------------------------------------------
    * Solve the system using GMRES
    *-----------------------------------------------------------*/

   if ((solver_id > 229) && (solver_id < 240))
   {
      time_index = hypre_InitializeTiming("GMRES Setup");
      hypre_BeginTiming(time_index);

      HYPRE_StructGMRESCreate(comm, &struct_solver);
      HYPRE_GMRESSetMaxIter( (HYPRE_Solver)struct_solver, 100 );
      HYPRE_GMRESSetTol( (HYPRE_Solver)struct_solver, tol );
      HYPRE_GMRESSetRelChange( (HYPRE_Solver)struct_solver, 0 );
      HYPRE_GMRESSetPrintLevel( (HYPRE_Solver)struct_solver, 1 );
      HYPRE_GMRESSetLogging( (HYPRE_Solver)struct_solver, 1 );

      if (solver_id == 230)
      {
         /* use symmetric SMG as preconditioner */
         HYPRE_StructSMGCreate(comm, &struct_precond);
         HYPRE_StructSMGSetMemoryUse(struct_precond, 0);
         HYPRE_StructSMGSetMaxIter(struct_precond, 1);
         HYPRE_StructSMGSetTol(struct_precond, 0.0);
         HYPRE_StructSMGSetZeroGuess(struct_precond);
         HYPRE_StructSMGSetNumPreRelax(struct_precond, n_pre);
         HYPRE_StructSMGSetNumPostRelax(struct_precond, n_post);
         HYPRE_StructSMGSetPrintLevel(struct_precond, 0);
         HYPRE_StructSMGSetLogging(struct_precond, 0);
         HYPRE_GMRESSetPrecond( (HYPRE_Solver)struct_solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSolve,
                                (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSetup,
                                (HYPRE_Solver)struct_precond);
      }

      else if (solver_id == 231)
      {
         /* use symmetric PFMG as preconditioner */
         HYPRE_StructPFMGCreate(comm, &struct_precond);
         HYPRE_StructPFMGSetMaxIter(struct_precond, 1);
         HYPRE_StructPFMGSetTol(struct_precond, 0.0);
         HYPRE_StructPFMGSetZeroGuess(struct_precond);
         HYPRE_StructPFMGSetRAPType(struct_precond, rap);
         HYPRE_StructPFMGSetRelaxType(struct_precond, relax);
         if (usr_jacobi_weight)
         {
            HYPRE_StructPFMGSetJacobiWeight(struct_precond, jacobi_weight);
         }
         HYPRE_StructPFMGSetNumPreRelax(struct_precond, n_pre);
         HYPRE_StructPFMGSetNumPostRelax(struct_precond, n_post);
         HYPRE_StructPFMGSetSkipRelax(struct_precond, skip);
         /*HYPRE_StructPFMGSetDxyz(struct_precond, dxyz);*/
         HYPRE_StructPFMGSetPrintLevel(struct_precond, 0);
         HYPRE_StructPFMGSetLogging(struct_precond, 0);
         HYPRE_GMRESSetPrecond( (HYPRE_Solver)struct_solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSolve,
                                (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSetup,
                                (HYPRE_Solver)struct_precond);
      }
      else if (solver_id == 232)
      {
         /* use symmetric SparseMSG as preconditioner */
         HYPRE_StructSparseMSGCreate(comm, &struct_precond);
         HYPRE_StructSparseMSGSetMaxIter(struct_precond, 1);
         HYPRE_StructSparseMSGSetJump(struct_precond, jump);
         HYPRE_StructSparseMSGSetTol(struct_precond, 0.0);
         HYPRE_StructSparseMSGSetZeroGuess(struct_precond);
         HYPRE_StructSparseMSGSetRelaxType(struct_precond, relax);
         if (usr_jacobi_weight)
         {
            HYPRE_StructSparseMSGSetJacobiWeight(struct_precond, jacobi_weight);
         }
         HYPRE_StructSparseMSGSetNumPreRelax(struct_precond, n_pre);
         HYPRE_StructSparseMSGSetNumPostRelax(struct_precond, n_post);
         HYPRE_StructSparseMSGSetPrintLevel(struct_precond, 0);
         HYPRE_StructSparseMSGSetLogging(struct_precond, 0);
         HYPRE_GMRESSetPrecond( (HYPRE_Solver)struct_solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_StructSparseMSGSolve,
                                (HYPRE_PtrToSolverFcn) HYPRE_StructSparseMSGSetup,
                                (HYPRE_Solver)struct_precond);
      }

      else if (solver_id == 237)
      {
         /* use two-step Jacobi as preconditioner */
         HYPRE_StructJacobiCreate(comm, &struct_precond);
         HYPRE_StructJacobiSetMaxIter(struct_precond, 2);
         HYPRE_StructJacobiSetTol(struct_precond, 0.0);
         HYPRE_StructJacobiSetZeroGuess(struct_precond);
         HYPRE_GMRESSetPrecond( (HYPRE_Solver)struct_solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_StructJacobiSolve,
                                (HYPRE_PtrToSolverFcn) HYPRE_StructJacobiSetup,
                                (HYPRE_Solver)struct_precond);
      }

      else if (solver_id == 238)
      {
         /* use diagonal scaling as preconditioner */
         struct_precond = NULL;
         HYPRE_GMRESSetPrecond( (HYPRE_Solver)struct_solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_StructDiagScale,
                                (HYPRE_PtrToSolverFcn) HYPRE_StructDiagScaleSetup,
                                (HYPRE_Solver)struct_precond);
      }

      HYPRE_GMRESSetup
      ( (HYPRE_Solver)struct_solver, (HYPRE_Matrix)sA, (HYPRE_Vector)sb,
        (HYPRE_Vector)sx );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("GMRES Solve");
      hypre_BeginTiming(time_index);

      HYPRE_GMRESSolve
      ( (HYPRE_Solver)struct_solver, (HYPRE_Matrix)sA, (HYPRE_Vector)sb,
        (HYPRE_Vector)sx);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_GMRESGetNumIterations( (HYPRE_Solver)struct_solver, &num_iterations);
      HYPRE_GMRESGetFinalRelativeResidualNorm( (HYPRE_Solver)struct_solver, &final_res_norm);
      HYPRE_StructGMRESDestroy(struct_solver);

      if (solver_id == 230)
      {
         HYPRE_StructSMGDestroy(struct_precond);
      }
      else if (solver_id == 231)
      {
         HYPRE_StructPFMGDestroy(struct_precond);
      }
      else if (solver_id == 232)
      {
         HYPRE_StructSparseMSGDestroy(struct_precond);
      }
      else if (solver_id == 237)
      {
         HYPRE_StructJacobiDestroy(struct_precond);
      }
   }
   /*-----------------------------------------------------------
    * Solve the system using BiCGTAB
    *-----------------------------------------------------------*/

   if ((solver_id > 239) && (solver_id < 250))
   {
      time_index = hypre_InitializeTiming("BiCGSTAB Setup");
      hypre_BeginTiming(time_index);

      HYPRE_StructBiCGSTABCreate(comm, &struct_solver);
      HYPRE_BiCGSTABSetMaxIter( (HYPRE_Solver)struct_solver, 100 );
      HYPRE_BiCGSTABSetTol( (HYPRE_Solver)struct_solver, tol );
      HYPRE_BiCGSTABSetPrintLevel( (HYPRE_Solver)struct_solver, 1 );
      HYPRE_BiCGSTABSetLogging( (HYPRE_Solver)struct_solver, 1 );

      if (solver_id == 240)
      {
         /* use symmetric SMG as preconditioner */
         HYPRE_StructSMGCreate(comm, &struct_precond);
         HYPRE_StructSMGSetMemoryUse(struct_precond, 0);
         HYPRE_StructSMGSetMaxIter(struct_precond, 1);
         HYPRE_StructSMGSetTol(struct_precond, 0.0);
         HYPRE_StructSMGSetZeroGuess(struct_precond);
         HYPRE_StructSMGSetNumPreRelax(struct_precond, n_pre);
         HYPRE_StructSMGSetNumPostRelax(struct_precond, n_post);
         HYPRE_StructSMGSetPrintLevel(struct_precond, 0);
         HYPRE_StructSMGSetLogging(struct_precond, 0);
         HYPRE_BiCGSTABSetPrecond( (HYPRE_Solver)struct_solver,
                                   (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSolve,
                                   (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSetup,
                                   (HYPRE_Solver)struct_precond);
      }

      else if (solver_id == 241)
      {
         /* use symmetric PFMG as preconditioner */
         HYPRE_StructPFMGCreate(comm, &struct_precond);
         HYPRE_StructPFMGSetMaxIter(struct_precond, 1);
         HYPRE_StructPFMGSetTol(struct_precond, 0.0);
         HYPRE_StructPFMGSetZeroGuess(struct_precond);
         HYPRE_StructPFMGSetRAPType(struct_precond, rap);
         HYPRE_StructPFMGSetRelaxType(struct_precond, relax);
         if (usr_jacobi_weight)
         {
            HYPRE_StructPFMGSetJacobiWeight(struct_precond, jacobi_weight);
         }
         HYPRE_StructPFMGSetNumPreRelax(struct_precond, n_pre);
         HYPRE_StructPFMGSetNumPostRelax(struct_precond, n_post);
         HYPRE_StructPFMGSetSkipRelax(struct_precond, skip);
         /*HYPRE_StructPFMGSetDxyz(struct_precond, dxyz);*/
         HYPRE_StructPFMGSetPrintLevel(struct_precond, 0);
         HYPRE_StructPFMGSetLogging(struct_precond, 0);
         HYPRE_BiCGSTABSetPrecond( (HYPRE_Solver)struct_solver,
                                   (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSolve,
                                   (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSetup,
                                   (HYPRE_Solver)struct_precond);
      }

      else if (solver_id == 242)
      {
         /* use symmetric SparseMSG as preconditioner */
         HYPRE_StructSparseMSGCreate(comm, &struct_precond);
         HYPRE_StructSparseMSGSetMaxIter(struct_precond, 1);
         HYPRE_StructSparseMSGSetJump(struct_precond, jump);
         HYPRE_StructSparseMSGSetTol(struct_precond, 0.0);
         HYPRE_StructSparseMSGSetZeroGuess(struct_precond);
         HYPRE_StructSparseMSGSetRelaxType(struct_precond, relax);
         if (usr_jacobi_weight)
         {
            HYPRE_StructSparseMSGSetJacobiWeight(struct_precond, jacobi_weight);
         }
         HYPRE_StructSparseMSGSetNumPreRelax(struct_precond, n_pre);
         HYPRE_StructSparseMSGSetNumPostRelax(struct_precond, n_post);
         HYPRE_StructSparseMSGSetPrintLevel(struct_precond, 0);
         HYPRE_StructSparseMSGSetLogging(struct_precond, 0);
         HYPRE_BiCGSTABSetPrecond( (HYPRE_Solver)struct_solver,
                                   (HYPRE_PtrToSolverFcn) HYPRE_StructSparseMSGSolve,
                                   (HYPRE_PtrToSolverFcn) HYPRE_StructSparseMSGSetup,
                                   (HYPRE_Solver)struct_precond);
      }

      else if (solver_id == 247)
      {
         /* use two-step Jacobi as preconditioner */
         HYPRE_StructJacobiCreate(comm, &struct_precond);
         HYPRE_StructJacobiSetMaxIter(struct_precond, 2);
         HYPRE_StructJacobiSetTol(struct_precond, 0.0);
         HYPRE_StructJacobiSetZeroGuess(struct_precond);
         HYPRE_BiCGSTABSetPrecond( (HYPRE_Solver)struct_solver,
                                   (HYPRE_PtrToSolverFcn) HYPRE_StructJacobiSolve,
                                   (HYPRE_PtrToSolverFcn) HYPRE_StructJacobiSetup,
                                   (HYPRE_Solver)struct_precond);
      }

      else if (solver_id == 248)
      {
         /* use diagonal scaling as preconditioner */
         struct_precond = NULL;
         HYPRE_BiCGSTABSetPrecond( (HYPRE_Solver)struct_solver,
                                   (HYPRE_PtrToSolverFcn) HYPRE_StructDiagScale,
                                   (HYPRE_PtrToSolverFcn) HYPRE_StructDiagScaleSetup,
                                   (HYPRE_Solver)struct_precond);
      }

      HYPRE_BiCGSTABSetup
      ( (HYPRE_Solver)struct_solver, (HYPRE_Matrix)sA, (HYPRE_Vector)sb,
        (HYPRE_Vector)sx );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("BiCGSTAB Solve");
      hypre_BeginTiming(time_index);

      HYPRE_BiCGSTABSolve
      ( (HYPRE_Solver)struct_solver, (HYPRE_Matrix)sA, (HYPRE_Vector)sb,
        (HYPRE_Vector)sx);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_BiCGSTABGetNumIterations( (HYPRE_Solver)struct_solver, &num_iterations);
      HYPRE_BiCGSTABGetFinalRelativeResidualNorm( (HYPRE_Solver)struct_solver, &final_res_norm);
      HYPRE_StructBiCGSTABDestroy(struct_solver);

      if (solver_id == 240)
      {
         HYPRE_StructSMGDestroy(struct_precond);
      }
      else if (solver_id == 241)
      {
         HYPRE_StructPFMGDestroy(struct_precond);
      }
      else if (solver_id == 242)
      {
         HYPRE_StructSparseMSGDestroy(struct_precond);
      }
      else if (solver_id == 247)
      {
         HYPRE_StructJacobiDestroy(struct_precond);
      }
   }

   /*-----------------------------------------------------------
    * Gather the solution vector
    *-----------------------------------------------------------*/

   HYPRE_SStructVectorGather(x);

   /*-----------------------------------------------------------
    * Print the solution and other info
    *-----------------------------------------------------------*/

   if (print_system)
   {
      FILE *file;
      char  filename[255];

      HYPRE_SStructVectorPrint("sstruct.out.x", x, 0);

      /* print out with shared data replicated */
      if (!read_fromfile_flag)
      {
         values   = hypre_TAlloc(HYPRE_Real, data.max_boxsize, HYPRE_MEMORY_HOST);
         d_values = hypre_TAlloc(HYPRE_Real, data.max_boxsize, memory_location);
         for (part = 0; part < data.nparts; part++)
         {
            pdata = data.pdata[part];
            for (var = 0; var < pdata.nvars; var++)
            {
               hypre_sprintf(filename, "sstruct.out.xx.%02d.%02d.%05d", part, var, myid);
               if ((file = fopen(filename, "w")) == NULL)
               {
                  hypre_printf("Error: can't open output file %s\n", filename);
                  exit(1);
               }
               for (box = 0; box < pdata.nboxes; box++)
               {
                  GetVariableBox(pdata.ilowers[box], pdata.iuppers[box],
                                 pdata.vartypes[var], ilower, iupper);
                  HYPRE_SStructVectorGetBoxValues(x, part, ilower, iupper,
                                                  var, d_values);
                  hypre_TMemcpy(values, d_values, HYPRE_Real, data.max_boxsize,
                                HYPRE_MEMORY_HOST, memory_location);
                  hypre_fprintf(file, "\nBox %d:\n\n", box);
                  size = 1;
                  for (j = 0; j < data.ndim; j++)
                  {
                     size *= (iupper[j] - ilower[j] + 1);
                  }
                  for (j = 0; j < size; j++)
                  {
                     hypre_fprintf(file, "%.14e\n", values[j]);
                  }
               }
               fflush(file);
               fclose(file);
            }
         }
         hypre_TFree(values, HYPRE_MEMORY_HOST);
         hypre_TFree(d_values, memory_location);
      }
   }

   if (myid == 0 /* begin lobpcg */ && !lobpcgFlag /* end lobpcg */)
   {
      hypre_printf("\n");
      hypre_printf("Iterations = %d\n", num_iterations);
      hypre_printf("Final Relative Residual Norm = %e\n", final_res_norm);
      hypre_printf("\n");
   }

   /*-----------------------------------------------------------
    * Verify GetBoxValues()
    *-----------------------------------------------------------*/

#if 0
   {
      HYPRE_SStructVector   xnew;
      HYPRE_ParVector       par_xnew;
      HYPRE_StructVector    sxnew;
      HYPRE_Real            rnorm, bnorm;

      HYPRE_SStructVectorCreate(comm, grid, &xnew);
      HYPRE_SStructVectorSetObjectType(xnew, object_type);
      HYPRE_SStructVectorInitialize(xnew);

      /* get/set replicated shared data */
      values = hypre_TAlloc(HYPRE_Real,  data.max_boxsize, HYPRE_MEMORY_HOST);
      for (part = 0; part < data.nparts; part++)
      {
         pdata = data.pdata[part];
         for (var = 0; var < pdata.nvars; var++)
         {
            for (box = 0; box < pdata.nboxes; box++)
            {
               GetVariableBox(pdata.ilowers[box], pdata.iuppers[box],
                              pdata.vartypes[var], ilower, iupper);
               HYPRE_SStructVectorGetBoxValues(x, part, ilower, iupper,
                                               var, values);
               HYPRE_SStructVectorSetBoxValues(xnew, part, ilower, iupper,
                                               var, values);
            }
         }
      }
      hypre_TFree(values, HYPRE_MEMORY_HOST);

      HYPRE_SStructVectorAssemble(xnew);

      /* Compute residual norm - this if/else is due to a bug in SStructMatvec */
      if (object_type == HYPRE_SSTRUCT)
      {
         HYPRE_SStructInnerProd(b, b, &bnorm);
         hypre_SStructMatvec(-1.0, A, xnew, 1.0, b);
         HYPRE_SStructInnerProd(b, b, &rnorm);
      }
      else if (object_type == HYPRE_PARCSR)
      {
         bnorm = hypre_ParVectorInnerProd(par_b, par_b);
         HYPRE_SStructVectorGetObject(xnew, (void **) &par_xnew);
         HYPRE_ParCSRMatrixMatvec(-1.0, par_A, par_xnew, 1.0, par_b );
         rnorm = hypre_ParVectorInnerProd(par_b, par_b);
      }
      else if (object_type == HYPRE_STRUCT)
      {
         bnorm = hypre_StructInnerProd(sb, sb);
         HYPRE_SStructVectorGetObject(xnew, (void **) &sxnew);
         hypre_StructMatvec(-1.0, sA, sxnew, 1.0, sb);
         rnorm = hypre_StructInnerProd(sb, sb);
      }
      bnorm = hypre_sqrt(bnorm);
      rnorm = hypre_sqrt(rnorm);

      if (myid == 0)
      {
         hypre_printf("\n");
         hypre_printf("solver relnorm = %16.14e\n", final_res_norm);
         hypre_printf("check  relnorm = %16.14e, bnorm = %16.14e, rnorm = %16.14e\n",
                      (rnorm / bnorm), bnorm, rnorm);
         hypre_printf("\n");
      }

      HYPRE_SStructVectorDestroy(xnew);
   }
#endif

   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

   HYPRE_SStructMatrixDestroy(A);
   HYPRE_SStructVectorDestroy(b);
   HYPRE_SStructVectorDestroy(x);
   if (gradient_matrix)
   {
      for (s = 0; s < data.ndim; s++)
      {
         HYPRE_SStructStencilDestroy(G_stencils[s]);
      }
      hypre_TFree(G_stencils, HYPRE_MEMORY_HOST);
      HYPRE_SStructGraphDestroy(G_graph);
      HYPRE_SStructGridDestroy(G_grid);
      HYPRE_SStructMatrixDestroy(G);
   }

   if (!read_fromfile_flag)
   {
      HYPRE_SStructGridDestroy(grid);
      HYPRE_SStructGraphDestroy(graph);

      for (s = 0; s < data.nstencils; s++)
      {
         HYPRE_SStructStencilDestroy(stencils[s]);
      }
      hypre_TFree(stencils, HYPRE_MEMORY_HOST);

      DestroyData(data);
      hypre_TFree(parts, HYPRE_MEMORY_HOST);
      hypre_TFree(refine, HYPRE_MEMORY_HOST);
      hypre_TFree(distribute, HYPRE_MEMORY_HOST);
      hypre_TFree(block, HYPRE_MEMORY_HOST);
   }
   /*hypre_FinalizeMemoryDebug(); */

   /* Finalize Hypre */
   HYPRE_Finalize();

   /* Finalize MPI */
   hypre_MPI_Finalize();

#if defined(HYPRE_USING_MEMORY_TRACKER)
   if (memory_location == HYPRE_MEMORY_HOST)
   {
      if (hypre_total_bytes[hypre_MEMORY_DEVICE] || hypre_total_bytes[hypre_MEMORY_UNIFIED])
      {
         hypre_printf("Error: nonzero GPU memory allocated with the HOST mode\n");
         hypre_assert(0);
      }
   }
#endif

   return (0);
}
