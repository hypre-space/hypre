/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.52 $
 ***********************************************************************EHEADER*/

#include "hypre_system.h"

#include "_hypre_sstruct_mv.h"

#define DEBUG 0

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

static ProblemData  static_problem_data;

typedef struct
{
   HYPRE_Int  print_system;

} SystemData;
 
static SystemData  static_system_data;

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

   switch(vartype)
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
      array[i] = strtod(sdata_ptr, &sdata_ptr);
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
         index[i] += index[i+6];
      }
   }
   sdata_ptr += strcspn(sdata_ptr, ")") + 1;

   for (i = 0; i < ndim; i++)
   {
      if (sign[i] == '+')
      {
         index[i+3] = 1;
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
      sdata = hypre_TAlloc(char, memchunk);
      sdata_line = fgets(sdata, maxline, file);

      s= 0;
      while (sdata_line != NULL)
      {
         sdata_size += strlen(sdata_line) + 1;

         /* allocate more space, if necessary */
         if ((sdata_size + maxline) > s)
         {
            sdata = hypre_TReAlloc(sdata, char, (sdata_size + memchunk));
            s= sdata_size + memchunk;
         }
         
         /* read the next input line */
         sdata_line = fgets((sdata + sdata_size), maxline, file);
      }
   }

   /* broadcast the data size */
   hypre_MPI_Bcast(&sdata_size, 1, HYPRE_MPI_INT, 0, hypre_MPI_COMM_WORLD);

   /* broadcast the data */
   sdata = hypre_TReAlloc(sdata, char, sdata_size);
   hypre_MPI_Bcast(sdata, sdata_size, hypre_MPI_CHAR, 0, hypre_MPI_COMM_WORLD);

   /*-----------------------------------------------------------
    * Parse the data and fill ProblemData structure
    *-----------------------------------------------------------*/

   data.max_boxsize = 0;
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
            data.pdata = hypre_CTAlloc(ProblemPartData, data.nparts);
         }
         else if ( strcmp(key, "GridSetExtents:") == 0 )
         {
            part = strtol(sdata_ptr, &sdata_ptr, 10);
            pdata = data.pdata[part];
            if ((pdata.nboxes % 10) == 0)
            {
               size = pdata.nboxes + 10;
               pdata.ilowers =
                  hypre_TReAlloc(pdata.ilowers, ProblemIndex, size);
               pdata.iuppers =
                  hypre_TReAlloc(pdata.iuppers, ProblemIndex, size);
               pdata.boxsizes =
                  hypre_TReAlloc(pdata.boxsizes, HYPRE_Int, size);
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
               il *= pdata.ilowers[pdata.nboxes][i+3];
               iu *= pdata.iuppers[pdata.nboxes][i+3];
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
            pdata.vartypes = hypre_CTAlloc(HYPRE_SStructVariable, pdata.nvars);
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
                  hypre_TReAlloc(pdata.glue_shared, HYPRE_Int, size);
               pdata.glue_ilowers =
                  hypre_TReAlloc(pdata.glue_ilowers, ProblemIndex, size);
               pdata.glue_iuppers =
                  hypre_TReAlloc(pdata.glue_iuppers, ProblemIndex, size);
               pdata.glue_offsets =
                  hypre_TReAlloc(pdata.glue_offsets, Index, size);
               pdata.glue_nbor_parts =
                  hypre_TReAlloc(pdata.glue_nbor_parts, HYPRE_Int, size);
               pdata.glue_nbor_ilowers =
                  hypre_TReAlloc(pdata.glue_nbor_ilowers, ProblemIndex, size);
               pdata.glue_nbor_iuppers =
                  hypre_TReAlloc(pdata.glue_nbor_iuppers, ProblemIndex, size);
               pdata.glue_nbor_offsets =
                  hypre_TReAlloc(pdata.glue_nbor_offsets, Index, size);
               pdata.glue_index_maps =
                  hypre_TReAlloc(pdata.glue_index_maps, Index, size);
               pdata.glue_index_dirs =
                  hypre_TReAlloc(pdata.glue_index_dirs, Index, size);
               pdata.glue_primaries =
                  hypre_TReAlloc(pdata.glue_primaries, HYPRE_Int, size);
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
            data.stencil_sizes   = hypre_CTAlloc(HYPRE_Int, data.nstencils);
            data.stencil_offsets = hypre_CTAlloc(Index *, data.nstencils);
            data.stencil_vars    = hypre_CTAlloc(HYPRE_Int *, data.nstencils);
            data.stencil_values  = hypre_CTAlloc(HYPRE_Real *, data.nstencils);
            SScanIntArray(sdata_ptr, &sdata_ptr,
                          data.nstencils, data.stencil_sizes);
            for (s = 0; s < data.nstencils; s++)
            {
               data.stencil_offsets[s] =
                  hypre_CTAlloc(Index, data.stencil_sizes[s]);
               data.stencil_vars[s] =
                  hypre_CTAlloc(HYPRE_Int, data.stencil_sizes[s]);
               data.stencil_values[s] =
                  hypre_CTAlloc(HYPRE_Real, data.stencil_sizes[s]);
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
            data.stencil_values[s][entry] = strtod(sdata_ptr, &sdata_ptr);
         }
         else if ( strcmp(key, "RhsSet:") == 0 )
         {
            if (data.rhs_true == 0)
            {
               data.rhs_true = 1;
            }
            data.rhs_value = strtod(sdata_ptr, &sdata_ptr);
         }
         else if ( strcmp(key, "FEMStencilCreate:") == 0 )
         {
            if (data.nstencils > 0)
            {
               hypre_printf("Stencil and FEMStencil cannot be used together\n");
               exit(1);
            }
            data.fem_nvars = strtol(sdata_ptr, &sdata_ptr, 10);
            data.fem_offsets = hypre_CTAlloc(Index, data.fem_nvars);
            data.fem_vars = hypre_CTAlloc(HYPRE_Int, data.fem_nvars);
            data.fem_values_full = hypre_CTAlloc(HYPRE_Real *, data.fem_nvars);
            for (i = 0; i < data.fem_nvars; i++)
            {
               data.fem_values_full[i] = hypre_CTAlloc(HYPRE_Real, data.fem_nvars);
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
               data.fem_rhs_values = hypre_CTAlloc(HYPRE_Real, data.fem_nvars);
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
               pdata.stencil_num = hypre_CTAlloc(HYPRE_Int, pdata.nvars);
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
                  hypre_TReAlloc(pdata.graph_ilowers, ProblemIndex, size);
               pdata.graph_iuppers =
                  hypre_TReAlloc(pdata.graph_iuppers, ProblemIndex, size);
               pdata.graph_strides =
                  hypre_TReAlloc(pdata.graph_strides, Index, size);
               pdata.graph_vars =
                  hypre_TReAlloc(pdata.graph_vars, HYPRE_Int, size);
               pdata.graph_to_parts =
                  hypre_TReAlloc(pdata.graph_to_parts, HYPRE_Int, size);
               pdata.graph_to_ilowers =
                  hypre_TReAlloc(pdata.graph_to_ilowers, ProblemIndex, size);
               pdata.graph_to_iuppers =
                  hypre_TReAlloc(pdata.graph_to_iuppers, ProblemIndex, size);
               pdata.graph_to_strides =
                  hypre_TReAlloc(pdata.graph_to_strides, Index, size);
               pdata.graph_to_vars =
                  hypre_TReAlloc(pdata.graph_to_vars, HYPRE_Int, size);
               pdata.graph_index_maps =
                  hypre_TReAlloc(pdata.graph_index_maps, Index, size);
               pdata.graph_index_signs =
                  hypre_TReAlloc(pdata.graph_index_signs, Index, size);
               pdata.graph_entries =
                  hypre_TReAlloc(pdata.graph_entries, HYPRE_Int, size);
               pdata.graph_values =
                  hypre_TReAlloc(pdata.graph_values, HYPRE_Real, size);
               pdata.graph_boxsizes =
                  hypre_TReAlloc(pdata.graph_boxsizes, HYPRE_Int, size);
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
               strtod(sdata_ptr, &sdata_ptr);
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
                  hypre_TReAlloc(data.symmetric_parts, HYPRE_Int, size);
               data.symmetric_vars =
                  hypre_TReAlloc(data.symmetric_vars, HYPRE_Int, size);
               data.symmetric_to_vars =
                  hypre_TReAlloc(data.symmetric_to_vars, HYPRE_Int, size);
               data.symmetric_booleans =
                  hypre_TReAlloc(data.symmetric_booleans, HYPRE_Int, size);
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
                  hypre_TReAlloc(pdata.matset_ilowers, ProblemIndex, size);
               pdata.matset_iuppers =
                  hypre_TReAlloc(pdata.matset_iuppers, ProblemIndex, size);
               pdata.matset_strides =
                  hypre_TReAlloc(pdata.matset_strides, Index, size);
               pdata.matset_vars =
                  hypre_TReAlloc(pdata.matset_vars, HYPRE_Int, size);
               pdata.matset_entries =
                  hypre_TReAlloc(pdata.matset_entries, HYPRE_Int, size);
               pdata.matset_values =
                  hypre_TReAlloc(pdata.matset_values, HYPRE_Real, size);
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
               strtod(sdata_ptr, &sdata_ptr);
            pdata.matset_nboxes++;
            data.pdata[part] = pdata;
         }
         else if ( strcmp(key, "MatrixAddToValues:") == 0 )
         {
            part = strtol(sdata_ptr, &sdata_ptr, 10);
            pdata = data.pdata[part];
            if ((pdata.matadd_nboxes% 10) == 0)
            {
               size = pdata.matadd_nboxes+10;
               pdata.matadd_ilowers=
                  hypre_TReAlloc(pdata.matadd_ilowers, ProblemIndex, size);
               pdata.matadd_iuppers=
                  hypre_TReAlloc(pdata.matadd_iuppers, ProblemIndex, size);
               pdata.matadd_vars=
                  hypre_TReAlloc(pdata.matadd_vars, HYPRE_Int, size);
               pdata.matadd_nentries=
                  hypre_TReAlloc(pdata.matadd_nentries, HYPRE_Int, size);
               pdata.matadd_entries=
                  hypre_TReAlloc(pdata.matadd_entries, HYPRE_Int *, size);
               pdata.matadd_values=
                  hypre_TReAlloc(pdata.matadd_values, HYPRE_Real *, size);
            }
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.matadd_ilowers[pdata.matadd_nboxes]);
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.matadd_iuppers[pdata.matadd_nboxes]);
            pdata.matadd_vars[pdata.matadd_nboxes]=
               strtol(sdata_ptr, &sdata_ptr, 10);
            i= strtol(sdata_ptr, &sdata_ptr, 10);
            pdata.matadd_nentries[pdata.matadd_nboxes]= i;
            pdata.matadd_entries[pdata.matadd_nboxes] =
               hypre_TAlloc(HYPRE_Int, i);
            SScanIntArray(sdata_ptr, &sdata_ptr, i,
                          (HYPRE_Int*) pdata.matadd_entries[pdata.matadd_nboxes]);
            pdata.matadd_values[pdata.matadd_nboxes] =
               hypre_TAlloc(HYPRE_Real, i);
            SScanDblArray(sdata_ptr, &sdata_ptr, i,
                          (HYPRE_Real *) pdata.matadd_values[pdata.matadd_nboxes]);
            pdata.matadd_nboxes++;
            data.pdata[part] = pdata;
         }
         else if ( strcmp(key, "FEMMatrixAddToValues:") == 0 )
         {
            part = strtol(sdata_ptr, &sdata_ptr, 10);
            pdata = data.pdata[part];
            if ((pdata.fem_matadd_nboxes% 10) == 0)
            {
               size = pdata.fem_matadd_nboxes+10;
               pdata.fem_matadd_ilowers=
                  hypre_TReAlloc(pdata.fem_matadd_ilowers, ProblemIndex, size);
               pdata.fem_matadd_iuppers=
                  hypre_TReAlloc(pdata.fem_matadd_iuppers, ProblemIndex, size);
               pdata.fem_matadd_nrows=
                  hypre_TReAlloc(pdata.fem_matadd_nrows, HYPRE_Int, size);
               pdata.fem_matadd_rows=
                  hypre_TReAlloc(pdata.fem_matadd_rows, HYPRE_Int *, size);
               pdata.fem_matadd_ncols=
                  hypre_TReAlloc(pdata.fem_matadd_ncols, HYPRE_Int, size);
               pdata.fem_matadd_cols=
                  hypre_TReAlloc(pdata.fem_matadd_cols, HYPRE_Int *, size);
               pdata.fem_matadd_values=
                  hypre_TReAlloc(pdata.fem_matadd_values, HYPRE_Real *, size);
            }
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.fem_matadd_ilowers[pdata.fem_matadd_nboxes]);
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.fem_matadd_iuppers[pdata.fem_matadd_nboxes]);
            i= strtol(sdata_ptr, &sdata_ptr, 10);
            pdata.fem_matadd_nrows[pdata.fem_matadd_nboxes]= i;
            pdata.fem_matadd_rows[pdata.fem_matadd_nboxes] = hypre_TAlloc(HYPRE_Int, i);
            SScanIntArray(sdata_ptr, &sdata_ptr, i,
                          (HYPRE_Int*) pdata.fem_matadd_rows[pdata.fem_matadd_nboxes]);
            j= strtol(sdata_ptr, &sdata_ptr, 10);
            pdata.fem_matadd_ncols[pdata.fem_matadd_nboxes]= j;
            pdata.fem_matadd_cols[pdata.fem_matadd_nboxes] = hypre_TAlloc(HYPRE_Int, j);
            SScanIntArray(sdata_ptr, &sdata_ptr, j,
                          (HYPRE_Int*) pdata.fem_matadd_cols[pdata.fem_matadd_nboxes]);
            pdata.fem_matadd_values[pdata.fem_matadd_nboxes] =
               hypre_TAlloc(HYPRE_Real, i*j);
            SScanDblArray(sdata_ptr, &sdata_ptr, i*j,
                          (HYPRE_Real *) pdata.fem_matadd_values[pdata.fem_matadd_nboxes]);
            pdata.fem_matadd_nboxes++;
            data.pdata[part] = pdata;
         }
         else if ( strcmp(key, "RhsAddToValues:") == 0 )
         {
            part = strtol(sdata_ptr, &sdata_ptr, 10);
            pdata = data.pdata[part];
            if ((pdata.rhsadd_nboxes% 10) == 0)
            {
               size = pdata.rhsadd_nboxes+10;
               pdata.rhsadd_ilowers=
                  hypre_TReAlloc(pdata.rhsadd_ilowers, ProblemIndex, size);
               pdata.rhsadd_iuppers=
                  hypre_TReAlloc(pdata.rhsadd_iuppers, ProblemIndex, size);
               pdata.rhsadd_vars=
                  hypre_TReAlloc(pdata.rhsadd_vars, HYPRE_Int, size);
               pdata.rhsadd_values=
                  hypre_TReAlloc(pdata.rhsadd_values, HYPRE_Real, size);
            }
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.rhsadd_ilowers[pdata.rhsadd_nboxes]);
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.rhsadd_iuppers[pdata.rhsadd_nboxes]);
            pdata.rhsadd_vars[pdata.rhsadd_nboxes]=
               strtol(sdata_ptr, &sdata_ptr, 10);
            pdata.rhsadd_values[pdata.rhsadd_nboxes] =
               strtod(sdata_ptr, &sdata_ptr);
            pdata.rhsadd_nboxes++;
            data.pdata[part] = pdata;
         }
         else if ( strcmp(key, "FEMRhsAddToValues:") == 0 )
         {
            part = strtol(sdata_ptr, &sdata_ptr, 10);
            pdata = data.pdata[part];
            if ((pdata.fem_rhsadd_nboxes% 10) == 0)
            {
               size = pdata.fem_rhsadd_nboxes+10;
               pdata.fem_rhsadd_ilowers=
                  hypre_TReAlloc(pdata.fem_rhsadd_ilowers, ProblemIndex, size);
               pdata.fem_rhsadd_iuppers=
                  hypre_TReAlloc(pdata.fem_rhsadd_iuppers, ProblemIndex, size);
               pdata.fem_rhsadd_values=
                  hypre_TReAlloc(pdata.fem_rhsadd_values, HYPRE_Real *, size);
            }
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.fem_rhsadd_ilowers[pdata.fem_rhsadd_nboxes]);
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.fem_rhsadd_iuppers[pdata.fem_rhsadd_nboxes]);
            pdata.fem_rhsadd_values[pdata.fem_rhsadd_nboxes] =
               hypre_TAlloc(HYPRE_Real, data.fem_nvars);
            SScanDblArray(sdata_ptr, &sdata_ptr, data.fem_nvars,
                          (HYPRE_Real *) pdata.fem_rhsadd_values[pdata.fem_rhsadd_nboxes]);
            pdata.fem_rhsadd_nboxes++;
            data.pdata[part] = pdata;
         }
         else if ( strcmp(key, "ProcessPoolCreate:") == 0 )
         {
            data.ndists++;
            data.dist_npools= hypre_TReAlloc(data.dist_npools, HYPRE_Int, data.ndists);
            data.dist_pools= hypre_TReAlloc(data.dist_pools, HYPRE_Int *, data.ndists);
            data.dist_npools[data.ndists-1] = strtol(sdata_ptr, &sdata_ptr, 10);
            data.dist_pools[data.ndists-1] = hypre_CTAlloc(HYPRE_Int, data.nparts);
#if 0
            data.npools = strtol(sdata_ptr, &sdata_ptr, 10);
            data.pools = hypre_CTAlloc(HYPRE_Int, data.nparts);
#endif
         }
         else if ( strcmp(key, "ProcessPoolSetPart:") == 0 )
         {
            i = strtol(sdata_ptr, &sdata_ptr, 10);
            part = strtol(sdata_ptr, &sdata_ptr, 10);
            data.dist_pools[data.ndists-1][part] = i;
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

      data.fem_ivalues_full = hypre_CTAlloc(HYPRE_Int *, data.fem_nvars);
      data.fem_ordering = hypre_CTAlloc(HYPRE_Int, (1+data.ndim)*data.fem_nvars);
      data.fem_sparsity = hypre_CTAlloc(HYPRE_Int, 2*data.fem_nvars*data.fem_nvars);
      data.fem_values   = hypre_CTAlloc(HYPRE_Real, data.fem_nvars*data.fem_nvars);

      for (i = 0; i < data.fem_nvars; i++)
      {
         data.fem_ivalues_full[i] = hypre_CTAlloc(HYPRE_Int, data.fem_nvars);
         k = (1+data.ndim)*i;
         data.fem_ordering[k] = data.fem_vars[i];
         for (d = 0; d < data.ndim; d++)
         {
            data.fem_ordering[k+1+d] = data.fem_offsets[i][d];
         }
         for (j = 0; j < data.fem_nvars; j++)
         {
            if (data.fem_values_full[i][j] != 0.0)
            {
               k = 2*data.fem_nsparse;
               data.fem_sparsity[k]   = i;
               data.fem_sparsity[k+1] = j;
               data.fem_values[data.fem_nsparse] = data.fem_values_full[i][j];
               data.fem_ivalues_full[i][j] = data.fem_nsparse;
               data.fem_nsparse ++;
            }
         }
      }
   }

   hypre_TFree(sdata);

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
   index[0] = m[0]*index[0] + (m[0]-1)*index[3];
   index[1] = m[1]*index[1] + (m[1]-1)*index[4];
   index[2] = m[2]*index[2] + (m[2]-1)*index[5];
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
   pool_procs = hypre_CTAlloc(HYPRE_Int, (data.npools+1));
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
            hypre_TFree(pdata.matadd_entries[box]);
            hypre_TFree(pdata.matadd_values[box]);
         }
         pdata.matadd_nboxes = 0;
         for (box = 0; box < pdata.fem_matadd_nboxes; box++)
         {
            hypre_TFree(pdata.fem_matadd_rows[box]);
            hypre_TFree(pdata.fem_matadd_cols[box]);
            hypre_TFree(pdata.fem_matadd_values[box]);
         }
         pdata.fem_matadd_nboxes = 0;
         pdata.rhsadd_nboxes = 0;
         for (box = 0; box < pdata.fem_rhsadd_nboxes; box++)
         {
            hypre_TFree(pdata.fem_rhsadd_values[box]);
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
            r = (pid - p - q*m[0]) / (m[0]*m[1]);

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

               pdata.ilowers[box][0] = pdata.ilowers[box][0] + p*n[0];
               pdata.ilowers[box][1] = pdata.ilowers[box][1] + q*n[1];
               pdata.ilowers[box][2] = pdata.ilowers[box][2] + r*n[2];
               pdata.iuppers[box][0] = pdata.iuppers[box][0] + p*n[0];
               pdata.iuppers[box][1] = pdata.iuppers[box][1] + q*n[1];
               pdata.iuppers[box][2] = pdata.iuppers[box][2] + r*n[2];
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
               hypre_TFree(pdata.matadd_entries[box]);
               hypre_TFree(pdata.matadd_values[box]);
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
               hypre_TFree(pdata.fem_matadd_rows[box]);
               hypre_TFree(pdata.fem_matadd_cols[box]);
               hypre_TFree(pdata.fem_matadd_values[box]);
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
               hypre_TFree(pdata.fem_rhsadd_values[box]);
            }
            pdata.fem_rhsadd_nboxes = i;
         }

         /* refine and block boxes */
         m[0] = block[part][0];
         m[1] = block[part][1];
         m[2] = block[part][2];
         if ( (m[0] * m[1] * m[2]) > 1)
         {
            pdata.ilowers = hypre_TReAlloc(pdata.ilowers, ProblemIndex,
                                           m[0]*m[1]*m[2]*pdata.nboxes);
            pdata.iuppers = hypre_TReAlloc(pdata.iuppers, ProblemIndex,
                                           m[0]*m[1]*m[2]*pdata.nboxes);
            pdata.boxsizes = hypre_TReAlloc(pdata.boxsizes, HYPRE_Int,
                                            m[0]*m[1]*m[2]*pdata.nboxes);
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
                        pdata.ilowers[i][0] = pdata.ilowers[box][0] + p*n[0];
                        pdata.ilowers[i][1] = pdata.ilowers[box][1] + q*n[1];
                        pdata.ilowers[i][2] = pdata.ilowers[box][2] + r*n[2];
                        pdata.iuppers[i][0] = pdata.iuppers[box][0] + p*n[0];
                        pdata.iuppers[i][1] = pdata.iuppers[box][1] + q*n[1];
                        pdata.iuppers[i][2] = pdata.iuppers[box][2] + r*n[2];
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
            pdata.nboxes *= m[0]*m[1]*m[2];

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
               size*= (pdata.matset_iuppers[box][i] -
                       pdata.matset_ilowers[box][i] + 1);
            }
            pdata.max_boxsize = hypre_max(pdata.max_boxsize, size);
         }
         for (box = 0; box < pdata.matadd_nboxes; box++)
         {
            size = 1;
            for (i = 0; i < 3; i++)
            {
               size*= (pdata.matadd_iuppers[box][i] -
                       pdata.matadd_ilowers[box][i] + 1);
            }
            pdata.max_boxsize = hypre_max(pdata.max_boxsize, size);
         }
         for (box = 0; box < pdata.fem_matadd_nboxes; box++)
         {
            size = 1;
            for (i = 0; i < 3; i++)
            {
               size*= (pdata.fem_matadd_iuppers[box][i] -
                       pdata.fem_matadd_ilowers[box][i] + 1);
            }
            pdata.max_boxsize = hypre_max(pdata.max_boxsize, size);
         }
         for (box = 0; box < pdata.rhsadd_nboxes; box++)
         {
            size = 1;
            for (i = 0; i < 3; i++)
            {
               size*= (pdata.rhsadd_iuppers[box][i] -
                       pdata.rhsadd_ilowers[box][i] + 1);
            }
            pdata.max_boxsize = hypre_max(pdata.max_boxsize, size);
         }
         for (box = 0; box < pdata.fem_rhsadd_nboxes; box++)
         {
            size = 1;
            for (i = 0; i < 3; i++)
            {
               size*= (pdata.fem_rhsadd_iuppers[box][i] -
                       pdata.fem_rhsadd_ilowers[box][i] + 1);
            }
            pdata.max_boxsize = hypre_max(pdata.max_boxsize, size);
         }

         /* refine periodicity */
         pdata.periodic[0] *= refine[part][0]*block[part][0]*distribute[part][0];
         pdata.periodic[1] *= refine[part][1]*block[part][1]*distribute[part][1];
         pdata.periodic[2] *= refine[part][2]*block[part][2]*distribute[part][2];
      }

      if (pdata.nboxes == 0)
      {
         hypre_TFree(pdata.ilowers);
         hypre_TFree(pdata.iuppers);
         hypre_TFree(pdata.boxsizes);
         pdata.max_boxsize = 0;
      }

      if (pdata.glue_nboxes == 0)
      {
         hypre_TFree(pdata.glue_shared);
         hypre_TFree(pdata.glue_ilowers);
         hypre_TFree(pdata.glue_iuppers);
         hypre_TFree(pdata.glue_offsets);
         hypre_TFree(pdata.glue_nbor_parts);
         hypre_TFree(pdata.glue_nbor_ilowers);
         hypre_TFree(pdata.glue_nbor_iuppers);
         hypre_TFree(pdata.glue_nbor_offsets);
         hypre_TFree(pdata.glue_index_maps);
         hypre_TFree(pdata.glue_index_dirs);
         hypre_TFree(pdata.glue_primaries);
      }

      if (pdata.graph_nboxes == 0)
      {
         hypre_TFree(pdata.graph_ilowers);
         hypre_TFree(pdata.graph_iuppers);
         hypre_TFree(pdata.graph_strides);
         hypre_TFree(pdata.graph_vars);
         hypre_TFree(pdata.graph_to_parts);
         hypre_TFree(pdata.graph_to_ilowers);
         hypre_TFree(pdata.graph_to_iuppers);
         hypre_TFree(pdata.graph_to_strides);
         hypre_TFree(pdata.graph_to_vars);
         hypre_TFree(pdata.graph_index_maps);
         hypre_TFree(pdata.graph_index_signs);
         hypre_TFree(pdata.graph_entries);
         hypre_TFree(pdata.graph_values);
         hypre_TFree(pdata.graph_boxsizes);
      }

      if (pdata.matset_nboxes == 0)
      {
         hypre_TFree(pdata.matset_ilowers);
         hypre_TFree(pdata.matset_iuppers);
         hypre_TFree(pdata.matset_strides);
         hypre_TFree(pdata.matset_vars);
         hypre_TFree(pdata.matset_entries);
         hypre_TFree(pdata.matset_values);
      }

      if (pdata.matadd_nboxes == 0)
      {
         hypre_TFree(pdata.matadd_ilowers);
         hypre_TFree(pdata.matadd_iuppers);
         hypre_TFree(pdata.matadd_vars);
         hypre_TFree(pdata.matadd_nentries);
         hypre_TFree(pdata.matadd_entries);
         hypre_TFree(pdata.matadd_values);
      }

      if (pdata.fem_matadd_nboxes == 0)
      {
         hypre_TFree(pdata.fem_matadd_ilowers);
         hypre_TFree(pdata.fem_matadd_iuppers);
         hypre_TFree(pdata.fem_matadd_nrows);
         hypre_TFree(pdata.fem_matadd_ncols);
         hypre_TFree(pdata.fem_matadd_rows);
         hypre_TFree(pdata.fem_matadd_cols);
         hypre_TFree(pdata.fem_matadd_values);
      }

      if (pdata.rhsadd_nboxes == 0)
      {
         hypre_TFree(pdata.rhsadd_ilowers);
         hypre_TFree(pdata.rhsadd_iuppers);
         hypre_TFree(pdata.rhsadd_vars);
         hypre_TFree(pdata.rhsadd_values);
      }

      if (pdata.fem_rhsadd_nboxes == 0)
      {
         hypre_TFree(pdata.fem_rhsadd_ilowers);
         hypre_TFree(pdata.fem_rhsadd_iuppers);
         hypre_TFree(pdata.fem_rhsadd_values);
      }

      data.pdata[part] = pdata;
   }

   data.max_boxsize = 0;
   for (part = 0; part < data.nparts; part++)
   {
      data.max_boxsize =
         hypre_max(data.max_boxsize, data.pdata[part].max_boxsize);
   }

   hypre_TFree(pool_procs);

   *data_ptr = data; 
   return 0;
}

/*--------------------------------------------------------------------------
 * Destroy data
 *--------------------------------------------------------------------------*/

HYPRE_Int
DestroyData( ProblemData   data )
{
   ProblemPartData  pdata;
   HYPRE_Int        part, box, s, i;

   for (part = 0; part < data.nparts; part++)
   {
      pdata = data.pdata[part];

      if (pdata.nboxes > 0)
      {
         hypre_TFree(pdata.ilowers);
         hypre_TFree(pdata.iuppers);
         hypre_TFree(pdata.boxsizes);
      }

      if (pdata.nvars > 0)
      {
         hypre_TFree(pdata.vartypes);
      }

      if (pdata.add_nvars > 0)
      {
         hypre_TFree(pdata.add_indexes);
         hypre_TFree(pdata.add_vartypes);
      }

      if (pdata.glue_nboxes > 0)
      {
         hypre_TFree(pdata.glue_shared);
         hypre_TFree(pdata.glue_ilowers);
         hypre_TFree(pdata.glue_iuppers);
         hypre_TFree(pdata.glue_offsets);
         hypre_TFree(pdata.glue_nbor_parts);
         hypre_TFree(pdata.glue_nbor_ilowers);
         hypre_TFree(pdata.glue_nbor_iuppers);
         hypre_TFree(pdata.glue_nbor_offsets);
         hypre_TFree(pdata.glue_index_maps);
         hypre_TFree(pdata.glue_index_dirs);
         hypre_TFree(pdata.glue_primaries);
      }

      if (pdata.nvars > 0)
      {
         hypre_TFree(pdata.stencil_num);
      }

      if (pdata.graph_nboxes > 0)
      {
         hypre_TFree(pdata.graph_ilowers);
         hypre_TFree(pdata.graph_iuppers);
         hypre_TFree(pdata.graph_strides);
         hypre_TFree(pdata.graph_vars);
         hypre_TFree(pdata.graph_to_parts);
         hypre_TFree(pdata.graph_to_ilowers);
         hypre_TFree(pdata.graph_to_iuppers);
         hypre_TFree(pdata.graph_to_strides);
         hypre_TFree(pdata.graph_to_vars);
         hypre_TFree(pdata.graph_index_maps);
         hypre_TFree(pdata.graph_index_signs);
         hypre_TFree(pdata.graph_entries);
         hypre_TFree(pdata.graph_values);
         hypre_TFree(pdata.graph_boxsizes);
      }

      if (pdata.matset_nboxes > 0)
      {
         hypre_TFree(pdata.matset_ilowers);
         hypre_TFree(pdata.matset_iuppers);
         hypre_TFree(pdata.matset_strides);
         hypre_TFree(pdata.matset_vars);
         hypre_TFree(pdata.matset_entries);
         hypre_TFree(pdata.matset_values);
      }

      if (pdata.matadd_nboxes > 0)
      {
         hypre_TFree(pdata.matadd_ilowers);
         hypre_TFree(pdata.matadd_iuppers);
         hypre_TFree(pdata.matadd_vars);
         hypre_TFree(pdata.matadd_nentries);
         for (box = 0; box < pdata.matadd_nboxes; box++)
         {
            hypre_TFree(pdata.matadd_entries[box]);
            hypre_TFree(pdata.matadd_values[box]);
         }
         hypre_TFree(pdata.matadd_entries);
         hypre_TFree(pdata.matadd_values);
      }

      if (pdata.fem_matadd_nboxes > 0)
      {
         hypre_TFree(pdata.fem_matadd_ilowers);
         hypre_TFree(pdata.fem_matadd_iuppers);
         hypre_TFree(pdata.fem_matadd_nrows);
         hypre_TFree(pdata.fem_matadd_ncols);
         for (box = 0; box < pdata.fem_matadd_nboxes; box++)
         {
            hypre_TFree(pdata.fem_matadd_rows[box]);
            hypre_TFree(pdata.fem_matadd_cols[box]);
            hypre_TFree(pdata.fem_matadd_values[box]);
         }
         hypre_TFree(pdata.fem_matadd_rows);
         hypre_TFree(pdata.fem_matadd_cols);
         hypre_TFree(pdata.fem_matadd_values);
      }

      if (pdata.rhsadd_nboxes > 0)
      {
         hypre_TFree(pdata.rhsadd_ilowers);
         hypre_TFree(pdata.rhsadd_iuppers);
         hypre_TFree(pdata.rhsadd_vars);
         hypre_TFree(pdata.rhsadd_values);
      }

      if (pdata.fem_rhsadd_nboxes > 0)
      {
         hypre_TFree(pdata.fem_rhsadd_ilowers);
         hypre_TFree(pdata.fem_rhsadd_iuppers);
         for (box = 0; box < pdata.fem_rhsadd_nboxes; box++)
         {
            hypre_TFree(pdata.fem_rhsadd_values[box]);
         }
         hypre_TFree(pdata.fem_rhsadd_values);
      }
   }
   hypre_TFree(data.pdata);

   if (data.nstencils > 0)
   {
      for (s = 0; s < data.nstencils; s++)
      {
         hypre_TFree(data.stencil_offsets[s]);
         hypre_TFree(data.stencil_vars[s]);
         hypre_TFree(data.stencil_values[s]);
      }
      hypre_TFree(data.stencil_sizes);
      hypre_TFree(data.stencil_offsets);
      hypre_TFree(data.stencil_vars);
      hypre_TFree(data.stencil_values);
   }

   if (data.fem_nvars > 0)
   {
      for (s = 0; s < data.fem_nvars; s++)
      {
         hypre_TFree(data.fem_values_full[s]);
         hypre_TFree(data.fem_ivalues_full[s]);
      }
      hypre_TFree(data.fem_offsets);
      hypre_TFree(data.fem_vars);
      hypre_TFree(data.fem_values_full);
      hypre_TFree(data.fem_ivalues_full);
      hypre_TFree(data.fem_ordering);
      hypre_TFree(data.fem_sparsity);
      hypre_TFree(data.fem_values);
   }

   if (data.fem_rhs_true > 0)
   {
      hypre_TFree(data.fem_rhs_values);
   }

   if (data.symmetric_num > 0)
   {
      hypre_TFree(data.symmetric_parts);
      hypre_TFree(data.symmetric_vars);
      hypre_TFree(data.symmetric_to_vars);
      hypre_TFree(data.symmetric_booleans);
   }

   for (i = 0; i < data.ndists; i++)
   {
      hypre_TFree(data.dist_pools[i]);
   }
   hypre_TFree(data.dist_pools);
   hypre_TFree(data.dist_npools);

   return 0;
}

/*--------------------------------------------------------------------------
 * Routine to load cosine function
 *--------------------------------------------------------------------------*/

HYPRE_Int
SetCosineVector(   HYPRE_Real  scale,
                   Index   ilower,
                   Index   iupper,
                   HYPRE_Real *values)
{
   HYPRE_Int    i, j, k;
   HYPRE_Int    count = 0;

   for (k = ilower[2]; k <= iupper[2]; k++)
   {
      for (j = ilower[1]; j <= iupper[1]; j++)
      {
         for (i = ilower[0]; i <= iupper[0]; i++)
         {
            values[count] = scale * cos((i+j+k)/10.0);
            count++;
         }
      }
   }

   return(0);
}

/*--------------------------------------------------------------------------
 * SStruct System Options
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DriveSystemSStructHelp( )
{
   hypre_printf("System SStruct Options: [-in <filename>] [<options>]\n");
   hypre_printf("\n");
   hypre_printf("  -in <filename>      : input file (default is `%s')\n",
                infile_default);
   hypre_printf("  -pt <pt1> <pt2> ... : set part(s) for subsequent options\n");
   hypre_printf("  -pooldist <p>       : pool distribution to use\n");
   hypre_printf("  -r <rx> <ry> <rz>   : refine part(s)\n");
   hypre_printf("  -P <Px> <Py> <Pz>   : refine and distribute part(s)\n");
   hypre_printf("  -b <bx> <by> <bz>   : refine and block part(s)\n");
   hypre_printf("  -rhsfromcosine      : solution is cos function (default)\n");
   hypre_printf("  -rhsone             : rhs is vector with unit components\n");
   hypre_printf("  -sym <s>            : symmetric storage (1) or not (0)\n");
   hypre_printf("  -gradient           : build gradient matrix\n");
   hypre_printf("  -print              : print out the system\n");
   hypre_printf("\n");

   return 0;
}

/*--------------------------------------------------------------------------
 * Driver for the semi-structured (SStruct) system interface
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_DriveSystemSStructCreate(
   char       *argv[],
   HYPRE_Int   argi,
   HYPRE_Int   argn,
   HYPRE_Int   object_type,
   HYPRE_SStructMatrix *A_ptr,
   HYPRE_SStructVector *b_ptr,
   HYPRE_SStructVector *x_ptr )
{
   char                 *infile;
   ProblemData           global_data;
   ProblemData           data;
   ProblemPartData       pdata;
   HYPRE_Int             nparts;
   HYPRE_Int             pooldist;
   HYPRE_Int            *parts;
   Index                *refine;
   Index                *distribute;
   Index                *block;
   HYPRE_Int             print_system;
   HYPRE_Int             cosine;
   HYPRE_Real            scale;
                        
   HYPRE_SStructGrid     grid, G_grid;
   HYPRE_SStructStencil *stencils, *G_stencils;
   HYPRE_SStructGraph    graph, G_graph;
   HYPRE_SStructMatrix   A, G;
   HYPRE_SStructVector   b;
   HYPRE_SStructVector   x;

   Index                 ilower, iupper;
   Index                 index, to_index;
   HYPRE_Real           *values;

   HYPRE_Int             num_procs, myid;
   HYPRE_Int             time_index;

   HYPRE_Int             arg_index, part, var, box, s, entry, i, j, k, size;
   HYPRE_Int             sym, row, col;
   HYPRE_Int             gradient_matrix;
                        
   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);

   /*-----------------------------------------------------------
    * Read input file
    *-----------------------------------------------------------*/

   arg_index = argi;

   /* parse command line for input file name */
   infile = infile_default;
   if (arg_index < argn)
   {
      if ( strcmp(argv[arg_index], "-in") == 0 )
      {
         arg_index++;
         infile = argv[arg_index++];
      }
   }

   ReadData(infile, &global_data);

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/

   nparts = global_data.nparts;
   pooldist = 0;

   parts      = hypre_TAlloc(HYPRE_Int, nparts);
   refine     = hypre_TAlloc(Index, nparts);
   distribute = hypre_TAlloc(Index, nparts);
   block      = hypre_TAlloc(Index, nparts);
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

   cosine = 1;
   if (global_data.rhs_true || global_data.fem_rhs_true)
   {
      cosine = 0;
   }
   sym = 1;
   gradient_matrix = 0;
   print_system = 0;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/

   while (arg_index < argn)
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
      else if ( strcmp(argv[arg_index], "-sym") == 0 )
      {
         arg_index++;
         sym = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-gradient") == 0 )
      {
         arg_index++;
         gradient_matrix = 1;;
      }
      else if ( strcmp(argv[arg_index], "-print") == 0 )
      {
         arg_index++;
         print_system = 1;
      }
      else
      {
         arg_index++;
      }
   }

   /*-----------------------------------------------------------
    * Distribute data
    *-----------------------------------------------------------*/

   DistributeData(global_data, pooldist, refine, distribute, block,
                  num_procs, myid, &data);

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if (object_type == HYPRE_STRUCT)
   {
      pdata = data.pdata[0];
      if (nparts > 1)
      {
         if (myid == 0)
         {
            hypre_printf("Error: Invalid num parts for Struct system\n");
         }
         exit(1);
      }
      if (pdata.nvars > 1)
      {
         if (myid == 0)
         {
            hypre_printf("Error: Invalid num vars for Struct system\n");
         }
         exit(1);
      }
   }

   /*-----------------------------------------------------------
    * Set up the grid
    *-----------------------------------------------------------*/

   time_index = hypre_InitializeTiming("SStruct System");
   hypre_BeginTiming(time_index);

   HYPRE_SStructGridCreate(hypre_MPI_COMM_WORLD, data.ndim, data.nparts, &grid);
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

   stencils = hypre_CTAlloc(HYPRE_SStructStencil, data.nstencils);
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
    * Set up the graph
    *-----------------------------------------------------------*/

   HYPRE_SStructGraphCreate(hypre_MPI_COMM_WORLD, grid, &graph);

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

   values = hypre_TAlloc(HYPRE_Real, hypre_max(data.max_boxsize, data.fem_nsparse));

   HYPRE_SStructMatrixCreate(hypre_MPI_COMM_WORLD, graph, &A);

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
               for (box = 0; box < pdata.nboxes; box++)
               {
                  GetVariableBox(pdata.ilowers[box], pdata.iuppers[box],
                                 pdata.vartypes[var], ilower, iupper);
                  HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
                                                  var, 1, &i, values);
               }
            }
         }
      }
   }
   else if (data.fem_nvars > 0)
   {
      /* FEMStencilSetRow: add to stencil values */
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
                                                     data.fem_values);
                  }
               }
            }
         }
      }
   }

   /* GraphAddEntries: set non-stencil entries */
   for (part = 0; part < data.nparts; part++)
   {
      pdata = data.pdata[part];
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
                                               &pdata.graph_values[box]);
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
         size= 1;
         for (j = 0; j < 3; j++)
         {
            size*= (pdata.matset_iuppers[box][j] -
                    pdata.matset_ilowers[box][j] + 1);
         }
         for (j = 0; j < size; j++)
         {
            values[j] = pdata.matset_values[box];
         }
         HYPRE_SStructMatrixSetBoxValues(A, part,
                                         pdata.matset_ilowers[box],
                                         pdata.matset_iuppers[box],
                                         pdata.matset_vars[box],
                                         1, &pdata.matset_entries[box],
                                         values);
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
            size*= (pdata.matadd_iuppers[box][j] -
                    pdata.matadd_ilowers[box][j] + 1);
         }

         for (entry = 0; entry < pdata.matadd_nentries[box]; entry++)
         {
            for (j = 0; j < size; j++)
            {
               values[j] = pdata.matadd_values[box][entry];
            }
          
            HYPRE_SStructMatrixAddToBoxValues(A, part, 
                                              pdata.matadd_ilowers[box],
                                              pdata.matadd_iuppers[box],
                                              pdata.matadd_vars[box],
                                              1, &pdata.matadd_entries[box][entry],
                                              values);
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
         for (index[2] = pdata.fem_matadd_ilowers[box][2];
              index[2] <= pdata.fem_matadd_iuppers[box][2]; index[2]++)
         {
            for (index[1] = pdata.fem_matadd_ilowers[box][1];
                 index[1] <= pdata.fem_matadd_iuppers[box][1]; index[1]++)
            {
               for (index[0] = pdata.fem_matadd_ilowers[box][0];
                    index[0] <= pdata.fem_matadd_iuppers[box][0]; index[0]++)
               {
                  HYPRE_SStructMatrixAddFEMValues(A, part, index, values);
               }
            }
         }
      }
   }

   HYPRE_SStructMatrixAssemble(A);

   /*-----------------------------------------------------------
    * Set up the linear system
    *-----------------------------------------------------------*/

   HYPRE_SStructVectorCreate(hypre_MPI_COMM_WORLD, grid, &b);

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
                                            var, values);
         }
      }
   }

   /* Add values for FEMRhsSet */
   if (data.fem_rhs_true)
   {
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
                                                     data.fem_rhs_values);
                  }
               }
            }
         }
      }
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
            size*= (pdata.rhsadd_iuppers[box][j] -
                    pdata.rhsadd_ilowers[box][j] + 1);
         }

         for (j = 0; j < size; j++)
         {
            values[j] = pdata.rhsadd_values[box];
         }
          
         HYPRE_SStructVectorAddToBoxValues(b, part, 
                                           pdata.rhsadd_ilowers[box],
                                           pdata.rhsadd_iuppers[box],
                                           pdata.rhsadd_vars[box], values);
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

   HYPRE_SStructVectorCreate(hypre_MPI_COMM_WORLD, grid, &x);

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
            scale = (part+1.0)*(var+1.0);
            for (box = 0; box < pdata.nboxes; box++)
            {
               GetVariableBox(pdata.ilowers[box], pdata.iuppers[box],
                              var, ilower, iupper);
               SetCosineVector(scale, ilower, iupper, values);
               HYPRE_SStructVectorSetBoxValues(x, part, ilower, iupper,
                                               var, values);
            }
         }
      }
   }

   HYPRE_SStructVectorAssemble(x);

   hypre_EndTiming(time_index);
   hypre_PrintTiming("System times", hypre_MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();

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
         HYPRE_ParCSRMatrix    par_A;
         HYPRE_ParVector       par_b;
         HYPRE_ParVector       par_x;

         HYPRE_SStructMatrixGetObject(A, (void **) &par_A);
         HYPRE_SStructVectorGetObject(b, (void **) &par_b);
         HYPRE_SStructVectorGetObject(x, (void **) &par_x);

         /* Apply A to cosine vector to yield righthand side */
         HYPRE_ParCSRMatrixMatvec(1.0, par_A, par_x, 0.0, par_b );
         /* Reset initial guess to zero */
         HYPRE_ParCSRMatrixMatvec(0.0, par_A, par_b, 0.0, par_x );
      }
      else if (object_type == HYPRE_STRUCT)
      {
         HYPRE_StructMatrix    sA;
         HYPRE_StructVector    sb;
         HYPRE_StructVector    sx;

         HYPRE_SStructMatrixGetObject(A, (void **) &sA);
         HYPRE_SStructVectorGetObject(b, (void **) &sb);
         HYPRE_SStructVectorGetObject(x, (void **) &sx);

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
      HYPRE_Int offsets[3][2][3] = { {{0,0,0}, {-1,0,0}},
                                     {{0,0,0}, {0,-1,0}},
                                     {{0,0,0}, {0,0,-1}} };
      HYPRE_Real stencil_values[2] = {1.0, -1.0};

      /* Set up the domain grid */

      HYPRE_SStructGridCreate(hypre_MPI_COMM_WORLD, data.ndim, data.nparts,
                              &G_grid);
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

      G_stencils = hypre_CTAlloc(HYPRE_SStructStencil, data.ndim);
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

      HYPRE_SStructGraphCreate(hypre_MPI_COMM_WORLD, grid, &G_graph);
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
      
      HYPRE_SStructMatrixCreate(hypre_MPI_COMM_WORLD, G_graph, &G);
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
               for (box = 0; box < pdata.nboxes; box++)
               {
                  GetVariableBox(pdata.ilowers[box], pdata.iuppers[box],
                                 pdata.vartypes[var], ilower, iupper);
                  HYPRE_SStructMatrixSetBoxValues(G, part, ilower, iupper,
                                                  var, 1, &i, values);
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

   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

   hypre_TFree(values);

   HYPRE_SStructGridDestroy(grid);
   for (s = 0; s < data.nstencils; s++)
   {
      HYPRE_SStructStencilDestroy(stencils[s]);
   }
   hypre_TFree(stencils);
   HYPRE_SStructGraphDestroy(graph);
   if (gradient_matrix)
   {
      for (s = 0; s < data.ndim; s++)
      {
         HYPRE_SStructStencilDestroy(G_stencils[s]);
      }
      hypre_TFree(G_stencils);
      HYPRE_SStructGraphDestroy(G_graph);
      HYPRE_SStructGridDestroy(G_grid);
      HYPRE_SStructMatrixDestroy(G);
   }

   hypre_TFree(parts);
   hypre_TFree(refine);
   hypre_TFree(distribute);
   hypre_TFree(block);

   static_problem_data = data;
   static_system_data.print_system = print_system;

   *A_ptr = A;
   *b_ptr = b;
   *x_ptr = x;

   return (0);
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DriveSystemSStructDestroy(
   HYPRE_SStructMatrix A,
   HYPRE_SStructVector b,
   HYPRE_SStructVector x )
{
   HYPRE_Int         print_system = static_system_data.print_system;
   ProblemData       data         = static_problem_data;

   ProblemPartData   pdata;
   Index             ilower, iupper;
   HYPRE_Int         part, var, box, j, size;
   HYPRE_Real       *values;

   /*-----------------------------------------------------------
    * Gather the solution vector
    *-----------------------------------------------------------*/

   HYPRE_SStructVectorGather(x);

   /*-----------------------------------------------------------
    * Print the solution and other info
    *-----------------------------------------------------------*/

   if (print_system)
   {
      FILE      *file;
      char       filename[255];
      HYPRE_Int  myid;

      hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);

      HYPRE_SStructVectorPrint("sstruct.out.x", x, 0);

      /* print out with shared data replicated */
      values = hypre_TAlloc(HYPRE_Real, data.max_boxsize);
      for (part = 0; part < data.nparts; part++)
      {
         pdata = data.pdata[part];
         for (var = 0; var < pdata.nvars; var++)
         {
            hypre_sprintf(filename, "sstruct.out.xx.%02d.%02d.%05d",
                          part, var, myid);
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
                                               var, values);
               hypre_fprintf(file, "\nBox %d:\n\n", box);
               size = 1;
               for (j = 0; j < data.ndim; j++)
               {
                  size*= (iupper[j] - ilower[j] + 1);
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
      hypre_TFree(values);
   }

   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

   HYPRE_SStructMatrixDestroy(A);
   HYPRE_SStructVectorDestroy(b);
   HYPRE_SStructVectorDestroy(x);

   DestroyData(data);

   return (0);
}
