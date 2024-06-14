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
#include "HYPRE_krylov.h"
#include "_hypre_sstruct_mv.h"
#include "_hypre_sstruct_ls.h"

#define DEBUG 0

/*--------------------------------------------------------------------------
 * Data structures
 *--------------------------------------------------------------------------*/

char infile_default[50] = "sstruct.in.default";

typedef HYPRE_Int Index[3];
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

   /* for GridSetNeighborBox */
   HYPRE_Int              glue_nboxes;
   ProblemIndex          *glue_ilowers;
   ProblemIndex          *glue_iuppers;
   HYPRE_Int             *glue_nbor_parts;
   ProblemIndex          *glue_nbor_ilowers;
   ProblemIndex          *glue_nbor_iuppers;
   Index                 *glue_index_maps;

   /* for GraphSetStencil */
   HYPRE_Int             *stencil_num;

   /* for GraphAddEntries */
   HYPRE_Int              graph_nentries;
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

   HYPRE_Int              matrix_nentries;
   ProblemIndex          *matrix_ilowers;
   ProblemIndex          *matrix_iuppers;
   Index                 *matrix_strides;
   HYPRE_Int             *matrix_vars;
   HYPRE_Int             *matrix_entries;
   HYPRE_Int              matrix_values_size;
   HYPRE_Real            *matrix_values;
   HYPRE_Real            *d_matrix_values;

   Index                  periodic;

} ProblemPartData;

typedef struct
{
   HYPRE_Int              ndim;
   HYPRE_Int              nparts;
   ProblemPartData       *pdata;
   HYPRE_Int              max_boxsize;

   HYPRE_MemoryLocation   memory_location;

   HYPRE_Int              nstencils;
   HYPRE_Int             *stencil_sizes;
   Index                **stencil_offsets;
   HYPRE_Int            **stencil_vars;
   HYPRE_Real           **stencil_values;

   HYPRE_Int              symmetric_nentries;
   HYPRE_Int             *symmetric_parts;
   HYPRE_Int             *symmetric_vars;
   HYPRE_Int             *symmetric_to_vars;
   HYPRE_Int             *symmetric_booleans;

   HYPRE_Int              ns_symmetric;

   Index                  rfactor;

   HYPRE_Int              npools;
   HYPRE_Int             *pools;   /* array of size nparts */

} ProblemData;

/*--------------------------------------------------------------------------
 * Read routines
 *--------------------------------------------------------------------------*/

HYPRE_Int
SScanIntArray( char        *sdata_ptr,
               char       **sdata_ptr_ptr,
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

   HYPRE_Int          part, var, entry, s, i, il, iu;

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
   data.symmetric_nentries = 0;
   data.symmetric_parts    = NULL;
   data.symmetric_vars     = NULL;
   data.symmetric_to_vars  = NULL;
   data.symmetric_booleans = NULL;
   data.ns_symmetric = 0;

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
            SScanIntArray(sdata_ptr, &sdata_ptr,
                          pdata.nvars, (HYPRE_Int *) pdata.vartypes);
            data.pdata[part] = pdata;
         }
         else if ( strcmp(key, "GridAddVariables:") == 0 )
         {
            /* TODO */
            hypre_printf("GridAddVariables not yet implemented!\n");
            exit(1);
         }
         else if ( strcmp(key, "GridSetNeighborBox:") == 0 )
         {
            part = strtol(sdata_ptr, &sdata_ptr, 10);
            pdata = data.pdata[part];
            if ((pdata.glue_nboxes % 10) == 0)
            {
               size = pdata.glue_nboxes + 10;
               pdata.glue_ilowers =
                  hypre_TReAlloc(pdata.glue_ilowers,  ProblemIndex,  size, HYPRE_MEMORY_HOST);
               pdata.glue_iuppers =
                  hypre_TReAlloc(pdata.glue_iuppers,  ProblemIndex,  size, HYPRE_MEMORY_HOST);
               pdata.glue_nbor_parts =
                  hypre_TReAlloc(pdata.glue_nbor_parts,  HYPRE_Int,  size, HYPRE_MEMORY_HOST);
               pdata.glue_nbor_ilowers =
                  hypre_TReAlloc(pdata.glue_nbor_ilowers,  ProblemIndex,  size, HYPRE_MEMORY_HOST);
               pdata.glue_nbor_iuppers =
                  hypre_TReAlloc(pdata.glue_nbor_iuppers,  ProblemIndex,  size, HYPRE_MEMORY_HOST);
               pdata.glue_index_maps =
                  hypre_TReAlloc(pdata.glue_index_maps,  Index,  size, HYPRE_MEMORY_HOST);
            }
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.glue_ilowers[pdata.glue_nboxes]);
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.glue_iuppers[pdata.glue_nboxes]);
            pdata.glue_nbor_parts[pdata.glue_nboxes] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.glue_nbor_ilowers[pdata.glue_nboxes]);
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.glue_nbor_iuppers[pdata.glue_nboxes]);
            SScanIntArray(sdata_ptr, &sdata_ptr, data.ndim,
                          pdata.glue_index_maps[pdata.glue_nboxes]);
            for (i = data.ndim; i < 3; i++)
            {
               pdata.glue_index_maps[pdata.glue_nboxes][i] = i;
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
            if ((pdata.graph_nentries % 10) == 0)
            {
               size = pdata.graph_nentries + 10;
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
                              pdata.graph_ilowers[pdata.graph_nentries]);
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.graph_iuppers[pdata.graph_nentries]);
            SScanIntArray(sdata_ptr, &sdata_ptr, data.ndim,
                          pdata.graph_strides[pdata.graph_nentries]);
            for (i = data.ndim; i < 3; i++)
            {
               pdata.graph_strides[pdata.graph_nentries][i] = 1;
            }
            pdata.graph_vars[pdata.graph_nentries] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            pdata.graph_to_parts[pdata.graph_nentries] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.graph_to_ilowers[pdata.graph_nentries]);
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.graph_to_iuppers[pdata.graph_nentries]);
            SScanIntArray(sdata_ptr, &sdata_ptr, data.ndim,
                          pdata.graph_to_strides[pdata.graph_nentries]);
            for (i = data.ndim; i < 3; i++)
            {
               pdata.graph_to_strides[pdata.graph_nentries][i] = 1;
            }
            pdata.graph_to_vars[pdata.graph_nentries] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            SScanIntArray(sdata_ptr, &sdata_ptr, data.ndim,
                          pdata.graph_index_maps[pdata.graph_nentries]);
            for (i = data.ndim; i < 3; i++)
            {
               pdata.graph_index_maps[pdata.graph_nentries][i] = i;
            }
            for (i = 0; i < 3; i++)
            {
               pdata.graph_index_signs[pdata.graph_nentries][i] = 1;
               if ( pdata.graph_to_iuppers[pdata.graph_nentries][i] <
                    pdata.graph_to_ilowers[pdata.graph_nentries][i] )
               {
                  pdata.graph_index_signs[pdata.graph_nentries][i] = -1;
               }
            }
            pdata.graph_entries[pdata.graph_nentries] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            pdata.graph_values[pdata.graph_nentries] =
               (HYPRE_Real)strtod(sdata_ptr, &sdata_ptr);
            pdata.graph_boxsizes[pdata.graph_nentries] = 1;
            for (i = 0; i < 3; i++)
            {
               pdata.graph_boxsizes[pdata.graph_nentries] *=
                  (pdata.graph_iuppers[pdata.graph_nentries][i] -
                   pdata.graph_ilowers[pdata.graph_nentries][i] + 1);
            }
            pdata.graph_nentries++;
            data.pdata[part] = pdata;
         }
         else if ( strcmp(key, "MatrixSetSymmetric:") == 0 )
         {
            if ((data.symmetric_nentries % 10) == 0)
            {
               size = data.symmetric_nentries + 10;
               data.symmetric_parts =
                  hypre_TReAlloc(data.symmetric_parts,  HYPRE_Int,  size, HYPRE_MEMORY_HOST);
               data.symmetric_vars =
                  hypre_TReAlloc(data.symmetric_vars,  HYPRE_Int,  size, HYPRE_MEMORY_HOST);
               data.symmetric_to_vars =
                  hypre_TReAlloc(data.symmetric_to_vars,  HYPRE_Int,  size, HYPRE_MEMORY_HOST);
               data.symmetric_booleans =
                  hypre_TReAlloc(data.symmetric_booleans,  HYPRE_Int,  size, HYPRE_MEMORY_HOST);
            }
            data.symmetric_parts[data.symmetric_nentries] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            data.symmetric_vars[data.symmetric_nentries] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            data.symmetric_to_vars[data.symmetric_nentries] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            data.symmetric_booleans[data.symmetric_nentries] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            data.symmetric_nentries++;
         }
         else if ( strcmp(key, "MatrixSetNSSymmetric:") == 0 )
         {
            data.ns_symmetric = strtol(sdata_ptr, &sdata_ptr, 10);
         }
         else if ( strcmp(key, "MatrixSetValues:") == 0 )
         {
            part = strtol(sdata_ptr, &sdata_ptr, 10);
            pdata = data.pdata[part];
            if ((pdata.matrix_nentries % 10) == 0)
            {
               size = pdata.matrix_nentries + 10;
               pdata.matrix_ilowers =
                  hypre_TReAlloc(pdata.matrix_ilowers,  ProblemIndex,  size, HYPRE_MEMORY_HOST);
               pdata.matrix_iuppers =
                  hypre_TReAlloc(pdata.matrix_iuppers,  ProblemIndex,  size, HYPRE_MEMORY_HOST);
               pdata.matrix_strides =
                  hypre_TReAlloc(pdata.matrix_strides,  Index,  size, HYPRE_MEMORY_HOST);
               pdata.matrix_vars =
                  hypre_TReAlloc(pdata.matrix_vars,  HYPRE_Int,  size, HYPRE_MEMORY_HOST);
               pdata.matrix_entries =
                  hypre_TReAlloc(pdata.matrix_entries,  HYPRE_Int,  size, HYPRE_MEMORY_HOST);
               pdata.matrix_values =
                  hypre_TReAlloc(pdata.matrix_values,  HYPRE_Real,  size, HYPRE_MEMORY_HOST);
               pdata.d_matrix_values =
                  hypre_TReAlloc_v2(pdata.d_matrix_values, HYPRE_Real, pdata.matrix_values_size,
                                    HYPRE_Real, size, memory_location);
               pdata.matrix_values_size = size;
            }
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.matrix_ilowers[pdata.matrix_nentries]);
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.matrix_iuppers[pdata.matrix_nentries]);
            SScanIntArray(sdata_ptr, &sdata_ptr, data.ndim,
                          pdata.matrix_strides[pdata.matrix_nentries]);
            for (i = data.ndim; i < 3; i++)
            {
               pdata.matrix_strides[pdata.matrix_nentries][i] = 1;
            }
            pdata.matrix_vars[pdata.matrix_nentries] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            pdata.matrix_entries[pdata.matrix_nentries] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            pdata.matrix_values[pdata.matrix_nentries] =
               (HYPRE_Real)strtod(sdata_ptr, &sdata_ptr);
            pdata.matrix_nentries++;
            data.pdata[part] = pdata;
         }

         else if ( strcmp(key, "rfactor:") == 0 )
         {
            SScanIntArray(sdata_ptr, &sdata_ptr, data.ndim, data.rfactor);
            for (i = data.ndim; i < 3; i++)
            {
               data.rfactor[i] = 1;
            }
         }

         else if ( strcmp(key, "ProcessPoolCreate:") == 0 )
         {
            data.npools = strtol(sdata_ptr, &sdata_ptr, 10);
            data.pools = hypre_CTAlloc(HYPRE_Int,  data.nparts, HYPRE_MEMORY_HOST);
         }
         else if ( strcmp(key, "ProcessPoolSetPart:") == 0 )
         {
            i = strtol(sdata_ptr, &sdata_ptr, 10);
            part = strtol(sdata_ptr, &sdata_ptr, 10);
            data.pools[part] = i;
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
   HYPRE_Int        pool, part, box, entry, p, q, r, i, d, dmap, sign, size;
   Index            m, mmap, n;
   ProblemIndex     int_ilower, int_iupper;

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
         pdata.glue_nboxes = 0;
         pdata.graph_nentries = 0;
         pdata.matrix_nentries = 0;
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

            for (entry = 0; entry < pdata.graph_nentries; entry++)
            {
               MapProblemIndex(pdata.graph_ilowers[entry], m);
               MapProblemIndex(pdata.graph_iuppers[entry], m);
               mmap[0] = m[pdata.graph_index_maps[entry][0]];
               mmap[1] = m[pdata.graph_index_maps[entry][1]];
               mmap[2] = m[pdata.graph_index_maps[entry][2]];
               MapProblemIndex(pdata.graph_to_ilowers[entry], mmap);
               MapProblemIndex(pdata.graph_to_iuppers[entry], mmap);
            }
            for (entry = 0; entry < pdata.matrix_nentries; entry++)
            {
               MapProblemIndex(pdata.matrix_ilowers[entry], m);
               MapProblemIndex(pdata.matrix_iuppers[entry], m);
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
            for (entry = 0; entry < pdata.graph_nentries; entry++)
            {
               MapProblemIndex(pdata.graph_ilowers[entry], m);
               MapProblemIndex(pdata.graph_iuppers[entry], m);
               mmap[0] = m[pdata.graph_index_maps[entry][0]];
               mmap[1] = m[pdata.graph_index_maps[entry][1]];
               mmap[2] = m[pdata.graph_index_maps[entry][2]];
               MapProblemIndex(pdata.graph_to_ilowers[entry], mmap);
               MapProblemIndex(pdata.graph_to_iuppers[entry], mmap);

               for (box = 0; box < pdata.nboxes; box++)
               {
                  size = IntersectBoxes(pdata.graph_ilowers[entry],
                                        pdata.graph_iuppers[entry],
                                        pdata.ilowers[box],
                                        pdata.iuppers[box],
                                        int_ilower, int_iupper);
                  if (size > 0)
                  {
                     /* if there is an intersection, it is the only one */
                     for (d = 0; d < 3; d++)
                     {
                        dmap = pdata.graph_index_maps[entry][d];
                        sign = pdata.graph_index_signs[entry][d];
                        pdata.graph_to_iuppers[i][dmap] =
                           pdata.graph_to_ilowers[entry][dmap] + sign *
                           (int_iupper[d] - pdata.graph_ilowers[entry][d]);
                        pdata.graph_to_ilowers[i][dmap] =
                           pdata.graph_to_ilowers[entry][dmap] + sign *
                           (int_ilower[d] - pdata.graph_ilowers[entry][d]);
                        pdata.graph_ilowers[i][d] = int_ilower[d];
                        pdata.graph_iuppers[i][d] = int_iupper[d];
                        pdata.graph_strides[i][d] =
                           pdata.graph_strides[entry][d];
                        pdata.graph_to_strides[i][d] =
                           pdata.graph_to_strides[entry][d];
                        pdata.graph_index_maps[i][d]  = dmap;
                        pdata.graph_index_signs[i][d] = sign;
                     }
                     for (d = 3; d < 9; d++)
                     {
                        pdata.graph_ilowers[i][d] =
                           pdata.graph_ilowers[entry][d];
                        pdata.graph_iuppers[i][d] =
                           pdata.graph_iuppers[entry][d];
                        pdata.graph_to_ilowers[i][d] =
                           pdata.graph_to_ilowers[entry][d];
                        pdata.graph_to_iuppers[i][d] =
                           pdata.graph_to_iuppers[entry][d];
                     }
                     pdata.graph_vars[i]     = pdata.graph_vars[entry];
                     pdata.graph_to_parts[i] = pdata.graph_to_parts[entry];
                     pdata.graph_to_vars[i]  = pdata.graph_to_vars[entry];
                     pdata.graph_entries[i]  = pdata.graph_entries[entry];
                     pdata.graph_values[i]   = pdata.graph_values[entry];
                     i++;
                     break;
                  }
               }
            }
            pdata.graph_nentries = i;

            i = 0;
            for (entry = 0; entry < pdata.matrix_nentries; entry++)
            {
               MapProblemIndex(pdata.matrix_ilowers[entry], m);
               MapProblemIndex(pdata.matrix_iuppers[entry], m);

               for (box = 0; box < pdata.nboxes; box++)
               {
                  size = IntersectBoxes(pdata.matrix_ilowers[entry],
                                        pdata.matrix_iuppers[entry],
                                        pdata.ilowers[box],
                                        pdata.iuppers[box],
                                        int_ilower, int_iupper);
                  if (size > 0)
                  {
                     /* if there is an intersection, it is the only one */
                     for (d = 0; d < 3; d++)
                     {
                        pdata.matrix_ilowers[i][d] = int_ilower[d];
                        pdata.matrix_iuppers[i][d] = int_iupper[d];
                        pdata.matrix_strides[i][d] =
                           pdata.matrix_strides[entry][d];
                     }
                     for (d = 3; d < 9; d++)
                     {
                        pdata.matrix_ilowers[i][d] =
                           pdata.matrix_ilowers[entry][d];
                        pdata.matrix_iuppers[i][d] =
                           pdata.matrix_iuppers[entry][d];
                     }
                     pdata.matrix_vars[i]    = pdata.matrix_vars[entry];
                     pdata.matrix_entries[i]  = pdata.matrix_entries[entry];
                     pdata.matrix_values[i]   = pdata.matrix_values[entry];
                     i++;
                     break;
                  }
               }
            }
            pdata.matrix_nentries = i;
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

            for (entry = 0; entry < pdata.graph_nentries; entry++)
            {
               MapProblemIndex(pdata.graph_ilowers[entry], m);
               MapProblemIndex(pdata.graph_iuppers[entry], m);
               mmap[0] = m[pdata.graph_index_maps[entry][0]];
               mmap[1] = m[pdata.graph_index_maps[entry][1]];
               mmap[2] = m[pdata.graph_index_maps[entry][2]];
               MapProblemIndex(pdata.graph_to_ilowers[entry], mmap);
               MapProblemIndex(pdata.graph_to_iuppers[entry], mmap);
            }
            for (entry = 0; entry < pdata.matrix_nentries; entry++)
            {
               MapProblemIndex(pdata.matrix_ilowers[entry], m);
               MapProblemIndex(pdata.matrix_iuppers[entry], m);
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
         for (box = 0; box < pdata.graph_nentries; box++)
         {
            pdata.graph_boxsizes[box] = 1;
            for (i = 0; i < 3; i++)
            {
               pdata.graph_boxsizes[box] *=
                  (pdata.graph_iuppers[box][i] -
                   pdata.graph_ilowers[box][i] + 1);
            }
         }
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
         hypre_TFree(pdata.glue_ilowers, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.glue_iuppers, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.glue_nbor_parts, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.glue_nbor_ilowers, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.glue_nbor_iuppers, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.glue_index_maps, HYPRE_MEMORY_HOST);
      }

      if (pdata.graph_nentries == 0)
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

      if (pdata.matrix_nentries == 0)
      {
         hypre_TFree(pdata.matrix_ilowers, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.matrix_iuppers, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.matrix_strides, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.matrix_vars, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.matrix_entries, HYPRE_MEMORY_HOST);
         pdata.matrix_values_size = 0;
         hypre_TFree(pdata.matrix_values, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.d_matrix_values, memory_location);
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
   HYPRE_Int        part, s;

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
         hypre_TFree(pdata.glue_ilowers, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.glue_iuppers, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.glue_nbor_parts, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.glue_nbor_ilowers, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.glue_nbor_iuppers, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.glue_index_maps, HYPRE_MEMORY_HOST);
      }

      if (pdata.nvars > 0)
      {
         hypre_TFree(pdata.stencil_num, HYPRE_MEMORY_HOST);
      }

      if (pdata.graph_nentries > 0)
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
         hypre_TFree(pdata.graph_values, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.d_graph_values, memory_location);
         hypre_TFree(pdata.graph_boxsizes, HYPRE_MEMORY_HOST);
      }

      if (pdata.matrix_nentries > 0)
      {
         hypre_TFree(pdata.matrix_ilowers, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.matrix_iuppers, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.matrix_strides, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.matrix_vars, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.matrix_entries, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.matrix_values, HYPRE_MEMORY_HOST);
         hypre_TFree(pdata.d_matrix_values, memory_location);
      }

   }
   hypre_TFree(data.pdata, HYPRE_MEMORY_HOST);

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

   if (data.symmetric_nentries > 0)
   {
      hypre_TFree(data.symmetric_parts, HYPRE_MEMORY_HOST);
      hypre_TFree(data.symmetric_vars, HYPRE_MEMORY_HOST);
      hypre_TFree(data.symmetric_to_vars, HYPRE_MEMORY_HOST);
      hypre_TFree(data.symmetric_booleans, HYPRE_MEMORY_HOST);
   }

   hypre_TFree(data.pools, HYPRE_MEMORY_HOST);

   return 0;
}

/*--------------------------------------------------------------------------
 * Compute new box based on variable type
 *--------------------------------------------------------------------------*/

HYPRE_Int
GetVariableBox( Index        cell_ilower,
                Index        cell_iupper,
                HYPRE_Int    int_vartype,
                Index        var_ilower,
                Index        var_iupper )
{
   HYPRE_Int ierr = 0;
   HYPRE_SStructVariable  vartype = (HYPRE_SStructVariable) int_vartype;

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
 * Print usage info
 *--------------------------------------------------------------------------*/

HYPRE_Int
PrintUsage( char       *progname,
            HYPRE_Int   myid )
{
   if ( myid == 0 )
   {
      hypre_printf("\n");
      hypre_printf("Usage: %s [<options>]\n", progname);
      hypre_printf("\n");
      hypre_printf("  -in <filename> : input file (default is `%s')\n",
                   infile_default);
      hypre_printf("\n");
      hypre_printf("  -pt <pt1> <pt2> ... : set part(s) for subsequent options\n");
      hypre_printf("  -r <rx> <ry> <rz>   : refine part(s)\n");
      hypre_printf("  -P <Px> <Py> <Pz>   : refine and distribute part(s)\n");
      hypre_printf("  -b <bx> <by> <bz>   : refine and block part(s)\n");
      hypre_printf("  -solver <ID>        : solver ID (default = 39)\n");
      hypre_printf("  -print             : print out the system\n");
      hypre_printf("  -sym <s>           : Struct- symmetric storage (1) or not (0)\n");

      hypre_printf("\n");
   }

   return 0;
}

/*--------------------------------------------------------------------------
 * Test driver for semi-structured matrix interface
 *--------------------------------------------------------------------------*/

hypre_int
main( hypre_int argc,
      char     *argv[] )
{
   char                 *infile;
   ProblemData           global_data;
   ProblemData           data;
   ProblemPartData       pdata;
   HYPRE_Int             nparts;
   HYPRE_Int            *parts;
   Index                *refine;
   Index                *distribute;
   Index                *block;
   HYPRE_Int             solver_id;
   HYPRE_Int             print_system;

   HYPRE_SStructGrid     grid;
   HYPRE_SStructStencil *stencils;
   HYPRE_SStructGraph    graph;
   HYPRE_SStructMatrix   A;
   HYPRE_ParCSRMatrix    T, parA;
   HYPRE_SStructVector   b;
   HYPRE_SStructVector   x;
   HYPRE_ParVector       parb, parx;
   HYPRE_SStructSolver   solver;

   HYPRE_StructGrid      cell_grid;
   hypre_Box            *bounding_box;
   HYPRE_Real            h;

   HYPRE_Int           **bdryRanks, *bdryRanksCnt;

   Index                 ilower, iupper;
   Index                 index, to_index;
   HYPRE_Real           *values;
   HYPRE_Real           *d_values;

   HYPRE_Int             num_iterations;
   HYPRE_Real            final_res_norm;

   HYPRE_Int             num_procs, myid;
   HYPRE_Int             time_index;

   HYPRE_Int             arg_index, part, box, var, entry, s, i, j, k;

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

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   /* Initialize MPI */
   hypre_MPI_Init(&argc, &argv);
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);

   /*-----------------------------------------------------------------
    * GPU Device binding
    * Must be done before HYPRE_Initialize() and should not be changed after
    *-----------------------------------------------------------------*/
   hypre_bind_device_id(-1, myid, num_procs, hypre_MPI_COMM_WORLD);

   /*-----------------------------------------------------------
    * Initialize : must be the first HYPRE function to call
    *-----------------------------------------------------------*/
   HYPRE_Initialize();
   HYPRE_DeviceInitialize();

   /*-----------------------------------------------------------
    * Read input file
    *-----------------------------------------------------------*/

   arg_index = 1;

   /* parse command line for input file name */
   infile = infile_default;
   if (argc > 1)
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

   parts      = hypre_TAlloc(HYPRE_Int, nparts, HYPRE_MEMORY_HOST);
   refine     = hypre_TAlloc(Index, nparts, HYPRE_MEMORY_HOST);
   distribute = hypre_TAlloc(Index, nparts, HYPRE_MEMORY_HOST);
   block      = hypre_TAlloc(Index, nparts, HYPRE_MEMORY_HOST);
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

   solver_id = 0;
   print_system = 0;

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
         solver_id = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-print") == 0 )
      {
         arg_index++;
         print_system = 1;
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
      else if ( strcmp(argv[arg_index], "-help") == 0 )
      {
         PrintUsage(argv[0], myid);
         exit(1);
         break;
      }
      else
      {
         break;
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

   /*-----------------------------------------------------------
    * Distribute data
    *-----------------------------------------------------------*/

   DistributeData(global_data, refine, distribute, block,
                  num_procs, myid, &data);

   /*-----------------------------------------------------------
    * Synchronize so that timings make sense
    *-----------------------------------------------------------*/

   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);

   /*-----------------------------------------------------------
    * Set up the grid
    *-----------------------------------------------------------*/

   time_index = hypre_InitializeTiming("SStruct Interface");
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

      /* GridSetNeighborBox */
      for (box = 0; box < pdata.glue_nboxes; box++)
      {
         hypre_printf("Error: No longer supporting SetNeighborBox\n");
#if 0
         HYPRE_SStructGridSetNeighborBox(grid, part,
                                         pdata.glue_ilowers[box],
                                         pdata.glue_iuppers[box],
                                         pdata.glue_nbor_parts[box],
                                         pdata.glue_nbor_ilowers[box],
                                         pdata.glue_nbor_iuppers[box],
                                         pdata.glue_index_maps[box]);
#endif
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
      for (i = 0; i < data.stencil_sizes[s]; i++)
      {
         HYPRE_SStructStencilSetEntry(stencils[s], i,
                                      data.stencil_offsets[s][i],
                                      data.stencil_vars[s][i]);
      }
   }

   /*-----------------------------------------------------------
    * Set up the graph
    *-----------------------------------------------------------*/

   HYPRE_SStructGraphCreate(hypre_MPI_COMM_WORLD, grid, &graph);
   HYPRE_SStructGraphSetObjectType(graph, HYPRE_PARCSR);

   for (part = 0; part < data.nparts; part++)
   {
      pdata = data.pdata[part];

      /* set stencils */
      for (var = 0; var < pdata.nvars; var++)
      {
         HYPRE_SStructGraphSetStencil(graph, part, var,
                                      stencils[pdata.stencil_num[var]]);
      }

      /* add entries */
      for (entry = 0; entry < pdata.graph_nentries; entry++)
      {
         for (index[2] = pdata.graph_ilowers[entry][2];
              index[2] <= pdata.graph_iuppers[entry][2];
              index[2] += pdata.graph_strides[entry][2])
         {
            for (index[1] = pdata.graph_ilowers[entry][1];
                 index[1] <= pdata.graph_iuppers[entry][1];
                 index[1] += pdata.graph_strides[entry][1])
            {
               for (index[0] = pdata.graph_ilowers[entry][0];
                    index[0] <= pdata.graph_iuppers[entry][0];
                    index[0] += pdata.graph_strides[entry][0])
               {
                  for (i = 0; i < 3; i++)
                  {
                     j = pdata.graph_index_maps[entry][i];
                     k = index[i] - pdata.graph_ilowers[entry][i];
                     k /= pdata.graph_strides[entry][i];
                     k *= pdata.graph_index_signs[entry][i];
                     to_index[j] = pdata.graph_to_ilowers[entry][j] +
                                   k * pdata.graph_to_strides[entry][j];
                  }
                  HYPRE_SStructGraphAddEntries(graph, part, index,
                                               pdata.graph_vars[entry],
                                               pdata.graph_to_parts[entry],
                                               to_index,
                                               pdata.graph_to_vars[entry]);
               }
            }
         }
      }
   }

   HYPRE_SStructGraphAssemble(graph);

   /*-----------------------------------------------------------
    * Set up the matrix
    *-----------------------------------------------------------*/

   values = hypre_CTAlloc(HYPRE_Real, data.max_boxsize, HYPRE_MEMORY_HOST);
   d_values = hypre_TAlloc(HYPRE_Real, data.max_boxsize, memory_location);

   HYPRE_SStructMatrixCreate(hypre_MPI_COMM_WORLD, graph, &A);

   /* TODO HYPRE_SStructMatrixSetSymmetric(A, 1); */
   for (entry = 0; entry < data.symmetric_nentries; entry++)
   {
      HYPRE_SStructMatrixSetSymmetric(A,
                                      data.symmetric_parts[entry],
                                      data.symmetric_vars[entry],
                                      data.symmetric_to_vars[entry],
                                      data.symmetric_booleans[entry]);
   }
   HYPRE_SStructMatrixSetNSSymmetric(A, data.ns_symmetric);

   HYPRE_SStructMatrixSetObjectType(A, HYPRE_PARCSR);
   HYPRE_SStructMatrixInitialize(A);

   for (part = 0; part < data.nparts; part++)
   {
      pdata = data.pdata[part];
      cell_grid = hypre_SStructPGridCellSGrid(hypre_SStructGridPGrid(grid, part));
      bounding_box = hypre_StructGridBoundingBox(cell_grid);

      h = (HYPRE_Real) (hypre_BoxIMax(bounding_box)[0] - hypre_BoxIMin(bounding_box)[0]);
      for (i = 1; i < data.ndim; i++)
      {
         if ((hypre_BoxIMax(bounding_box)[i] - hypre_BoxIMin(bounding_box)[i]) > h)
         {
            h = (HYPRE_Real) (hypre_BoxIMax(bounding_box)[i] - hypre_BoxIMin(bounding_box)[i]);
         }
      }
      h = 1.0 / h;

      /* set stencil values */
      for (var = 0; var < pdata.nvars; var++)
      {
         s = pdata.stencil_num[var];
         for (i = 0; i < data.stencil_sizes[s]; i++)
         {
            for (j = 0; j < pdata.max_boxsize; j++)
            {
               values[j] = h * data.stencil_values[s][i];
            }
            if (i < 9)
            {
               for (j = 0; j < pdata.max_boxsize; j++)
               {
                  values[j] += data.stencil_values[s + data.ndim][i] / h;
               }
            }

            hypre_TMemcpy(d_values, values, HYPRE_Real, data.max_boxsize,
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

      hypre_TMemcpy(pdata.d_graph_values, pdata.graph_values, HYPRE_Real, pdata.graph_values_size,
                    memory_location, HYPRE_MEMORY_HOST);

      /* set non-stencil entries */
      for (entry = 0; entry < pdata.graph_nentries; entry++)
      {
         for (index[2] = pdata.graph_ilowers[entry][2];
              index[2] <= pdata.graph_iuppers[entry][2];
              index[2] += pdata.graph_strides[entry][2])
         {
            for (index[1] = pdata.graph_ilowers[entry][1];
                 index[1] <= pdata.graph_iuppers[entry][1];
                 index[1] += pdata.graph_strides[entry][1])
            {
               for (index[0] = pdata.graph_ilowers[entry][0];
                    index[0] <= pdata.graph_iuppers[entry][0];
                    index[0] += pdata.graph_strides[entry][0])
               {
                  HYPRE_SStructMatrixSetValues(A, part, index,
                                               pdata.graph_vars[entry],
                                               1, &pdata.graph_entries[entry],
                                               &pdata.d_graph_values[entry]);
               }
            }
         }
      }
   }

   /* reset matrix values:
    *   NOTE THAT THE matrix_ilowers & matrix_iuppers MUST BE IN TERMS OF THE
    *   CHOOSEN VAR_TYPE INDICES, UNLIKE THE EXTENTS OF THE GRID< WHICH ARE
    *   IN TEMS OF THE CELL VARTYPE INDICES.
    */
   for (part = 0; part < data.nparts; part++)
   {
      pdata = data.pdata[part];

      hypre_TMemcpy(pdata.d_matrix_values, pdata.matrix_values, HYPRE_Real, pdata.matrix_values_size,
                    memory_location, HYPRE_MEMORY_HOST);

      for (entry = 0; entry < pdata.matrix_nentries; entry++)
      {
         for (index[2] = pdata.matrix_ilowers[entry][2];
              index[2] <= pdata.matrix_iuppers[entry][2];
              index[2] += pdata.matrix_strides[entry][2])
         {
            for (index[1] = pdata.matrix_ilowers[entry][1];
                 index[1] <= pdata.matrix_iuppers[entry][1];
                 index[1] += pdata.matrix_strides[entry][1])
            {
               for (index[0] = pdata.matrix_ilowers[entry][0];
                    index[0] <= pdata.matrix_iuppers[entry][0];
                    index[0] += pdata.matrix_strides[entry][0])
               {
                  HYPRE_SStructMatrixSetValues(A, part, index,
                                               pdata.matrix_vars[entry],
                                               1, &pdata.matrix_entries[entry],
                                               &pdata.d_matrix_values[entry]);
               }
            }
         }
      }
   }

   HYPRE_SStructMatrixAssemble(A);
   HYPRE_MaxwellGrad(grid, &T);

   /* eliminate the physical boundary points */
   HYPRE_SStructMatrixGetObject(A, (void **) &parA);
   HYPRE_SStructMaxwellPhysBdy(&grid, 1, data.rfactor,
                               &bdryRanks, &bdryRanksCnt);

   HYPRE_SStructMaxwellEliminateRowsCols(parA, bdryRanksCnt[0], bdryRanks[0]);

   /*{
      hypre_MaxwellOffProcRow **OffProcRows;
      hypre_SStructSharedDOF_ParcsrMatRowsComm(grid,
                                               (hypre_ParCSRMatrix *) parA,
                                               &i,
                                               &OffProcRows);
      for (j= 0; j< i; j++)
      {
         hypre_MaxwellOffProcRowDestroy((void *) OffProcRows[j]);
      }
      hypre_TFree(OffProcRows, HYPRE_MEMORY_HOST);
    }*/

   /*-----------------------------------------------------------
    * Set up the linear system
    *-----------------------------------------------------------*/

   HYPRE_SStructVectorCreate(hypre_MPI_COMM_WORLD, grid, &b);
   HYPRE_SStructVectorSetObjectType(b, HYPRE_PARCSR);

   HYPRE_SStructVectorInitialize(b);
   for (j = 0; j < data.max_boxsize; j++)
   {
      values[j] = hypre_sin((HYPRE_Real)(j + 1));
      values[j] = (HYPRE_Real) hypre_Rand();
      values[j] = (HYPRE_Real) j;
   }

   hypre_TMemcpy(d_values, values, HYPRE_Real, data.max_boxsize,
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
   HYPRE_SStructVectorAssemble(b);

   HYPRE_SStructVectorCreate(hypre_MPI_COMM_WORLD, grid, &x);
   HYPRE_SStructVectorSetObjectType(x, HYPRE_PARCSR);

   HYPRE_SStructVectorInitialize(x);
   for (j = 0; j < data.max_boxsize; j++)
   {
      values[j] = 0.0;
   }

   hypre_TMemcpy(d_values, values, HYPRE_Real, data.max_boxsize,
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
            HYPRE_SStructVectorSetBoxValues(x, part, ilower, iupper,
                                            var, d_values);
         }
      }
   }
   HYPRE_SStructVectorAssemble(x);

   HYPRE_SStructVectorGetObject(x, (void **) &parx);
   HYPRE_SStructVectorGetObject(b, (void **) &parb);
   HYPRE_SStructMaxwellZeroVector(parx, bdryRanks[0], bdryRanksCnt[0]);
   HYPRE_SStructMaxwellZeroVector(parb, bdryRanks[0], bdryRanksCnt[0]);

   hypre_TFree(bdryRanks[0], HYPRE_MEMORY_HOST);
   hypre_TFree(bdryRanks, HYPRE_MEMORY_HOST);
   hypre_TFree(bdryRanksCnt, HYPRE_MEMORY_HOST);

   hypre_EndTiming(time_index);
   hypre_PrintTiming("SStruct Interface", hypre_MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();

   /*-----------------------------------------------------------
    * Print out the system and initial guess
    *-----------------------------------------------------------*/

   if (print_system)
   {
      HYPRE_SStructMatrixPrint("sstruct.out.A",  A, 0);
      HYPRE_SStructVectorPrint("sstruct.out.b",  b, 0);
      HYPRE_SStructVectorPrint("sstruct.out.x0", x, 0);
   }

   /*-----------------------------------------------------------
    * Debugging code
    *-----------------------------------------------------------*/

   hypre_TFree(values, HYPRE_MEMORY_HOST);
   hypre_TFree(d_values, memory_location);

   if (solver_id == 1)
   {
      time_index = hypre_InitializeTiming("Maxwell Setup");
      hypre_BeginTiming(time_index);

      HYPRE_SStructMaxwellCreate(hypre_MPI_COMM_WORLD, &solver);
      HYPRE_SStructMaxwellSetMaxIter(solver, 20);
      HYPRE_SStructMaxwellSetTol(solver, 1.0e-8);
      HYPRE_SStructMaxwellSetRelChange(solver, 0);
      HYPRE_SStructMaxwellSetNumPreRelax(solver, 1);
      HYPRE_SStructMaxwellSetNumPostRelax(solver, 1);
      HYPRE_SStructMaxwellSetRfactors(solver, data.rfactor);
      HYPRE_SStructMaxwellSetGrad(solver, T);
      /*HYPRE_SStructMaxwellSetConstantCoef(solver, 1);*/
      HYPRE_SStructMaxwellSetPrintLevel(solver, 1);
      HYPRE_SStructMaxwellSetLogging(solver, 1);
      HYPRE_SStructMaxwellSetup(solver, A, b, x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("Maxwell Solve");
      hypre_BeginTiming(time_index);

      HYPRE_SStructMaxwellSolve(solver, A, b, x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_SStructMaxwellGetNumIterations(solver, &num_iterations);
      HYPRE_SStructMaxwellGetFinalRelativeResidualNorm(
         solver, &final_res_norm);
      HYPRE_SStructMaxwellDestroy(solver);
   }

   HYPRE_SStructVectorGather(x);

   /*-----------------------------------------------------------
    * Print the solution and other info
    *-----------------------------------------------------------*/

   if (print_system)
   {
      HYPRE_SStructVectorPrint("sstruct.out.x", x, 0);
   }

   if (myid == 0)
   {
      hypre_printf("\n");
      hypre_printf("Iterations = %d\n", num_iterations);
      hypre_printf("Final Relative Residual Norm = %e\n", final_res_norm);
      hypre_printf("\n");
   }

   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

   HYPRE_SStructGridDestroy(grid);
   for (s = 0; s < data.nstencils; s++)
   {
      HYPRE_SStructStencilDestroy(stencils[s]);
   }
   hypre_TFree(stencils, HYPRE_MEMORY_HOST);
   HYPRE_SStructGraphDestroy(graph);
   HYPRE_SStructMatrixDestroy(A);
   HYPRE_ParCSRMatrixDestroy(T);
   HYPRE_SStructVectorDestroy(b);
   HYPRE_SStructVectorDestroy(x);


   DestroyData(data);

   hypre_TFree(parts, HYPRE_MEMORY_HOST);
   hypre_TFree(refine, HYPRE_MEMORY_HOST);
   hypre_TFree(distribute, HYPRE_MEMORY_HOST);
   hypre_TFree(block, HYPRE_MEMORY_HOST);

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
