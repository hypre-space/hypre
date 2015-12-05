/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.23 $
 ***********************************************************************EHEADER*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "_hypre_utilities.h"
#include "_hypre_sstruct_mv.h"
 
#include "bHYPRE_SStructMatrix.h"
#include "bHYPRE_SStructVector.h"
#include "bHYPRE_Operator.h"
#include "bHYPRE_Solver.h"
#include "bHYPRE_StructSMG.h"
#include "bHYPRE_StructPFMG.h"
#include "bHYPRE_PCG.h"
#include "bHYPRE_GMRES.h"
#include "bHYPRE_BoomerAMG.h"
#include "bHYPRE_ParaSails.h"
#include "bHYPRE_SStructGrid.h"
#include "bHYPRE_SStructStencil.h"
#include "bHYPRE_SStructGraph.h"
#include "bHYPRE_SStructParCSRMatrix.h"
#include "bHYPRE_SStructParCSRVector.h"
#include "bHYPRE_StructMatrix.h"
#include "bHYPRE_StructVector.h"
#include "bHYPRE_IJParCSRMatrix.h"
#include "bHYPRE_IJParCSRVector.h"
#include "bHYPRE_SStructVariable.h"
#include "bHYPRE_SStructSplit.h"
#include "bHYPRE_SStructDiagScale.h"
#include "bHYPRE_IdentitySolver.h"
#include "sidl_Exception.h"

#define DEBUG 0
#define DO_THIS_LATER 0

/*--------------------------------------------------------------------------
 * Test driver for semistructured matrix interface
 * Modified to use the Babel interface.
 * Not finished, but -solver 40 and 200 work.
 *--------------------------------------------------------------------------*/

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
   HYPRE_Int                    nboxes;
   ProblemIndex          *ilowers;
   ProblemIndex          *iuppers;
   HYPRE_Int                   *boxsizes;
   HYPRE_Int                    max_boxsize;

   /* for GridSetVariables */
   HYPRE_Int                    nvars;
   enum bHYPRE_SStructVariable__enum *vartypes;

   /* for GridAddVariables */
   HYPRE_Int                    add_nvars;
   ProblemIndex          *add_indexes;
   enum bHYPRE_SStructVariable__enum *add_vartypes;

   /* for GridSetNeighborBox */
   HYPRE_Int                    glue_nboxes;
   ProblemIndex          *glue_ilowers;
   ProblemIndex          *glue_iuppers;
   HYPRE_Int                   *glue_nbor_parts;
   ProblemIndex          *glue_nbor_ilowers;
   ProblemIndex          *glue_nbor_iuppers;
   Index                 *glue_index_maps;

   /* for GraphSetStencil */
   HYPRE_Int                   *stencil_num;

   /* for GraphAddEntries */
   HYPRE_Int                    graph_nentries;
   ProblemIndex          *graph_ilowers;
   ProblemIndex          *graph_iuppers;
   Index                 *graph_strides;
   HYPRE_Int                   *graph_vars;
   HYPRE_Int                   *graph_to_parts;
   ProblemIndex          *graph_to_ilowers;
   ProblemIndex          *graph_to_iuppers;
   Index                 *graph_to_strides;
   HYPRE_Int                   *graph_to_vars;
   Index                 *graph_index_maps;
   Index                 *graph_index_signs;
   HYPRE_Int                   *graph_entries;
   double                *graph_values;
   HYPRE_Int                   *graph_boxsizes;

   HYPRE_Int                    matrix_nentries;
   ProblemIndex          *matrix_ilowers;
   ProblemIndex          *matrix_iuppers;
   Index                 *matrix_strides;
   HYPRE_Int                   *matrix_vars;
   HYPRE_Int                   *matrix_entries;
   double                *matrix_values;

   Index                  periodic;

} ProblemPartData;
 
typedef struct
{
   HYPRE_Int              ndim;
   HYPRE_Int              nparts;
   ProblemPartData *pdata;
   HYPRE_Int              max_boxsize;

   HYPRE_Int              nstencils;
   HYPRE_Int             *stencil_sizes;
   Index          **stencil_offsets;
   HYPRE_Int            **stencil_vars;
   double         **stencil_values;

   HYPRE_Int              symmetric_nentries;
   HYPRE_Int             *symmetric_parts;
   HYPRE_Int             *symmetric_vars;
   HYPRE_Int             *symmetric_to_vars;
   HYPRE_Int             *symmetric_booleans;

   HYPRE_Int              ns_symmetric;

   HYPRE_Int              npools;
   HYPRE_Int             *pools;   /* array of size nparts */

} ProblemData;
 
/*--------------------------------------------------------------------------
 * Compute new box based on variable type
 *--------------------------------------------------------------------------*/

HYPRE_Int
GetVariableBox( Index  cell_ilower,
                Index  cell_iupper,
                HYPRE_Int    int_vartype,
                Index  var_ilower,
                Index  var_iupper )
{
   HYPRE_Int ierr = 0;
   enum bHYPRE_SStructVariable__enum vartype = (enum bHYPRE_SStructVariable__enum) int_vartype;

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
SScanProblemIndex( char          *sdata_ptr,
                   char         **sdata_ptr_ptr,
                   HYPRE_Int            ndim,
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

   HYPRE_Int                myid;
   FILE              *file;

   char              *sdata = NULL;
   char              *sdata_line;
   char              *sdata_ptr;
   HYPRE_Int                sdata_size;
   HYPRE_Int                size;
   HYPRE_Int                memchunk = 10000;
   HYPRE_Int                maxline  = 250;

   char               key[250];

   HYPRE_Int                part, var, entry, s, i, il, iu;

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
            pdata.vartypes = hypre_CTAlloc(enum bHYPRE_SStructVariable__enum, pdata.nvars);
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
                  hypre_TReAlloc(pdata.glue_ilowers, ProblemIndex, size);
               pdata.glue_iuppers =
                  hypre_TReAlloc(pdata.glue_iuppers, ProblemIndex, size);
               pdata.glue_nbor_parts =
                  hypre_TReAlloc(pdata.glue_nbor_parts, HYPRE_Int, size);
               pdata.glue_nbor_ilowers =
                  hypre_TReAlloc(pdata.glue_nbor_ilowers, ProblemIndex, size);
               pdata.glue_nbor_iuppers =
                  hypre_TReAlloc(pdata.glue_nbor_iuppers, ProblemIndex, size);
               pdata.glue_index_maps =
                  hypre_TReAlloc(pdata.glue_index_maps, Index, size);
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
            data.stencil_sizes   = hypre_CTAlloc(HYPRE_Int, data.nstencils);
            data.stencil_offsets = hypre_CTAlloc(Index *, data.nstencils);
            data.stencil_vars    = hypre_CTAlloc(HYPRE_Int *, data.nstencils);
            data.stencil_values  = hypre_CTAlloc(double *, data.nstencils);
            SScanIntArray(sdata_ptr, &sdata_ptr,
                          data.nstencils, data.stencil_sizes);
            for (s = 0; s < data.nstencils; s++)
            {
               data.stencil_offsets[s] =
                  hypre_CTAlloc(Index, data.stencil_sizes[s]);
               data.stencil_vars[s] =
                  hypre_CTAlloc(HYPRE_Int, data.stencil_sizes[s]);
               data.stencil_values[s] =
                  hypre_CTAlloc(double, data.stencil_sizes[s]);
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
            if ((pdata.graph_nentries % 10) == 0)
            {
               size = pdata.graph_nentries + 10;
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
                  hypre_TReAlloc(pdata.graph_values, double, size);
               pdata.graph_boxsizes =
                  hypre_TReAlloc(pdata.graph_boxsizes, HYPRE_Int, size);
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
               strtod(sdata_ptr, &sdata_ptr);
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
                  hypre_TReAlloc(data.symmetric_parts, HYPRE_Int, size);
               data.symmetric_vars =
                  hypre_TReAlloc(data.symmetric_vars, HYPRE_Int, size);
               data.symmetric_to_vars =
                  hypre_TReAlloc(data.symmetric_to_vars, HYPRE_Int, size);
               data.symmetric_booleans =
                  hypre_TReAlloc(data.symmetric_booleans, HYPRE_Int, size);
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
                  hypre_TReAlloc(pdata.matrix_ilowers, ProblemIndex, size);
               pdata.matrix_iuppers =
                  hypre_TReAlloc(pdata.matrix_iuppers, ProblemIndex, size);
               pdata.matrix_strides =
                  hypre_TReAlloc(pdata.matrix_strides, Index, size);
               pdata.matrix_vars =
                  hypre_TReAlloc(pdata.matrix_vars, HYPRE_Int, size);
               pdata.matrix_entries =
                  hypre_TReAlloc(pdata.matrix_entries, HYPRE_Int, size);
               pdata.matrix_values =
                  hypre_TReAlloc(pdata.matrix_values, double, size);
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
               strtod(sdata_ptr, &sdata_ptr);
            pdata.matrix_nentries++;
            data.pdata[part] = pdata;
         }
         else if ( strcmp(key, "ProcessPoolCreate:") == 0 )
         {
            data.npools = strtol(sdata_ptr, &sdata_ptr, 10);
            data.pools = hypre_CTAlloc(HYPRE_Int, data.nparts);
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
                Index        *refine,
                Index        *distribute,
                Index        *block,
                HYPRE_Int           num_procs,
                HYPRE_Int           myid,
                ProblemData  *data_ptr )
{
   ProblemData      data = global_data;
   ProblemPartData  pdata;
   HYPRE_Int             *pool_procs;
   HYPRE_Int              np, pid;
   HYPRE_Int              pool, part, box, entry, p, q, r, i, d;
   HYPRE_Int              dmap, sign, size;
   Index            m, mmap, n;
   ProblemIndex     ilower, iupper, int_ilower, int_iupper;

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
                  /* first convert the box extents based on vartype */
                  GetVariableBox(pdata.ilowers[box], pdata.iuppers[box],
                                 pdata.vartypes[pdata.graph_vars[entry]],
                                 ilower, iupper);
                  size = IntersectBoxes(pdata.graph_ilowers[entry],
                                        pdata.graph_iuppers[entry],
                                        ilower, iupper,
                                        int_ilower, int_iupper);
                  if (size > 0)
                  {
                     /* if there is an intersection, it is the only one */
                     for (d = 0; d < 3; d++)
                     {
                        dmap = pdata.graph_index_maps[entry][d];
                        sign = pdata.graph_index_signs[entry][d];
                        pdata.graph_to_ilowers[i][dmap] =
                           pdata.graph_to_ilowers[entry][dmap] +
                           sign * pdata.graph_to_strides[entry][d] *
                           ((int_ilower[d] - pdata.graph_ilowers[entry][d]) /
                            pdata.graph_strides[entry][d]);
                        pdata.graph_to_iuppers[i][dmap] =
                           pdata.graph_to_iuppers[entry][dmap] +
                           sign * pdata.graph_to_strides[entry][d] *
                           ((int_iupper[d] - pdata.graph_iuppers[entry][d]) /
                            pdata.graph_strides[entry][d]);
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
                  /* first convert the box extents based on vartype */
                  GetVariableBox(pdata.ilowers[box], pdata.iuppers[box],
                                 pdata.vartypes[pdata.matrix_vars[entry]],
                                 ilower, iupper);
                  size = IntersectBoxes(pdata.matrix_ilowers[entry],
                                        pdata.matrix_iuppers[entry],
                                        ilower, iupper,
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
                     pdata.matrix_vars[i]     = pdata.matrix_vars[entry];
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
         hypre_TFree(pdata.ilowers);
         hypre_TFree(pdata.iuppers);
         hypre_TFree(pdata.boxsizes);
         pdata.max_boxsize = 0;
      }

      if (pdata.glue_nboxes == 0)
      {
         hypre_TFree(pdata.glue_ilowers);
         hypre_TFree(pdata.glue_iuppers);
         hypre_TFree(pdata.glue_nbor_parts);
         hypre_TFree(pdata.glue_nbor_ilowers);
         hypre_TFree(pdata.glue_nbor_iuppers);
         hypre_TFree(pdata.glue_index_maps);
      }

      if (pdata.graph_nentries == 0)
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

      if (pdata.matrix_nentries == 0)
      {
         hypre_TFree(pdata.matrix_ilowers);
         hypre_TFree(pdata.matrix_iuppers);
         hypre_TFree(pdata.matrix_strides);
         hypre_TFree(pdata.matrix_vars);
         hypre_TFree(pdata.matrix_entries);
         hypre_TFree(pdata.matrix_values);
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
   HYPRE_Int              part, s;

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
         hypre_TFree(pdata.glue_ilowers);
         hypre_TFree(pdata.glue_iuppers);
         hypre_TFree(pdata.glue_nbor_parts);
         hypre_TFree(pdata.glue_nbor_ilowers);
         hypre_TFree(pdata.glue_nbor_iuppers);
         hypre_TFree(pdata.glue_index_maps);
      }

      if (pdata.nvars > 0)
      {
         hypre_TFree(pdata.stencil_num);
      }

      if (pdata.graph_nentries > 0)
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

      if (pdata.matrix_nentries > 0)
      {
         hypre_TFree(pdata.matrix_ilowers);
         hypre_TFree(pdata.matrix_iuppers);
         hypre_TFree(pdata.matrix_strides);
         hypre_TFree(pdata.matrix_vars);
         hypre_TFree(pdata.matrix_entries);
         hypre_TFree(pdata.matrix_values);
      }

   }
   hypre_TFree(data.pdata);

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

   if (data.symmetric_nentries > 0)
   {
      hypre_TFree(data.symmetric_parts);
      hypre_TFree(data.symmetric_vars);
      hypre_TFree(data.symmetric_to_vars);
      hypre_TFree(data.symmetric_booleans);
   }

   hypre_TFree(data.pools);

   return 0;
}

/*--------------------------------------------------------------------------
 * Routine to load cosine function
 *--------------------------------------------------------------------------*/

HYPRE_Int
SetCosineVector(   double  scale,
                   Index   ilower,
                   Index   iupper,
                   double *values)
{
   HYPRE_Int          i,j,k;
   HYPRE_Int          count = 0;

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
 * Print usage info
 *--------------------------------------------------------------------------*/

HYPRE_Int
PrintUsage( char *progname,
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
      hypre_printf("                         3 - SysPFMG\n");
      hypre_printf("                        10 - PCG with SMG split precond\n");
      hypre_printf("                        11 - PCG with PFMG split precond\n");
      hypre_printf("                        13 - PCG with SysPFMG precond\n");
      hypre_printf("                        18 - PCG with diagonal scaling\n");
      hypre_printf("                        19 - PCG\n");
      hypre_printf("                        20 - PCG with BoomerAMG precond\n");
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
      hypre_printf("                        120- PCG with hybrid precond\n");
      hypre_printf("                        200- Struct SMG (default)\n");
      hypre_printf("                        201- Struct PFMG\n");
      hypre_printf("                        202- Struct SparseMSG\n");
      hypre_printf("                        203- Struct PFMG constant coefficients\n");
      hypre_printf("                        204- Struct PFMG constant coefficients variable diagonal\n");
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
      hypre_printf(">>> The only implemented solvers are 10,11,18,19,20,22,40 and 200.<<<\n");
      hypre_printf("  -print             : print out the system\n");
      hypre_printf("  -rhsfromcosine     : solution is cosine function (default)\n");
      hypre_printf("  -rhsone            : rhs is vector with unit components\n");
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
      hypre_printf("  -skip <s>          : Struct- skip levels in PFMG (0 or 1)\n");
      hypre_printf("  -sym <s>           : Struct- symmetric storage (1) or not (0)\n");
      hypre_printf("  -jump <num>        : Struct- num levels to jump in SparseMSG\n");
      hypre_printf("  -solver_type <ID>  : Struct- solver type for Hybrid\n");
      hypre_printf("                        1 - PCG (default)\n");
      hypre_printf("                        2 - GMRES\n");
      hypre_printf("  -cf <cf>           : Struct- convergence factor for Hybrid\n");

      hypre_printf("\n");
   }

   return 0;
}

/*--------------------------------------------------------------------------
 * Test driver for semi-structured matrix interface
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
main( HYPRE_Int   argc,
      char *argv[] )
{
   char                 *infile;
   ProblemData           global_data;
   ProblemData           data;
   ProblemPartData       pdata;
   HYPRE_Int                   nparts;
   HYPRE_Int                  *parts;
   Index                *refine;
   Index                *distribute;
   Index                *block;
   HYPRE_Int                   solver_id, object_type;
   HYPRE_Int                   print_system;
   HYPRE_Int                   cosine, struct_cosine;
   double                scale;
   MPI_Comm              mpi_comm = hypre_MPI_COMM_WORLD;
                        
   bHYPRE_MPICommunicator bmpicomm;
   bHYPRE_SStructGrid     b_grid;
   bHYPRE_SStructStencil *b_stencils;
   bHYPRE_SStructGraph    b_graph;
   bHYPRE_Operator b_A_O;
   bHYPRE_SStructMatrix   b_A;
   bHYPRE_SStructVector   b_b;
   bHYPRE_SStructVector   b_x;
   bHYPRE_SStructParCSRMatrix   b_spA;
   bHYPRE_SStructParCSRVector   b_spb;
   bHYPRE_SStructParCSRVector   b_spx;
   bHYPRE_IJParCSRMatrix        b_pA;
   bHYPRE_IJParCSRVector        b_pb;
   bHYPRE_IJParCSRVector        b_px;
   bHYPRE_Vector   bV_x;
   bHYPRE_Vector   bV_b;
   bHYPRE_Solver          b_precond;

   bHYPRE_StructSMG       b_solver_SMG;
   bHYPRE_PCG             b_solver_PCG;
   bHYPRE_GMRES           b_solver_GMRES;
   bHYPRE_BoomerAMG       b_boomeramg;
   bHYPRE_ParaSails       b_parasails;
   bHYPRE_SStructSplit    solver_Split;
   bHYPRE_SStructDiagScale solver_DS;
   bHYPRE_IdentitySolver  solver_Id;

   bHYPRE_StructMatrix    b_sA;
   bHYPRE_StructVector    b_sb;
   bHYPRE_StructVector    b_sx;

   sidl_BaseInterface     b_BI;

   HYPRE_Int ierr = 0;

   Index                 ilower, iupper;
   Index                 index, to_index;
   double               *values;

   HYPRE_Int                   num_iterations;
   double                final_res_norm;
                         
   HYPRE_Int                   num_procs, myid;
   HYPRE_Int                   time_index;
                         
   HYPRE_Int                   n_pre, n_post;
   HYPRE_Int                   skip;
   HYPRE_Int                   sym;
   HYPRE_Int                   rap;
   HYPRE_Int                   relax;
   HYPRE_Int                   jump;
   HYPRE_Int                   solver_type;

   double                cf_tol;

   HYPRE_Int                   arg_index, part, box, var, entry, s, i, j, k;
   sidl_BaseInterface _ex1 = NULL;
   sidl_BaseInterface * _ex = &_ex1;
                        
   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   /* Initialize MPI */
   hypre_MPI_Init(&argc, &argv);

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);
   bmpicomm = bHYPRE_MPICommunicator_CreateC( (void *)(&mpi_comm), _ex );

   hypre_InitMemoryDebug(myid);

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

   skip  = 0;
   sym   = 1;
   rap   = 0;
   relax = 1;
   jump  = 0;
   solver_type = 1;
   cf_tol = 0.90;

   nparts = global_data.nparts;

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

   solver_id = 39;
   print_system = 0;
   cosine = 1;
   struct_cosine = 0;

   n_pre  = 1;
   n_post = 1;

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
      else if ( strcmp(argv[arg_index], "-rhsone") == 0 )
      {
         arg_index++;
         cosine = 0;
      }
      else if ( strcmp(argv[arg_index], "-rhsfromcosine") == 0 )
      {
         arg_index++;
         cosine = 1;
         struct_cosine = 1;
      }
      else if ( strcmp(argv[arg_index], "-print") == 0 )
      {
         arg_index++;
         print_system = 1;
      }
      else if ( strcmp(argv[arg_index], "-skip") == 0 )
      {
         arg_index++;
         skip = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-v") == 0 )
      {
         arg_index++;
         n_pre = atoi(argv[arg_index++]);
         n_post = atoi(argv[arg_index++]);
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
      else if ( strcmp(argv[arg_index], "-sym") == 0 )
      {
         arg_index++;
         sym = atoi(argv[arg_index++]);
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
      else if ( strcmp(argv[arg_index], "-cf") == 0 )
      {
         arg_index++;
         cf_tol = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-help") == 0 )
      {
         PrintUsage(argv[0], myid);
         bHYPRE_MPICommunicator_deleteRef( bmpicomm, _ex );
         hypre_MPI_Finalize();
         exit(1);
         break;
      }
      else
      {
         break;
      }
   }

   /*-----------------------------------------------------------
    * Print driver parameters TODO
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
   }

   /*-----------------------------------------------------------
    * Distribute data
    *-----------------------------------------------------------*/

   DistributeData(global_data, refine, distribute, block,
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
            hypre_printf("Warning: Invalid number of parts for Struct Solver. Part 0 taken. \n");
         }
      }

      if (pdata.nvars > 1)
      {
         if (!myid)
         {
            hypre_printf("Error: Invalid number of nvars for Struct Solver \n");
            hypre_printf( "Try a different file with your -in parameter.\n" );
         }
         exit(1);
      }
   }

   /*-----------------------------------------------------------
    * Synchronize so that timings make sense
    *-----------------------------------------------------------*/

   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);

   /*-----------------------------------------------------------
    * Set up the grid
    *-----------------------------------------------------------*/
   hypre_assert( ierr == 0 );

   time_index = hypre_InitializeTiming("SStruct Interface");
   hypre_BeginTiming(time_index);

   b_grid = bHYPRE_SStructGrid_Create( bmpicomm, data.ndim, data.nparts, _ex );

   for (part = 0; part < data.nparts; part++)
   {
      pdata = data.pdata[part];
      for (box = 0; box < pdata.nboxes; box++)
      {
         bHYPRE_SStructGrid_SetExtents( b_grid, part, pdata.ilowers[box],
                                        pdata.iuppers[box], data.ndim, _ex );
      }

      for ( var=0; var<pdata.nvars; ++var )
      {
         bHYPRE_SStructGrid_SetVariable( b_grid, part, var, pdata.nvars, pdata.vartypes[var], _ex );
      }

      /* GridAddVariabes */

      /* GridSet_NeighborBox */
      for (box = 0; box < pdata.glue_nboxes; box++)
      {
         bHYPRE_SStructGrid_SetNeighborBox(
            b_grid, part,
            pdata.glue_ilowers[box], pdata.glue_iuppers[box],
            pdata.glue_nbor_parts[box],
            pdata.glue_nbor_ilowers[box], pdata.glue_nbor_iuppers[box],
            pdata.glue_index_maps[box], data.ndim, _ex );
      }

      bHYPRE_SStructGrid_SetPeriodic( b_grid, part, pdata.periodic, data.ndim, _ex );
   }

   bHYPRE_SStructGrid_Assemble( b_grid, _ex );

   /*-----------------------------------------------------------
    * Set up the stencils
    *-----------------------------------------------------------*/

   b_stencils = hypre_CTAlloc( bHYPRE_SStructStencil, data.nstencils );
   for (s = 0; s < data.nstencils; s++)
   {
      b_stencils[s] = bHYPRE_SStructStencil_Create( data.ndim, data.stencil_sizes[s], _ex );

      for (i = 0; i < data.stencil_sizes[s]; i++)
      {
         bHYPRE_SStructStencil_SetEntry( b_stencils[s], i,
                                         data.stencil_offsets[s][i], data.ndim,
                                         data.stencil_vars[s][i], _ex );
      }
   }

   /*-----------------------------------------------------------
    * Set object type
    *-----------------------------------------------------------*/

   object_type = HYPRE_SSTRUCT;

   if ( ((solver_id >= 20) && (solver_id < 30)) ||
        ((solver_id >= 40) && (solver_id < 50)) ||
        ((solver_id >= 60) && (solver_id < 70)) ||
        (solver_id == 120))
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

   b_graph = bHYPRE_SStructGraph_Create( bmpicomm, b_grid, _ex );

   bHYPRE_SStructGraph_SetObjectType( b_graph, object_type, _ex );

   for (part = 0; part < data.nparts; part++)
   {
      pdata = data.pdata[part];

      /* set stencils */
      for (var = 0; var < pdata.nvars; var++)
      {
         bHYPRE_SStructGraph_SetStencil( b_graph, part, var,
                                         b_stencils[pdata.stencil_num[var]], _ex );
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
                  bHYPRE_SStructGraph_AddEntries
                     ( b_graph, part, index, 3, pdata.graph_vars[entry],
                       pdata.graph_to_parts[entry], to_index,
                       pdata.graph_to_vars[entry], _ex );
               }
            }
         }
      }
   }

   bHYPRE_SStructGraph_Assemble( b_graph, _ex );

   /*-----------------------------------------------------------
    * Set up the matrix
    *-----------------------------------------------------------*/
   hypre_assert( ierr == 0 );

   values = hypre_TAlloc(double, data.max_boxsize);

   if ( object_type == HYPRE_PARCSR )
   {
      b_spA = bHYPRE_SStructParCSRMatrix_Create( bmpicomm, b_graph, _ex );

      /* TODO HYPRE_SStructMatrixSetSymmetric(A, 1); */
      for (entry = 0; entry < data.symmetric_nentries; entry++)
      {
         bHYPRE_SStructParCSRMatrix_SetSymmetric( b_spA,
                                                  data.symmetric_parts[entry],
                                                  data.symmetric_vars[entry],
                                                  data.symmetric_to_vars[entry],
                                                  data.symmetric_booleans[entry], _ex );
      }
      bHYPRE_SStructParCSRMatrix_SetNSSymmetric( b_spA, data.ns_symmetric, _ex );
      /* this Initialize calls SetObjectType */
      bHYPRE_SStructParCSRMatrix_Initialize( b_spA, _ex );
   }

   else
   {
      b_A = bHYPRE_SStructMatrix_Create( bmpicomm, b_graph, _ex );

      /* TODO HYPRE_SStructMatrixSetSymmetric(A, 1); */
      for (entry = 0; entry < data.symmetric_nentries; entry++)
      {
         bHYPRE_SStructMatrix_SetSymmetric( b_A,
                                            data.symmetric_parts[entry],
                                            data.symmetric_vars[entry],
                                            data.symmetric_to_vars[entry],
                                            data.symmetric_booleans[entry], _ex );
      }
      bHYPRE_SStructMatrix_SetNSSymmetric( b_A, data.ns_symmetric, _ex );
       /* this Initialize can't call SetObjectType because object_type could be
          HYPRE_SSTRUCT or HYPRE_STRUCT */
      bHYPRE_SStructMatrix_SetObjectType(b_A, object_type, _ex );
      bHYPRE_SStructMatrix_Initialize( b_A, _ex );
   }

   for (part = 0; part < data.nparts; part++)
   {
      pdata = data.pdata[part];

      /* set stencil values */
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

               if ( object_type == HYPRE_PARCSR )
               {
               bHYPRE_SStructParCSRMatrix_SetBoxValues
                  ( b_spA, part, ilower, iupper, data.ndim, var,
                    1, &i, values, pdata.max_boxsize, _ex );
               }
               else
               {
                  bHYPRE_SStructMatrix_SetBoxValues
                     ( b_A, part, ilower, iupper, data.ndim, var,
                       1, &i, values, pdata.max_boxsize, _ex );
               }
            }
         }
      }

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
                  if ( object_type == HYPRE_PARCSR )
                  {
                     bHYPRE_SStructParCSRMatrix_SetValues
                        ( b_spA, part, index, 3, pdata.graph_vars[entry],
                          1, &(pdata.graph_entries[entry]),
                          &(pdata.graph_values[entry]), _ex );
                  }
                  else
                  {
                     bHYPRE_SStructMatrix_SetValues
                        ( b_A, part, index, 3, pdata.graph_vars[entry],
                          1, &(pdata.graph_entries[entry]),
                          &(pdata.graph_values[entry]), _ex );
                  }
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
                  if ( object_type == HYPRE_PARCSR )
                  {
                     bHYPRE_SStructParCSRMatrix_SetValues
                        ( b_spA, part, index, 3, pdata.matrix_vars[entry],
                          1, &(pdata.matrix_entries[entry]), &(pdata.matrix_values[entry]), _ex );
                  }
                  else
                  {
                     bHYPRE_SStructMatrix_SetValues
                        ( b_A, part, index, 3, pdata.matrix_vars[entry],
                          1, &(pdata.matrix_entries[entry]), &(pdata.matrix_values[entry]), _ex );
                  }
               }
            }
         }
      }
   }

   if ( object_type == HYPRE_PARCSR )
   {
      bHYPRE_SStructParCSRMatrix_Assemble( b_spA, _ex );

      bHYPRE_SStructParCSRMatrix_GetObject( b_spA, &b_BI, _ex );
      b_pA = bHYPRE_IJParCSRMatrix__cast( b_BI, _ex );
      sidl_BaseInterface_deleteRef( b_BI, _ex );
   }
   else if ( object_type == HYPRE_STRUCT )
   {
      bHYPRE_SStructMatrix_Assemble( b_A, _ex );

      bHYPRE_SStructMatrix_GetObject( b_A, &b_BI, _ex );
      b_sA = bHYPRE_StructMatrix__cast( b_BI, _ex );
      sidl_BaseInterface_deleteRef( b_BI, _ex );
   }
   else if ( object_type == HYPRE_SSTRUCT )
   {
      bHYPRE_SStructMatrix_Assemble( b_A, _ex );
   }

   /*-----------------------------------------------------------
    * Set up the linear system
    *-----------------------------------------------------------*/
   hypre_assert( ierr == 0 );

   /************* b ***************/

   if ( object_type == HYPRE_PARCSR )
   {
      b_spb = bHYPRE_SStructParCSRVector_Create( bmpicomm, b_grid, _ex );
      /* this Initialize calls SetObjectType */
      bHYPRE_SStructParCSRVector_Initialize( b_spb, _ex );
   }
   else
   {
      b_b = bHYPRE_SStructVector_Create( bmpicomm, b_grid, _ex );
       /* this Initialize can't call SetObjectType because object_type could be
          HYPRE_SSTRUCT or HYPRE_STRUCT */
      bHYPRE_SStructVector_SetObjectType(b_b, object_type, _ex );
      bHYPRE_SStructVector_Initialize( b_b, _ex );
   }


   for (j = 0; j < data.max_boxsize; j++)
   {
      values[j] = 1.0;
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

            if ( object_type == HYPRE_PARCSR )
            {
               bHYPRE_SStructParCSRVector_SetBoxValues
                  ( b_spb, part, ilower, iupper, data.ndim, var, values, pdata.max_boxsize, _ex );
            }
            else
            {
               bHYPRE_SStructVector_SetBoxValues
                  ( b_b, part, ilower, iupper, data.ndim, var, values, pdata.max_boxsize, _ex );
            }
         }
      }
   }


   if ( object_type == HYPRE_PARCSR )
   {
      bHYPRE_SStructParCSRVector_Assemble( b_spb, _ex );

      bHYPRE_SStructParCSRVector_GetObject( b_spb, &b_BI, _ex );
      b_pb = bHYPRE_IJParCSRVector__cast( b_BI, _ex );
      sidl_BaseInterface_deleteRef( b_BI, _ex );
   }
   else if ( object_type == HYPRE_STRUCT )
   {
      bHYPRE_SStructVector_Assemble( b_b, _ex );

      bHYPRE_SStructVector_GetObject( b_b, &b_BI, _ex );
      b_sb = bHYPRE_StructVector__cast( b_BI, _ex );
      sidl_BaseInterface_deleteRef( b_BI, _ex );
   }
   else if ( object_type == HYPRE_SSTRUCT )
   {
      bHYPRE_SStructVector_Assemble( b_b, _ex );
   }

   /************* x ***************/

   if ( object_type == HYPRE_PARCSR )
   {
      b_spx = bHYPRE_SStructParCSRVector_Create( bmpicomm, b_grid, _ex );
      /* this Initialize calls SetObjectType */
      bHYPRE_SStructParCSRVector_Initialize( b_spx, _ex );
   }
   else
   {
      b_x = bHYPRE_SStructVector_Create( bmpicomm, b_grid, _ex );
       /* this Initialize can't call SetObjectType because object_type could be
          HYPRE_SSTRUCT or HYPRE_STRUCT */
      bHYPRE_SStructVector_SetObjectType(b_x, object_type, _ex );
      bHYPRE_SStructVector_Initialize( b_x, _ex );
   }

   for (j = 0; j < data.max_boxsize; j++)
   {
      values[j] = 0.0;
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

            if ( object_type == HYPRE_PARCSR )
            {
               bHYPRE_SStructParCSRVector_SetBoxValues
                  ( b_spx, part, ilower, iupper, data.ndim, var,
                    values, pdata.max_boxsize, _ex );
            }
            else
            {
               bHYPRE_SStructVector_SetBoxValues
                  ( b_x, part, ilower, iupper, data.ndim, var,
                    values, pdata.max_boxsize, _ex );
            }
         }
      }
   }

   if ( object_type == HYPRE_PARCSR )
   {
      bHYPRE_SStructParCSRVector_Assemble( b_spx, _ex );

      bHYPRE_SStructParCSRVector_GetObject( b_spx, &b_BI, _ex );
      b_px = bHYPRE_IJParCSRVector__cast( b_BI, _ex );
      sidl_BaseInterface_deleteRef( b_BI, _ex );
   }
   else if ( object_type == HYPRE_STRUCT )
   {
      bHYPRE_SStructVector_Assemble( b_x, _ex );

      bHYPRE_SStructVector_GetObject( b_x, &b_BI, _ex );
      b_sx = bHYPRE_StructVector__cast( b_BI, _ex );
      sidl_BaseInterface_deleteRef( b_BI, _ex );
   }
   else if ( object_type == HYPRE_SSTRUCT )
   {
      bHYPRE_SStructVector_Assemble( b_x, _ex );
   }

   hypre_EndTiming(time_index);
   hypre_PrintTiming("SStruct Interface", hypre_MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();

   /*-----------------------------------------------------------
    * If requested, reset linear system so that it has
    * exact solution:
    *
    *   u(part,var,i,j,k) = (part+1)*(var+1)*cosine[(i+j+k)/10]
    * 
    *-----------------------------------------------------------*/

   if ( object_type == HYPRE_STRUCT )
   {
      cosine= struct_cosine;
   }

   hypre_assert( ierr == 0 );
   if (cosine)
   {
      if ( object_type == HYPRE_PARCSR )
      {
         for (part = 0; part < data.nparts; part++)
         {
            pdata = data.pdata[part];
            for (var = 0; var < pdata.nvars; var++)
            {
               scale = (part+1.0)*(var+1.0);
               for (box = 0; box < pdata.nboxes; box++)
               {
                  GetVariableBox(pdata.ilowers[box], pdata.iuppers[box], var,
                                 ilower, iupper);
                  SetCosineVector(scale, ilower, iupper, values);
                  bHYPRE_SStructParCSRVector_SetBoxValues(
                     b_spx, part, ilower, iupper, data.ndim, var, values, 1, _ex );
                  /* ... last arg should be size of values, 1 may work even if wrong */
               }
            }
         }

         bHYPRE_SStructParCSRVector_Assemble( b_spx, _ex );

         /* Apply A to cosine vector to yield righthand side, b=A*x */
         bV_b = bHYPRE_Vector__cast( b_spb, _ex );
         bV_x = bHYPRE_Vector__cast( b_spx, _ex );
         ierr += bHYPRE_SStructParCSRMatrix_Apply( b_spA, bV_x, &bV_b, _ex );
         /* Reset initial guess to zero, x=0 */
         ierr += bHYPRE_SStructParCSRVector_Clear( b_spx, _ex );
         bHYPRE_Vector_deleteRef( bV_b, _ex );
         bHYPRE_Vector_deleteRef( bV_x, _ex );
      }
      else
      {
         for (part = 0; part < data.nparts; part++)
         {
            pdata = data.pdata[part];
            for (var = 0; var < pdata.nvars; var++)
            {
               scale = (part+1.0)*(var+1.0);
               for (box = 0; box < pdata.nboxes; box++)
               {
                  GetVariableBox(pdata.ilowers[box], pdata.iuppers[box], var,
                                 ilower, iupper);
                  SetCosineVector(scale, ilower, iupper, values);
                  bHYPRE_SStructVector_SetBoxValues(
                     b_x, part, ilower, iupper, data.ndim, var, values, 1, _ex );
                  /* ... last arg should be size of values, 1 may work even if wrong */
               }
            }
         }
         bHYPRE_SStructVector_Assemble( b_x, _ex );

         /* Apply A to cosine vector to yield righthand side, b=A*x */
         bV_b = bHYPRE_Vector__cast( b_b, _ex );
         bV_x = bHYPRE_Vector__cast( b_x, _ex );
         ierr += bHYPRE_SStructMatrix_Apply( b_A, bV_x, &bV_b, _ex );
         /* Reset initial guess to zero, x=0 */
         ierr += bHYPRE_SStructVector_Clear( b_x, _ex );
         bHYPRE_Vector_deleteRef( bV_b, _ex );
         bHYPRE_Vector_deleteRef( bV_x, _ex );
      }
      hypre_assert( ierr == 0 );
   }

   /*-----------------------------------------------------------
    * Print out the system and initial guess
    *-----------------------------------------------------------*/

      if ( object_type == HYPRE_PARCSR )
      {
         bHYPRE_SStructParCSRMatrix_Print( b_spA, "sstruct_b.out.A", 0, _ex );
         bHYPRE_SStructParCSRVector_Print( b_spb, "sstruct_b.out.b", 0, _ex );
         bHYPRE_SStructParCSRVector_Print( b_spx, "sstruct_b.out.x0", 0, _ex );
      }
      else
      {
         bHYPRE_SStructMatrix_Print( b_A, "sstruct_b.out.A", 0, _ex );
         bHYPRE_SStructVector_Print( b_b, "sstruct_b.out.b", 0, _ex );
         bHYPRE_SStructVector_Print( b_x, "sstruct_b.out.x0", 0, _ex );
      }

   /*-----------------------------------------------------------
    * Debugging code
    *-----------------------------------------------------------*/

#if DO_THIS_LATER
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
               GetVariableBox(pdata.ilowers[box], pdata.iuppers[box], var,
                              ilower, iupper);
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
               GetVariableBox(pdata.ilowers[box], pdata.iuppers[box], var,
                              ilower, iupper);
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
#endif /* DEBUG */
#endif /* DO_THIS_LATER */

   hypre_TFree(values);

   /*-----------------------------------------------------------
    * Solve the system using SysPFMG
    *-----------------------------------------------------------*/

   if (solver_id == 3)
   {
      hypre_assert( "solver 3 not implemented"==0 );
#if DO_THIS_LATER
      time_index = hypre_InitializeTiming("SysPFMG Setup");
      hypre_BeginTiming(time_index);

      HYPRE_SStructSysPFMGCreate(hypre_MPI_COMM_WORLD, &solver);
      HYPRE_SStructSysPFMGSetMaxIter(solver, 100);
      HYPRE_SStructSysPFMGSetTol(solver, 1.0e-6);
      HYPRE_SStructSysPFMGSetRelChange(solver, 0);
      /* weighted Jacobi = 1; red-black GS = 2 */
      HYPRE_SStructSysPFMGSetRelaxType(solver, relax);
      HYPRE_SStructSysPFMGSetNumPreRelax(solver, n_pre);
      HYPRE_SStructSysPFMGSetNumPostRelax(solver, n_post);
      HYPRE_SStructSysPFMGSetSkipRelax(solver, skip);
      /*HYPRE_StructPFMGSetDxyz(solver, dxyz);*/
      HYPRE_SStructSysPFMGSetPrintLevel(solver, 1);
      HYPRE_SStructSysPFMGSetLogging(solver, 1);
      HYPRE_SStructSysPFMGSetup(solver, A, b, x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("SysPFMG Solve");
      hypre_BeginTiming(time_index);

      HYPRE_SStructSysPFMGSolve(solver, A, b, x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_SStructSysPFMGGetNumIterations(solver, &num_iterations);
      HYPRE_SStructSysPFMGGetFinalRelativeResidualNorm(
                                           solver, &final_res_norm);
      HYPRE_SStructSysPFMGDestroy(solver);
#endif /* DO_THIS_LATER */
   }

   /*-----------------------------------------------------------
    * Solve the system using PCG
    *-----------------------------------------------------------*/

   if ((solver_id >= 10) && (solver_id < 20))
   {
      time_index = hypre_InitializeTiming("PCG Setup");
      hypre_BeginTiming(time_index);

      b_A_O = bHYPRE_Operator__cast( b_A, _ex );
      b_solver_PCG = bHYPRE_PCG_Create( bmpicomm, b_A_O, _ex );
      ierr += bHYPRE_PCG_SetIntParameter( b_solver_PCG, "MaxIter", 100, _ex );
      ierr += bHYPRE_PCG_SetDoubleParameter( b_solver_PCG, "Tolerance", 1.0e-06, _ex );
      ierr += bHYPRE_PCG_SetIntParameter( b_solver_PCG, "TwoNorm", 1, _ex );
      ierr += bHYPRE_PCG_SetIntParameter( b_solver_PCG, "RelChange", 0, _ex );
      ierr += bHYPRE_PCG_SetIntParameter( b_solver_PCG, "PrintLevel", 1, _ex );
      hypre_assert( ierr==0 );

      bV_b = bHYPRE_Vector__cast( b_b, _ex );
      bV_x = bHYPRE_Vector__cast( b_x, _ex );

      if ((solver_id == 10) || (solver_id == 11))
      {
         /* use Split solver as preconditioner */
         solver_Split = bHYPRE_SStructSplit_Create( bmpicomm, b_A_O, _ex );
         ierr += bHYPRE_SStructSplit_SetIntParameter( solver_Split, "MaxIter", 1, _ex );
         ierr += bHYPRE_SStructSplit_SetDoubleParameter(
            solver_Split, "Tolerance", 0, _ex );
         ierr += bHYPRE_SStructSplit_SetIntParameter(
            solver_Split, "ZeroGuess", 1, _ex );
         if (solver_id == 10)
         {
            ierr += bHYPRE_SStructSplit_SetStringParameter(
               solver_Split, "StructSolver", "SMG", _ex );
         }
         else if (solver_id == 11)
         {
            ierr += bHYPRE_SStructSplit_SetStringParameter(
               solver_Split, "StructSolver", "PFMG", _ex );
         }
         b_precond = (bHYPRE_Solver) bHYPRE_SStructDiagScale__cast2
            ( solver_Split, "bHYPRE.Solver", _ex ); 
         ierr += bHYPRE_PCG_SetPreconditioner( b_solver_PCG, b_precond, _ex );
         hypre_assert( ierr==0 );
      }

      else if (solver_id == 13)
      {
         hypre_assert( "solver 13 not implemented"==0 );
#if DO_THIS_LATER
         /* use SysPFMG solver as preconditioner */
         HYPRE_SStructSysPFMGCreate(hypre_MPI_COMM_WORLD, &precond);
         HYPRE_SStructSysPFMGSetMaxIter(precond, 1);
         HYPRE_SStructSysPFMGSetTol(precond, 0.0);
         HYPRE_SStructSysPFMGSetZeroGuess(precond);
         /* weighted Jacobi = 1; red-black GS = 2 */
         HYPRE_SStructSysPFMGSetRelaxType(precond, relax);
         HYPRE_SStructSysPFMGSetNumPreRelax(precond, n_pre);
         HYPRE_SStructSysPFMGSetNumPostRelax(precond, n_post);
         HYPRE_SStructSysPFMGSetSkipRelax(precond, skip);
         /*HYPRE_StructPFMGSetDxyz(precond, dxyz);*/
         HYPRE_PCGSetPrecond( (HYPRE_Solver) solver,
                              (HYPRE_PtrToSolverFcn) HYPRE_SStructSysPFMGSolve,
                              (HYPRE_PtrToSolverFcn) HYPRE_SStructSysPFMGSetup,
                              (HYPRE_Solver) precond);

#endif /* DO_THIS_LATER */
      }
      else if (solver_id == 18)
      {
         /* use diagonal scaling as preconditioner */
         solver_DS = bHYPRE_SStructDiagScale_Create( bmpicomm, b_A_O, _ex );
         ierr += bHYPRE_SStructDiagScale_Setup( solver_DS, bV_b, bV_x, _ex );
         b_precond = (bHYPRE_Solver) bHYPRE_SStructDiagScale__cast2
            ( solver_DS, "bHYPRE.Solver", _ex ); 
         ierr += bHYPRE_PCG_SetPreconditioner( b_solver_PCG, b_precond, _ex );
         hypre_assert( ierr==0 );
      }
      else if (solver_id == 19)
      {
         /* no preconditioner */
         solver_Id = bHYPRE_IdentitySolver_Create( bmpicomm, _ex );
         ierr += bHYPRE_IdentitySolver_Setup( solver_Id, bV_b, bV_x, _ex );
         b_precond = (bHYPRE_Solver) bHYPRE_SStructDiagScale__cast2
            ( solver_Id, "bHYPRE.Solver", _ex ); 
         ierr += bHYPRE_PCG_SetPreconditioner( b_solver_PCG, b_precond, _ex );
         hypre_assert( ierr==0 );
      }
      else
         hypre_assert( "unknown solver"==0 );

      ierr += bHYPRE_PCG_Setup( b_solver_PCG, bV_b, bV_x, _ex );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      time_index = hypre_InitializeTiming("PCG Solve");
      hypre_BeginTiming(time_index);

      ierr += bHYPRE_PCG_Apply( b_solver_PCG, bV_b, &bV_x, _ex );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      ierr += bHYPRE_PCG_GetIntValue( b_solver_PCG, "NumIterations", &num_iterations, _ex );
      ierr += bHYPRE_PCG_GetDoubleValue( b_solver_PCG, "RelResidualNorm", &final_res_norm, _ex );

      bHYPRE_Operator_deleteRef( b_A_O, _ex );
      bHYPRE_Solver_deleteRef( b_precond, _ex );
      bHYPRE_Vector_deleteRef( bV_b, _ex );
      bHYPRE_Vector_deleteRef( bV_x, _ex );
      if ((solver_id == 10) || (solver_id == 11))
      {
         bHYPRE_SStructSplit_deleteRef( solver_Split, _ex );
      }
      else if (solver_id == 13)
      {
         /*HYPRE_SStructSysPFMGDestroy(precond);*/
      }
      else if (solver_id == 18)
      {
         bHYPRE_SStructDiagScale_deleteRef( solver_DS, _ex );
      }
      else if (solver_id == 19)
      {
         bHYPRE_IdentitySolver_deleteRef( solver_Id, _ex );
      }

      bHYPRE_PCG_deleteRef( b_solver_PCG, _ex );

   }

   /*-----------------------------------------------------------
    * Solve the system using ParCSR version of PCG
    *-----------------------------------------------------------*/

   if ((solver_id >= 20) && (solver_id < 30))
   {
      time_index = hypre_InitializeTiming("PCG Setup");
      hypre_BeginTiming(time_index);

      b_A_O = bHYPRE_Operator__cast( b_pA, _ex );
      b_solver_PCG = bHYPRE_PCG_Create( bmpicomm, b_A_O, _ex );
      ierr += bHYPRE_PCG_SetIntParameter( b_solver_PCG, "MaxIter", 100, _ex );
      ierr += bHYPRE_PCG_SetIntParameter( b_solver_PCG, "TwoNorm", 1, _ex );
      ierr += bHYPRE_PCG_SetIntParameter( b_solver_PCG, "RelChange", 0, _ex );
      ierr += bHYPRE_PCG_SetDoubleParameter( b_solver_PCG, "Tol", 1.0e-6, _ex );
      ierr += bHYPRE_PCG_SetIntParameter( b_solver_PCG, "PrintLevel", 1, _ex );

      hypre_assert( ierr==0 );

      if (solver_id == 20)
      {
         /* use BoomerAMG as preconditioner */
         b_boomeramg = bHYPRE_BoomerAMG_Create( bmpicomm, b_pA, _ex );
         ierr += bHYPRE_BoomerAMG_SetIntParameter( b_boomeramg, "CoarsenType", 6, _ex );
         ierr += bHYPRE_BoomerAMG_SetIntParameter( b_boomeramg, "PrintLevel", 1, _ex );
         ierr += bHYPRE_BoomerAMG_SetDoubleParameter(
            b_boomeramg, "StrongThreshold", 0.25, _ex );
         ierr += bHYPRE_BoomerAMG_SetIntParameter( b_boomeramg, "MaxIter", 1, _ex );
         ierr += bHYPRE_BoomerAMG_SetDoubleParameter(
            b_boomeramg, "Tol", 0.0, _ex );
         ierr += bHYPRE_BoomerAMG_SetStringParameter(
            b_boomeramg, "PrintFileName", "sstruct.out.log", _ex );

         b_precond = (bHYPRE_Solver) bHYPRE_BoomerAMG__cast2(
            b_boomeramg, "bHYPRE.Solver", _ex );
         ierr += bHYPRE_PCG_SetPreconditioner( b_solver_PCG, b_precond, _ex );
         hypre_assert( ierr==0 );
      }
      else if (solver_id == 22)
      {
         /* use ParaSails as preconditioner */
         b_parasails = bHYPRE_ParaSails_Create( bmpicomm, b_pA, _ex );
         ierr += bHYPRE_ParaSails_SetDoubleParameter( b_parasails, "Thresh", 0.1, _ex );
         ierr += bHYPRE_ParaSails_SetIntParameter( b_parasails, "Nlevels", 1, _ex );

         b_precond = (bHYPRE_Solver) bHYPRE_ParaSails__cast2(
            b_parasails, "bHYPRE.Solver", _ex );
         ierr += bHYPRE_PCG_SetPreconditioner( b_solver_PCG, b_precond, _ex );
         hypre_assert( ierr==0 );
      }
      else if (solver_id == 28)
      {
         hypre_assert( "solver 28 not implemented"==0 );
#if DO_THIS_LATER
         /* use diagonal scaling as preconditioner */
         /* solver 18 does the same thing, but using SStructDiagScale */
         par_precond = NULL;
         HYPRE_PCGSetPrecond(  par_solver,
                              (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale,
                              (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScaleSetup,
                               par_precond );
#endif /* DO_THIS_LATER */
      }
      else
         hypre_assert( "solver not implemented"==0 );

      bV_b = bHYPRE_Vector__cast( b_pb, _ex );
      bV_x = bHYPRE_Vector__cast( b_px, _ex );
      ierr += bHYPRE_PCG_Setup( b_solver_PCG, bV_b, bV_x, _ex );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      time_index = hypre_InitializeTiming("PCG Solve");
      hypre_BeginTiming(time_index);

      ierr += bHYPRE_PCG_Apply( b_solver_PCG, bV_b, &bV_x, _ex );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();


      ierr += bHYPRE_PCG_GetIntValue(
         b_solver_PCG, "NumIterations", &num_iterations, _ex );
      ierr += bHYPRE_PCG_GetDoubleValue(
         b_solver_PCG, "RelResidualNorm", &final_res_norm, _ex );

      bHYPRE_Operator_deleteRef( b_A_O, _ex );
      bHYPRE_Solver_deleteRef( b_precond, _ex );
      bHYPRE_Vector_deleteRef( bV_b, _ex );
      bHYPRE_Vector_deleteRef( bV_x, _ex );
      if (solver_id == 20)
      {
         bHYPRE_BoomerAMG_deleteRef( b_boomeramg, _ex );
      }
      else if (solver_id == 22)
      {
         bHYPRE_ParaSails_deleteRef( b_parasails, _ex );
      }
      bHYPRE_PCG_deleteRef( b_solver_PCG, _ex );
      hypre_assert( ierr==0 );
   }

   /*-----------------------------------------------------------
    * Solve the system using GMRES  (struct matrix and vector type)
    *-----------------------------------------------------------*/

   if ((solver_id >= 30) && (solver_id < 40))  /* includes 39, plain GMRES, the default */
   {
      hypre_assert( "solvers 30-39 not implemented"==0 );
#if DO_THIS_LATER
      time_index = hypre_InitializeTiming("GMRES Setup");
      hypre_BeginTiming(time_index);

      HYPRE_SStructGMRESCreate(hypre_MPI_COMM_WORLD, &solver);
      HYPRE_GMRESSetKDim( (HYPRE_Solver) solver, 5 );
      HYPRE_GMRESSetMaxIter( (HYPRE_Solver) solver, 100 );
      HYPRE_GMRESSetTol( (HYPRE_Solver) solver, 1.0e-06 );
      HYPRE_GMRESSetPrintLevel( (HYPRE_Solver) solver, 1 );
      HYPRE_GMRESSetLogging( (HYPRE_Solver) solver, 1 );

      if ((solver_id == 30) || (solver_id == 31))
      {
         /* use Split solver as preconditioner */
         HYPRE_SStructSplitCreate(hypre_MPI_COMM_WORLD, &precond);
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
      hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      time_index = hypre_InitializeTiming("GMRES Solve");
      hypre_BeginTiming(time_index);

      HYPRE_GMRESSolve( (HYPRE_Solver) solver, (HYPRE_Matrix) A,
                        (HYPRE_Vector) b, (HYPRE_Vector) x );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_GMRESGetNumIterations( (HYPRE_Solver) solver, &num_iterations );
      HYPRE_GMRESGetFinalRelativeResidualNorm( (HYPRE_Solver) solver, &final_res_norm );
      HYPRE_SStructGMRESDestroy(solver);

      if ((solver_id == 30) || (solver_id == 31))
      {
         HYPRE_SStructSplitDestroy(precond);
      }
#endif /* DO_THIS_LATER */
   }

   /*-----------------------------------------------------------
    * Solve the system using ParCSR version of GMRES
    *-----------------------------------------------------------*/

   if ((solver_id >= 40) && (solver_id < 50))
   {
      time_index = hypre_InitializeTiming("GMRES Setup");
      hypre_BeginTiming(time_index);

      b_A_O = bHYPRE_Operator__cast( b_pA, _ex );
      b_solver_GMRES = bHYPRE_GMRES_Create( bmpicomm, b_A_O, _ex );
      bHYPRE_GMRES_SetIntParameter( b_solver_GMRES, "KDim", 5, _ex );
      bHYPRE_GMRES_SetIntParameter( b_solver_GMRES, "MaxIter", 100, _ex );
      bHYPRE_GMRES_SetDoubleParameter( b_solver_GMRES, "Tolerance", 1.0e-06, _ex );
      bHYPRE_GMRES_SetIntParameter( b_solver_GMRES, "PrintLevel", 1, _ex );
      bHYPRE_GMRES_SetIntParameter( b_solver_GMRES, "Logging", 1, _ex );

      hypre_assert( ierr==0 );

      if (solver_id == 40)
      {
         /* use BoomerAMG as preconditioner */
         b_boomeramg = bHYPRE_BoomerAMG_Create( bmpicomm, b_pA, _ex );
         bHYPRE_BoomerAMG_SetIntParameter( b_boomeramg, "CoarsenType", 6, _ex );
         bHYPRE_BoomerAMG_SetIntParameter( b_boomeramg, "StrongThreshold", 0.25, _ex );
         bHYPRE_BoomerAMG_SetDoubleParameter( b_boomeramg, "Tolerance", 0.0, _ex );
         bHYPRE_BoomerAMG_SetIntParameter( b_boomeramg, "PrintLevel", 1, _ex );
         bHYPRE_BoomerAMG_SetStringParameter( b_boomeramg,
                                              "PrintFileName", "sstruct.out.log", _ex);
         bHYPRE_BoomerAMG_SetIntParameter( b_boomeramg, "MaxIterations", 1, _ex );

         b_precond = (bHYPRE_Solver) bHYPRE_BoomerAMG__cast2
            ( b_boomeramg, "bHYPRE.Solver", _ex ); 
         bHYPRE_GMRES_SetPreconditioner( b_solver_GMRES, b_precond, _ex );
      }
      else if (solver_id == 41)
      {
         hypre_assert( "solver 41 not implemented"==0 );
#if DO_THIS_LATER
         /* use EUCLID as preconditioner */
#endif /* DO_THIS_LATER */
      }

      else if (solver_id == 42)
      {
         hypre_assert( "solver 41 not implemented"==0 );
#if DO_THIS_LATER
         /* use ParaSails as preconditioner */
         HYPRE_ParCSRParaSailsCreate(hypre_MPI_COMM_WORLD, &par_precond ); 
	 HYPRE_ParCSRParaSailsSetParams(par_precond, 0.1, 1);
	 HYPRE_ParCSRParaSailsSetSym(par_precond, 0);
         HYPRE_GMRESSetPrecond( par_solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_ParCSRParaSailsSolve,
                                (HYPRE_PtrToSolverFcn) HYPRE_ParCSRParaSailsSetup,
                                par_precond);
#endif /* DO_THIS_LATER */
      }

      bV_b = bHYPRE_Vector__cast( b_pb, _ex );
      bV_x = bHYPRE_Vector__cast( b_px, _ex );
      ierr += bHYPRE_GMRES_Setup( b_solver_GMRES, bV_b, bV_x, _ex );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      time_index = hypre_InitializeTiming("GMRES Solve");
      hypre_BeginTiming(time_index);

      bHYPRE_GMRES_Apply( b_solver_GMRES, bV_b, &bV_x, _ex );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      ierr += bHYPRE_GMRES_GetIntValue( b_solver_GMRES, "NumIterations", &num_iterations, _ex );
      ierr += bHYPRE_GMRES_GetDoubleValue( b_solver_GMRES, "RelResidualNorm", &final_res_norm, _ex );

      bHYPRE_Solver_deleteRef( b_precond, _ex );
      bHYPRE_Operator_deleteRef( b_A_O, _ex );
      bHYPRE_Vector_deleteRef( bV_b, _ex );
      bHYPRE_Vector_deleteRef( bV_x, _ex );
      if (solver_id == 40)
      {
         bHYPRE_BoomerAMG_deleteRef( b_boomeramg, _ex );
      }
#if DO_THIS_LATER
      else if (solver_id == 41)
      {
         HYPRE_ParCSREuclidDestroy(par_precond);
      }
      else if (solver_id == 42)
      {
         HYPRE_ParCSRParaSailsDestroy(par_precond);
      }
#endif /* DO_THIS_LATER */

      bHYPRE_GMRES_deleteRef( b_solver_GMRES, _ex );
   }

   /*-----------------------------------------------------------
    * Solve the system using BiCGSTAB
    *-----------------------------------------------------------*/

   if ((solver_id >= 50) && (solver_id < 60))
   {
      hypre_assert( "solvers 50-59 not implemented"==0 );
#if DO_THIS_LATER
      time_index = hypre_InitializeTiming("BiCGSTAB Setup");
      hypre_BeginTiming(time_index);

      HYPRE_SStructBiCGSTABCreate(hypre_MPI_COMM_WORLD, &solver);
      HYPRE_BiCGSTABSetMaxIter( (HYPRE_Solver) solver, 100 );
      HYPRE_BiCGSTABSetTol( (HYPRE_Solver) solver, 1.0e-06 );
      HYPRE_BiCGSTABSetPrintLevel( (HYPRE_Solver) solver, 1 );
      HYPRE_BiCGSTABSetLogging( (HYPRE_Solver) solver, 1 );

      if ((solver_id == 50) || (solver_id == 51))
      {
         /* use Split solver as preconditioner */
         HYPRE_SStructSplitCreate(hypre_MPI_COMM_WORLD, &precond);
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
      hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      time_index = hypre_InitializeTiming("BiCGSTAB Solve");
      hypre_BeginTiming(time_index);

      HYPRE_BiCGSTABSolve( (HYPRE_Solver) solver, (HYPRE_Matrix) A,
                        (HYPRE_Vector) b, (HYPRE_Vector) x );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_BiCGSTABGetNumIterations( (HYPRE_Solver) solver, &num_iterations );
      HYPRE_BiCGSTABGetFinalRelativeResidualNorm( (HYPRE_Solver) solver, &final_res_norm );
      HYPRE_SStructBiCGSTABDestroy(solver);

      if ((solver_id == 50) || (solver_id == 51))
      {
         HYPRE_SStructSplitDestroy(precond);
      }
#endif /* DO_THIS_LATER */
   }

   /*-----------------------------------------------------------
    * Solve the system using ParCSR version of BiCGSTAB
    *-----------------------------------------------------------*/

   if ((solver_id >= 60) && (solver_id < 70))
   {
      hypre_assert( "solvers 60-69 not implemented"==0 );
#if DO_THIS_LATER
      time_index = hypre_InitializeTiming("BiCGSTAB Setup");
      hypre_BeginTiming(time_index);

      HYPRE_ParCSRBiCGSTABCreate(hypre_MPI_COMM_WORLD, &par_solver);
      HYPRE_BiCGSTABSetMaxIter(par_solver, 100);
      HYPRE_BiCGSTABSetTol(par_solver, 1.0e-06);
      HYPRE_BiCGSTABSetPrintLevel(par_solver, 1);
      HYPRE_BiCGSTABSetLogging(par_solver, 1);

      if (solver_id == 60)
      {
         /* use BoomerAMG as preconditioner */
         HYPRE_BoomerAMGCreate(&par_precond); 
         HYPRE_BoomerAMGSetCoarsenType(par_precond, 6);
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
         /* use EUCLID as preconditioner */
      }

      else if (solver_id == 62)
      {
         /* use ParaSails as preconditioner */
         HYPRE_ParCSRParaSailsCreate(hypre_MPI_COMM_WORLD, &par_precond ); 
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
      hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      time_index = hypre_InitializeTiming("BiCGSTAB Solve");
      hypre_BeginTiming(time_index);

      HYPRE_BiCGSTABSolve( par_solver, (HYPRE_Matrix) par_A,
                        (HYPRE_Vector) par_b, (HYPRE_Vector) par_x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_BiCGSTABGetNumIterations( par_solver, &num_iterations);
      HYPRE_BiCGSTABGetFinalRelativeResidualNorm( par_solver,
                                               &final_res_norm);
      HYPRE_ParCSRBiCGSTABDestroy(par_solver);

      if (solver_id == 60)
      {
         HYPRE_BoomerAMGDestroy(par_precond);
      }
      else if (solver_id == 61)
      {
         HYPRE_ParCSREuclidDestroy(par_precond);
      }
      else if (solver_id == 62)
      {
         HYPRE_ParCSRParaSailsDestroy(par_precond);
      }
#endif /* DO_THIS_LATER */
   }

   /*-----------------------------------------------------------
    * Solve the system using ParCSR hybrid DSCG/BoomerAMG
    *-----------------------------------------------------------*/

   if (solver_id == 120) 
   {
      hypre_assert( "solver 120 not implemented"==0 );
#if DO_THIS_LATER
      time_index = hypre_InitializeTiming("Hybrid Setup");
      hypre_BeginTiming(time_index);

      HYPRE_ParCSRHybridCreate(&par_solver);
      HYPRE_ParCSRHybridSetTol(par_solver, 1.0e-06);
      HYPRE_ParCSRHybridSetTwoNorm(par_solver, 1);
      HYPRE_ParCSRHybridSetRelChange(par_solver, 0);
      HYPRE_ParCSRHybridSetPrintLevel(par_solver,1);
      HYPRE_ParCSRHybridSetLogging(par_solver,1);
      HYPRE_ParCSRHybridSetup(par_solver,par_A,par_b,par_x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      time_index = hypre_InitializeTiming("Hybrid Solve");
      hypre_BeginTiming(time_index);

      HYPRE_ParCSRHybridSolve(par_solver,par_A,par_b,par_x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index); 
      hypre_ClearTiming();

      HYPRE_ParCSRHybridGetNumIterations(par_solver, &num_iterations);
      HYPRE_ParCSRHybridGetFinalRelativeResidualNorm(
                                           par_solver, &final_res_norm);

      HYPRE_ParCSRHybridDestroy(par_solver);
#endif /* DO_THIS_LATER */
   }

   /*-----------------------------------------------------------
    * Solve the system using Struct solvers
    *-----------------------------------------------------------*/

   if (solver_id == 200)  /* struct SMG */
   {
      time_index = hypre_InitializeTiming("SMG Setup");
      hypre_BeginTiming(time_index);

      b_solver_SMG = bHYPRE_StructSMG_Create( bmpicomm, b_sA, _ex );
      bHYPRE_StructSMG_SetIntParameter( b_solver_SMG, "MemoryUse", 0, _ex );
      bHYPRE_StructSMG_SetIntParameter( b_solver_SMG, "MaxIter", 50, _ex );
      bHYPRE_StructSMG_SetDoubleParameter( b_solver_SMG, "Tol", 1.0e-6, _ex );
      bHYPRE_StructSMG_SetIntParameter( b_solver_SMG, "RelChange", 0, _ex );
      bHYPRE_StructSMG_SetIntParameter( b_solver_SMG, "NumPreRelax", n_pre, _ex );
      bHYPRE_StructSMG_SetIntParameter( b_solver_SMG, "NumPostRelax", n_post, _ex );
      bHYPRE_StructSMG_SetIntParameter( b_solver_SMG, "PrintLevel", 1, _ex );
      bHYPRE_StructSMG_SetIntParameter( b_solver_SMG, "Logging", 1, _ex );

      bV_b = bHYPRE_Vector__cast( b_sb, _ex );
      bV_x = bHYPRE_Vector__cast( b_sx, _ex );
      ierr += bHYPRE_StructSMG_Setup( b_solver_SMG, bV_b, bV_x, _ex );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("SMG Solve");
      hypre_BeginTiming(time_index);

      bHYPRE_StructSMG_Apply( b_solver_SMG, bV_b, &bV_x, _ex );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      ierr += bHYPRE_StructSMG_GetIntValue( b_solver_SMG, "NumIterations", &num_iterations, _ex );
      ierr += bHYPRE_StructSMG_GetDoubleValue( b_solver_SMG, "RelResidualNorm", &final_res_norm, _ex );

      bHYPRE_StructSMG_deleteRef( b_solver_SMG, _ex );
      bHYPRE_Vector_deleteRef( bV_b, _ex );
      bHYPRE_Vector_deleteRef( bV_x, _ex );
   }

   else if ( solver_id == 201 || solver_id == 203 || solver_id == 204 )
   {
      hypre_assert( "solvers 201,203,204 not implemented"==0 );
#if DO_THIS_LATER
      time_index = hypre_InitializeTiming("PFMG Setup");
      hypre_BeginTiming(time_index);

      HYPRE_StructPFMGCreate(hypre_MPI_COMM_WORLD, &struct_solver);
      HYPRE_StructPFMGSetMaxIter(struct_solver, 50);
      HYPRE_StructPFMGSetTol(struct_solver, 1.0e-06);
      HYPRE_StructPFMGSetRelChange(struct_solver, 0);
      HYPRE_StructPFMGSetRAPType(struct_solver, rap);
      HYPRE_StructPFMGSetRelaxType(struct_solver, relax);
      HYPRE_StructPFMGSetNumPreRelax(struct_solver, n_pre);
      HYPRE_StructPFMGSetNumPostRelax(struct_solver, n_post);
      HYPRE_StructPFMGSetSkipRelax(struct_solver, skip);
      /*HYPRE_StructPFMGSetDxyz(struct_solver, dxyz);*/
      HYPRE_StructPFMGSetPrintLevel(struct_solver, 1);
      HYPRE_StructPFMGSetLogging(struct_solver, 1);
      HYPRE_StructPFMGSetup(struct_solver, sA, sb, sx);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("PFMG Solve");
      hypre_BeginTiming(time_index);

      HYPRE_StructPFMGSolve(struct_solver, sA, sb, sx);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_StructPFMGGetNumIterations(struct_solver, &num_iterations);
      HYPRE_StructPFMGGetFinalRelativeResidualNorm(struct_solver, &final_res_norm);
      HYPRE_StructPFMGDestroy(struct_solver);
#endif /* DO_THIS_LATER */
   }

   /*-----------------------------------------------------------
    * Solve the system using SparseMSG
    *-----------------------------------------------------------*/

   else if (solver_id == 202)
   {
      hypre_assert( "solver 202 not implemented"==0 );
#if DO_THIS_LATER
      time_index = hypre_InitializeTiming("SparseMSG Setup");
      hypre_BeginTiming(time_index);

      HYPRE_StructSparseMSGCreate(hypre_MPI_COMM_WORLD, &struct_solver);
      HYPRE_StructSparseMSGSetMaxIter(struct_solver, 50);
      HYPRE_StructSparseMSGSetJump(struct_solver, jump);
      HYPRE_StructSparseMSGSetTol(struct_solver, 1.0e-06);
      HYPRE_StructSparseMSGSetRelChange(struct_solver, 0);
      HYPRE_StructSparseMSGSetRelaxType(struct_solver, relax);
      HYPRE_StructSparseMSGSetNumPreRelax(struct_solver, n_pre);
      HYPRE_StructSparseMSGSetNumPostRelax(struct_solver, n_post);
      HYPRE_StructSparseMSGSetPrintLevel(struct_solver, 1);
      HYPRE_StructSparseMSGSetLogging(struct_solver, 1);
      HYPRE_StructSparseMSGSetup(struct_solver, sA, sb, sx);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("SparseMSG Solve");
      hypre_BeginTiming(time_index);

      HYPRE_StructSparseMSGSolve(struct_solver, sA, sb, sx);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_StructSparseMSGGetNumIterations(struct_solver, &num_iterations);
      HYPRE_StructSparseMSGGetFinalRelativeResidualNorm(struct_solver,
                                                        &final_res_norm);
      HYPRE_StructSparseMSGDestroy(struct_solver);
#endif /* DO_THIS_LATER */
   }

   /*-----------------------------------------------------------
    * Solve the system using CG
    *-----------------------------------------------------------*/

   if ((solver_id > 209) && (solver_id < 220))
   {
      hypre_assert( "solvers 210-219 not implemented"==0 );
#if DO_THIS_LATER
      time_index = hypre_InitializeTiming("PCG Setup");
      hypre_BeginTiming(time_index);

      HYPRE_StructPCGCreate(hypre_MPI_COMM_WORLD, &struct_solver);
      HYPRE_PCGSetMaxIter( (HYPRE_Solver)struct_solver, 50 );
      HYPRE_PCGSetTol( (HYPRE_Solver)struct_solver, 1.0e-06 );
      HYPRE_PCGSetTwoNorm( (HYPRE_Solver)struct_solver, 1 );
      HYPRE_PCGSetRelChange( (HYPRE_Solver)struct_solver, 0 );
      HYPRE_PCGSetPrintLevel( (HYPRE_Solver)struct_solver, 1 );

      if (solver_id == 210)
      {
         /* use symmetric SMG as preconditioner */
         HYPRE_StructSMGCreate(hypre_MPI_COMM_WORLD, &struct_precond);
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
         HYPRE_StructPFMGCreate(hypre_MPI_COMM_WORLD, &struct_precond);
         HYPRE_StructPFMGSetMaxIter(struct_precond, 1);
         HYPRE_StructPFMGSetTol(struct_precond, 0.0);
         HYPRE_StructPFMGSetZeroGuess(struct_precond);
         HYPRE_StructPFMGSetRAPType(struct_precond, rap);
         HYPRE_StructPFMGSetRelaxType(struct_precond, relax);
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
         HYPRE_StructSparseMSGCreate(hypre_MPI_COMM_WORLD, &struct_precond);
         HYPRE_StructSparseMSGSetMaxIter(struct_precond, 1);
         HYPRE_StructSparseMSGSetJump(struct_precond, jump);
         HYPRE_StructSparseMSGSetTol(struct_precond, 0.0);
         HYPRE_StructSparseMSGSetZeroGuess(struct_precond);
         HYPRE_StructSparseMSGSetRelaxType(struct_precond, relax);
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
         HYPRE_StructJacobiCreate(hypre_MPI_COMM_WORLD, &struct_precond);
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
#ifdef HYPRE_USE_PTHREADS
         for (i = 0; i < hypre_NumThreads; i++)
         {
            struct_precond[i] = NULL;
         }
#else
         struct_precond = NULL;
#endif /* HYPRE_USE_PTHREADS */
         HYPRE_PCGSetPrecond( (HYPRE_Solver) struct_solver,
                              (HYPRE_PtrToSolverFcn) HYPRE_StructDiagScale,
                              (HYPRE_PtrToSolverFcn) HYPRE_StructDiagScaleSetup,
                              (HYPRE_Solver) struct_precond);
      }

      HYPRE_PCGSetup
         ( (HYPRE_Solver)struct_solver, (HYPRE_Matrix)sA, (HYPRE_Vector)sb, 
                                                          (HYPRE_Vector)sx );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("PCG Solve");
      hypre_BeginTiming(time_index);

      HYPRE_PCGSolve
         ( (HYPRE_Solver) struct_solver, (HYPRE_Matrix)sA, (HYPRE_Vector)sb, 
                                                           (HYPRE_Vector)sx);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
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
#endif /* DO_THIS_LATER */
   }

   /*-----------------------------------------------------------
    * Solve the system using Hybrid
    *-----------------------------------------------------------*/

   if ((solver_id > 219) && (solver_id < 230))
   {
      hypre_assert( "solvers 220-229 not implemented"==0 );
#if DO_THIS_LATER
      time_index = hypre_InitializeTiming("Hybrid Setup");
      hypre_BeginTiming(time_index);

      HYPRE_StructHybridCreate(hypre_MPI_COMM_WORLD, &struct_solver);
      HYPRE_StructHybridSetDSCGMaxIter(struct_solver, 100);
      HYPRE_StructHybridSetPCGMaxIter(struct_solver, 50);
      HYPRE_StructHybridSetTol(struct_solver, 1.0e-06);
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

      if (solver_id == 220)
      {
         /* use symmetric SMG as preconditioner */
         HYPRE_StructSMGCreate(hypre_MPI_COMM_WORLD, &struct_precond);
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
         HYPRE_StructPFMGCreate(hypre_MPI_COMM_WORLD, &struct_precond);
         HYPRE_StructPFMGSetMaxIter(struct_precond, 1);
         HYPRE_StructPFMGSetTol(struct_precond, 0.0);
         HYPRE_StructPFMGSetZeroGuess(struct_precond);
         HYPRE_StructPFMGSetRAPType(struct_precond, rap);
         HYPRE_StructPFMGSetRelaxType(struct_precond, relax);
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
         HYPRE_StructSparseMSGCreate(hypre_MPI_COMM_WORLD, &struct_precond);
         HYPRE_StructSparseMSGSetJump(struct_precond, jump);
         HYPRE_StructSparseMSGSetMaxIter(struct_precond, 1);
         HYPRE_StructSparseMSGSetTol(struct_precond, 0.0);
         HYPRE_StructSparseMSGSetZeroGuess(struct_precond);
         HYPRE_StructSparseMSGSetRelaxType(struct_precond, relax);
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
      hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("Hybrid Solve");
      hypre_BeginTiming(time_index);

      HYPRE_StructHybridSolve(struct_solver, sA, sb, sx);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
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
#endif /* DO_THIS_LATER */
   }

   /*-----------------------------------------------------------
    * Solve the system using GMRES
    *-----------------------------------------------------------*/

   if ((solver_id > 229) && (solver_id < 240))
   {
      hypre_assert( "solvers 230-239 not implemented"==0 );
#if DO_THIS_LATER
      time_index = hypre_InitializeTiming("GMRES Setup");
      hypre_BeginTiming(time_index);

      HYPRE_StructGMRESCreate(hypre_MPI_COMM_WORLD, &struct_solver);
      HYPRE_GMRESSetMaxIter( (HYPRE_Solver)struct_solver, 50 );
      HYPRE_GMRESSetTol( (HYPRE_Solver)struct_solver, 1.0e-06 );
      HYPRE_GMRESSetRelChange( (HYPRE_Solver)struct_solver, 0 );
      HYPRE_GMRESSetPrintLevel( (HYPRE_Solver)struct_solver, 1 );
      HYPRE_GMRESSetLogging( (HYPRE_Solver)struct_solver, 1 );

      if (solver_id == 230)
      {
         /* use symmetric SMG as preconditioner */
         HYPRE_StructSMGCreate(hypre_MPI_COMM_WORLD, &struct_precond);
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
         HYPRE_StructPFMGCreate(hypre_MPI_COMM_WORLD, &struct_precond);
         HYPRE_StructPFMGSetMaxIter(struct_precond, 1);
         HYPRE_StructPFMGSetTol(struct_precond, 0.0);
         HYPRE_StructPFMGSetZeroGuess(struct_precond);
         HYPRE_StructPFMGSetRAPType(struct_precond, rap);
         HYPRE_StructPFMGSetRelaxType(struct_precond, relax);
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
         HYPRE_StructSparseMSGCreate(hypre_MPI_COMM_WORLD, &struct_precond);
         HYPRE_StructSparseMSGSetMaxIter(struct_precond, 1);
         HYPRE_StructSparseMSGSetJump(struct_precond, jump);
         HYPRE_StructSparseMSGSetTol(struct_precond, 0.0);
         HYPRE_StructSparseMSGSetZeroGuess(struct_precond);
         HYPRE_StructSparseMSGSetRelaxType(struct_precond, relax);
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
         HYPRE_StructJacobiCreate(hypre_MPI_COMM_WORLD, &struct_precond);
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
#ifdef HYPRE_USE_PTHREADS
         for (i = 0; i < hypre_NumThreads; i++)
         {
            struct_precond[i] = NULL;
         }
#else
         struct_precond = NULL;
#endif /* HYPRE_USE_PTHREADS */
         HYPRE_GMRESSetPrecond( (HYPRE_Solver)struct_solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_StructDiagScale,
                                (HYPRE_PtrToSolverFcn) HYPRE_StructDiagScaleSetup,
                                (HYPRE_Solver)struct_precond);
      }

      HYPRE_GMRESSetup
         ( (HYPRE_Solver)struct_solver, (HYPRE_Matrix)sA, (HYPRE_Vector)sb, 
                                                          (HYPRE_Vector)sx );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("GMRES Solve");
      hypre_BeginTiming(time_index);

      HYPRE_GMRESSolve
         ( (HYPRE_Solver)struct_solver, (HYPRE_Matrix)sA, (HYPRE_Vector)sb, 
                                                          (HYPRE_Vector)sx);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
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
#endif /* DO_THIS_LATER */
   }

   /*-----------------------------------------------------------
    * Solve the system using BiCGTAB
    *-----------------------------------------------------------*/

   if ((solver_id > 239) && (solver_id < 250))
   {
      hypre_assert( "solvers 240-249 not implemented"==0 );
#if DO_THIS_LATER
      time_index = hypre_InitializeTiming("BiCGSTAB Setup");
      hypre_BeginTiming(time_index);

      HYPRE_StructBiCGSTABCreate(hypre_MPI_COMM_WORLD, &struct_solver);
      HYPRE_BiCGSTABSetMaxIter( (HYPRE_Solver)struct_solver, 50 );
      HYPRE_BiCGSTABSetTol( (HYPRE_Solver)struct_solver, 1.0e-06 );
      HYPRE_BiCGSTABSetPrintLevel( (HYPRE_Solver)struct_solver, 1 );
      HYPRE_BiCGSTABSetLogging( (HYPRE_Solver)struct_solver, 1 );

      if (solver_id == 240)
      {
         /* use symmetric SMG as preconditioner */
         HYPRE_StructSMGCreate(hypre_MPI_COMM_WORLD, &struct_precond);
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
         HYPRE_StructPFMGCreate(hypre_MPI_COMM_WORLD, &struct_precond);
         HYPRE_StructPFMGSetMaxIter(struct_precond, 1);
         HYPRE_StructPFMGSetTol(struct_precond, 0.0);
         HYPRE_StructPFMGSetZeroGuess(struct_precond);
         HYPRE_StructPFMGSetRAPType(struct_precond, rap);
         HYPRE_StructPFMGSetRelaxType(struct_precond, relax);
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
         HYPRE_StructSparseMSGCreate(hypre_MPI_COMM_WORLD, &struct_precond);
         HYPRE_StructSparseMSGSetMaxIter(struct_precond, 1);
         HYPRE_StructSparseMSGSetJump(struct_precond, jump);
         HYPRE_StructSparseMSGSetTol(struct_precond, 0.0);
         HYPRE_StructSparseMSGSetZeroGuess(struct_precond);
         HYPRE_StructSparseMSGSetRelaxType(struct_precond, relax);
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
         HYPRE_StructJacobiCreate(hypre_MPI_COMM_WORLD, &struct_precond);
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
#ifdef HYPRE_USE_PTHREADS
         for (i = 0; i < hypre_NumThreads; i++)
         {
            struct_precond[i] = NULL;
         }
#else
         struct_precond = NULL;
#endif /* HYPRE_USE_PTHREADS */
         HYPRE_BiCGSTABSetPrecond( (HYPRE_Solver)struct_solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_StructDiagScale,
                                (HYPRE_PtrToSolverFcn) HYPRE_StructDiagScaleSetup,
                                (HYPRE_Solver)struct_precond);
      }

      HYPRE_BiCGSTABSetup
         ( (HYPRE_Solver)struct_solver, (HYPRE_Matrix)sA, (HYPRE_Vector)sb, 
                                                          (HYPRE_Vector)sx );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("BiCGSTAB Solve");
      hypre_BeginTiming(time_index);

      HYPRE_BiCGSTABSolve
         ( (HYPRE_Solver)struct_solver, (HYPRE_Matrix)sA, (HYPRE_Vector)sb,
                                                          (HYPRE_Vector)sx);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
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
#endif /* DO_THIS_LATER */
   }

   /*-----------------------------------------------------------
    * Gather the solution vector
    *-----------------------------------------------------------*/

   if ( object_type == HYPRE_PARCSR )
   {
      bHYPRE_SStructParCSRVector_Gather( b_spx, _ex );
   }
   else
   {
      bHYPRE_SStructVector_Gather( b_x, _ex );
   }

   /*-----------------------------------------------------------
    * Print the solution and other info
    *-----------------------------------------------------------*/

   if (print_system)
   {
      if ( object_type == HYPRE_PARCSR )
      {
         bHYPRE_SStructParCSRVector_Print( b_spx, "sstruct_b.out.x", 0, _ex );
      }
      else
      {
         bHYPRE_SStructVector_Print( b_x, "sstruct_b.out.x", 0, _ex );
      }
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

   bHYPRE_SStructGraph_deleteRef( b_graph, _ex );
   bHYPRE_SStructGrid_deleteRef( b_grid, _ex );
   for (s = 0; s < data.nstencils; s++)
      bHYPRE_SStructStencil_deleteRef( b_stencils[s], _ex );
   hypre_TFree( b_stencils );

   if ( object_type == HYPRE_PARCSR )
   {
      bHYPRE_IJParCSRMatrix_deleteRef( b_pA, _ex );
      bHYPRE_IJParCSRVector_deleteRef( b_pb, _ex );
      bHYPRE_IJParCSRVector_deleteRef( b_px, _ex );
      bHYPRE_SStructParCSRMatrix_deleteRef( b_spA, _ex);
      bHYPRE_SStructParCSRVector_deleteRef( b_spb, _ex );
      bHYPRE_SStructParCSRVector_deleteRef( b_spx, _ex );
   }
   else if ( object_type == HYPRE_STRUCT )
   {
      bHYPRE_SStructMatrix_deleteRef( b_A, _ex );
      bHYPRE_SStructVector_deleteRef( b_b, _ex );
      bHYPRE_SStructVector_deleteRef( b_x, _ex );
      bHYPRE_StructMatrix_deleteRef( b_sA, _ex );
      bHYPRE_StructVector_deleteRef( b_sb, _ex );
      bHYPRE_StructVector_deleteRef( b_sx, _ex );
   }
   else if ( object_type == HYPRE_SSTRUCT )
   {
      bHYPRE_SStructMatrix_deleteRef( b_A, _ex );
      bHYPRE_SStructVector_deleteRef( b_b, _ex );
      bHYPRE_SStructVector_deleteRef( b_x, _ex );
   }

   DestroyData(data);

   hypre_TFree(parts);
   hypre_TFree(refine);
   hypre_TFree(distribute);
   hypre_TFree(block);

   hypre_FinalizeMemoryDebug();

   /* Finalize MPI */
   bHYPRE_MPICommunicator_deleteRef( bmpicomm, _ex );
   hypre_MPI_Finalize();

   return (0);

EXIT: return 1;
}
