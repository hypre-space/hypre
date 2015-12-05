/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.10 $
 ***********************************************************************EHEADER*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "_hypre_utilities.h"
#include "HYPRE_sstruct_ls.h"
#include "HYPRE_krylov.h"
#include "_hypre_sstruct_mv.h"
#include "_hypre_sstruct_ls.h"
 
#define DEBUG 0

#if DEBUG
#include "_hypre_sstruct_mv.h"
#endif

/*--------------------------------------------------------------------------
 * Data structures
 *--------------------------------------------------------------------------*/

char infile_default[50] = "sstruct_default.in";

typedef HYPRE_Int Index[3];
typedef HYPRE_Int ProblemIndex[9];  /* last 3 digits are shifts */

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
   HYPRE_SStructVariable *vartypes;

   /* for GridAddVariables */
   HYPRE_Int                    add_nvars;
   ProblemIndex          *add_indexes;
   HYPRE_SStructVariable *add_vartypes;

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

   HYPRE_Int                    matrix_nentries;
   ProblemIndex          *matrix_ilowers;
   ProblemIndex          *matrix_iuppers;
   Index                 *matrix_strides;
   HYPRE_Int                   *matrix_vars;
   HYPRE_Int                   *matrix_entries;
   double                *matrix_values;

   Index                  periodic;

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

   double               **graph_fine_to_coarse_data;
   double               **graph_coarse_to_fine_data;

   /* for amr structure */
   HYPRE_Int                    fac_plevel;
   Index                  fac_prefinement;

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

   HYPRE_Int              npools;
   HYPRE_Int             *pools;   /* array of size nparts */

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

   HYPRE_Int                part, var, entry, s, i;

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

      s= memchunk;
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
            if ( (pdata.ilowers[pdata.nboxes][3]*
                  pdata.ilowers[pdata.nboxes][4]*
                  pdata.ilowers[pdata.nboxes][5] != 0) ||
                 (pdata.iuppers[pdata.nboxes][3]*
                  pdata.iuppers[pdata.nboxes][4]*
                  pdata.iuppers[pdata.nboxes][5] != 1) )
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
            SScanIntArray(sdata_ptr, &sdata_ptr,
                          pdata.nvars, pdata.vartypes);
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
            pdata.glue_nboxes++;
            data.pdata[part] = pdata;
         }
         else if ( strcmp(key, "GridSetPeriodic:") == 0 )
         {
            part = strtol(sdata_ptr, &sdata_ptr, 10);
            pdata = data.pdata[part];
            SScanIntArray(sdata_ptr, &sdata_ptr, data.ndim, pdata.periodic);
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
       
         else if ( strcmp(key, "FacParts:") == 0 )
         {
            part = strtol(sdata_ptr, &sdata_ptr, 10);
            pdata = data.pdata[part];
            pdata.fac_plevel= strtol(sdata_ptr, &sdata_ptr, 10);
            SScanIntArray(sdata_ptr, &sdata_ptr, data.ndim, pdata.fac_prefinement);
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
   HYPRE_Int              pool, part, box, entry, p, q, r, i, d, dmap, sign, size;
   HYPRE_Int              mod_lower, mod_upper;
   Index            m, mmap, n, ones, mod_graph;
   ProblemIndex     ilower, iupper, int_ilower, int_iupper;

   ones[0]= 1;
   ones[1]= 1;
   ones[2]= 1;

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

               for (d= 0; d< 3; d++)
               {
                  mod_graph[d]= pdata.graph_ilowers[entry][d] % pdata.graph_strides[entry][d];
               }

               for (box = 0; box < pdata.nboxes; box++)
               {
                  GetVariableBox(pdata.ilowers[box], pdata.iuppers[box],
                                 pdata.vartypes[pdata.graph_vars[entry]],
                                 ilower, iupper);
                  size = IntersectBoxes(pdata.graph_ilowers[entry],
                                        pdata.graph_iuppers[entry],
                                        ilower, iupper,
                                        int_ilower, int_iupper);

                  /* adjust int_ilower & int_iupper by a mod shift to get the extents
                     correctly */
                  if (size > 0)
                  {
                     for (d= 0; d< 3; d++)
                     {
                        mod_lower= int_ilower[d] % pdata.graph_strides[entry][d];
                        if (mod_graph[d] >= mod_lower)
                        {
                           int_ilower[d]+= (mod_graph[d] - mod_lower);
                        }
                        else
                        {
                           int_ilower[d]+= pdata.graph_strides[entry][d] + (mod_graph[d] - mod_lower);
                        }
                        mod_lower= mod_graph[d];

                        mod_upper= int_iupper[d] % pdata.graph_strides[entry][d];
                        if (mod_lower >= mod_upper)
                        {
                           int_iupper[d]-= (mod_lower-mod_upper);
                        }
                        else
                        {
                           int_iupper[d]+= (mod_upper-mod_lower) - pdata.graph_strides[entry][d];
                        }

                        if (int_iupper[d] - int_ilower[d] < 0)
                        {
                           size= -1;
                           break;
                        }
                     }
                  }

                  if (size > 0)
                  {
                     /* if there is an intersection, it is the only one */
                     for (d = 0; d < 3; d++)
                     {
                        dmap = pdata.graph_index_maps[entry][d];
                        sign = pdata.graph_index_signs[entry][d];
                        size =(int_iupper[d] - pdata.graph_iuppers[entry][d])/
                               pdata.graph_strides[entry][d];
                        pdata.graph_to_iuppers[i][dmap] =
                           pdata.graph_to_ilowers[entry][dmap] + sign*
                           size*pdata.graph_to_strides[entry][d];
                        size= (int_ilower[d] - pdata.graph_ilowers[entry][d])/
                               pdata.graph_strides[entry][d];
                        pdata.graph_to_ilowers[i][dmap] =
                           pdata.graph_to_ilowers[entry][dmap] + sign *
                           size*pdata.graph_to_strides[entry][d];
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

      if (pdata.matrix_nentries == 0)
      {
         hypre_TFree(pdata.matrix_ilowers);
         hypre_TFree(pdata.matrix_iuppers);
         hypre_TFree(pdata.matrix_strides);
         hypre_TFree(pdata.matrix_vars);
         hypre_TFree(pdata.matrix_entries);
         hypre_TFree(pdata.matrix_values);
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

   hypre_TFree(data.pools);

   return 0;
}

/*--------------------------------------------------------------------------
 * Test driver for semi-structured matrix interface
 *--------------------------------------------------------------------------*/
 
hypre_int
main( hypre_int argc,
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
   HYPRE_Int                   solver_id;
   HYPRE_Int                   print_system;
                        
   HYPRE_SStructGrid     grid;
   HYPRE_SStructStencil *stencils;
   HYPRE_SStructGraph    graph;
   HYPRE_SStructMatrix   A, A_amg;
   HYPRE_StructMatrix    sA;
   HYPRE_SStructVector   b, b_amg;
   HYPRE_SStructVector   x, x_amg;
   HYPRE_StructVector    sx;
   HYPRE_SStructSolver   solver;

   HYPRE_ParCSRMatrix    par_A;
   HYPRE_ParVector       par_b;
   HYPRE_ParVector       par_x;
   HYPRE_Solver          par_solver;
   HYPRE_Solver          par_precond;

   Index                 ilower, iupper;
   Index                 index, to_index;
   double               *values, *box_values;

   HYPRE_Int                  *plevels;
   Index                *prefinements;

   HYPRE_Int                   num_iterations;
   double                final_res_norm;
                         
   HYPRE_Int                   num_procs, myid;
   HYPRE_Int                   time_index;
                         
   HYPRE_Int                   n_pre, n_post;
   HYPRE_Int                   skip;

   HYPRE_Int                   arg_index, part, box, var, entry, s, i, j, k;

                        
   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   /* Initialize MPI */
   hypre_MPI_Init(&argc, &argv);

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);

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

   skip = 0;

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
      else
      {
         break;
      }
   }

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

   stencils = hypre_CTAlloc(HYPRE_SStructStencil, data.nstencils);
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
    * Set up the amr structure
    *-----------------------------------------------------------*/
   plevels     = hypre_TAlloc(HYPRE_Int, data.nparts);
   prefinements= hypre_TAlloc(Index, data.nparts);
   for (part = 0; part < data.nparts; part++)
   {
      pdata = data.pdata[part];
      plevels[part]= pdata.fac_plevel;
      for (i= 0; i< data.ndim; i++)
      {
         prefinements[part][i]= pdata.fac_prefinement[i];
      }
   }

   n_pre  = prefinements[data.nparts-1][0]-1;
   n_post = prefinements[data.nparts-1][0]-1;
   /*-----------------------------------------------------------
    * Set up the graph
    *-----------------------------------------------------------*/

   HYPRE_SStructGraphCreate(hypre_MPI_COMM_WORLD, grid, &graph);
   if ( ((solver_id >= 20) && (solver_id <= 30)) ||
        ((solver_id >= 40) && (solver_id < 60)) )
   {
       HYPRE_SStructGraphSetObjectType(graph, HYPRE_PARCSR);
   }

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

   values = hypre_TAlloc(double, data.max_boxsize);

   HYPRE_SStructMatrixCreate(hypre_MPI_COMM_WORLD, graph, &A);
   HYPRE_SStructMatrixInitialize(A);

   if ( ((solver_id >= 20) && (solver_id <= 30)) ||
        ((solver_id >= 40) && (solver_id < 60)) )
   {
      HYPRE_SStructMatrixCreate(hypre_MPI_COMM_WORLD, graph, &A_amg);
      HYPRE_SStructMatrixSetObjectType(A_amg, HYPRE_PARCSR);
      HYPRE_SStructMatrixInitialize(A_amg);
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
               HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
                                               var, 1, &i, values);
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
                  HYPRE_SStructMatrixSetValues(A, part, index,
                                               pdata.graph_vars[entry],
                                               1, &pdata.graph_entries[entry],
                                               &pdata.graph_values[entry]);
               }
            }
         }
      }

      if ( ((solver_id >= 20) && (solver_id <= 30)) ||
           ((solver_id >= 40) && (solver_id < 60)) )
      {
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
                      HYPRE_SStructMatrixSetValues(A_amg, part, index,
                                                   pdata.graph_vars[entry],
                                                   1, &pdata.graph_entries[entry],
                                                  &pdata.graph_values[entry]);
                  }
               }
            }
         }
      }

   }

   /* reset matrix values */
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
                  HYPRE_SStructMatrixSetValues(A, part, index,
                                               pdata.matrix_vars[entry],
                                               1, &pdata.matrix_entries[entry],
                                               &pdata.matrix_values[entry]);
               }
            }
         }
      }
   }

   /* reset matrix values so that stencil connections between two parts are zeroed */
   for (part= data.nparts-1; part> 0; part--)
   {
      /*hypre_FacZeroCFSten(hypre_SStructMatrixPMatrix(A, part),
                          hypre_SStructMatrixPMatrix(A, part-1),
                          grid,
                          part,
                          prefinements[part]);
       hypre_FacZeroFCSten(hypre_SStructMatrixPMatrix(A, part),
                          grid,
                          part); 
       hypre_ZeroAMRMatrixData(A, part-1, prefinements[part]); */

      HYPRE_SStructFACZeroCFSten(A, grid, part, prefinements[part]);
      HYPRE_SStructFACZeroFCSten(A,
                                 grid,
                                 part);
      HYPRE_SStructFACZeroAMRMatrixData(A, part-1, prefinements[part]);
   }
                          
   HYPRE_SStructMatrixAssemble(A);

   if ( ((solver_id >= 20) && (solver_id <= 30)) ||
        ((solver_id >= 40) && (solver_id < 60)) )
   {
      for (part= data.nparts-1; part>= 0; part--)
      {
         pdata = data.pdata[part];

         for (var = 0; var < pdata.nvars; var++)
         {
            sA= hypre_SStructPMatrixSMatrix(hypre_SStructMatrixPMatrix(A, part),
                                            var, var);
            s = pdata.stencil_num[var];
            for (i = 0; i < data.stencil_sizes[s]; i++)
            {
               for (box = 0; box < pdata.nboxes; box++)
               {
                  box_values= hypre_TAlloc(double, data.max_boxsize);
                  GetVariableBox(pdata.ilowers[box], pdata.iuppers[box],
                                 pdata.vartypes[var], ilower, iupper);
                  HYPRE_StructMatrixGetBoxValues(sA, ilower, iupper, 1,
                                                &i, box_values);

                  HYPRE_SStructMatrixSetBoxValues(A_amg, part, ilower, iupper,
                                                  var, 1, &i, box_values);
                  hypre_TFree(box_values);
               }
            }
         }
      }

      HYPRE_SStructMatrixAssemble(A_amg);
      HYPRE_SStructMatrixGetObject(A_amg, (void **) &par_A);
   }
   /*-----------------------------------------------------------
    * Set up the linear system
    *-----------------------------------------------------------*/

   HYPRE_SStructVectorCreate(hypre_MPI_COMM_WORLD, grid, &b);
   HYPRE_SStructVectorInitialize(b);

   if ( ((solver_id >= 20) && (solver_id <= 30)) ||
        ((solver_id >= 40) && (solver_id < 60)) )
   {
      HYPRE_SStructVectorCreate(hypre_MPI_COMM_WORLD, grid, &b_amg);
      HYPRE_SStructVectorSetObjectType(b_amg, HYPRE_PARCSR);
      HYPRE_SStructVectorInitialize(b_amg);
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
 
            GetVariableBox(pdata.ilowers[box], pdata.iuppers[box], var,
                           ilower, iupper);
            HYPRE_SStructVectorSetBoxValues(b, part, ilower, iupper,
                                            var, values);
         }
      }
   }
   /*hypre_ZeroAMRVectorData(b, plevels, prefinements);*/
   HYPRE_SStructFACZeroAMRVectorData(b, plevels, prefinements);

   HYPRE_SStructVectorAssemble(b);

   if ( ((solver_id >= 20) && (solver_id <= 30)) ||
        ((solver_id >= 40) && (solver_id < 60)) )
   {
      for (part = 0; part < data.nparts; part++)
      {
         pdata = data.pdata[part];
         for (var = 0; var < pdata.nvars; var++)
         {
            sx= hypre_SStructPVectorSVector(hypre_SStructVectorPVector(b, part),
                                            var);
            for (box = 0; box < pdata.nboxes; box++)
            {
               box_values= hypre_TAlloc(double, data.max_boxsize);
               GetVariableBox(pdata.ilowers[box], pdata.iuppers[box], var,
                              ilower, iupper);
               HYPRE_StructVectorGetBoxValues(sx, ilower, iupper, box_values); 
               HYPRE_SStructVectorSetBoxValues(b_amg, part, ilower, iupper,
                                               var, values);
               hypre_TFree(box_values);
            }
         }
      }

      HYPRE_SStructVectorAssemble(b_amg);
      HYPRE_SStructVectorGetObject(b_amg, (void **) &par_b);
   }


   HYPRE_SStructVectorCreate(hypre_MPI_COMM_WORLD, grid, &x);
   HYPRE_SStructVectorInitialize(x);

   if ( ((solver_id >= 20) && (solver_id <= 30)) ||
        ((solver_id >= 40) && (solver_id < 60)) )
   {
      HYPRE_SStructVectorCreate(hypre_MPI_COMM_WORLD, grid, &x_amg);
      HYPRE_SStructVectorSetObjectType(x_amg, HYPRE_PARCSR);
      HYPRE_SStructVectorInitialize(x_amg);
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
            GetVariableBox(pdata.ilowers[box], pdata.iuppers[box], var,
                           ilower, iupper);
            HYPRE_SStructVectorSetBoxValues(x, part, ilower, iupper,
                                            var, values);
         }
      }
   }
   /*hypre_ZeroAMRVectorData(x, plevels, prefinements);*/
   HYPRE_SStructFACZeroAMRVectorData(x, plevels, prefinements);
   HYPRE_SStructVectorAssemble(x);


   if ( ((solver_id >= 20) && (solver_id <= 30)) ||
        ((solver_id >= 40) && (solver_id < 60)) )
   {
      for (part = 0; part < data.nparts; part++)
      {
         pdata = data.pdata[part];
         for (var = 0; var < pdata.nvars; var++)
         {
            sx= hypre_SStructPVectorSVector(hypre_SStructVectorPVector(x, part),
                                            var);
            for (box = 0; box < pdata.nboxes; box++)
            {
               box_values= hypre_TAlloc(double, data.max_boxsize);
               GetVariableBox(pdata.ilowers[box], pdata.iuppers[box], var,
                              ilower, iupper);
               HYPRE_StructVectorGetBoxValues(sx, ilower, iupper, box_values); 
               HYPRE_SStructVectorSetBoxValues(x_amg, part, ilower, iupper,
                                               var, values);
               hypre_TFree(box_values);
            }
         }
      }

      HYPRE_SStructVectorAssemble(x_amg);
      HYPRE_SStructVectorGetObject(x_amg, (void **) &par_x);
   }
 

   hypre_EndTiming(time_index);
   hypre_PrintTiming("SStruct Interface", hypre_MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();

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
#endif

   hypre_TFree(values);

   /*-----------------------------------------------------------
    * Print out the system and initial guess
    *-----------------------------------------------------------*/
   if (print_system)
   {
      HYPRE_SStructVectorPrint("sstruct.out.b",  b, 0);
      HYPRE_SStructVectorPrint("sstruct.out.x0", x, 0);
   }

   /*-----------------------------------------------------------
    * Solve the system using FAC
    *-----------------------------------------------------------*/

   if ((solver_id >=50) && (solver_id <100))
   {
      if (solver_id < 100)
      {
        /* n_pre+= n_post;*/
        /* n_post= 0;*/
      }
      time_index = hypre_InitializeTiming("FAC Setup");
      hypre_BeginTiming(time_index);

      HYPRE_SStructFACCreate(hypre_MPI_COMM_WORLD, &solver);
      HYPRE_SStructFACSetMaxLevels(solver, data.nparts);
      HYPRE_SStructFACSetMaxIter(solver, 20);
      HYPRE_SStructFACSetTol(solver, 1.0e-06);
      HYPRE_SStructFACSetPLevels(solver, data.nparts, plevels);
      HYPRE_SStructFACSetPRefinements(solver, data.nparts, prefinements);
      HYPRE_SStructFACSetRelChange(solver, 0);
      if (solver_id > 90)
      {
         HYPRE_SStructFACSetRelaxType(solver, 2);
      }
      else
      {
         HYPRE_SStructFACSetRelaxType(solver, 1);
      }
      HYPRE_SStructFACSetNumPreRelax(solver, n_pre);
      HYPRE_SStructFACSetNumPostRelax(solver, n_post);
      HYPRE_SStructFACSetCoarseSolverType(solver, 2);
      HYPRE_SStructFACSetLogging(solver, 1);
      HYPRE_SStructFACSetup2(solver, A, b, x);
      
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("FAC Solve");
      hypre_BeginTiming(time_index);

      if (solver_id > 90)
      {
         HYPRE_SStructFACSolve3(solver, A, b, x);
      }
      else
      {
         HYPRE_SStructFACSolve3(solver, A, b, x);
      }

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_SStructFACGetNumIterations(solver, &num_iterations);
      HYPRE_SStructFACGetFinalRelativeResidualNorm(
                                           solver, &final_res_norm);
      HYPRE_SStructFACDestroy2(solver);
   }



   /*-----------------------------------------------------------
    * Solve the system using PCG
    *-----------------------------------------------------------*/
   if ((solver_id >= 20) && (solver_id < 30))
   {
      time_index = hypre_InitializeTiming("PCG Setup");
      hypre_BeginTiming(time_index);

      HYPRE_ParCSRPCGCreate(hypre_MPI_COMM_WORLD, &par_solver);
      HYPRE_PCGSetMaxIter( par_solver, 200 );
      HYPRE_PCGSetTol( par_solver, 1.0e-06 );
      HYPRE_PCGSetTwoNorm( par_solver, 1 );
      HYPRE_PCGSetRelChange( par_solver, 0 );
      HYPRE_PCGSetPrintLevel( par_solver, 1 );

      if (solver_id == 20)
      {
         /* use BoomerAMG as preconditioner */
         HYPRE_BoomerAMGCreate(&par_precond);
         HYPRE_BoomerAMGSetCoarsenType(par_precond, 6);
         HYPRE_BoomerAMGSetStrongThreshold(par_precond, 0.5);
         HYPRE_BoomerAMGSetTol(par_precond, 1.0e-06);
         HYPRE_BoomerAMGSetPrintLevel(par_precond, 1);
         HYPRE_BoomerAMGSetPrintFileName(par_precond, "sstruct.out.log");
         /*HYPRE_BoomerAMGSetLogging(par_precond, 1, "sstruct.out.log");*/
         HYPRE_BoomerAMGSetMaxIter(par_precond, 1);
         HYPRE_PCGSetPrecond( par_solver,
                             (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                             (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
                             par_precond );
      }
      HYPRE_PCGSetup( par_solver, (HYPRE_Matrix) par_A,
                      (HYPRE_Vector) par_b, (HYPRE_Vector) par_x );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("PCG Solve");
      hypre_BeginTiming(time_index);

      HYPRE_PCGSolve( par_solver, (HYPRE_Matrix) par_A,
                      (HYPRE_Vector) par_b, (HYPRE_Vector) par_x );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_PCGGetNumIterations( par_solver, &num_iterations );
      HYPRE_PCGGetFinalRelativeResidualNorm( par_solver, &final_res_norm );
      HYPRE_ParCSRPCGDestroy(par_solver);

      if (solver_id == 20)
      {
         HYPRE_BoomerAMGDestroy(par_precond);
      }
   }

   /*-----------------------------------------------------------
    * Solve the system using ParCSR version of GMRES
    *-----------------------------------------------------------*/
   if ((solver_id >= 40) && (solver_id < 50))
   {
      time_index = hypre_InitializeTiming("GMRES Setup");
      hypre_BeginTiming(time_index);

      HYPRE_ParCSRGMRESCreate(hypre_MPI_COMM_WORLD, &par_solver);
      HYPRE_GMRESSetKDim(par_solver, 5);
      HYPRE_GMRESSetMaxIter(par_solver, 200);
      HYPRE_GMRESSetTol(par_solver, 1.0e-06);
      HYPRE_GMRESSetLogging(par_solver, 1);

      if (solver_id == 40)
      {
         /* use BoomerAMG as preconditioner */
         HYPRE_BoomerAMGCreate(&par_precond);
         HYPRE_BoomerAMGSetCoarsenType(par_precond, 6);
         HYPRE_BoomerAMGSetStrongThreshold(par_precond, 0.);
         HYPRE_BoomerAMGSetTruncFactor(par_precond, 0.3);
         HYPRE_BoomerAMGSetTol(par_precond, 1.0e-06);
         HYPRE_BoomerAMGSetPrintLevel(par_precond, 1);
         HYPRE_BoomerAMGSetPrintFileName(par_precond, "sstruct.out.log");
         HYPRE_BoomerAMGSetMaxIter(par_precond, 1);
         HYPRE_GMRESSetPrecond( par_solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                                (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
                                par_precond);
      }

      HYPRE_GMRESSetup( par_solver, (HYPRE_Matrix) par_A,
                        (HYPRE_Vector) par_b, (HYPRE_Vector) par_x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("GMRES Solve");
      hypre_BeginTiming(time_index);

      HYPRE_GMRESSolve( par_solver, (HYPRE_Matrix) par_A,
                        (HYPRE_Vector) par_b, (HYPRE_Vector) par_x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_GMRESGetNumIterations( par_solver, &num_iterations);
      HYPRE_GMRESGetFinalRelativeResidualNorm( par_solver,
                                               &final_res_norm);
      HYPRE_ParCSRGMRESDestroy(par_solver);

      if (solver_id == 40)
      {
         HYPRE_BoomerAMGDestroy(par_precond);
      }
   }

   if (solver_id == 30)
   {
      time_index = hypre_InitializeTiming("AMG Setup");
      hypre_BeginTiming(time_index);

      HYPRE_BoomerAMGCreate(&par_solver);
      HYPRE_BoomerAMGSetCoarsenType(par_solver, 6);
      HYPRE_BoomerAMGSetStrongThreshold(par_solver, 0.);
      HYPRE_BoomerAMGSetTruncFactor(par_solver, 0.3);
      /*HYPRE_BoomerAMGSetMaxLevels(par_solver, 4);*/
      HYPRE_BoomerAMGSetTol(par_solver, 1.0e-06);
      HYPRE_BoomerAMGSetPrintLevel(par_solver, 1);
      HYPRE_BoomerAMGSetPrintFileName(par_solver, "sstruct.out.log");
      HYPRE_BoomerAMGSetMaxIter(par_solver, 200);
      HYPRE_BoomerAMGSetup(par_solver, par_A, par_b, par_x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("BoomerAMG Solve");
      hypre_BeginTiming(time_index);

      HYPRE_BoomerAMGSolve(par_solver, par_A, par_b, par_x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_BoomerAMGGetNumIterations(par_solver, &num_iterations);
      HYPRE_BoomerAMGGetFinalRelativeResidualNorm(par_solver,
                                                 &final_res_norm);

      HYPRE_BoomerAMGDestroy(par_solver);
   }

   /*-----------------------------------------------------------
    * Gather the solution vector
    *-----------------------------------------------------------*/

   HYPRE_SStructVectorGather(x);

   /*-----------------------------------------------------------
    * Print the solution and other info
    *-----------------------------------------------------------*/

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
   hypre_TFree(stencils);

   HYPRE_SStructGraphDestroy(graph);

   if ( ((solver_id >= 20) && (solver_id <= 30)) ||
        ((solver_id >= 40) && (solver_id < 60)) )
   {
      HYPRE_SStructMatrixDestroy(A_amg);
      HYPRE_SStructVectorDestroy(b_amg);
      HYPRE_SStructVectorDestroy(x_amg);
   }

   HYPRE_SStructMatrixDestroy(A);
   HYPRE_SStructVectorDestroy(b);
   HYPRE_SStructVectorDestroy(x);


   hypre_TFree(plevels);
   hypre_TFree(prefinements);

   DestroyData(data);

   hypre_TFree(parts);
   hypre_TFree(refine);
   hypre_TFree(distribute);
   hypre_TFree(block);

   hypre_FinalizeMemoryDebug();

   /* Finalize MPI */
   hypre_MPI_Finalize();

   return (0);
}
