/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/




#include <stdio.h>
 
#include "general.h"
#include "matrix.h"
#include "io.h"

#define  hypre_NDIMU(nv)  (50*nv)
#define  hypre_NDIMA(na)  (6*na)


/*--------------------------------------------------------------------------
 * RYSMP reads old-style ysmp files (with `1 1')
 *--------------------------------------------------------------------------*/
 
hypre_Matrix   *RYSMP(file_name)
char     *file_name;
{
   hypre_Matrix  *matrix;
 
   FILE    *fp;
 
   double  *data;
   int     *ia;
   int     *ja;
   int      size;
   
   int      j;
 
 
   /*----------------------------------------------------------
    * Read in the data
    *----------------------------------------------------------*/
 
   fp = fopen(file_name, "r");
 
    /* read in junk line */
    fscanf(fp, "%*[^\n]\n");

   fscanf(fp, "%d", &size);
 
   ia = hypre_CTAlloc(int, hypre_NDIMU(size+1));
   for (j = 0; j < size+1; j++)
      fscanf(fp, "%d", &ia[j]);
 
   ja = hypre_CTAlloc(int, hypre_NDIMA(ia[size]-1));
   for (j = 0; j < ia[size]-1; j++)
      fscanf(fp, "%d", &ja[j]);
 
   data = hypre_CTAlloc(double, hypre_NDIMA(ia[size]-1));
   for (j = 0; j < ia[size]-1; j++)
      fscanf(fp, "%le", &data[j]);
 
   fclose(fp);
 
   /*----------------------------------------------------------
    * Create the matrix structure
    *----------------------------------------------------------*/
 
   matrix = hypre_NewMatrix(data, ia, ja, size);
 
   return matrix;
}

/*--------------------------------------------------------------------------
 *  Program kills the junk line `1 1' in old-style ysmp  files
 *--------------------------------------------------------------------------*/

int   main(argc, argv)
int   argc;
char *argv[];
{
   char    *run_name;

   char     file_name_old[255];
   char     file_name_new[255];

   FILE    *fp;

   hypre_Matrix  *A;

   /*-------------------------------------------------------
    * Check that the number of command args is correct
    *-------------------------------------------------------*/

   if (argc < 2)
   {
      fprintf(stderr, "Usage:  kill_ones  <ysmp file name>\n");
      exit(1);
   }

   run_name = argv[1];

   sprintf(file_name_old, "%s", run_name);
   sprintf(file_name_new, "%s.new", run_name);

   A = RYSMP(file_name_old);
   hypre_WriteYSMP(file_name_new,A);

   return 0;
}

