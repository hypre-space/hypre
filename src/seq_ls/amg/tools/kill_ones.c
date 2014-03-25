/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
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
 
   HYPRE_Real  *data;
   HYPRE_Int     *ia;
   HYPRE_Int     *ja;
   HYPRE_Int      size;
   
   HYPRE_Int      j;
 
 
   /*----------------------------------------------------------
    * Read in the data
    *----------------------------------------------------------*/
 
   fp = fopen(file_name, "r");
 
    /* read in junk line */
    hypre_fscanf(fp, "%*[^\n]\n");

   hypre_fscanf(fp, "%d", &size);
 
   ia = hypre_CTAlloc(HYPRE_Int, hypre_NDIMU(size+1));
   for (j = 0; j < size+1; j++)
      hypre_fscanf(fp, "%d", &ia[j]);
 
   ja = hypre_CTAlloc(HYPRE_Int, hypre_NDIMA(ia[size]-1));
   for (j = 0; j < ia[size]-1; j++)
      hypre_fscanf(fp, "%d", &ja[j]);
 
   data = hypre_CTAlloc(HYPRE_Real, hypre_NDIMA(ia[size]-1));
   for (j = 0; j < ia[size]-1; j++)
      hypre_fscanf(fp, "%le", &data[j]);
 
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

HYPRE_Int   main(argc, argv)
HYPRE_Int   argc;
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
      hypre_fprintf(stderr, "Usage:  kill_ones  <ysmp file name>\n");
      exit(1);
   }

   run_name = argv[1];

   hypre_sprintf(file_name_old, "%s", run_name);
   hypre_sprintf(file_name_new, "%s.new", run_name);

   A = RYSMP(file_name_old);
   hypre_WriteYSMP(file_name_new,A);

   return 0;
}

