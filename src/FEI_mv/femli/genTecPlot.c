/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include <stdio.h>

main(int argc, char **argv)
{
   int    i, j, ncnt, nElems, nNodes, *elemNodeList, nrows, ncols;
   int    vecNum=0;
   double *nodalCoord, *eVec;
   char   filename[100];
   FILE   *fp;

   printf("argument 1 : vector number (0 - m)\n");
   printf("argument 2 : file 1 (eigenfile)\n");
   printf("argument 3 : file 2 (elemConn file)\n");
   printf("argument 4 : file 3 (nodal coordinate file)\n");

   if (argc >= 2) sscanf(argv[1],"%d", &vecNum);
   printf("vecNum = %d\n", vecNum);
   if (argc >= 3) strcpy(filename, argv[2]);
   else           strcpy(filename,"eVec");
   fp = fopen(filename, "r");
   if ( fp == NULL )
   {
      printf("ERROR : %s file not found.\n", filename);
      exit(1);
   }
   fscanf(fp,"%d %d", &nrows, &ncols);
   eVec = hypre_TAlloc(double, nrows*ncols, HYPRE_MEMORY_HOST);
   for (i = 0; i < nrows; i++)
      for (j = 0; j < ncols; j++) fscanf(fp,"%lg", &eVec[i+j*nrows]);
   fclose(fp);

   if (argc >= 4) strcpy(filename, argv[3]);
   else           strcpy(filename,"elemNodeList");
   fp = fopen(filename, "r");
   if ( fp == NULL )
   {
      printf("ERROR : %s file not found.\n", filename);
      exit(1);
   }
   fscanf(fp,"%d", &nElems);
   elemNodeList = hypre_TAlloc(int, nElems*8, HYPRE_MEMORY_HOST);
   ncnt = 0;
   for (i = 0; i < nElems; i++)
   {
      for (j = 0; j < 8; j++)
      {
         fscanf(fp,"%d %d %d", &elemNodeList[ncnt],&elemNodeList[ncnt],
                &elemNodeList[ncnt]);
         elemNodeList[ncnt] = elemNodeList[ncnt] / 3;
         ncnt++;
      }
   }
   fclose(fp);

   if (argc >= 5) strcpy(filename, argv[4]);
   else           strcpy(filename,"nodalCoord");
   fp = fopen(filename, "r");
   if ( fp == NULL )
   {
      printf("ERROR : %s file not found.\n", filename);
      exit(1);
   }
   fscanf(fp,"%d", &nNodes);
   nodalCoord = hypre_TAlloc(double, nNodes*3, HYPRE_MEMORY_HOST);
   ncnt = 0;
   for (i = 0; i < nNodes; i++)
   {
      fscanf(fp,"%lg %lg %lg", &nodalCoord[ncnt],&nodalCoord[ncnt+1],
             &nodalCoord[ncnt+2]);
      ncnt += 3;
   }
   fclose(fp);

   if (argc >= 6) strcpy(filename, argv[5]);
   else           strcpy(filename,"tplotout.dat");
   printf("outputfile = %s\n", filename);
   fp = fopen(filename, "w");
   fprintf(fp, "TITLE = \"ALE3D TBAR Data\"\n");
   fprintf(fp, "VARIABLES = \"X\" \"Y\" \"Z\" \"U\" \"V\" \"W\"\n");
   fprintf(fp, "ZONE N=%d, E=%d, F=FEPOINT, ET=BRICK\n", nNodes, nElems);
   fprintf(fp, "\n");
   for (i = 0; i < nNodes; i++)
   {
      fprintf(fp, "%16.8e %16.8e %16.8e %16.8e %16.8e %16.8e\n",
              nodalCoord[i*3],nodalCoord[i*3+1],nodalCoord[i*3+2],
              eVec[i*3+vecNum*nrows], eVec[i*3+1+vecNum*nrows], 
              eVec[i*3+2+vecNum*nrows]);
   }
   fprintf(fp, "\n");
   ncnt = 0;
   for (i = 0; i < nElems; i++)
   {
      for (j = 0; j < 8; j++)
         fprintf(fp, "%7d ", elemNodeList[ncnt++]);
      fprintf(fp, "\n");
   }
   fclose(fp);
}

