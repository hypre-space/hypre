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





/******************************************************************************
 *
 * Constructors and destructors for problem structure.
 *
 *****************************************************************************/

#include "headers.h"


/*--------------------------------------------------------------------------
 * NewProblem
 *--------------------------------------------------------------------------*/

Problem  *NewProblem(file_name)
char     *file_name;
{
   Problem  *problem;

   FILE     *fp;

   HYPRE_Int      num_variables;

   hypre_Matrix  *A;
   hypre_Vector  *f;
   hypre_Vector  *u;

   HYPRE_Int      num_unknowns;
   HYPRE_Int      num_points;

   HYPRE_Int     *iu;
   HYPRE_Int     *ip;
   HYPRE_Int     *iv;

   char     temp_file_name[256];
   FILE    *temp_fp;
   HYPRE_Int      flag;
   double  *data;
   double   temp_d;

   HYPRE_Int      i, j, k;


   /*----------------------------------------------------------
    * Allocate the problem structure
    *----------------------------------------------------------*/

   problem = hypre_CTAlloc(Problem, 1);

   /*----------------------------------------------------------
    * Open the problem file
    *----------------------------------------------------------*/

   fp = fopen(file_name, "r");

   /*----------------------------------------------------------
    * num_variables
    *----------------------------------------------------------*/

   hypre_fscanf(fp, "%d", &num_variables);

   /*----------------------------------------------------------
    * A
    *----------------------------------------------------------*/

   hypre_fscanf(fp, "%s", temp_file_name);
   A = ReadYSMP(temp_file_name);

   if (hypre_MatrixSize(A) != num_variables)
   {
      hypre_printf("hypre_Matrix is of incompatible size\n");
      exit(1);
   }

   hypre_sprintf(ProblemAInput(problem), "%s", temp_file_name);

   /*----------------------------------------------------------
    * f
    *----------------------------------------------------------*/

   hypre_fscanf(fp, "%d", &flag);
   ProblemFFlag(problem) = flag;

   if (flag == 0)
   {
      hypre_fscanf(fp, "%s", temp_file_name);
      f = ReadVec(temp_file_name);

      if (hypre_VectorSize(f) != num_variables)
      {
	 hypre_printf("Right hand side is of incompatible\n");
	 exit(1);
      }

      hypre_sprintf(ProblemFInput(problem), "%s", temp_file_name);
   }
   else
   {
      data = hypre_CTAlloc(double, hypre_NDIMU(num_variables));

      if (flag == 1)
      {
	 hypre_fscanf(fp, "%le", &temp_d);
	 for (i = 0; i < num_variables; i++)
	    data[i] = temp_d;

	 hypre_sprintf(ProblemFInput(problem), "%e", temp_d);
      }
      else if (flag == 2)
      {
	 for (i = 0; i < num_variables; i++)
	    data[i] = hypre_Rand();
      }

      f = hypre_NewVector(data, num_variables);
   }

   /*----------------------------------------------------------
    * u
    *----------------------------------------------------------*/

   hypre_fscanf(fp, "%d", &flag);
   ProblemUFlag(problem) = flag;

   if (flag == 0)
   {
      hypre_fscanf(fp, "%s", temp_file_name);
      u = ReadVec(temp_file_name);

      if (hypre_VectorSize(u) != num_variables)
      {
	 hypre_printf("Initial guess is of incompatible size\n");
	 exit(1);
      }

      hypre_sprintf(ProblemUInput(problem), "%s", temp_file_name);
   }
   else
   {
      data = hypre_CTAlloc(double, hypre_NDIMU(num_variables));

      if (flag == 1)
      {
	 hypre_fscanf(fp, "%le", &temp_d);
	 for (i = 0; i < num_variables; i++)
	    data[i] = temp_d;

	 hypre_sprintf(ProblemUInput(problem), "%e", temp_d);
      }
      else if (flag == 2)
      {
	 for (i = 0; i < num_variables; i++)
	    data[i] = hypre_Rand();
      }

      u = hypre_NewVector(data, num_variables);
   }

   /*----------------------------------------------------------
    * num_unknowns, num_points
    *----------------------------------------------------------*/

   hypre_fscanf(fp, "%d%d", &num_unknowns, &num_points);

   /*----------------------------------------------------------
    * iu, ip, iv
    *----------------------------------------------------------*/

   hypre_fscanf(fp, "%d", &flag);
   ProblemIUPVFlag(problem) = flag;

   if (flag == 0)
   {
      iu = hypre_CTAlloc(HYPRE_Int, hypre_NDIMU(num_variables));
      ip = hypre_CTAlloc(HYPRE_Int, hypre_NDIMU(num_variables));
      iv = hypre_CTAlloc(HYPRE_Int, hypre_NDIMP(num_points+1));

      hypre_fscanf(fp, "%s", temp_file_name);
      temp_fp = fopen(temp_file_name, "r");

      for (j = 0; j < num_variables; j++)
	 hypre_fscanf(temp_fp, "%d", &iu[j]);
      for (j = 0; j < num_variables; j++)
	 hypre_fscanf(temp_fp, "%d", &ip[j]);
      for (j = 0; j < num_points+1; j++)
	 hypre_fscanf(temp_fp, "%d", &iv[j]);

      fclose(temp_fp);

      hypre_sprintf(ProblemIUPVInput(problem), "%s", temp_file_name);
   }
   else
   {
      iu = NULL;
      ip = NULL;
      iv = NULL;
   }


   /*----------------------------------------------------------
    * Close the problem file
    *----------------------------------------------------------*/

   fclose(fp);

   /*----------------------------------------------------------
    * Set the problem structure
    *----------------------------------------------------------*/

   ProblemNumVariables(problem) = num_variables;

   ProblemA(problem)   		= A;
   ProblemF(problem)   		= f;
   ProblemU(problem)   		= u;

   ProblemNumUnknowns(problem)  = num_unknowns;
   ProblemNumPoints(problem)    = num_points;

   ProblemIU(problem)           = iu;
   ProblemIP(problem)           = ip;
   ProblemIV(problem)           = iv;

   return problem;
}

/*--------------------------------------------------------------------------
 * FreeProblem
 *--------------------------------------------------------------------------*/

void     FreeProblem(problem)
Problem  *problem;
{
   if (problem)
   {
      hypre_TFree(ProblemIU(problem));
      hypre_TFree(ProblemIP(problem));
      hypre_TFree(ProblemIV(problem));
      hypre_TFree(problem);
   }
}

/*--------------------------------------------------------------------------
 * WriteProblem
 *--------------------------------------------------------------------------*/

void     WriteProblem(file_name, problem)
char    *file_name;
Problem *problem;
{
   FILE    *fp;

   HYPRE_Int      flag;
   char     temp_file_name[256];
   double   temp_d;


   /*----------------------------------------------------------
    * Open the output file
    *----------------------------------------------------------*/
   
   fp = fopen(file_name, "a");
   hypre_fprintf(fp, "\nPROBLEM INFORMATION: \n\n");
  
   /*----------------------------------------------------------
    * num_variables
    *----------------------------------------------------------*/

   hypre_fprintf(fp, "    Number of variables: %d \n", ProblemNumVariables(problem));
  
   /*----------------------------------------------------------
    * A
    *----------------------------------------------------------*/

   hypre_sscanf(ProblemAInput(problem), "%s", temp_file_name); 
   hypre_fprintf(fp, "    Input matrix file: %s \n", temp_file_name);

   /*----------------------------------------------------------
    * f
    *----------------------------------------------------------*/

   flag = ProblemFFlag(problem);
   if (flag == 0)
   {
      hypre_sscanf(ProblemFInput(problem), "%s", temp_file_name); 
      hypre_fprintf(fp, "    Right-hand side file name: %s \n", temp_file_name);
   }
   else if (flag == 1)
   {
      hypre_sscanf(ProblemFInput(problem), "%le", &temp_d);   
      hypre_fprintf(fp, "    Right-hand side constant with value: %e \n", temp_d);
   }
   else if (flag == 2)
   {
      hypre_fprintf(fp, "    Right-hand side is random vector. \n");
   }

   /*----------------------------------------------------------
    * u
    *----------------------------------------------------------*/

   flag = ProblemUFlag(problem);
   if (flag == 0)
   {
      hypre_sscanf(ProblemUInput(problem), "%s", temp_file_name); 
      hypre_fprintf(fp, "    Initial guess file name: %s \n", temp_file_name);
   }
   else if (flag == 1)
   {
      hypre_sscanf(ProblemUInput(problem), "%le", &temp_d);   
      hypre_fprintf(fp, "    Initial guess constant with value: %e \n", temp_d);
   }
   else if (flag == 2)
   {
      hypre_fprintf(fp, "    Initial guess is random vector. \n");
   }

   /*----------------------------------------------------------
    * num_unknowns, num_points
    *----------------------------------------------------------*/

   hypre_fprintf(fp, "    Number of unknown functions: %d \n",
	   ProblemNumUnknowns(problem));
   hypre_fprintf(fp, "    Number of unknown points: %d \n",
	   ProblemNumPoints(problem));

   /*----------------------------------------------------------
    * iu, ip, iv
    *----------------------------------------------------------*/

   flag = ProblemIUPVFlag(problem);
   if (flag == 0)
   { 
      hypre_sscanf(ProblemIUPVInput(problem), "%s", temp_file_name); 
      hypre_fprintf(fp, "    iu, iv, ip read from file: %s \n", temp_file_name);
   }
   else
   {
      hypre_fprintf(fp, "    Pointers iu, iv, ip defined in standard way. \n");
   }

 
   /*----------------------------------------------------------
    * Close the output file
    *----------------------------------------------------------*/

   fclose(fp);

   return;
}

