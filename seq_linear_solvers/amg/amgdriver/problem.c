/*BHEADER**********************************************************************
 * (c) 1996   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

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

   int      num_variables;

   hypre_Matrix  *A;
   hypre_Vector  *f;
   hypre_Vector  *u;

   int      num_unknowns;
   int      num_points;

   int     *iu;
   int     *ip;
   int     *iv;

   char     temp_file_name[256];
   FILE    *temp_fp;
   int      flag;
   double  *data;
   double   temp_d;

   int      i, j, k;


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

   fscanf(fp, "%d", &num_variables);

   /*----------------------------------------------------------
    * A
    *----------------------------------------------------------*/

   fscanf(fp, "%s", temp_file_name);
   A = ReadYSMP(temp_file_name);

   if (hypre_MatrixSize(A) != num_variables)
   {
      printf("hypre_Matrix is of incompatible size\n");
      exit(1);
   }

   sprintf(ProblemAInput(problem), "%s", temp_file_name);

   /*----------------------------------------------------------
    * f
    *----------------------------------------------------------*/

   fscanf(fp, "%d", &flag);
   ProblemFFlag(problem) = flag;

   if (flag == 0)
   {
      fscanf(fp, "%s", temp_file_name);
      f = ReadVec(temp_file_name);

      if (hypre_VectorSize(f) != num_variables)
      {
	 printf("Right hand side is of incompatible\n");
	 exit(1);
      }

      sprintf(ProblemFInput(problem), "%s", temp_file_name);
   }
   else
   {
      data = hypre_CTAlloc(double, hypre_NDIMU(num_variables));

      if (flag == 1)
      {
	 fscanf(fp, "%le", &temp_d);
	 for (i = 0; i < num_variables; i++)
	    data[i] = temp_d;

	 sprintf(ProblemFInput(problem), "%e", temp_d);
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

   fscanf(fp, "%d", &flag);
   ProblemUFlag(problem) = flag;

   if (flag == 0)
   {
      fscanf(fp, "%s", temp_file_name);
      u = ReadVec(temp_file_name);

      if (hypre_VectorSize(u) != num_variables)
      {
	 printf("Initial guess is of incompatible size\n");
	 exit(1);
      }

      sprintf(ProblemUInput(problem), "%s", temp_file_name);
   }
   else
   {
      data = hypre_CTAlloc(double, hypre_NDIMU(num_variables));

      if (flag == 1)
      {
	 fscanf(fp, "%le", &temp_d);
	 for (i = 0; i < num_variables; i++)
	    data[i] = temp_d;

	 sprintf(ProblemUInput(problem), "%e", temp_d);
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

   fscanf(fp, "%d%d", &num_unknowns, &num_points);

   /*----------------------------------------------------------
    * iu, ip, iv
    *----------------------------------------------------------*/

   fscanf(fp, "%d", &flag);
   ProblemIUPVFlag(problem) = flag;

   if (flag == 0)
   {
      iu = hypre_CTAlloc(int, hypre_NDIMU(num_variables));
      ip = hypre_CTAlloc(int, hypre_NDIMU(num_variables));
      iv = hypre_CTAlloc(int, hypre_NDIMP(num_points+1));

      fscanf(fp, "%s", temp_file_name);
      temp_fp = fopen(temp_file_name, "r");

      for (j = 0; j < num_variables; j++)
	 fscanf(temp_fp, "%d", &iu[j]);
      for (j = 0; j < num_variables; j++)
	 fscanf(temp_fp, "%d", &ip[j]);
      for (j = 0; j < num_points+1; j++)
	 fscanf(temp_fp, "%d", &iv[j]);

      fclose(temp_fp);

      sprintf(ProblemIUPVInput(problem), "%s", temp_file_name);
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

   int      flag;
   char     temp_file_name[256];
   double   temp_d;


   /*----------------------------------------------------------
    * Open the output file
    *----------------------------------------------------------*/
   
   fp = fopen(file_name, "a");
   fprintf(fp, "\nPROBLEM INFORMATION: \n\n");
  
   /*----------------------------------------------------------
    * num_variables
    *----------------------------------------------------------*/

   fprintf(fp, "    Number of variables: %d \n", ProblemNumVariables(problem));
  
   /*----------------------------------------------------------
    * A
    *----------------------------------------------------------*/

   sscanf(ProblemAInput(problem), "%s", temp_file_name); 
   fprintf(fp, "    Input matrix file: %s \n", temp_file_name);

   /*----------------------------------------------------------
    * f
    *----------------------------------------------------------*/

   flag = ProblemFFlag(problem);
   if (flag == 0)
   {
      sscanf(ProblemFInput(problem), "%s", temp_file_name); 
      fprintf(fp, "    Right-hand side file name: %s \n", temp_file_name);
   }
   else if (flag == 1)
   {
      sscanf(ProblemFInput(problem), "%le", &temp_d);   
      fprintf(fp, "    Right-hand side constant with value: %e \n", temp_d);
   }
   else if (flag == 2)
   {
      fprintf(fp, "    Right-hand side is random vector. \n");
   }

   /*----------------------------------------------------------
    * u
    *----------------------------------------------------------*/

   flag = ProblemUFlag(problem);
   if (flag == 0)
   {
      sscanf(ProblemUInput(problem), "%s", temp_file_name); 
      fprintf(fp, "    Initial guess file name: %s \n", temp_file_name);
   }
   else if (flag == 1)
   {
      sscanf(ProblemUInput(problem), "%le", &temp_d);   
      fprintf(fp, "    Initial guess constant with value: %e \n", temp_d);
   }
   else if (flag == 2)
   {
      fprintf(fp, "    Initial guess is random vector. \n");
   }

   /*----------------------------------------------------------
    * num_unknowns, num_points
    *----------------------------------------------------------*/

   fprintf(fp, "    Number of unknown functions: %d \n",
	   ProblemNumUnknowns(problem));
   fprintf(fp, "    Number of unknown points: %d \n",
	   ProblemNumPoints(problem));

   /*----------------------------------------------------------
    * iu, ip, iv
    *----------------------------------------------------------*/

   flag = ProblemIUPVFlag(problem);
   if (flag == 0)
   { 
      sscanf(ProblemIUPVInput(problem), "%s", temp_file_name); 
      fprintf(fp, "    iu, iv, ip read from file: %s \n", temp_file_name);
   }
   else
   {
      fprintf(fp, "    Pointers iu, iv, ip defined in standard way. \n");
   }

 
   /*----------------------------------------------------------
    * Close the output file
    *----------------------------------------------------------*/

   fclose(fp);

   return;
}

