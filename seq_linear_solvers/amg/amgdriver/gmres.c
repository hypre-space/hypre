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
 * Generalized Minimum Residual
 *
 *****************************************************************************/

#include "headers.h"
#include "gmres.h"


/*--------------------------------------------------------------------------
 * SPGMRATimes
 *--------------------------------------------------------------------------*/

int       SPGMRATimes(A_data_arg, v_arg, z_arg)
void     *A_data_arg;
N_Vector  v_arg;
N_Vector  z_arg;
{
   hypre_Matrix  *A = A_data_arg;
   hypre_Vector  *x;
   hypre_Vector  *y;


   x = hypre_NewVector(N_VDATA(v_arg), N_VLENGTH(v_arg));
   y = hypre_NewVector(N_VDATA(z_arg), N_VLENGTH(z_arg));

   hypre_Matvec(1.0, A, x, 0.0, y);

   hypre_TFree(x);
   hypre_TFree(y);

   return 0;
}

/*--------------------------------------------------------------------------
 * SPGMRPSolve
 *--------------------------------------------------------------------------*/

int       SPGMRPSolve(P_data_arg, r_arg,  z_arg, lr_arg)
void     *P_data_arg;
N_Vector  r_arg;
N_Vector  z_arg;
int       lr_arg;
{
   SPGMRPData  *P_data = P_data_arg;
   int        (*precond)()   = (P_data -> precond);
   void        *precond_data = (P_data -> precond_data);
   hypre_Vector  *s                = (P_data -> s);
   hypre_Vector  *r                = (P_data -> r);

   hypre_Vector  *s_temp;
   hypre_Vector  *r_temp;


   s_temp = hypre_NewVector(N_VDATA(z_arg), N_VLENGTH(z_arg));
   r_temp = hypre_NewVector(N_VDATA(r_arg), N_VLENGTH(r_arg));
   hypre_CopyVector(s_temp, s);
   hypre_CopyVector(r_temp, r);

   /* s = C*r */
   hypre_InitVector(s, 0.0);
   precond(s, r, 0.0, precond_data);

   hypre_CopyVector(r, r_temp);
   hypre_CopyVector(s, s_temp);
   hypre_TFree(r_temp);
   hypre_TFree(s_temp);

   return 0;
}

/*--------------------------------------------------------------------------
 * GMRES
 *--------------------------------------------------------------------------*/

void     GMRES(x_arg, b_arg, tol_arg, data_arg)
hypre_Vector  *x_arg;
hypre_Vector  *b_arg;
double   tol_arg;
void    *data_arg;
{
   GMRESData *gmres_data   = data_arg;

   int        max_restarts = GMRESDataMaxRestarts(gmres_data);

   void      *A_data       = GMRESDataAData(gmres_data);
   void      *P_data       = GMRESDataPData(gmres_data);
   SpgmrMem   spgmr_mem    = GMRESDataSpgmrMem(gmres_data);

   N_Vector   x;
   N_Vector   b;

   real       delta;
   real       norm, rel_norm, b_norm;

   int        nli;
   int        nps;

   /* logging variables */
   FILE      *log_fp;


   /*-----------------------------------------------------------------------
    * Start gmres solve
    *-----------------------------------------------------------------------*/

   b_norm = sqrt(hypre_InnerProd(b_arg, b_arg));
   delta  = tol_arg*b_norm;

   N_VMAKE(x, hypre_VectorData(x_arg), hypre_VectorSize(x_arg));
   N_VMAKE(b, hypre_VectorData(b_arg), hypre_VectorSize(b_arg));

   SpgmrSolve(spgmr_mem, A_data, x, b, RIGHT, MODIFIED_GS, delta,
	      max_restarts, P_data, NULL, NULL, SPGMRATimes, SPGMRPSolve,
	      &norm, &nli, &nps);

   hypre_TFree(x);
   hypre_TFree(b);

   rel_norm = b_norm ? (norm / b_norm) : 0;

#if 1
   printf("Iterations = %d: ||r||_2 = %e, ||r||_2/||b||_2 = %e\n",
	  nli, norm, rel_norm);
#endif

   /*-----------------------------------------------------------------------
    * Print log
    *-----------------------------------------------------------------------*/

   log_fp = fopen(GlobalsLogFileName, "a");

   fprintf(log_fp, "\nGMRES INFO:\n\n");

   fprintf(log_fp, "Iters       ||r||_2    ||r||_2/||b||_2\n");
   fprintf(log_fp, "-----    ------------    ------------\n");
   
   fprintf(log_fp, "% 5d    %e    %e\n", nli, norm, rel_norm);
   
   fclose(log_fp);
}

/*--------------------------------------------------------------------------
 * GMRESSetup
 *--------------------------------------------------------------------------*/

void      GMRESSetup(A, precond, precond_data, data)
hypre_Matrix   *A;
int     (*precond)();
void     *precond_data;
void     *data;
{
   GMRESData  *gmres_data = data;

   SPGMRPData *P_data;

   int         size;
   double     *darray;


   GMRESDataAData(gmres_data)    = (void *) A;

   size = hypre_MatrixSize(A);

   P_data = hypre_CTAlloc(SPGMRPData, 1);
   (P_data -> precond)        = precond;
   (P_data -> precond_data)   = precond_data;
   darray = hypre_CTAlloc(double, hypre_NDIMU(size));
   (P_data -> s) = hypre_NewVector(darray, size);
   darray = hypre_CTAlloc(double, hypre_NDIMU(size));
   (P_data -> r) = hypre_NewVector(darray, size);
   GMRESDataPData(gmres_data) = (void *) P_data;

   GMRESDataSpgmrMem(gmres_data) = 
      SpgmrMalloc(size, GMRESDataMaxKrylov(gmres_data), NULL);
}

/*--------------------------------------------------------------------------
 * NewGMRESData
 *--------------------------------------------------------------------------*/

void     *NewGMRESData(problem, solver, log_file_name)
Problem  *problem;
Solver   *solver;
char     *log_file_name;
{
   GMRESData  *gmres_data;

   gmres_data = hypre_CTAlloc(GMRESData, 1);

   GMRESDataMaxKrylov(gmres_data)   = SolverGMRESMaxKrylov(solver);
   GMRESDataMaxRestarts(gmres_data) = SolverGMRESMaxRestarts(solver);

   GMRESDataLogFileName(gmres_data) = log_file_name;

   return (void *)gmres_data;
}

/*--------------------------------------------------------------------------
 * FreeGMRESData
 *--------------------------------------------------------------------------*/

void   FreeGMRESData(data)
void  *data;
{
   GMRESData  *gmres_data = data;

   SPGMRPData *P_data = GMRESDataPData(gmres_data);


   if (gmres_data)
   {
      hypre_FreeVector(P_data -> s);
      hypre_FreeVector(P_data -> r);
      hypre_TFree(P_data);
      SpgmrFree(GMRESDataSpgmrMem(gmres_data));
      hypre_TFree(gmres_data);
   }
}

