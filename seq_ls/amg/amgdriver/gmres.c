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
   Matrix  *A = A_data_arg;
   Vector  *x;
   Vector  *y;


   x = NewVector(N_VDATA(v_arg), N_VLENGTH(v_arg));
   y = NewVector(N_VDATA(z_arg), N_VLENGTH(z_arg));

   Matvec(1.0, A, x, 0.0, y);

   tfree(x);
   tfree(y);

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
   void       (*precond)()   = (P_data -> precond);
   void        *precond_data = (P_data -> precond_data);
   Vector  *s                = (P_data -> s);
   Vector  *r                = (P_data -> r);

   Vector  *s_temp;
   Vector  *r_temp;


   s_temp = NewVector(N_VDATA(z_arg), N_VLENGTH(z_arg));
   r_temp = NewVector(N_VDATA(r_arg), N_VLENGTH(r_arg));
   CopyVector(s_temp, s);
   CopyVector(r_temp, r);

   /* s = C*r */
   InitVector(s, 0.0);
   precond(s, r, 0.0, precond_data);

   CopyVector(r, r_temp);
   CopyVector(s, s_temp);
   tfree(r_temp);
   tfree(s_temp);

   return 0;
}

/*--------------------------------------------------------------------------
 * GMRES
 *--------------------------------------------------------------------------*/

void     GMRES(x_arg, b_arg, tol_arg, data_arg)
Vector  *x_arg;
Vector  *b_arg;
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

   b_norm = sqrt(InnerProd(b_arg, b_arg));
   delta  = tol_arg*b_norm;

   N_VMAKE(x, VectorData(x_arg), VectorSize(x_arg));
   N_VMAKE(b, VectorData(b_arg), VectorSize(b_arg));

   SpgmrSolve(spgmr_mem, A_data, x, b, RIGHT, MODIFIED_GS, delta,
	      max_restarts, P_data, NULL, NULL, SPGMRATimes, SPGMRPSolve,
	      &norm, &nli, &nps);

   tfree(x);
   tfree(b);

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
Matrix   *A;
void    (*precond)();
void     *precond_data;
void     *data;
{
   GMRESData  *gmres_data = data;

   SPGMRPData *P_data;

   int         size;
   double     *darray;


   GMRESDataAData(gmres_data)    = (void *) A;

   size = MatrixSize(A);

   P_data = ctalloc(SPGMRPData, 1);
   (P_data -> precond)        = precond;
   (P_data -> precond_data)   = precond_data;
   darray = ctalloc(double, NDIMU(size));
   (P_data -> s) = NewVector(darray, size);
   darray = ctalloc(double, NDIMU(size));
   (P_data -> r) = NewVector(darray, size);
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

   gmres_data = ctalloc(GMRESData, 1);

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
      FreeVector(P_data -> s);
      FreeVector(P_data -> r);
      tfree(P_data);
      SpgmrFree(GMRESDataSpgmrMem(gmres_data));
      tfree(gmres_data);
   }
}

