
/******************************************************
 *
 *  File:  Hypre_PCG.c
 *
 *********************************************************/

#include "Hypre_PCG_Skel.h" 
#include "Hypre_PCG_Data.h" 

#include "math.h"
#include "utilities.h"

#ifndef max
#define max(a,b)  (((a)<(b)) ? (b) : (a))
#endif

/* *************************************************
 * Constructor
 *    Allocate Memory for private data
 *    and initialize here
 ***************************************************/
void Hypre_PCG_constructor(Hypre_PCG this) {

/*
  int size;

   size = sizeof( struct Hypre_PCG_private_type );
   this->d_table = (struct Hypre_PCG_private_type *) malloc( size );
   if ( this->d_table==NULL ) {
      hypre_OutOfMemory( size );
      return;
   };
*/
   this->d_table = hypre_CTAlloc( struct Hypre_PCG_private_type, 1 );

   /* Space for the contained data is allocated through Setup (p,s,r,norms,
      relnorms).  Space for matvec will have been allocated by the function
      calling Setup, and space for preconditioner will have been allocated by
      the function calling SetPreconditioner.*/

} /* end constructor */

/* *************************************************
 *  Destructor
 *      deallocate memory for private data here.
 *   This frees memory which had been allocated in constructor or Setup.
 ***************************************************/
void Hypre_PCG_destructor(Hypre_PCG this) {

   Hypre_PCG_Private pcg_data = Hypre_PCG_getPrivate(this);

/* let the memory leak to avoid a possible Babel bug...
   Hypre_Vector_Delete( pcg_data->p );
   Hypre_Vector_Delete( pcg_data->s );
   Hypre_Vector_Delete( pcg_data->r );
*/

   if ((pcg_data -> logging) > 0)
   {
      hypre_TFree( pcg_data->norms );
      hypre_TFree( pcg_data->rel_norms );
   };

   hypre_TFree( pcg_data );

   return;

} /* end destructor */


/* ********************************************************
 * impl_Hypre_PCG_Apply
 *
 * This code is a generic version based on 
 * hypre_PCGSolve and hypre_KrylovSolve, functions that
 * were written specifically for struct{matrix,vector} and
 * parcsr{matrix,vector}, respectively.
 *
 * We use the following convergence test as the default (see Ashby, Holst,
 * Manteuffel, and Saylor):
 *
 *       ||e||_A                           ||r||_C
 *       -------  <=  [kappa_A(C*A)]^(1/2) -------  < tol
 *       ||x||_A                           ||b||_C
 *
 * where we let (for the time being) kappa_A(CA) = 1.
 * We implement the test as:
 *
 *       gamma = <C*r,r>  <  (tol^2)*<C*b,b> = eps
 *
 **********************************************************/

int
impl_Hypre_PCG_Apply(Hypre_PCG this, Hypre_Vector b, Hypre_Vector* xp)
{
   Hypre_PCG_Private pcg_data = Hypre_PCG_getPrivate (this);

   Hypre_Vector x = *xp;

   double          tol          = (pcg_data -> tol);
   double          cf_tol       = (pcg_data -> cf_tol);
   int             max_iter     = (pcg_data -> max_iter);
   int             two_norm     = (pcg_data -> two_norm);
   int             rel_change   = (pcg_data -> rel_change);

   Hypre_Vector    p            = (pcg_data -> p);
   Hypre_Vector    s            = (pcg_data -> s);
   Hypre_Vector    r            = (pcg_data -> r);
   Hypre_LinearOperator  matvec = (pcg_data -> matvec);
   Hypre_Solver    precond      = (pcg_data -> preconditioner);
   Hypre_LinearOperator precond_solver = Hypre_Solver_castTo( precond,
                                                              "Hypre_LinearOperator" );

   int             logging      = (pcg_data -> logging);
   double         *norms        = (pcg_data -> norms);
   double         *rel_norms    = (pcg_data -> rel_norms);
                
   double          alpha, beta;
   double          gamma, gamma_old;
   double          bi_prod, i_prod, eps;
   double          pi_prod, xi_prod;
                
   double          i_prod_0;
   double          cf_ave_0 = 0.0;
   double          cf_ave_1 = 0.0;
   double          weight;

   double          guard_zero_residual; 

   int             i = 0;
   int             ierr = 0;

   /*-----------------------------------------------------------------------
    * With relative change convergence test on, it is possible to attempt
    * another iteration with a zero residual. This causes the parameter
    * alpha to go NaN. The guard_zero_residual parameter is to circumvent
    * this. Perhaps it should be set to something non-zero (but small).
    *-----------------------------------------------------------------------*/

   guard_zero_residual = 0.0;

   /*-----------------------------------------------------------------------
    * Start pcg solve
    *-----------------------------------------------------------------------*/

   /* compute eps */
   if (two_norm)
   {
      /* bi_prod = <b,b> */
      ierr += Hypre_Vector_Dot (b, b, &bi_prod);
   }
   else
   {
      /* bi_prod = <C*b,b> */
      ierr += Hypre_Vector_Clear (p);
      ierr += Hypre_Solver_Apply (precond, b, &p);
      ierr += Hypre_Vector_Dot (p, b, &bi_prod);
   }
   eps = (tol*tol)*bi_prod;

   /* Check to see if the rhs vector b is zero */
   if (bi_prod == 0.0)
   {
      /* Set x equal to zero and return */
      ierr += Hypre_Vector_Copy (x, b);
      if (logging > 0)
      {
         norms[0]     = 0.0;
         rel_norms[i] = 0.0;
      }
      ierr = 0;
      return ierr;
   }

   /* r = b - Ax */
   ierr += Hypre_LinearOperator_Apply (matvec, x, &r);
   ierr += Hypre_Vector_Scale (r, -1.0);
   ierr += Hypre_Vector_Axpy (r, 1.0, b);
 
   /* Set initial residual norm */
   if (logging > 0 || cf_tol > 0.0)
   {
      ierr += Hypre_Vector_Dot (r, r, &i_prod_0);
      if (logging > 0) norms[0] = sqrt(i_prod_0);
   }

   /* p = C*r */
   ierr += Hypre_Vector_Clear (p);
   ierr += Hypre_Solver_Apply (precond, r, &p);

   /* gamma = <r,p> */
   ierr += Hypre_Vector_Dot (r, p, &gamma);

   while ((i+1) <= max_iter)
   {
      i++;

      /* s = A*p */
      ierr += Hypre_LinearOperator_Apply (matvec, p, &s);

      /* alpha = gamma / <s,p> */
      ierr += Hypre_Vector_Dot (s, p, &alpha);
      alpha = gamma / alpha;

      gamma_old = gamma;

      /* x = x + alpha*p */
      ierr += Hypre_Vector_Axpy (x, alpha, p);

      /* r = r - alpha*s */
      ierr += Hypre_Vector_Axpy (r, -alpha, s);
         
      /* s = C*r */
      ierr += Hypre_Vector_Clear (s);
/* doesn't work, likely Babel bug (jfp 031600)...
      printf("ierr += Hypre_LinearOperator_Apply (precond_solver, r, &s)\n");
      printf("precond=%i, precond_solver=%i\n", precond, precond_solver );
      ierr += Hypre_LinearOperator_Apply (precond_solver, r, &s);
 do it without the cast to LinearOperator...
*/
      ierr += Hypre_Solver_Apply (precond, r, &s);

      /* gamma = <r,s> */
      ierr += Hypre_Vector_Dot (r, s, &gamma);

      /* set i_prod for convergence test */
      if (two_norm)
         ierr += Hypre_Vector_Dot (r, r, &i_prod);
      else
         i_prod = gamma;

#if 0
      if (two_norm)
         printf("Iter (%d): ||r||_2 = %e, ||r||_2/||b||_2 = %e\n",
                i, sqrt(i_prod), (bi_prod ? sqrt(i_prod/bi_prod) : 0));
      else
         printf("Iter (%d): ||r||_C = %e, ||r||_C/||b||_C = %e\n",
                i, sqrt(i_prod), (bi_prod ? sqrt(i_prod/bi_prod) : 0));
#endif
 
      /* log norm info */
      if (logging > 0)
      {
         norms[i]     = sqrt(i_prod);
         rel_norms[i] = bi_prod ? sqrt(i_prod/bi_prod) : 0;
      }

      /* check for convergence */
      if (i_prod < eps)
      {
         if (rel_change && i_prod > guard_zero_residual)
         {
            ierr += Hypre_Vector_Dot (p, p, &pi_prod);
            ierr += Hypre_Vector_Dot (x, x, &xi_prod);
            if ((alpha*alpha*pi_prod/xi_prod) < (eps/bi_prod))
               break;
         }
         else
         {
            break;
         }
      }

      /*--------------------------------------------------------------------
       * Optional test to see if adequate progress is being made.
       * The average convergence factor is recorded and compared
       * against the tolerance 'cf_tol'. The weighting factor is  
       * intended to pay more attention to the test when an accurate
       * estimate for average convergence factor is available.  
       *--------------------------------------------------------------------*/

      if (cf_tol > 0.0)
      {
         cf_ave_0 = cf_ave_1;
         cf_ave_1 = pow( i_prod / i_prod_0, 1.0/(2.0*i)); 

         weight   = fabs(cf_ave_1 - cf_ave_0);
         weight   = weight / max(cf_ave_1, cf_ave_0);
         weight   = 1.0 - weight;
#if 0
         printf("I = %d: cf_new = %e, cf_old = %e, weight = %e\n",
                i, cf_ave_1, cf_ave_0, weight );
#endif
         if (weight * cf_ave_1 > cf_tol) break;
      }

      /* beta = gamma / gamma_old */
      beta = gamma / gamma_old;

      /* p = s + beta p */
      ierr += Hypre_Vector_Scale (p, beta);   
      ierr += Hypre_Vector_Axpy (p, 1.0, s);
   }


#if 0
   if (two_norm)
      printf("Iterations = %d: ||r||_2 = %e, ||r||_2/||b||_2 = %e\n",
             i, sqrt(i_prod), (bi_prod ? sqrt(i_prod/bi_prod) : 0));
   else
      printf("Iterations = %d: ||r||_C = %e, ||r||_C/||b||_C = %e\n",
             i, sqrt(i_prod), (bi_prod ? sqrt(i_prod/bi_prod) : 0));
#endif

   /*-----------------------------------------------------------------------
    * Print log
    *-----------------------------------------------------------------------*/

#if 0
   if (logging > 0)
   {
      if (two_norm)
      {
         printf("Iters       ||r||_2    ||r||_2/||b||_2\n");
         printf("-----    ------------    ------------ \n");
      }
      else
      {
         printf("Iters       ||r||_C    ||r||_C/||b||_C\n");
         printf("-----    ------------    ------------ \n");
      }
      for (j = 1; j <= i; j++)
      {
         printf("% 5d    %e    %e\n", j, norms[j], rel_norms[j]);
      }
   }
#endif

   (pcg_data -> num_iterations) = i;

   return ierr;

} /* end impl_Hypre_PCGApply */


/* ********************************************************
 * impl_Hypre_PCG_GetSystemOperator
 **********************************************************/
Hypre_LinearOperator  
impl_Hypre_PCG_GetSystemOperator(Hypre_PCG this) 
{

   Hypre_PCG_Private pcg_data = Hypre_PCG_getPrivate(this);

   return (pcg_data->matvec);

} /* end impl_Hypre_PCGGetSystemOperator */


/* ********************************************************
 * impl_Hypre_PCGGetResidual
 **********************************************************/
Hypre_Vector  impl_Hypre_PCG_GetResidual(Hypre_PCG this) {
   Hypre_PCG_Private pcg_data = Hypre_PCG_getPrivate (this);
   return pcg_data->r;
} /* end impl_Hypre_PCGGetResidual */

/* ********************************************************
 * impl_Hypre_PCGGetConvergenceInfo
 **********************************************************/
int  impl_Hypre_PCG_GetConvergenceInfo(Hypre_PCG this, char* name, double* value) {
   int ivalue;
   Hypre_PCG_Private pcg_data = Hypre_PCG_getPrivate (this);

   if ( !strcmp(name,"number of iterations") ) {
      *value = pcg_data->num_iterations;
      return 0;
   }
   else if ( !strcmp(name,"residual norm") ) {
      if (pcg_data->logging > 0) {
         *value = pcg_data->rel_norms[pcg_data->num_iterations];
         return 0;
      }
      else {
         *value = -1;
         return -1;
      }
   }
   else {
      printf(
         "Don't understand keyword %s to Hypre_PCG_GetConvergenceInfo\n",
         name );
      *value = 0;
      return -1;
   }
} /* end impl_Hypre_PCGGetConvergenceInfo */

/* ********************************************************
 * impl_Hypre_PCGGetPreconditioner
 **********************************************************/
Hypre_Solver  impl_Hypre_PCG_GetPreconditioner(Hypre_PCG this) {
   Hypre_PCG_Private pcg_data = Hypre_PCG_getPrivate (this);
   return pcg_data->preconditioner;
} /* end impl_Hypre_PCGGetPreconditioner */

/* ********************************************************
 * impl_Hypre_PCGGetDoubleParameter
 **********************************************************/
double  impl_Hypre_PCG_GetDoubleParameter(Hypre_PCG this, char* name) {
   Hypre_PCG_Private pcg_data = Hypre_PCG_getPrivate (this);

   if ( !strcmp( name, "tol" ) ) {
      return pcg_data->tol;
   }
   else if ( !strcmp( name, "cf_tol" ) ) {
      return pcg_data->cf_tol;
   }
   else {
      printf( "Don't understand keyword %s to Hypre_PCG_GetDoubleParameter\n",
              name );
      return -1;
   }
} /* end impl_Hypre_PCGGetDoubleParameter */

/* ********************************************************
 * impl_Hypre_PCGGetIntParameter
 **********************************************************/
int  impl_Hypre_PCG_GetIntParameter(Hypre_PCG this, char* name) {
   Hypre_PCG_Private pcg_data = Hypre_PCG_getPrivate (this);

   if ( !strcmp( name, "max_iter" ) ) {
      return pcg_data->max_iter;
   }
   else if ( !strcmp( name, "two_norm" ) || !strcmp( name, "2-norm" ) ) {
      return pcg_data->two_norm;
   }
   else if ( !strcmp( name, "rel_change" ) ||
             !strcmp( name, "relative change test" ) ) {
      return pcg_data->rel_change;
   }
   else if ( !strcmp( name, "num_iterations" ) ) {
      return pcg_data->num_iterations;
   }
   else if ( !strcmp( name, "logging" ) || !strcmp( name, "log" ) ) {
      return pcg_data->logging;
   }
   else {
      printf( "Don't understand keyword %s to Hypre_PCG_GetIntParameter\n",
              name );
      return -1;
   }
} /* end impl_Hypre_PCGGetIntParameter */

/* ********************************************************
 * impl_Hypre_PCGSetDoubleParameter
 **********************************************************/
int  impl_Hypre_PCG_SetDoubleParameter(Hypre_PCG this, char* name, double value) {
   Hypre_PCG_Private pcg_data = Hypre_PCG_getPrivate (this);
   if ( !strcmp( name, "tol" ) ) {
      pcg_data->tol = value;
      return 0;
   }
   else if ( !strcmp( name, "cf_tol" ) ) {
      pcg_data->cf_tol = value;
      return 0;
   }
   else {
      printf( "Don't understand keyword %s to Hypre_PCG_GetDoubleParameter\n",
              name );
      return -1;
   }
} /* end impl_Hypre_PCGSetDoubleParameter */

/* ********************************************************
 * impl_Hypre_PCGSetIntParameter
 **********************************************************/
int  impl_Hypre_PCG_SetIntParameter(Hypre_PCG this, char* name, int value)
{
   Hypre_PCG_Private pcg_data = Hypre_PCG_getPrivate (this);

   if ( !strcmp( name, "max_iter" ) ) {
      pcg_data->max_iter = value;
      return 0;
   }
   else if ( !strcmp( name, "two_norm" ) || !strcmp( name, "2-norm" ) ) {
      pcg_data->two_norm = value;
      return 0;
   }
   else if ( !strcmp( name, "rel_change" ) || !strcmp( name, "relative change test" ) ) {
      pcg_data->rel_change = value;;
      return 0;
   }
   else if ( !strcmp( name, "logging" ) || !strcmp( name, "log" ) ) {
      pcg_data->logging = value;
      return 0;
   }
   else {
      printf( "Don't understand keyword %s to Hypre_PCG_SetIntParameter\n",
              name );
      return -1;
   }
} /* end impl_Hypre_PCGSetIntParameter */

/* ********************************************************
 * impl_Hypre_PCGNew
 **********************************************************/
int  impl_Hypre_PCG_New(Hypre_PCG this, Hypre_MPI_Com comm) {

   Hypre_PCG_Private pcg_data = Hypre_PCG_getPrivate (this);

   (pcg_data -> tol)          = 1.0e-06;
   (pcg_data -> cf_tol)      = 0.0;
   (pcg_data -> max_iter)     = 1000;
   (pcg_data -> two_norm)     = 0;
   (pcg_data -> rel_change)   = 0;
   (pcg_data -> logging)      = 0;
   (pcg_data -> norms)        = NULL;
   (pcg_data -> rel_norms)    = NULL;

   return 0;
} /* end impl_Hypre_PCGNew */

/* ********************************************************
 * impl_Hypre_PCGConstructor
 **********************************************************/
Hypre_PCG  impl_Hypre_PCG_Constructor(Hypre_MPI_Com comm) {
/* Hypre_PCG_new calls Hypre_PCG_constructor and does a whole lot more ... */
   Hypre_PCG pcg = Hypre_PCG_new();
   Hypre_PCG_New( pcg, comm );
   return pcg;
} /* end impl_Hypre_PCGConstructor */


/* ********************************************************
 * impl_Hypre_PCG_Setup
 **********************************************************/
int
impl_Hypre_PCG_Setup(Hypre_PCG this, 
                     Hypre_LinearOperator A, 
                     Hypre_Vector b, 
                     Hypre_Vector x) 
{

   Hypre_PCG_Private pcg_data = Hypre_PCG_getPrivate (this);
   Hypre_Vector *vp;
   int ierr = 0;
   int max_iter = (pcg_data -> max_iter);

   /* Save LinearOperator */

   pcg_data->matvec = A;

   /* Create temporary vectors */

   ierr += Hypre_Vector_Clone (x, vp);
   pcg_data->p = *vp;
   ierr += Hypre_Vector_Clone (x, vp);
   pcg_data->s = *vp;
   ierr += Hypre_Vector_Clone (b, vp);
   pcg_data->r = *vp;

   /* Allocate space for log info */

   if ((pcg_data -> logging) > 0)
   {
      (pcg_data -> norms)     =
         hypre_CTAlloc(double, max_iter + 1);
      (pcg_data -> rel_norms) = hypre_CTAlloc(double, max_iter + 1);
   }

   return ierr;

} /* end impl_Hypre_PCGSetup */


/* ********************************************************
 * impl_Hypre_PCGGetConstructedObject
 **********************************************************/
Hypre_Solver  impl_Hypre_PCG_GetConstructedObject(Hypre_PCG this) {
   return (Hypre_Solver) Hypre_PCG_castTo( this, "Hypre_Solver" );
} /* end impl_Hypre_PCGGetConstructedObject */


/* ********************************************************
 * impl_Hypre_PCGSetPreconditioner
 **********************************************************/
int  
impl_Hypre_PCG_SetPreconditioner(Hypre_PCG this, Hypre_Solver precond) 
{

   Hypre_PCG_Private pcg_data = Hypre_PCG_getPrivate (this);

   pcg_data->preconditioner = precond;

   return 0;
   
} /* end impl_Hypre_PCGSetPreconditioner */
