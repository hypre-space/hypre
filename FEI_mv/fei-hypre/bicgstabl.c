/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * BiCGSTABL 
 *
 *****************************************************************************/

#include "utilities/utilities.h"
#include "HYPRE.h"
#include "IJ_matrix_vector/HYPRE_IJ_mv.h"
#include "parcsr_matrix_vector/HYPRE_parcsr_mv.h"
#include "parcsr_matrix_vector/parcsr_matrix_vector.h"
#include "parcsr_linear_solvers/parcsr_linear_solvers.h"
#include "parcsr_linear_solvers/HYPRE_parcsr_ls.h"
#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_BiCGSTABLData
 *--------------------------------------------------------------------------*/

typedef struct
{
   int      size;
   int      max_iter;
   int      stop_crit;
   double   tol;
   double   rel_residual_norm;

   void  *A;
   void  *r;
   void  *s;
   void  *y;
   void  *t;
   void  *tt;
   void  *st;
   void  *asm1;
   void  *as;
   void  *awt;
   void  *wt;
   void  *wh;
   void  *at;
   void  *xt;
   void  *t2;

   void  *matvec_data;

   int    (*precond)();
   int    (*precond_setup)();
   void    *precond_data;

   /* log info (always logged) */
   int      num_iterations;
 
   /* additional log info (logged when `logging' > 0) */
   int      logging;
   double  *norms;
   char    *log_file_name;

} hypre_BiCGSTABLData;

/*--------------------------------------------------------------------------
 * hypre_BiCGSTABLCreate
 *--------------------------------------------------------------------------*/
 
void * hypre_BiCGSTABLCreate( )
{
   hypre_BiCGSTABLData *bicgstab_data;
 
   bicgstab_data = hypre_CTAlloc(hypre_BiCGSTABLData, 1);
 
   /* set defaults */
   (bicgstab_data -> tol)            = 1.0e-06;
   (bicgstab_data -> size)           = 2;
   (bicgstab_data -> max_iter)       = 1000;
   (bicgstab_data -> stop_crit)      = 0; /* rel. residual norm */
   (bicgstab_data -> precond)        = hypre_KrylovIdentity;
   (bicgstab_data -> precond_setup)  = hypre_KrylovIdentitySetup;
   (bicgstab_data -> precond_data)   = NULL;
   (bicgstab_data -> logging)        = 0;
   (bicgstab_data -> s)              = NULL;
   (bicgstab_data -> y)              = NULL;
   (bicgstab_data -> t)              = NULL;
   (bicgstab_data -> tt)             = NULL;
   (bicgstab_data -> s)              = NULL;
   (bicgstab_data -> asm1)           = NULL;
   (bicgstab_data -> as)             = NULL;
   (bicgstab_data -> awt)            = NULL;
   (bicgstab_data -> wt)             = NULL;
   (bicgstab_data -> wh)             = NULL;
   (bicgstab_data -> at)             = NULL;
   (bicgstab_data -> xt)             = NULL;
   (bicgstab_data -> t2)             = NULL;
   (bicgstab_data -> matvec_data)    = NULL;
   (bicgstab_data -> norms)          = NULL;
   (bicgstab_data -> log_file_name)  = NULL;
 
   return (void *) bicgstab_data;
}

/*--------------------------------------------------------------------------
 * hypre_BiCGSTABLDestroy
 *--------------------------------------------------------------------------*/
 
int hypre_BiCGSTABLDestroy( void *bicgstab_vdata )
{
   hypre_BiCGSTABLData *bicgstab_data = bicgstab_vdata;
   int i, ierr = 0;
 
   if (bicgstab_data)
   {
      if ((bicgstab_data -> logging) > 0)
      {
         hypre_TFree(bicgstab_data -> norms);
      }
 
      hypre_KrylovMatvecDestroy(bicgstab_data -> matvec_data);
 
      hypre_KrylovDestroyVector(bicgstab_data -> r);
      hypre_KrylovDestroyVector(bicgstab_data -> s);
      hypre_KrylovDestroyVector(bicgstab_data -> y);
      hypre_KrylovDestroyVector(bicgstab_data -> t);
      hypre_KrylovDestroyVector(bicgstab_data -> tt);
      hypre_KrylovDestroyVector(bicgstab_data -> st);
      hypre_KrylovDestroyVector(bicgstab_data -> as);
      hypre_KrylovDestroyVector(bicgstab_data -> asm1);
      hypre_KrylovDestroyVector(bicgstab_data -> awt);
      hypre_KrylovDestroyVector(bicgstab_data -> wt);
      hypre_KrylovDestroyVector(bicgstab_data -> wh);
      hypre_KrylovDestroyVector(bicgstab_data -> at);
      hypre_KrylovDestroyVector(bicgstab_data -> xt);
      hypre_KrylovDestroyVector(bicgstab_data -> t2);
 
      hypre_TFree(bicgstab_data);
   }
 
   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_BiCGSTABLSetup
 *--------------------------------------------------------------------------*/
 
int hypre_BiCGSTABLSetup( void *bicgstab_vdata, void *A, void *b, void *x         )
{
   hypre_BiCGSTABLData *bicgstab_data     = bicgstab_vdata;
   int            max_iter         = (bicgstab_data -> max_iter);
   int          (*precond_setup)() = (bicgstab_data -> precond_setup);
   void          *precond_data     = (bicgstab_data -> precond_data);
   int            ierr = 0;
 
   (bicgstab_data -> A) = A;
 
   /*--------------------------------------------------
    * The arguments for NewVector are important to
    * maintain consistency between the setup and
    * compute phases of matvec and the preconditioner.
    *--------------------------------------------------*/
 
   if ((bicgstab_data -> r) == NULL)
      (bicgstab_data -> r) = hypre_KrylovCreateVector(b);
   if ((bicgstab_data -> s) == NULL)
      (bicgstab_data -> s) = hypre_KrylovCreateVector(b);
   if ((bicgstab_data -> y) == NULL)
      (bicgstab_data -> y) = hypre_KrylovCreateVector(b);
   if ((bicgstab_data -> t) == NULL)
      (bicgstab_data -> t) = hypre_KrylovCreateVector(b);
   if ((bicgstab_data -> tt) == NULL)
      (bicgstab_data -> tt) = hypre_KrylovCreateVector(b);
   if ((bicgstab_data -> st) == NULL)
      (bicgstab_data -> st) = hypre_KrylovCreateVector(b);
   if ((bicgstab_data -> asm1) == NULL)
      (bicgstab_data -> asm1) = hypre_KrylovCreateVector(b);
   if ((bicgstab_data -> as) == NULL)
      (bicgstab_data -> as) = hypre_KrylovCreateVector(b);
   if ((bicgstab_data -> awt) == NULL)
      (bicgstab_data -> awt) = hypre_KrylovCreateVector(b);
   if ((bicgstab_data -> wt) == NULL)
      (bicgstab_data -> wt) = hypre_KrylovCreateVector(b);
   if ((bicgstab_data -> wh) == NULL)
      (bicgstab_data -> wh) = hypre_KrylovCreateVector(b);
   if ((bicgstab_data -> at) == NULL)
      (bicgstab_data -> at) = hypre_KrylovCreateVector(b);
   if ((bicgstab_data -> xt) == NULL)
      (bicgstab_data -> xt) = hypre_KrylovCreateVector(b);
   if ((bicgstab_data -> t2) == NULL)
      (bicgstab_data -> t2) = hypre_KrylovCreateVector(b);
 
   if ((bicgstab_data -> matvec_data) == NULL)
      (bicgstab_data -> matvec_data) = hypre_KrylovMatvecCreate(A, x);
 
   precond_setup(precond_data, A, b, x);
 
   /*-----------------------------------------------------
    * Allocate space for log info
    *-----------------------------------------------------*/
 
   if ((bicgstab_data -> logging) > 0)
   {
      if ((bicgstab_data -> norms) == NULL)
         (bicgstab_data -> norms) = hypre_CTAlloc(double, max_iter + 1);
      if ((bicgstab_data -> log_file_name) == NULL)
         (bicgstab_data -> log_file_name) = "bicgstab.out.log";
   }
 
   return ierr;
}
 
/*--------------------------------------------------------------------------
 * hypre_BiCGSTABLSolve
 *-------------------------------------------------------------------------*/

int hypre_BiCGSTABLSolve(void  *bicgstab_vdata, void  *A, void  *b, void  *x)
{
   hypre_BiCGSTABLData  *bicgstab_data   = bicgstab_vdata;
   int               size         = (bicgstab_data -> size);
   int 		     max_iter     = (bicgstab_data -> max_iter);
   int 		     stop_crit    = (bicgstab_data -> stop_crit);
   double 	     accuracy     = (bicgstab_data -> tol);
   void              *matvec_data  = (bicgstab_data -> matvec_data);

   void             *r            = (bicgstab_data -> r);
   void             *s            = (bicgstab_data -> s);
   void             *y            = (bicgstab_data -> y);
   void             *t            = (bicgstab_data -> t);
   void             *tt           = (bicgstab_data -> tt);
   void             *wt           = (bicgstab_data -> wt);
   void             *awt          = (bicgstab_data -> awt);
   void             *asm1         = (bicgstab_data -> asm1);
   void             *as           = (bicgstab_data -> as);
   void             *wh           = (bicgstab_data -> wh);
   void             *xt           = (bicgstab_data -> xt);
   void             *at           = (bicgstab_data -> at);
   void             *st           = (bicgstab_data -> st);
   void             *t2           = (bicgstab_data -> t2);
   int 	           (*precond)()   = (bicgstab_data -> precond);
   int 	            *precond_data = (bicgstab_data -> precond_data);

   /* logging variables */
   int             logging        = (bicgstab_data -> logging);
   double         *norms          = (bicgstab_data -> norms);
   char           *log_file_name  = (bicgstab_data -> log_file_name);
   
   int        ierr = 0;
   int        iter, flag; 
   int        j, m; 
   int        my_id, num_procs;
   double     eta, chi, xi, psi, dtmp, dtmp2, r_norm, b_norm;
   double     A11, A12, A21, A22, B1, B2, omega, epsmac = 1.e-16; 
   double     epsilon, phi, delta, deltam1, omegam1;

   hypre_KrylovCommInfo(A,&my_id,&num_procs);
   if (logging > 0)
   {
      norms          = (bicgstab_data -> norms);
      log_file_name  = (bicgstab_data -> log_file_name);
   }

   /* initialize work arrays */
hypre_KrylovClearVector(x);
   hypre_KrylovCopyVector(b,r);

   /* compute initial residual */

   hypre_KrylovMatvec(matvec_data,-1.0, A, x, 1.0, r);
   r_norm = sqrt(hypre_KrylovInnerProd(r,r));
   b_norm = sqrt(hypre_KrylovInnerProd(b,b));
   if (logging > 0)
   {
      norms[0] = r_norm;
      if (my_id == 0)
      {
  	 printf("BiCGSTABL : L2 norm of b = %e\n", b_norm);
         if (b_norm == 0.0)
            printf("Rel_resid_norm actually contains the residual norm\n");
         printf("BiCGSTABL : Initial L2 norm of residual = %e\n", r_norm);
      }
      
   }
   iter = 0;

   if (b_norm > 0.0)
   {
      /* convergence criterion |r_i| <= accuracy*|b| if |b| > 0 */
      epsilon = accuracy * b_norm;
   }
   else
   {
      /* convergence criterion |r_i| <= accuracy*|r0| if |b| = 0 */
      epsilon = accuracy * r_norm;
   };

   /* convergence criterion |r_i| <= accuracy , absolute residual norm*/
   if (stop_crit) epsilon = accuracy;

   hypre_KrylovCopyVector(r,s);
   hypre_KrylovCopyVector(r,y);
   delta = hypre_KrylovInnerProd(r,y);
   precond(precond_data, A, s, t);
   hypre_KrylovMatvec(matvec_data,1.0,A,t,0.0,as);
   phi = hypre_KrylovInnerProd(y,as) / delta;
   omega = 0.0;
   
   while ( iter < max_iter && r_norm > epsilon )
   {
      iter++;

      omegam1 = omega;
      omega   = 1.0 / phi;

      if ( iter >= 2 ) 
      {
         hypre_KrylovCopyVector(awt,at);
         dtmp = - psi;
         hypre_KrylovAxpy(dtmp,asm1,at);
         hypre_KrylovCopyVector(wt,wh);
         dtmp = - omega;
         hypre_KrylovAxpy(dtmp,at,wh);
      }

      hypre_KrylovCopyVector(r,wt);
      dtmp = - omega;
      hypre_KrylovAxpy(dtmp,as,wt);

      if ( iter % 2 == 1 )
      {
         m = ( iter - 1 ) / 2;
         precond(precond_data, A, wt, t);
         hypre_KrylovMatvec(matvec_data,1.0,A,t,0.0,awt);
         dtmp = hypre_KrylovInnerProd(wt,awt);
         dtmp2 = hypre_KrylovInnerProd(awt,awt);
         chi = dtmp / dtmp2;
         hypre_KrylovCopyVector(wt,r);
         dtmp = - chi;
         hypre_KrylovAxpy(dtmp,awt,r);
         hypre_KrylovCopyVector(x,xt);
         hypre_KrylovAxpy(omega,s,x);
         hypre_KrylovAxpy(chi,wt,x);
         deltam1 = delta;
         delta = hypre_KrylovInnerProd(r,y);
         psi = - omega * delta / ( deltam1 * chi);
         hypre_KrylovCopyVector(s,st);
         hypre_KrylovCopyVector(s,t);
         dtmp = - chi;
         hypre_KrylovAxpy(dtmp,as,t);
         hypre_KrylovCopyVector(r,s);
         dtmp = - psi;
         hypre_KrylovAxpy(dtmp,t,s);
      }
      else
      {
         m = ( iter - 2 ) / 2;
         dtmp = - 1.0;
         hypre_KrylovCopyVector(wt,t2);
         hypre_KrylovAxpy(dtmp,wh,t2);
         precond(precond_data, A, wt, t);
         hypre_KrylovMatvec(matvec_data,1.0,A,t,0.0,awt);
         A11 = hypre_KrylovInnerProd(t2,t2);
         A12 = hypre_KrylovInnerProd(t2,awt);
         A21 = A12;
         A22 = hypre_KrylovInnerProd(awt,awt);
         B1  = hypre_KrylovInnerProd(t2,wh);
         B2  = hypre_KrylovInnerProd(awt,wh);
         flag = 0;
         if ( A21 > A11 )
         {
            dtmp = A11; A11 = A21; A21 = dtmp;
            dtmp = A12; A12 = A22; A22 = dtmp;
            flag = 1;
         }
         A21 = A12 / A11;
         A22 = A22 - A12 * A12 / A11;
         xi = B1; 
         eta = B2 - A21 * xi;
         eta = eta / A22;
         xi = (xi - A12 * eta) / A11;
         xi = - xi; 
         eta = -eta;
         if ( flag == 1 ) { dtmp = eta; eta = xi; xi = dtmp;}
         dtmp = 1.0 - xi;
         hypre_KrylovCopyVector(wh,r);
         hypre_KrylovScaleVector(dtmp,r);
         hypre_KrylovAxpy(xi,wt,r);
         hypre_KrylovAxpy(eta,awt,r);
         hypre_KrylovCopyVector(x,t);
         hypre_KrylovAxpy(omega,s,t);
         hypre_KrylovCopyVector(xt,x);
         hypre_KrylovAxpy(omegam1,st,x);
         hypre_KrylovAxpy(omega,tt,x);
         dtmp = 1.0 - xi;
         hypre_KrylovScaleVector(dtmp,x);
         hypre_KrylovAxpy(xi,t,x);
         dtmp = - eta;
         hypre_KrylovAxpy(dtmp,wt,x);
         deltam1 = delta;
         delta  = hypre_KrylovInnerProd(r,y);
         psi = omega * delta / ( deltam1 * eta);
         hypre_KrylovCopyVector(s,st);
         dtmp = 1.0 - xi;
         hypre_KrylovCopyVector(tt,t);
         hypre_KrylovAxpy(xi,s,t);
         hypre_KrylovAxpy(eta,as,t);
         hypre_KrylovCopyVector(r,s);
         dtmp = - psi;
         hypre_KrylovAxpy(dtmp,t,s);
      }

      hypre_KrylovCopyVector(wt,tt);
      dtmp = - psi;
      hypre_KrylovAxpy(dtmp,st,tt);
      hypre_KrylovCopyVector(as,asm1);
      precond(precond_data, A, s, t);
      hypre_KrylovMatvec(matvec_data,1.0,A,t,0.0,as);
      phi = hypre_KrylovInnerProd(as,y) / delta;
        
      precond(precond_data, A, x, t);
      hypre_KrylovMatvec(matvec_data,-1.0, A, t, 1.0, r);
      r_norm = hypre_KrylovInnerProd(r,r);
      if ( my_id == 0 && logging )
         printf(" BiCGSTAB2 : iter %4d - res. norm = %e \n", iter, r_norm);
   }
   precond(precond_data, A, x, t);
   hypre_KrylovCopyVector(t,x);

   (bicgstab_data -> num_iterations) = iter;
   if (b_norm > 0.0)
      (bicgstab_data -> rel_residual_norm) = r_norm/b_norm;
   if (b_norm == 0.0)
      (bicgstab_data -> rel_residual_norm) = r_norm;

   if (iter >= max_iter && r_norm > epsilon) ierr = 1;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_BiCGSTABLSetTol
 *--------------------------------------------------------------------------*/
 
int hypre_BiCGSTABLSetTol( void *bicgstab_vdata, double tol )
{
   hypre_BiCGSTABLData *bicgstab_data = bicgstab_vdata;
   int            ierr = 0;
 
   (bicgstab_data -> tol) = tol;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_BiCGSTABLSetMinIter
 *--------------------------------------------------------------------------*/
 
int hypre_BiCGSTABLSetSize( void *bicgstab_vdata, int size )
{
   hypre_BiCGSTABLData *bicgstab_data = bicgstab_vdata;
   int              ierr = 0;
 
   (bicgstab_data -> size) = size;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_BiCGSTABLSetMaxIter
 *--------------------------------------------------------------------------*/
 
int hypre_BiCGSTABLSetMaxIter( void *bicgstab_vdata, int max_iter )
{
   hypre_BiCGSTABLData *bicgstab_data = bicgstab_vdata;
   int              ierr = 0;
 
   (bicgstab_data -> max_iter) = max_iter;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_BiCGSTABLSetStopCrit
 *--------------------------------------------------------------------------*/
 
int hypre_BiCGSTABLSetStopCrit( void *bicgstab_vdata, double stop_crit )
{
   hypre_BiCGSTABLData *bicgstab_data = bicgstab_vdata;
   int            ierr = 0;
 
   (bicgstab_data -> stop_crit) = stop_crit;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_BiCGSTABLSetPrecond
 *--------------------------------------------------------------------------*/
 
int hypre_BiCGSTABLSetPrecond( void  *bicgstab_vdata, int  (*precond)(),
                       int  (*precond_setup)(), void  *precond_data )
{
   hypre_BiCGSTABLData *bicgstab_data = bicgstab_vdata;
   int              ierr = 0;
 
   (bicgstab_data -> precond)        = precond;
   (bicgstab_data -> precond_setup)  = precond_setup;
   (bicgstab_data -> precond_data)   = precond_data;
 
   return ierr;
}
 
/*--------------------------------------------------------------------------
 * hypre_BiCGSTABLSetLogging
 *--------------------------------------------------------------------------*/
 
int hypre_BiCGSTABLSetLogging( void *bicgstab_vdata, int logging)
{
   hypre_BiCGSTABLData *bicgstab_data = bicgstab_vdata;
   int              ierr = 0;
 
   (bicgstab_data -> logging) = logging;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_BiCGSTABLGetNumIterations
 *--------------------------------------------------------------------------*/
 
int hypre_BiCGSTABLGetNumIterations(void *bicgstab_vdata,int  *num_iterations)
{
   hypre_BiCGSTABLData *bicgstab_data = bicgstab_vdata;
   int              ierr = 0;
 
   *num_iterations = (bicgstab_data -> num_iterations);
 
   return ierr;
}
 
/*--------------------------------------------------------------------------
 * hypre_BiCGSTABLGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/
 
int hypre_BiCGSTABLGetFinalRelativeResidualNorm( void   *bicgstab_vdata,
                                         double *relative_residual_norm )
{
   hypre_BiCGSTABLData *bicgstab_data = bicgstab_vdata;
   int 		ierr = 0;
 
   *relative_residual_norm = (bicgstab_data -> rel_residual_norm);
   
   return ierr;
} 

/******************************************************************************
 ******************************************************************************
 ******************************************************************************
  haven't been verified to work yet
 *****************************************************************************/

#ifdef OLDSTUFF

/******************************************************************************
 *
 * BiCGSTABL 
 *
 *****************************************************************************/

#include "utilities/utilities.h"
#include "HYPRE.h"
#include "IJ_matrix_vector/HYPRE_IJ_mv.h"
#include "parcsr_matrix_vector/HYPRE_parcsr_mv.h"
#include "parcsr_matrix_vector/parcsr_matrix_vector.h"
#include "parcsr_linear_solvers/parcsr_linear_solvers.h"
#include "parcsr_linear_solvers/HYPRE_parcsr_ls.h"
#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_BiCGSTABLData
 *--------------------------------------------------------------------------*/

typedef struct
{
   int      size;
   int      max_iter;
   int      stop_crit;
   double   tol;
   double   rel_residual_norm;

   void  *A;
   void  *r;
   void  *rh;
   void  *rt;
   void  *rt1;
   void  *rt2;
   void  *rt3;
   void  *ut;
   void  *ut1;
   void  *ut2;
   void  *ut3;
   void  *t;
   void  *xh;

   void  *matvec_data;

   int    (*precond)();
   int    (*precond_setup)();
   void    *precond_data;

   /* log info (always logged) */
   int      num_iterations;
 
   /* additional log info (logged when `logging' > 0) */
   int      logging;
   double  *norms;
   char    *log_file_name;

} hypre_BiCGSTABLData;

/*--------------------------------------------------------------------------
 * hypre_BiCGSTABLCreate
 *--------------------------------------------------------------------------*/
 
void * hypre_BiCGSTABLCreate( )
{
   hypre_BiCGSTABLData *bicgstab_data;
 
   bicgstab_data = hypre_CTAlloc(hypre_BiCGSTABLData, 1);
 
   /* set defaults */
   (bicgstab_data -> tol)            = 1.0e-06;
   (bicgstab_data -> size)           = 2;
   (bicgstab_data -> max_iter)       = 1000;
   (bicgstab_data -> stop_crit)      = 0; /* rel. residual norm */
   (bicgstab_data -> precond)        = hypre_KrylovIdentity;
   (bicgstab_data -> precond_setup)  = hypre_KrylovIdentitySetup;
   (bicgstab_data -> precond_data)   = NULL;
   (bicgstab_data -> logging)        = 0;
   (bicgstab_data -> r)              = NULL;
   (bicgstab_data -> rh)             = NULL;
   (bicgstab_data -> rt)             = NULL;
   (bicgstab_data -> rt1)            = NULL;
   (bicgstab_data -> rt2)            = NULL;
   (bicgstab_data -> rt3)            = NULL;
   (bicgstab_data -> ut)             = NULL;
   (bicgstab_data -> ut1)            = NULL;
   (bicgstab_data -> ut2)            = NULL;
   (bicgstab_data -> ut3)            = NULL;
   (bicgstab_data -> xh)             = NULL;
   (bicgstab_data -> t)              = NULL;
   (bicgstab_data -> matvec_data)    = NULL;
   (bicgstab_data -> norms)          = NULL;
   (bicgstab_data -> log_file_name)  = NULL;
 
   return (void *) bicgstab_data;
}

/*--------------------------------------------------------------------------
 * hypre_BiCGSTABLDestroy
 *--------------------------------------------------------------------------*/
 
int hypre_BiCGSTABLDestroy( void *bicgstab_vdata )
{
   hypre_BiCGSTABLData *bicgstab_data = bicgstab_vdata;
   int i, ierr = 0;
 
   if (bicgstab_data)
   {
      if ((bicgstab_data -> logging) > 0)
      {
         hypre_TFree(bicgstab_data -> norms);
      }
 
      hypre_KrylovMatvecDestroy(bicgstab_data -> matvec_data);
 
      hypre_KrylovDestroyVector(bicgstab_data -> r);
      hypre_KrylovDestroyVector(bicgstab_data -> rh);
      hypre_KrylovDestroyVector(bicgstab_data -> rt);
      hypre_KrylovDestroyVector(bicgstab_data -> rt1);
      hypre_KrylovDestroyVector(bicgstab_data -> rt2);
      hypre_KrylovDestroyVector(bicgstab_data -> rt3);
      hypre_KrylovDestroyVector(bicgstab_data -> ut);
      hypre_KrylovDestroyVector(bicgstab_data -> ut1);
      hypre_KrylovDestroyVector(bicgstab_data -> ut2);
      hypre_KrylovDestroyVector(bicgstab_data -> ut3);
      hypre_KrylovDestroyVector(bicgstab_data -> xh);
      hypre_KrylovDestroyVector(bicgstab_data -> t);
 
      hypre_TFree(bicgstab_data);
   }
 
   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_BiCGSTABLSetup
 *--------------------------------------------------------------------------*/
 
int hypre_BiCGSTABLSetup( void *bicgstab_vdata, void *A, void *b, void *x         )
{
   hypre_BiCGSTABLData *bicgstab_data     = bicgstab_vdata;
   int            max_iter         = (bicgstab_data -> max_iter);
   int          (*precond_setup)() = (bicgstab_data -> precond_setup);
   void          *precond_data     = (bicgstab_data -> precond_data);
   int            ierr = 0;
 
   (bicgstab_data -> A) = A;
 
   /*--------------------------------------------------
    * The arguments for NewVector are important to
    * maintain consistency between the setup and
    * compute phases of matvec and the preconditioner.
    *--------------------------------------------------*/
 
   if ((bicgstab_data -> r) == NULL)
      (bicgstab_data -> r) = hypre_KrylovCreateVector(b);
   if ((bicgstab_data -> rh) == NULL)
      (bicgstab_data -> rh) = hypre_KrylovCreateVector(b);
   if ((bicgstab_data -> rt) == NULL)
      (bicgstab_data -> rt) = hypre_KrylovCreateVector(b);
   if ((bicgstab_data -> rt1) == NULL)
      (bicgstab_data -> rt1) = hypre_KrylovCreateVector(b);
   if ((bicgstab_data -> rt2) == NULL)
      (bicgstab_data -> rt2) = hypre_KrylovCreateVector(b);
   if ((bicgstab_data -> rt3) == NULL)
      (bicgstab_data -> rt3) = hypre_KrylovCreateVector(b);
   if ((bicgstab_data -> ut) == NULL)
      (bicgstab_data -> ut) = hypre_KrylovCreateVector(b);
   if ((bicgstab_data -> ut1) == NULL)
      (bicgstab_data -> ut1) = hypre_KrylovCreateVector(b);
   if ((bicgstab_data -> ut2) == NULL)
      (bicgstab_data -> ut2) = hypre_KrylovCreateVector(b);
   if ((bicgstab_data -> ut3) == NULL)
      (bicgstab_data -> ut3) = hypre_KrylovCreateVector(b);
   if ((bicgstab_data -> xh) == NULL)
      (bicgstab_data -> xh) = hypre_KrylovCreateVector(b);
   if ((bicgstab_data -> t) == NULL)
      (bicgstab_data -> t) = hypre_KrylovCreateVector(b);
 
   if ((bicgstab_data -> matvec_data) == NULL)
      (bicgstab_data -> matvec_data) = hypre_KrylovMatvecCreate(A, x);
 
   precond_setup(precond_data, A, b, x);
 
   /*-----------------------------------------------------
    * Allocate space for log info
    *-----------------------------------------------------*/
 
   if ((bicgstab_data -> logging) > 0)
   {
      if ((bicgstab_data -> norms) == NULL)
         (bicgstab_data -> norms) = hypre_CTAlloc(double, max_iter + 1);
      if ((bicgstab_data -> log_file_name) == NULL)
         (bicgstab_data -> log_file_name) = "bicgstab.out.log";
   }
 
   return ierr;
}
 
/*--------------------------------------------------------------------------
 * hypre_BiCGSTABLSolve
 *-------------------------------------------------------------------------*/

int hypre_BiCGSTABLSolve(void  *bicgstab_vdata, void  *A, void  *b, void  *x)
{
   hypre_BiCGSTABLData  *bicgstab_data   = bicgstab_vdata;
   int               size         = (bicgstab_data -> size);
   int 		     max_iter     = (bicgstab_data -> max_iter);
   int 		     stop_crit    = (bicgstab_data -> stop_crit);
   double 	     accuracy     = (bicgstab_data -> tol);
   void              *matvec_data  = (bicgstab_data -> matvec_data);
   double            mat[2][2], gammanp[2], gammap[2], sigma[2], tau[2][2];

   void             *r            = (bicgstab_data -> r);
   void             *rh           = (bicgstab_data -> rh);
   void             *rt           = (bicgstab_data -> rt);
   void             *rt1          = (bicgstab_data -> rt1);
   void             *rt2          = (bicgstab_data -> rt2);
   void             *rt3          = (bicgstab_data -> rt3);
   void             *ut           = (bicgstab_data -> ut);
   void             *ut1          = (bicgstab_data -> ut1);
   void             *ut2          = (bicgstab_data -> ut2);
   void             *ut3          = (bicgstab_data -> ut3);
   void             *xh           = (bicgstab_data -> xh);
   void             *t            = (bicgstab_data -> t);

   int 	           (*precond)()   = (bicgstab_data -> precond);
   int 	            *precond_data = (bicgstab_data -> precond_data);

   /* logging variables */
   int             logging        = (bicgstab_data -> logging);
   double         *norms          = (bicgstab_data -> norms);
   char           *log_file_name  = (bicgstab_data -> log_file_name);
   
   int        ierr = 0;
   int        iter; 
   int        j; 
   int        my_id, num_procs;
   double     alpha, beta, gamma, epsilon, rho, rho1, dtmp, r_norm, b_norm;
   double     gammapp[2], darray[2], epsmac = 1.e-16, omega; 

   hypre_KrylovCommInfo(A,&my_id,&num_procs);
   if (logging > 0)
   {
      norms          = (bicgstab_data -> norms);
      log_file_name  = (bicgstab_data -> log_file_name);
   }

   /* initialize work arrays */
hypre_KrylovClearVector(x);
   hypre_KrylovCopyVector(b,r);

   /* compute initial residual */

   hypre_KrylovMatvec(matvec_data,-1.0, A, x, 1.0, r);
   r_norm = sqrt(hypre_KrylovInnerProd(r,r));
   b_norm = sqrt(hypre_KrylovInnerProd(b,b));
   if (logging > 0)
   {
      norms[0] = r_norm;
      if (my_id == 0)
      {
  	 printf("BiCGSTABL : L2 norm of b = %e\n", b_norm);
         if (b_norm == 0.0)
            printf("Rel_resid_norm actually contains the residual norm\n");
         printf("BiCGSTABL : Initial L2 norm of residual = %e\n", r_norm);
      }
      
   }
   iter = 0;

   if (b_norm > 0.0)
   {
      /* convergence criterion |r_i| <= accuracy*|b| if |b| > 0 */
      epsilon = accuracy * b_norm;
   }
   else
   {
      /* convergence criterion |r_i| <= accuracy*|r0| if |b| = 0 */
      epsilon = accuracy * r_norm;
   };

   /* convergence criterion |r_i| <= accuracy , absolute residual norm*/
   if (stop_crit) epsilon = accuracy;

   hypre_KrylovCopyVector(r,rh);
   hypre_KrylovCopyVector(r,rt);
   hypre_KrylovCopyVector(x,xh);
   hypre_KrylovClearVector(ut);
   omega = rho = 1.0; alpha = 0.0;

   while ( iter < max_iter && r_norm > epsilon )
   {
      iter += size;

      hypre_KrylovCopyVector(ut,ut1);
      hypre_KrylovCopyVector(rt,rt1);
    
      rho = - omega * rho;    

      rho1 = hypre_KrylovInnerProd(rh,rt1);
      beta = alpha * rho1 / rho;
      rho = rho1;
      dtmp = -beta;
      hypre_KrylovScaleVector(dtmp,ut1);
      hypre_KrylovAxpy(1.0,rt1,ut1);
      precond(precond_data, A, ut1, t);
      hypre_KrylovMatvec(matvec_data,1.0,A,t,0.0,ut2);
      gamma = hypre_KrylovInnerProd(rh,ut2);
      alpha = rho / gamma; dtmp = -alpha;
      hypre_KrylovAxpy(dtmp,ut2,rt1);
      precond(precond_data, A, rt1, t);
      hypre_KrylovMatvec(matvec_data,1.0,A,t,0.0,rt2);
      hypre_KrylovAxpy(alpha,ut1,xh);

      rho1 = hypre_KrylovInnerProd(rh,rt2);
      beta = alpha * rho1 / rho;
      rho = rho1;
      dtmp = -beta;
      hypre_KrylovScaleVector(dtmp,ut1);
      hypre_KrylovAxpy(1.0,rt1,ut1);
      hypre_KrylovScaleVector(dtmp,ut2);
      hypre_KrylovAxpy(1.0,rt2,ut2);
      precond(precond_data, A, ut2, t);
      hypre_KrylovMatvec(matvec_data,1.0,A,t,0.0,ut3);
      gamma = hypre_KrylovInnerProd(rh,ut3);
      alpha = rho / gamma; dtmp = -alpha;
      hypre_KrylovAxpy(dtmp,ut2,rt1);
      hypre_KrylovAxpy(dtmp,ut3,rt2);
      precond(precond_data, A, rt2, t);
      hypre_KrylovMatvec(matvec_data,1.0,A,t,0.0,rt3);
      hypre_KrylovAxpy(alpha,ut1,xh);

      mat[0][0] = 0.0;
      mat[0][1] = 0.0;
      mat[1][0] = 0.0;
      mat[1][1] = 0.0;

      darray[0] = hypre_KrylovInnerProd(rt2,rt2);
      darray[1] = hypre_KrylovInnerProd(rt1,rt2);
      sigma[0]  = darray[0];
      mat[0][0] = sigma[0];
      gammap[0] = darray[1] / sigma[0];

      dtmp = hypre_KrylovInnerProd(rt2,rt3);
      tau[0][1] = dtmp / sigma[0];
      mat[0][1] = tau[0][1] * sigma[0];
      dtmp = -tau[0][1];
      hypre_KrylovAxpy(dtmp,rt2,rt3);
      darray[0] = hypre_KrylovInnerProd(rt3,rt3);
      darray[1] = hypre_KrylovInnerProd(rt1,rt3);
      sigma[1]  = darray[0];
      mat[1][1] = sigma[1];
      gammap[1] = darray[1] / sigma[1];

      gammanp[1] = gammap[1];
      omega = gammanp[1];
      gammanp[0] = gammap[0];
      gammanp[0] = gammanp[0] - tau[0][1] * gammanp[1];
      gammapp[0] = gammanp[1];

      dtmp = gammanp[0];
      hypre_KrylovAxpy(dtmp,rt1,xh);
      dtmp = - gammap[1];
      hypre_KrylovAxpy(dtmp,rt3,rt1);
      dtmp = - gammanp[1];
      hypre_KrylovAxpy(dtmp,ut3,ut1);
      dtmp = - gammanp[0];
      hypre_KrylovAxpy(dtmp,ut2,ut1);
      dtmp = gammapp[0];
      hypre_KrylovAxpy(dtmp,rt2,xh);
      dtmp = - gammap[0];
      hypre_KrylovAxpy(dtmp,rt2,rt1);

      hypre_KrylovCopyVector(ut1,ut);
      hypre_KrylovCopyVector(rt1,rt);
      hypre_KrylovCopyVector(xh,x);

      precond(precond_data, A, x, t);
      hypre_KrylovMatvec(matvec_data,-1.0, A, t, 1.0, r);
      r_norm = hypre_KrylovInnerProd(r,r);
      if ( my_id == 0 && logging )
         printf(" BiCGSTABL : iter %4d - res. norm = %e \n", iter, r_norm);
   }
   precond(precond_data, A, x, t);
   hypre_KrylovCopyVector(t,x);

   (bicgstab_data -> num_iterations) = iter;
   if (b_norm > 0.0)
      (bicgstab_data -> rel_residual_norm) = r_norm/b_norm;
   if (b_norm == 0.0)
      (bicgstab_data -> rel_residual_norm) = r_norm;

   if (iter >= max_iter && r_norm > epsilon) ierr = 1;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_BiCGSTABLSetTol
 *--------------------------------------------------------------------------*/
 
int hypre_BiCGSTABLSetTol( void *bicgstab_vdata, double tol )
{
   hypre_BiCGSTABLData *bicgstab_data = bicgstab_vdata;
   int            ierr = 0;
 
   (bicgstab_data -> tol) = tol;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_BiCGSTABLSetMinIter
 *--------------------------------------------------------------------------*/
 
int hypre_BiCGSTABLSetSize( void *bicgstab_vdata, int size )
{
   hypre_BiCGSTABLData *bicgstab_data = bicgstab_vdata;
   int              ierr = 0;
 
   (bicgstab_data -> size) = size;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_BiCGSTABLSetMaxIter
 *--------------------------------------------------------------------------*/
 
int hypre_BiCGSTABLSetMaxIter( void *bicgstab_vdata, int max_iter )
{
   hypre_BiCGSTABLData *bicgstab_data = bicgstab_vdata;
   int              ierr = 0;
 
   (bicgstab_data -> max_iter) = max_iter;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_BiCGSTABLSetStopCrit
 *--------------------------------------------------------------------------*/
 
int hypre_BiCGSTABLSetStopCrit( void *bicgstab_vdata, double stop_crit )
{
   hypre_BiCGSTABLData *bicgstab_data = bicgstab_vdata;
   int            ierr = 0;
 
   (bicgstab_data -> stop_crit) = stop_crit;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_BiCGSTABLSetPrecond
 *--------------------------------------------------------------------------*/
 
int hypre_BiCGSTABLSetPrecond( void  *bicgstab_vdata, int  (*precond)(),
                       int  (*precond_setup)(), void  *precond_data )
{
   hypre_BiCGSTABLData *bicgstab_data = bicgstab_vdata;
   int              ierr = 0;
 
   (bicgstab_data -> precond)        = precond;
   (bicgstab_data -> precond_setup)  = precond_setup;
   (bicgstab_data -> precond_data)   = precond_data;
 
   return ierr;
}
 
/*--------------------------------------------------------------------------
 * hypre_BiCGSTABLSetLogging
 *--------------------------------------------------------------------------*/
 
int hypre_BiCGSTABLSetLogging( void *bicgstab_vdata, int logging)
{
   hypre_BiCGSTABLData *bicgstab_data = bicgstab_vdata;
   int              ierr = 0;
 
   (bicgstab_data -> logging) = logging;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_BiCGSTABLGetNumIterations
 *--------------------------------------------------------------------------*/
 
int hypre_BiCGSTABLGetNumIterations(void *bicgstab_vdata,int  *num_iterations)
{
   hypre_BiCGSTABLData *bicgstab_data = bicgstab_vdata;
   int              ierr = 0;
 
   *num_iterations = (bicgstab_data -> num_iterations);
 
   return ierr;
}
 
/*--------------------------------------------------------------------------
 * hypre_BiCGSTABLGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/
 
int hypre_BiCGSTABLGetFinalRelativeResidualNorm( void   *bicgstab_vdata,
                                         double *relative_residual_norm )
{
   hypre_BiCGSTABLData *bicgstab_data = bicgstab_vdata;
   int 		ierr = 0;
 
   *relative_residual_norm = (bicgstab_data -> rel_residual_norm);
   
   return ierr;
} 

#endif

