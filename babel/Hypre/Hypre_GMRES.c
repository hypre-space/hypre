
/******************************************************
 *
 *  File:  Hypre_GMRES.c
 *
 *********************************************************/

#include "Hypre_GMRES_Skel.h" 
#include "Hypre_GMRES_Data.h" 
#include "Hypre_GMRES_Stub.h"
#include "math.h"
#include "utilities.h"
#include "Hypre_MPI_Com_Skel.h"
#include "Hypre_MPI_Com_Data.h"

/* *************************************************
 * Constructor
 *    Allocate Memory for private data
 *    and initialize here
 ***************************************************/
void Hypre_GMRES_constructor(Hypre_GMRES this) {
   this->Hypre_GMRES_data = hypre_CTAlloc( struct Hypre_GMRES_private_type, 1 );
   /* Setup allocates p,r,w,norms, log_file_name.  Calling functions
    allocate matvec and preconditioner */
} /* end constructor */

/* *************************************************
 *  Destructor
 *      deallocate memory for private data here.
 ***************************************************/
void Hypre_GMRES_destructor(Hypre_GMRES this) {
   hypre_TFree( this->Hypre_GMRES_data );

   return;
} /* end destructor */

/* ********************************************************
 * impl_Hypre_GMRESGetParameterDouble
 **********************************************************/
int impl_Hypre_GMRES_GetParameterDouble
( Hypre_GMRES this, char* name, double *value ) {
   Hypre_GMRES_Private gmr_data = Hypre_GMRES_getPrivate(this);

   if ( !strcmp( name, "tol" ) ) {
      *value = gmr_data->tol;
      return 0;
   }
   else {
      printf( "Don't understand keyword %s to Hypre_GMRES_GetParameterDouble\n",
              name );
      *value = -123.456;
      return 1;
   }
} /* end impl_Hypre_GMRESGetParameterDouble */

/* ********************************************************
 * impl_Hypre_GMRESGetParameterInt
 **********************************************************/
int  impl_Hypre_GMRES_GetParameterInt(Hypre_GMRES this, char* name, int *value) {
   Hypre_GMRES_Private gmr_data = Hypre_GMRES_getPrivate(this);

   if ( !strcmp( name, "k_dim" ) || !strcmp( name, "k dim" ) ) {
      *value = gmr_data->k_dim;
      return 0;
   }
   else if ( !strcmp( name, "min_iter" ) || !strcmp( name, "min iter" ) ) {
      *value = gmr_data->min_iter;
      return 0;
   }
   else if ( !strcmp( name, "max_iter" ) || !strcmp( name, "max iter" ) ) {
      *value = gmr_data->max_iter;
      return 0;
   }
   else if ( !strcmp( name, "logging" ) ) {
      *value = gmr_data->logging;
      return 0;
   }
   else if
      ( !strcmp( name, "stop_crit" ) || !strcmp( name, "stop crit" )
        || !strcmp( name, "stopping criterion" ) ) {
      *value = gmr_data->stop_crit;
      return 0;
   }
   else {
      printf( "Don't understand keyword %s to Hypre_GMRES_GetParameterInt\n",
              name );
      *value = -123456;
      return 1;
   }
} /* end impl_Hypre_GMRESGetParameterInt */


/* ********************************************************
 * impl_Hypre_GMRES_SetParameterDouble
 **********************************************************/
int  
impl_Hypre_GMRES_SetParameterDouble(Hypre_GMRES this, char* name, double value)
{
   Hypre_GMRES_Private gmr_data = Hypre_GMRES_getPrivate(this);

   if ( !strcmp( name, "tol" ) ) {
      gmr_data->tol = value;
      return 0;
   }
   else {
      printf( "Don't understand keyword %s to Hypre_GMRES_SetParameterDouble\n",
              name );
      return 1;
   }
} /* end impl_Hypre_GMRES_SetParameterDouble */


/* ********************************************************
 * impl_Hypre_GMRES_SetParameterInt
 **********************************************************/
int  
impl_Hypre_GMRES_SetParameterInt( Hypre_GMRES this, char* name, int value ) 
{
   Hypre_GMRES_Private gmr_data = Hypre_GMRES_getPrivate(this);

   if ( !strcmp( name, "k_dim" ) || !strcmp( name, "k dim" ) ) {
      return gmr_data->k_dim;
   }
   else if ( !strcmp( name, "min_iter" ) || !strcmp( name, "min iter" ) ) {
      return gmr_data->min_iter;
   }
   else if ( !strcmp( name, "max_iter" ) || !strcmp( name, "max iter" ) ) {
      return gmr_data->max_iter;
   }
   else if ( !strcmp( name, "logging" ) ) {
      return gmr_data->logging;
   }
   else if
      ( !strcmp( name, "stop_crit" ) || !strcmp( name, "stop crit" )
        || !strcmp( name, "stopping criterion" ) ) {
      return gmr_data->stop_crit;
   }
   else {
      printf( "Don't understand keyword %s to Hypre_GMRES_SetParameterInt\n",
              name );
      return 1;
   }
} /* end impl_Hypre_GMRES_SetParameterInt */


/* ********************************************************
 * impl_Hypre_GMRESSetParameterString
 **********************************************************/
int  impl_Hypre_GMRES_SetParameterString
( Hypre_GMRES this, char* name, char* value ) {
   printf("Don't understand keyword %s to Hypre_GMRES_SetParameterString\n",
          name );
} /* end impl_Hypre_GMRESSetParameterString */

/* ********************************************************
 * impl_Hypre_GMRESSetPreconditioner
 **********************************************************/
int  impl_Hypre_GMRES_SetPreconditioner
( Hypre_GMRES this, Hypre_Solver precond ) {
   Hypre_GMRES_Private gmr_data = Hypre_GMRES_getPrivate(this);

   gmr_data->preconditioner = precond;

   return 0;
   
} /* end impl_Hypre_GMRESSetPreconditioner */


/* ********************************************************
 * impl_Hypre_GMRESGetPreconditioner
 **********************************************************/
int impl_Hypre_GMRES_GetPreconditioner
( Hypre_GMRES this, Hypre_Solver* precond ) {
   Hypre_GMRES_Private gmr_data = Hypre_GMRES_getPrivate(this);
   *precond = gmr_data->preconditioner;
   return 0;
} /* end impl_Hypre_PCGGetPreconditioner */

/* ********************************************************
 * impl_Hypre_GMRES_Start
 **********************************************************/
int  
impl_Hypre_GMRES_Start( Hypre_GMRES this, Hypre_MPI_Com comm )
{
   Hypre_GMRES_Private gmr_data = Hypre_GMRES_getPrivate(this);

   (gmr_data -> k_dim)          = 5;
   (gmr_data -> tol)            = 1.0e-06;
   (gmr_data -> min_iter)       = 0;
   (gmr_data -> max_iter)       = 1000;
   (gmr_data -> stop_crit)      = 0; /* rel. residual norm */
   (gmr_data -> logging)        = 0;
   (gmr_data -> p)              = NULL;
   (gmr_data -> r)              = NULL;
   (gmr_data -> w)              = NULL;
   (gmr_data -> norms)          = NULL;
   (gmr_data -> log_file_name)  = NULL;
   (gmr_data -> comm )          = comm;

   return 0;
} /* end impl_Hypre_GMRES_Start */


/* ********************************************************
 * impl_Hypre_GMRESConstructor
 **********************************************************/
Hypre_GMRES  impl_Hypre_GMRES_Constructor(Hypre_MPI_Com comm) {
   Hypre_GMRES gmr = Hypre_GMRES_New();
   Hypre_GMRES_Start( gmr, comm );
   return gmr;
} /* end impl_Hypre_GMRESConstructor */


/* ********************************************************
 * impl_Hypre_GMRES_Setup
 **********************************************************/
int  
impl_Hypre_GMRES_Setup( Hypre_GMRES this,
                        Hypre_LinearOperator A,
                        Hypre_Vector b,
                        Hypre_Vector x )
{
   Hypre_GMRES_Private gmr_data = Hypre_GMRES_getPrivate(this);
   Hypre_Vector *vp;
   int ierr = 0;
   int max_iter = (gmr_data -> max_iter);
   int k_dim = gmr_data->k_dim;
   int i;

   /* Save LinearOperator */

   gmr_data->matvec = A;

   /* Create temporary vectors */

   if ((gmr_data->p) == NULL)
   {
      vp = hypre_CTAlloc( Hypre_Vector, k_dim+1 );
      /* ... This works because a Hypre_Vector is really just a pointer.
       If we needed to allocate space for the actual underlying objects,
       allocating something involving Hypre_Vector's couldn't work.*/
      for (i=0; i < k_dim+1; i++)
      {	
         ierr += Hypre_Vector_Clone (x, &(vp[i])); 
      }
      gmr_data->p = vp;
   }

   if ((gmr_data->r) == NULL)
   {
      ierr += Hypre_Vector_Clone (b, vp);
      gmr_data->r = *vp;
   }

   if ((gmr_data->w) == NULL)
   {
      ierr += Hypre_Vector_Clone (b, vp);
      gmr_data->w = *vp;
   }

   /* Allocate space for log info */

   if ((gmr_data -> logging) > 0)
   {
      if ((gmr_data -> norms) == NULL)
         (gmr_data -> norms) = hypre_CTAlloc(double, max_iter + 1);
      if ((gmr_data -> log_file_name) == NULL)
         (gmr_data -> log_file_name) = "gmres.out.log";
   }

   return ierr;
} /* end impl_Hypre_GMRES_Setup */


/* ********************************************************
 * impl_Hypre_GMRESGetConstructedObject
 **********************************************************/
int impl_Hypre_GMRES_GetConstructedObject(Hypre_GMRES this, Hypre_Solver* obj) {
   *obj = (Hypre_Solver) Hypre_GMRES_castTo( this, "Hypre_Solver" );
   return 0;
} /* end impl_Hypre_GMRESGetConstructedObject */


/* ********************************************************
 * impl_Hypre_GMRES_Apply
 **********************************************************/
int  
impl_Hypre_GMRES_Apply(Hypre_GMRES this, Hypre_Vector b, Hypre_Vector* xp) 
{
   Hypre_GMRES_Private gmr_data = Hypre_GMRES_getPrivate (this);

   Hypre_Vector x = *xp;

   int 		     k_dim        = (gmr_data -> k_dim);
   int               min_iter     = (gmr_data -> min_iter);
   int 		     max_iter     = (gmr_data -> max_iter);
   int 		     stop_crit    = (gmr_data -> stop_crit);
   double 	     accuracy     = (gmr_data -> tol);

   Hypre_Vector      r            = (gmr_data -> r);
   Hypre_Vector      w            = (gmr_data -> w);
   Hypre_Vector     *p            = (gmr_data -> p);

   Hypre_LinearOperator  matvec   = (gmr_data -> matvec);
   Hypre_Solver          precond  = (gmr_data -> preconditioner);

   /* logging variables */
   int             logging        = (gmr_data -> logging);
   double         *norms          = (gmr_data -> norms);
   char           *log_file_name  = (gmr_data -> log_file_name);
   /* FILE           *fp; */

   
   int        ierr = 0;
   int	      i, j, k;
   double     *rs, **hh, *c, *s;
   int        iter; 
   int        my_id;
   double     epsilon, gamma, t, r_norm, b_norm;
   double     epsmac = 1.e-16; 

   /* 
      This hardwired call to MPI_Comm_rank is correct only if my 
      communicator is MPI_COMM_WORLD.
      Should find rank with respect to the communicator associated with
      this solver, but

        (1) we don't currently save that communicator when it's passed
		to Start().  This is easily fixed. (jfp: done, 26apr00)
        (2) we need a GetCommunicator method, to ask a Solver what
		communicator it's associated with.  Maybe this method
		needs to be inherited from higher up, like from Hypre_Object
   MPI_Comm_rank (MPI_COMM_WORLD, &my_id);
   */
   /* >>>>> TO DO: put a corresponding function in the MPI_Com interface;
      call that, so we don't have to "know" what's in a MPI_Com's data table ... */
   MPI_Comm_rank( *(gmr_data->comm->Hypre_MPI_Com_data->hcom), &my_id );

   if (logging > 0)
   {
      norms          = (gmr_data -> norms);
      log_file_name  = (gmr_data -> log_file_name);
      /* fp = fopen(log_file_name,"w"); */
   }

   /* allocate work arrays */
   rs = hypre_CTAlloc(double,k_dim+1); 
   c = hypre_CTAlloc(double,k_dim); 
   s = hypre_CTAlloc(double,k_dim); 

   hh = hypre_CTAlloc(double*,k_dim+1); 
   for (i=0; i < k_dim+1; i++)
   {	
   	hh[i] = hypre_CTAlloc(double,k_dim); 
   }

/* compute initial residual */

   ierr += Hypre_LinearOperator_Apply (matvec, x, &p[0]);
   ierr += Hypre_Vector_Scale (p[0], -1.0);
   ierr += Hypre_Vector_Axpy (p[0], 1.0, b);

   ierr += Hypre_Vector_Dot (p[0], p[0], &r_norm);
   r_norm = sqrt(r_norm);
   ierr += Hypre_Vector_Dot (b, b, &b_norm);
   b_norm = sqrt(b_norm);
   if (logging > 0)
   {
      norms[0] = r_norm;
      if (my_id == 0)
      {
  	 printf("L2 norm of b: %e\n", b_norm);
         if (b_norm == 0.0)
            printf("Rel_resid_norm actually contains the residual norm\n");
         printf("Initial L2 norm of residual: %e\n", r_norm);
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
   if (stop_crit)
      epsilon = accuracy;

   while (iter < max_iter)
   {
   /* initialize first term of hessenberg system */

	rs[0] = r_norm;
        if (r_norm == 0.0)
        {
	   ierr = 0;
	   return ierr;
	}

	if (r_norm <= epsilon && iter >= min_iter) 
        {
                ierr += Hypre_LinearOperator_Apply (matvec, x, &r);
                ierr += Hypre_Vector_Scale (r, -1.0);
                ierr += Hypre_Vector_Axpy (r, 1.0, b);

                ierr += Hypre_Vector_Dot (r, r, &r_norm);
		r_norm = sqrt(r_norm);
		if (r_norm <= epsilon)
                {
                  if (logging > 0 && my_id == 0)
                     printf("Final L2 norm of residual: %e\n\n", r_norm);
                  break;
                }
	}

      	t = 1.0 / r_norm;
        ierr += Hypre_Vector_Scale (p[0], t);
	i = 0;
	while (i < k_dim && (r_norm > epsilon || iter < min_iter)
                         && iter < max_iter)
	{
		i++;
		iter++;
		ierr += Hypre_Vector_Clear (r);
		ierr += Hypre_Solver_Apply (precond, p[i-1], &r);
		ierr += Hypre_LinearOperator_Apply (matvec, r, &p[i]);
		/* modified Gram_Schmidt */
		for (j=0; j < i; j++)
		{
                   ierr += Hypre_Vector_Dot (p[j], p[i], &hh[j][i-1]);
                   ierr += Hypre_Vector_Axpy (p[i], -hh[j][i-1], p[j]);
		}
		ierr += Hypre_Vector_Dot (p[i], p[i], &t);
		t = sqrt(t);
		hh[i][i-1] = t;	
		if (t != 0.0)
		{
			t = 1.0/t;
			Hypre_Vector_Scale (p[i], t);
		}
		/* done with modified Gram_schmidt and Arnoldi step.
		   update factorization of hh */
		for (j = 1; j < i; j++)
		{
			t = hh[j-1][i-1];
			hh[j-1][i-1] = c[j-1]*t + s[j-1]*hh[j][i-1];		
			hh[j][i-1] = -s[j-1]*t + c[j-1]*hh[j][i-1];
		}
		gamma = sqrt(hh[i-1][i-1]*hh[i-1][i-1] + hh[i][i-1]*hh[i][i-1]);
		if (gamma == 0.0) gamma = epsmac;
		c[i-1] = hh[i-1][i-1]/gamma;
		s[i-1] = hh[i][i-1]/gamma;
		rs[i] = -s[i-1]*rs[i-1];
		rs[i-1] = c[i-1]*rs[i-1];
		/* determine residual norm */
		hh[i-1][i-1] = c[i-1]*hh[i-1][i-1] + s[i-1]*hh[i][i-1];
		r_norm = fabs(rs[i]);
		if (logging > 0)
		{
		   norms[iter] = r_norm;
		}
	}
	/* now compute solution, first solve upper triangular system */
	
	rs[i-1] = rs[i-1]/hh[i-1][i-1];
	for (k = i-2; k >= 0; k--)
	{
		t = rs[k];
		for (j = k+1; j < i; j++)
		{
			t -= hh[k][j]*rs[j];
		}
		rs[k] = t/hh[k][k];
	}
	/* form linear combination of p's to get solution */
	
	ierr += Hypre_Vector_Copy (w, p[0]);
	ierr += Hypre_Vector_Scale (w, rs[0]);
	for (j = 1; j < i; j++)
		ierr += Hypre_Vector_Axpy (w, rs[j], p[j]);

	ierr += Hypre_Vector_Clear (r);
	ierr += Hypre_Solver_Apply (precond, w, &r);

	ierr += Hypre_Vector_Axpy (x, 1.0, r);

/* check for convergence, evaluate actual residual */
	if (r_norm <= epsilon && iter >= min_iter) 
        {
                ierr += Hypre_LinearOperator_Apply (matvec, x, &r);
                ierr += Hypre_Vector_Scale (r, -1.0);
                ierr += Hypre_Vector_Axpy (r, 1.0, b);

		ierr += Hypre_Vector_Dot( r, r, &r_norm );
		r_norm = sqrt(r_norm);
		if (r_norm <= epsilon)
                {
                  if (logging > 0 && my_id == 0)
                     printf("Final L2 norm of residual: %e\n\n", r_norm);
                  break;
                }
		else
		{
		   ierr += Hypre_Vector_Copy (p[0],r);
		   i = 0;
		}
	}

/* compute residual vector and continue loop */

	for (j=i ; j > 0; j--)
	{
		rs[j-1] = -s[j-1]*rs[j];
		rs[j] = c[j-1]*rs[j];
	}

	if (i) ierr += Hypre_Vector_Axpy (p[0], rs[0]-1.0, p[0]);
	for (j=1; j < i+1; j++)
		ierr += Hypre_Vector_Axpy (p[0], rs[j], p[j]);	
   }

   if (logging > 0 && my_id == 0)
   {
      if (b_norm > 0.0)
         {printf("=============================================\n\n");
          printf("Iters     resid.norm     conv.rate  rel.res.norm\n");
          printf("-----    ------------    ---------- ------------\n");
      
          for (j = 1; j <= iter; j++)
          {
             printf("% 5d    %e    %f   %e\n", j, norms[j],norms[j]/norms[j-1],
 	             norms[j]/b_norm);
          }
          printf("\n\n"); }

      else
         {printf("=============================================\n\n");
          printf("Iters     resid.norm     conv.rate\n");
          printf("-----    ------------    ----------\n");
      
          for (j = 1; j <= iter; j++)
          {
             printf("% 5d    %e    %f\n", j, norms[j],norms[j]/norms[j-1]);
          }
          printf("\n\n"); };
   }

   (gmr_data -> num_iterations) = iter;
   if (b_norm > 0.0)
      (gmr_data -> rel_residual_norm) = r_norm/b_norm;
   if (b_norm == 0.0)
      (gmr_data -> rel_residual_norm) = r_norm;

   if (iter >= max_iter && r_norm > epsilon) ierr = 1;

   hypre_TFree(c); 
   hypre_TFree(s); 
   hypre_TFree(rs);
 
   for (i=0; i < k_dim+1; i++)
   {	
   	hypre_TFree(hh[i]);
   }
   hypre_TFree(hh); 

   return ierr;

} /* end impl_Hypre_GMRES_Apply */


/* ********************************************************
 * impl_Hypre_GMRESGetDims
 **********************************************************/
int  impl_Hypre_GMRES_GetDims(Hypre_GMRES this, int* m, int* n) {
   Hypre_GMRES_Private GMRES_data = Hypre_GMRES_getPrivate(this);
   Hypre_LinearOperator matvec = (GMRES_data->matvec);
   return Hypre_LinearOperator_GetDims( matvec, m, n );
} /* end impl_Hypre_GMRESGetDims */

/* ********************************************************
 * impl_Hypre_GMRESGetSystemOperator
 **********************************************************/
int impl_Hypre_GMRES_GetSystemOperator
( Hypre_GMRES this, Hypre_LinearOperator *op ) {
   Hypre_GMRES_Private gmr_data = Hypre_GMRES_getPrivate(this);
   *op = (gmr_data->matvec);
   return 0;
} /* end impl_Hypre_GMRESGetSystemOperator */

/* ********************************************************
 * impl_Hypre_GMRESGetResidual
 **********************************************************/
int impl_Hypre_GMRES_GetResidual( Hypre_GMRES this, Hypre_Vector* resid ) {
   Hypre_GMRES_Private gmr_data = Hypre_GMRES_getPrivate (this);
   *resid = gmr_data->r;
   return 0;
} /* end impl_Hypre_GMRESGetResidual */


/* ********************************************************
 * impl_Hypre_GMRESGetConvergenceInfo
 **********************************************************/
int  impl_Hypre_GMRES_GetConvergenceInfo
( Hypre_GMRES this, char* name, double* value ) {
   int ivalue;
   Hypre_GMRES_Private gmr_data = Hypre_GMRES_getPrivate (this);

   if ( !strcmp(name,"number of iterations") ) {
      *value = gmr_data->num_iterations;
      return 0;
   }
   else if ( !strcmp(name,"relative residual norm") ) {
      *value = gmr_data->rel_residual_norm;
      return 0;
   }
   else if ( !strcmp(name,"residual norm") ) {
      if (gmr_data->logging > 0) {
         *value = gmr_data->norms[gmr_data->num_iterations];
         return 0;
      }
      else {
         *value = -1;
         return 1;
      }
   }
   else {
      printf(
         "Don't understand keyword %s to Hypre_GMRES_GetConvergenceInfo\n",
         name );
      *value = 0;
      return 1;
   }

} /* end impl_Hypre_GMRESGetConvergenceInfo */

