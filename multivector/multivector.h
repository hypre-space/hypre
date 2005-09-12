#ifndef MULTIVECTOR_FUNCTION_PROTOTYPES
#define MULTIVECTOR_FUNCTION_PROTOTYPES

#include "interpreter.h"
#include "utilities.h"

/* abstract multivector */
typedef struct mv_MultiVector* mv_MultiVectorPtr;

/* The functions below simply call the respective functions pointed to
   in the HYPRE_InterfaceInterpreter structure */

#ifdef __cplusplus
extern "C" {
#endif

/* returns pointer to "real" multivector data (x->data) */
void *
mv_MultiVectorGetData (mv_MultiVectorPtr x);

  /* wraps our multivector structure around the data provided by user */
mv_MultiVectorPtr
mv_MultiVectorWrap( mv_InterfaceInterpreter* ii, void * data, int ownsData );

  /* creates a multivector of width n using sample vector */
mv_MultiVectorPtr 
mv_MultiVectorCreateFromSampleVector( void*, int n, void* sample );

  /* creates a multivector of the same shape as x; copies values
     if copyValues is non-zero */
mv_MultiVectorPtr 
mv_MultiVectorCreateCopy( mv_MultiVectorPtr x, int copyValues );

void 
mv_MultiVectorDestroy( mv_MultiVectorPtr );

int
mv_MultiVectorWidth( mv_MultiVectorPtr v );

int
mv_MultiVectorHeight( mv_MultiVectorPtr v );

  /* sets mask for v; all the subsequent operations
     apply only to masked vectors */
void
mv_MultiVectorSetMask( mv_MultiVectorPtr v, int* mask );

void 
mv_MultiVectorClear( mv_MultiVectorPtr );

void 
mv_MultiVectorSetRandom( mv_MultiVectorPtr v, int seed );

void 
mv_MultiVectorCopy( mv_MultiVectorPtr src, mv_MultiVectorPtr dest );

  /* computes y = a*x + y */
void 
mv_MultiVectorAxpy( double a, mv_MultiVectorPtr x, mv_MultiVectorPtr y ); 

  /* computes the matrix v = x'*y stored in fortran style: gh is the leading dimension,
     h the number of rows and w the number of columns (cf. blas or lapack) */
void 
mv_MultiVectorByMultiVector( mv_MultiVectorPtr x, mv_MultiVectorPtr y,
				int gh, int h, int w, double* v );

  /*computes the diagonal of x'*y stored in diag(mask) */
void 
mv_MultiVectorByMultiVectorDiag( mv_MultiVectorPtr, mv_MultiVectorPtr,
				   int* mask, int n, double* diag );

  /* computes y = x*v, where v is stored in fortran style */
void 
mv_MultiVectorByMatrix( mv_MultiVectorPtr x, 
			   int gh, int h, int w, double* v,
			   mv_MultiVectorPtr y );

  /* computes y = x*v + y, where v is stored in fortran style */
void 
mv_MultiVectorXapy( mv_MultiVectorPtr x, 
		       int gh, int h, int w, double* v,
		       mv_MultiVectorPtr y );

  /* computes y = x*diag(mask) */
void mv_MultiVectorByDiagonal( mv_MultiVectorPtr x, 
				  int* mask, int n, double* diag,
				  mv_MultiVectorPtr y );

  /* computes y = f(x) vector-by-vector */
void 
mv_MultiVectorEval( void (*f)( void*, void*, void* ), 
		       void* par,
		       mv_MultiVectorPtr x, 
		       mv_MultiVectorPtr y );

#ifdef __cplusplus
}
#endif

#endif /* MULTIVECTOR_FUNCTION_PROTOTYPES */
