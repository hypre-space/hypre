#ifndef MULTIVECTOR_FUNCTION_PROTOTYPES
#define MULTIVECTOR_FUNCTION_PROTOTYPES

#include "utilities.h"

#include "HYPRE_interpreter.h"

/* abstract multivector */
typedef struct
{
  void*	data;      /* the pointer to the actual multivector */
  int	ownsData;

  HYPRE_InterfaceInterpreter* interpreter; /* a structure that defines
					      multivector operations */
  
} hypre_MultiVector;

typedef struct hypre_MultiVector* hypre_MultiVectorPtr;

/* The functions below simply call the respective functions pointed to
   in the HYPRE_InterfaceInterpreter structure */

#ifdef __cplusplus
extern "C" {
#endif

  /* creates a multivector of width n using sample vector */
hypre_MultiVectorPtr 
hypre_MultiVectorCreateFromSampleVector( void*, int n, void* sample );

  /* creates a multivector of the same shape as x; copies values
     if copyValues is non-zero */
hypre_MultiVectorPtr 
hypre_MultiVectorCreateCopy( hypre_MultiVectorPtr x, int copyValues );

void 
hypre_MultiVectorDestroy( hypre_MultiVectorPtr );

int
hypre_MultiVectorWidth( hypre_MultiVectorPtr v );

int
hypre_MultiVectorHeight( hypre_MultiVectorPtr v );

  /* sets mask for v; all the subsequent operations exept Print 
     apply only to masked vectors */
void
hypre_MultiVectorSetMask( hypre_MultiVectorPtr v, int* mask );

void 
hypre_MultiVectorClear( hypre_MultiVectorPtr );

void 
hypre_MultiVectorSetRandom( hypre_MultiVectorPtr v, int seed );

void 
hypre_MultiVectorCopy( hypre_MultiVectorPtr src, hypre_MultiVectorPtr dest );

  /* computes y = a*x + y */
void 
hypre_MultiVectorAxpy( double a, hypre_MultiVectorPtr x, hypre_MultiVectorPtr y ); 

  /* computes the matrix v = x'*y stored in fortran style: gh is the leading dimension,
     h the number of rows and w the number of columns (cf. blas or lapack) */
void 
hypre_MultiVectorByMultiVector( hypre_MultiVectorPtr x, hypre_MultiVectorPtr y,
				int gh, int h, int w, double* v );

  /*computes the diagonal of x'*y stored in diag(mask) */
void 
hypre_MultiVectorByMultiVectorDiag( hypre_MultiVectorPtr, hypre_MultiVectorPtr,
				   int* mask, int n, double* diag );

  /* computes y = x*v, where v is stored in fortran style */
void 
hypre_MultiVectorByMatrix( hypre_MultiVectorPtr x, 
			   int gh, int h, int w, double* v,
			   hypre_MultiVectorPtr y );

  /* computes y = x*v + y, where v is stored in fortran style */
void 
hypre_MultiVectorXapy( hypre_MultiVectorPtr x, 
		       int gh, int h, int w, double* v,
		       hypre_MultiVectorPtr y );

  /* computes y = x*diag(mask) */
void hypre_MultiVectorByDiagonal( hypre_MultiVectorPtr x, 
				  int* mask, int n, double* diag,
				  hypre_MultiVectorPtr y );

  /* computes y = f(x) vector-by-vector */
void 
hypre_MultiVectorEval( void (*f)( void*, void*, void* ), 
		       void* par,
		       hypre_MultiVectorPtr x, 
		       hypre_MultiVectorPtr y );

int
hypre_MultiVectorPrint( hypre_MultiVectorPtr x, const char* fileName );

hypre_MultiVectorPtr 
hypre_MultiVectorRead( MPI_Comm comm, void*, const char* fileName );

#ifdef __cplusplus
}
#endif

#endif /* MULTIVECTOR_FUNCTION_PROTOTYPES */

