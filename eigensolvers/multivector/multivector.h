#ifndef MULTIVECTOR_FUNCTION_PROTOTYPES
#define MULTIVECTOR_FUNCTION_PROTOTYPES

#include "utilities.h"

#include "HYPRE_interpreter.h"

typedef struct
{
  long		numVectors;
  void**	vector;
  int		ownsVectors;
  
  HYPRE_InterfaceInterpreter* interpreter;
  
} hypre_MultiVector;

typedef struct hypre_MultiVector* hypre_MultiVectorPtr;

/*******************************************************************/
/*
The above is a temporary implementation of the hypre_MultiVector
data type, just to get things going with LOBPCG eigensolver.

A more proper implementation would be to define hypre_MultiParVector,
hypre_MultiStructVector and hypre_MultiSStructVector by adding a new 
record

int numVectors;

in hypre_ParVector, hypre_StructVector and hypre_SStructVector,
and increasing the size of data numVectors times. Respective
modifications of most vector operations are straightforward
(it is strongly suggested that BLAS routines are used wherever
possible), efficient implementation of matrix-by-multivector 
multiplication may be more difficult.

With the above implementation of hypre vectors, the definition
of hypre_MultiVector becomes simply

typedef struct
{
  void*	multiVector;
  HYPRE_InterfaceInterpreter* interpreter;  
} hypre_MultiVector;

with pointers to abstract multivector functions added to the structure
HYPRE_InterfaceInterpreter (cf. HYPRE_interpreter.h; particular values
are assigned to these pointers by functions 
HYPRE_ParCSRSetupInterpreter, HYPRE_StructSetupInterpreter and
int HYPRE_SStructSetupInterpreter),
and the multivector functions below become simply interfaces
of the form

void 
hypre_MultiVectorCopy( int* ms, hypre_MultiVectorPtr src,
int* md, hypre_MultiVectorPtr dest ) 
{
  (src->interpreter->MultiVectorCopy)( ms, src->multiVector, 
				       md, dest->multiVector );
}
*/
/*********************************************************************/

#ifdef __cplusplus
extern "C" {
#endif

hypre_MultiVectorPtr 
hypre_MultiVectorCreateFromSampleVector( HYPRE_InterfaceInterpreter*, int n, void* sample );

hypre_MultiVectorPtr 
hypre_MultiVectorCreateCopy( hypre_MultiVectorPtr, int copyValues );

void 
hypre_MultiVectorDestroy( hypre_MultiVectorPtr );

long
hypre_MultiVectorWidth( hypre_MultiVectorPtr v );

long
hypre_MultiVectorHeight( hypre_MultiVectorPtr v );

void 
hypre_MultiVectorClear( hypre_MultiVectorPtr );

void 
hypre_MultiVectorSetRandom( hypre_MultiVectorPtr v, int seed );

void 
hypre_MultiVectorSetVector( hypre_MultiVectorPtr mv, int i, void* v );

void 
hypre_MultiVectorGetVector( hypre_MultiVectorPtr mv, int i, void* v );

void 
hypre_MultiVectorCopy( int* ms, hypre_MultiVectorPtr src,
		       int* md, hypre_MultiVectorPtr dest );

void 
hypre_MultiVectorAxpy( double, 
		       int* mx, hypre_MultiVectorPtr,
		       int* my, hypre_MultiVectorPtr ); 

void 
hypre_MultiVectorByMultiVector(  
				int* mx, hypre_MultiVectorPtr,
				int* my, hypre_MultiVectorPtr,
				int gh, int h, int w, double* v );

void 
hypre_MultiVectorByMultiVectorDiag(  
				   int* xMask, hypre_MultiVectorPtr x,
				   int* yMask, hypre_MultiVectorPtr y,
				   int* dMask, int n, double* diag );

void 
hypre_MultiVectorByMatrix(  
			  int* ms, hypre_MultiVectorPtr, 
			  int gh, int h, int w, double* v,
			  int* md, hypre_MultiVectorPtr );

void 
hypre_MultiVectorXapy(  
		      int* ms, hypre_MultiVectorPtr, 
		      int gh, int h, int w, double* v,
		      int* md, hypre_MultiVectorPtr );

void hypre_MultiVectorByDiagonal(  
				 int* srcMask, hypre_MultiVectorPtr src, 
				 int* diagMask, int n, double* diag,
				 int* destMask, hypre_MultiVectorPtr dest );

void hypre_MultiVectorExplicitQR(  
				 int* xMask, hypre_MultiVectorPtr x, 
				 int rGHeight, int rHeight, 
				 int rWidth, double* rVal );
void 
hypre_MultiVectorEval( void (*f)( void*, void*, void* ), 
		       void* par,
		       int* xMask, hypre_MultiVectorPtr x, 
		       int* yMask, hypre_MultiVectorPtr y );

int
hypre_MultiVectorPrint( hypre_MultiVectorPtr x_, const char* fileName );

hypre_MultiVectorPtr 
hypre_MultiVectorRead( MPI_Comm comm, HYPRE_InterfaceInterpreter*, const char* fileName );

long 
aux_indexOrMaskCount( int n, int* indexOrMask, int* isMask );

void
aux_indexFromMask( int n, int* mask, int* index );

#ifdef __cplusplus
}
#endif

#endif /* MULTIVECTOR_FUNCTION_PROTOTYPES */

