
/******************************************************
 *
 *  File:  Hypre_GMRES.c
 *
 *********************************************************/

#include "Hypre_GMRES_Skel.h" 
#include "Hypre_GMRES_Data.h" 


/* *************************************************
 * Constructor
 *    Allocate Memory for private data
 *    and initialize here
 ***************************************************/
void Hypre_GMRES_constructor(Hypre_GMRES this) {

} /* end constructor */

/* *************************************************
 *  Destructor
 *      deallocate memory for private data here.
 ***************************************************/
void Hypre_GMRES_destructor(Hypre_GMRES this) {

} /* end destructor */

/* ********************************************************
 * impl_Hypre_GMRESGetDoubleParameter
 *       insert the library code below
 **********************************************************/
double  impl_Hypre_GMRES_GetDoubleParameter(Hypre_GMRES this, char* name) {
} /* end impl_Hypre_GMRESGetDoubleParameter */

/* ********************************************************
 * impl_Hypre_GMRESGetIntParameter
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_GMRES_GetIntParameter(Hypre_GMRES this, char* name) {
} /* end impl_Hypre_GMRESGetIntParameter */

/* ********************************************************
 * impl_Hypre_GMRESSetDoubleParameter
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_GMRES_SetDoubleParameter(Hypre_GMRES this, char* name, double value) {
} /* end impl_Hypre_GMRESSetDoubleParameter */

/* ********************************************************
 * impl_Hypre_GMRESSetIntParameter
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_GMRES_SetIntParameter(Hypre_GMRES this, char* name, int value) {
} /* end impl_Hypre_GMRESSetIntParameter */

/* ********************************************************
 * impl_Hypre_GMRESSetPreconditioner
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_GMRES_SetPreconditioner(Hypre_GMRES this, Hypre_Solver precond) {
} /* end impl_Hypre_GMRESSetPreconditioner */

/* ********************************************************
 * impl_Hypre_GMRESNew
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_GMRES_New(Hypre_GMRES this, Hypre_MPI_Com comm) {
} /* end impl_Hypre_GMRESNew */

/* ********************************************************
 * impl_Hypre_GMRESConstructor
 *       insert the library code below
 **********************************************************/
Hypre_GMRES  impl_Hypre_GMRES_Constructor(Hypre_MPI_Com comm) {
} /* end impl_Hypre_GMRESConstructor */

/* ********************************************************
 * impl_Hypre_GMRESSetup
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_GMRES_Setup(Hypre_GMRES this, Hypre_LinearOperator A, Hypre_Vector b, Hypre_Vector x) {
} /* end impl_Hypre_GMRESSetup */

/* ********************************************************
 * impl_Hypre_GMRESGetConstructedObject
 *       insert the library code below
 **********************************************************/
Hypre_Solver  impl_Hypre_GMRES_GetConstructedObject(Hypre_GMRES this) {
} /* end impl_Hypre_GMRESGetConstructedObject */

/* ********************************************************
 * impl_Hypre_GMRESApply
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_GMRES_Apply(Hypre_GMRES this, Hypre_Vector b, Hypre_Vector* x) {
} /* end impl_Hypre_GMRESApply */

/* ********************************************************
 * impl_Hypre_GMRESGetSystemOperator
 *       insert the library code below
 **********************************************************/
Hypre_LinearOperator  impl_Hypre_GMRES_GetSystemOperator(Hypre_GMRES this) {
} /* end impl_Hypre_GMRESGetSystemOperator */

/* ********************************************************
 * impl_Hypre_GMRESGetResidual
 *       insert the library code below
 **********************************************************/
Hypre_Vector  impl_Hypre_GMRES_GetResidual(Hypre_GMRES this) {
} /* end impl_Hypre_GMRESGetResidual */

/* ********************************************************
 * impl_Hypre_GMRESGetConvergenceInfo
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_GMRES_GetConvergenceInfo(Hypre_GMRES this, char* name, double* value) {
} /* end impl_Hypre_GMRESGetConvergenceInfo */

