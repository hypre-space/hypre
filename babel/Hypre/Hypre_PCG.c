
/******************************************************
 *
 *  File:  Hypre_PCG.c
 *
 *********************************************************/

#include "Hypre_PCG_Skel.h" 
#include "Hypre_PCG_Data.h" 


/* *************************************************
 * Constructor
 *    Allocate Memory for private data
 *    and initialize here
 ***************************************************/
void Hypre_PCG_constructor(Hypre_PCG this) {

} /* end constructor */

/* *************************************************
 *  Destructor
 *      deallocate memory for private data here.
 ***************************************************/
void Hypre_PCG_destructor(Hypre_PCG this) {

} /* end destructor */

/* ********************************************************
 * impl_Hypre_PCGApply
 *       insert the library code below
 **********************************************************/
void  impl_Hypre_PCG_Apply(Hypre_PCG this, Hypre_StructVector b, Hypre_StructVector* x) {
} /* end impl_Hypre_PCGApply */

/* ********************************************************
 * impl_Hypre_PCGGetSystemOperator
 *       insert the library code below
 **********************************************************/
Hypre_StructMatrix  impl_Hypre_PCG_GetSystemOperator(Hypre_PCG this) {
} /* end impl_Hypre_PCGGetSystemOperator */

/* ********************************************************
 * impl_Hypre_PCGGetResidual
 *       insert the library code below
 **********************************************************/
Hypre_StructVector  impl_Hypre_PCG_GetResidual(Hypre_PCG this) {
} /* end impl_Hypre_PCGGetResidual */

/* ********************************************************
 * impl_Hypre_PCGGetConvergenceInfo
 *       insert the library code below
 **********************************************************/
void  impl_Hypre_PCG_GetConvergenceInfo(Hypre_PCG this, char* name, double* value) {
} /* end impl_Hypre_PCGGetConvergenceInfo */

/* ********************************************************
 * impl_Hypre_PCGGetPreconditioner
 *       insert the library code below
 **********************************************************/
Hypre_Solver  impl_Hypre_PCG_GetPreconditioner(Hypre_PCG this) {
} /* end impl_Hypre_PCGGetPreconditioner */

/* ********************************************************
 * impl_Hypre_PCGSetSystemOperator
 *       insert the library code below
 **********************************************************/
void  impl_Hypre_PCG_SetSystemOperator(Hypre_PCG this, Hypre_StructMatrix op) {
} /* end impl_Hypre_PCGSetSystemOperator */

/* ********************************************************
 * impl_Hypre_PCGGetParameter
 *       insert the library code below
 **********************************************************/
double  impl_Hypre_PCG_GetParameter(Hypre_PCG this, char* name) {
} /* end impl_Hypre_PCGGetParameter */

/* ********************************************************
 * impl_Hypre_PCGSetParameter
 *       insert the library code below
 **********************************************************/
void  impl_Hypre_PCG_SetParameter(Hypre_PCG this, char* name, double value) {
} /* end impl_Hypre_PCGSetParameter */

/* ********************************************************
 * impl_Hypre_PCGNew
 *       insert the library code below
 **********************************************************/
void  impl_Hypre_PCG_New(Hypre_PCG this, Hypre_MPI_Com comm) {
} /* end impl_Hypre_PCGNew */

/* ********************************************************
 * impl_Hypre_PCGConstructor
 *       insert the library code below
 **********************************************************/
Hypre_PCG  impl_Hypre_PCG_Constructor(Hypre_MPI_Com comm) {
} /* end impl_Hypre_PCGConstructor */

/* ********************************************************
 * impl_Hypre_PCGSetup
 *       insert the library code below
 **********************************************************/
void  impl_Hypre_PCG_Setup(Hypre_PCG this, Hypre_StructMatrix A, Hypre_StructVector b, Hypre_StructVector x) {
} /* end impl_Hypre_PCGSetup */

/* ********************************************************
 * impl_Hypre_PCGGetConstructedObject
 *       insert the library code below
 **********************************************************/
Hypre_SolverBuilder  impl_Hypre_PCG_GetConstructedObject(Hypre_PCG this) {
} /* end impl_Hypre_PCGGetConstructedObject */

/* ********************************************************
 * impl_Hypre_PCGSetPreconditioner
 *       insert the library code below
 **********************************************************/
void  impl_Hypre_PCG_SetPreconditioner(Hypre_PCG this, Hypre_Solver precond) {
} /* end impl_Hypre_PCGSetPreconditioner */

