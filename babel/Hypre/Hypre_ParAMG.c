
/******************************************************
 *
 *  File:  Hypre_ParAMG.c
 *
 *********************************************************/

#include "Hypre_ParAMG_Skel.h" 
#include "Hypre_ParAMG_Data.h" 


/* *************************************************
 * Constructor
 *    Allocate Memory for private data
 *    and initialize here
 ***************************************************/
void Hypre_ParAMG_constructor(Hypre_ParAMG this) {

} /* end constructor */

/* *************************************************
 *  Destructor
 *      deallocate memory for private data here.
 ***************************************************/
void Hypre_ParAMG_destructor(Hypre_ParAMG this) {

} /* end destructor */

/* ********************************************************
 * impl_Hypre_ParAMGGetDoubleParameter
 *       insert the library code below
 **********************************************************/
double  impl_Hypre_ParAMG_GetDoubleParameter(Hypre_ParAMG this, char* name) {
} /* end impl_Hypre_ParAMGGetDoubleParameter */

/* ********************************************************
 * impl_Hypre_ParAMGGetIntParameter
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_ParAMG_GetIntParameter(Hypre_ParAMG this, char* name) {
} /* end impl_Hypre_ParAMGGetIntParameter */

/* ********************************************************
 * impl_Hypre_ParAMGSetDoubleParameter
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_ParAMG_SetDoubleParameter(Hypre_ParAMG this, char* name, double value) {
} /* end impl_Hypre_ParAMGSetDoubleParameter */

/* ********************************************************
 * impl_Hypre_ParAMGSetIntParameter
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_ParAMG_SetIntParameter(Hypre_ParAMG this, char* name, int value) {
} /* end impl_Hypre_ParAMGSetIntParameter */

/* ********************************************************
 * impl_Hypre_ParAMGNew
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_ParAMG_New(Hypre_ParAMG this, Hypre_MPI_Com comm) {
} /* end impl_Hypre_ParAMGNew */

/* ********************************************************
 * impl_Hypre_ParAMGConstructor
 *       insert the library code below
 **********************************************************/
Hypre_ParAMG  impl_Hypre_ParAMG_Constructor(Hypre_MPI_Com comm) {
} /* end impl_Hypre_ParAMGConstructor */

/* ********************************************************
 * impl_Hypre_ParAMGSetup
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_ParAMG_Setup(Hypre_ParAMG this, Hypre_LinearOperator A, Hypre_Vector b, Hypre_Vector x) {
} /* end impl_Hypre_ParAMGSetup */

/* ********************************************************
 * impl_Hypre_ParAMGGetConstructedObject
 *       insert the library code below
 **********************************************************/
Hypre_Solver  impl_Hypre_ParAMG_GetConstructedObject(Hypre_ParAMG this) {
} /* end impl_Hypre_ParAMGGetConstructedObject */

/* ********************************************************
 * impl_Hypre_ParAMGApply
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_ParAMG_Apply(Hypre_ParAMG this, Hypre_Vector b, Hypre_Vector* x) {
} /* end impl_Hypre_ParAMGApply */

/* ********************************************************
 * impl_Hypre_ParAMGGetSystemOperator
 *       insert the library code below
 **********************************************************/
Hypre_LinearOperator  impl_Hypre_ParAMG_GetSystemOperator(Hypre_ParAMG this) {
} /* end impl_Hypre_ParAMGGetSystemOperator */

/* ********************************************************
 * impl_Hypre_ParAMGGetResidual
 *       insert the library code below
 **********************************************************/
Hypre_Vector  impl_Hypre_ParAMG_GetResidual(Hypre_ParAMG this) {
} /* end impl_Hypre_ParAMGGetResidual */

/* ********************************************************
 * impl_Hypre_ParAMGGetConvergenceInfo
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_ParAMG_GetConvergenceInfo(Hypre_ParAMG this, char* name, double* value) {
} /* end impl_Hypre_ParAMGGetConvergenceInfo */

