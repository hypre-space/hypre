
/******************************************************
 *
 *  File:  Hypre_Pilut.c
 *
 *********************************************************/

#include "Hypre_Pilut_Skel.h" 
#include "Hypre_Pilut_Data.h" 


/* *************************************************
 * Constructor
 *    Allocate Memory for private data
 *    and initialize here
 ***************************************************/
void Hypre_Pilut_constructor(Hypre_Pilut this) {

} /* end constructor */

/* *************************************************
 *  Destructor
 *      deallocate memory for private data here.
 ***************************************************/
void Hypre_Pilut_destructor(Hypre_Pilut this) {

} /* end destructor */

/* ********************************************************
 * impl_Hypre_PilutGetDoubleParameter
 *       insert the library code below
 **********************************************************/
double  impl_Hypre_Pilut_GetDoubleParameter(Hypre_Pilut this, char* name) {
} /* end impl_Hypre_PilutGetDoubleParameter */

/* ********************************************************
 * impl_Hypre_PilutGetIntParameter
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_Pilut_GetIntParameter(Hypre_Pilut this, char* name) {
} /* end impl_Hypre_PilutGetIntParameter */

/* ********************************************************
 * impl_Hypre_PilutSetDoubleParameter
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_Pilut_SetDoubleParameter(Hypre_Pilut this, char* name, double value) {
} /* end impl_Hypre_PilutSetDoubleParameter */

/* ********************************************************
 * impl_Hypre_PilutSetIntParameter
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_Pilut_SetIntParameter(Hypre_Pilut this, char* name, int value) {
} /* end impl_Hypre_PilutSetIntParameter */

/* ********************************************************
 * impl_Hypre_PilutNew
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_Pilut_New(Hypre_Pilut this, Hypre_MPI_Com comm) {
} /* end impl_Hypre_PilutNew */

/* ********************************************************
 * impl_Hypre_PilutConstructor
 *       insert the library code below
 **********************************************************/
Hypre_Pilut  impl_Hypre_Pilut_Constructor(Hypre_MPI_Com comm) {
} /* end impl_Hypre_PilutConstructor */

/* ********************************************************
 * impl_Hypre_PilutSetup
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_Pilut_Setup(Hypre_Pilut this, Hypre_LinearOperator A, Hypre_Vector b, Hypre_Vector x) {
} /* end impl_Hypre_PilutSetup */

/* ********************************************************
 * impl_Hypre_PilutGetConstructedObject
 *       insert the library code below
 **********************************************************/
Hypre_Solver  impl_Hypre_Pilut_GetConstructedObject(Hypre_Pilut this) {
} /* end impl_Hypre_PilutGetConstructedObject */

/* ********************************************************
 * impl_Hypre_PilutApply
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_Pilut_Apply(Hypre_Pilut this, Hypre_Vector b, Hypre_Vector* x) {
} /* end impl_Hypre_PilutApply */

/* ********************************************************
 * impl_Hypre_PilutGetSystemOperator
 *       insert the library code below
 **********************************************************/
Hypre_LinearOperator  impl_Hypre_Pilut_GetSystemOperator(Hypre_Pilut this) {
} /* end impl_Hypre_PilutGetSystemOperator */

/* ********************************************************
 * impl_Hypre_PilutGetResidual
 *       insert the library code below
 **********************************************************/
Hypre_Vector  impl_Hypre_Pilut_GetResidual(Hypre_Pilut this) {
} /* end impl_Hypre_PilutGetResidual */

/* ********************************************************
 * impl_Hypre_PilutGetConvergenceInfo
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_Pilut_GetConvergenceInfo(Hypre_Pilut this, char* name, double* value) {
} /* end impl_Hypre_PilutGetConvergenceInfo */

