
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
 * impl_Hypre_PilutGetParameterDouble
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_Pilut_GetParameterDouble(Hypre_Pilut this, char* name, double* value) {
} /* end impl_Hypre_PilutGetParameterDouble */

/* ********************************************************
 * impl_Hypre_PilutGetParameterInt
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_Pilut_GetParameterInt(Hypre_Pilut this, char* name, int* value) {
} /* end impl_Hypre_PilutGetParameterInt */

/* ********************************************************
 * impl_Hypre_PilutSetParameterDouble
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_Pilut_SetParameterDouble(Hypre_Pilut this, char* name, double value) {
} /* end impl_Hypre_PilutSetParameterDouble */

/* ********************************************************
 * impl_Hypre_PilutSetParameterInt
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_Pilut_SetParameterInt(Hypre_Pilut this, char* name, int value) {
} /* end impl_Hypre_PilutSetParameterInt */

/* ********************************************************
 * impl_Hypre_PilutSetParameterString
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_Pilut_SetParameterString(Hypre_Pilut this, char* name, char* value) {
} /* end impl_Hypre_PilutSetParameterString */

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
int  impl_Hypre_Pilut_GetConstructedObject(Hypre_Pilut this, Hypre_Solver* obj) {
} /* end impl_Hypre_PilutGetConstructedObject */

/* ********************************************************
 * impl_Hypre_PilutApply
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_Pilut_Apply(Hypre_Pilut this, Hypre_Vector b, Hypre_Vector* x) {
} /* end impl_Hypre_PilutApply */

/* ********************************************************
 * impl_Hypre_PilutGetDims
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_Pilut_GetDims(Hypre_Pilut this, int* m, int* n) {
} /* end impl_Hypre_PilutGetDims */

/* ********************************************************
 * impl_Hypre_PilutGetSystemOperator
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_Pilut_GetSystemOperator(Hypre_Pilut this, Hypre_LinearOperator* op) {
} /* end impl_Hypre_PilutGetSystemOperator */

/* ********************************************************
 * impl_Hypre_PilutGetResidual
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_Pilut_GetResidual(Hypre_Pilut this, Hypre_Vector* resid) {
} /* end impl_Hypre_PilutGetResidual */

/* ********************************************************
 * impl_Hypre_PilutGetConvergenceInfo
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_Pilut_GetConvergenceInfo(Hypre_Pilut this, char* name, double* value) {
} /* end impl_Hypre_PilutGetConvergenceInfo */

