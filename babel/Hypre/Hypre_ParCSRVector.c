
/******************************************************
 *
 *  File:  Hypre_ParCSRVector.c
 *
 *********************************************************/

#include "Hypre_ParCSRVector_Skel.h" 
#include "Hypre_ParCSRVector_Data.h" 


/* *************************************************
 * Constructor
 *    Allocate Memory for private data
 *    and initialize here
 ***************************************************/
void Hypre_ParCSRVector_constructor(Hypre_ParCSRVector this) {

} /* end constructor */

/* *************************************************
 *  Destructor
 *      deallocate memory for private data here.
 ***************************************************/
void Hypre_ParCSRVector_destructor(Hypre_ParCSRVector this) {

} /* end destructor */

/* ********************************************************
 * impl_Hypre_ParCSRVectorClear
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_ParCSRVector_Clear(Hypre_ParCSRVector this) {
} /* end impl_Hypre_ParCSRVectorClear */

/* ********************************************************
 * impl_Hypre_ParCSRVectorCopy
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_ParCSRVector_Copy(Hypre_ParCSRVector this, Hypre_Vector x) {
} /* end impl_Hypre_ParCSRVectorCopy */

/* ********************************************************
 * impl_Hypre_ParCSRVectorClone
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_ParCSRVector_Clone(Hypre_ParCSRVector this, Hypre_Vector* x) {
} /* end impl_Hypre_ParCSRVectorClone */

/* ********************************************************
 * impl_Hypre_ParCSRVectorScale
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_ParCSRVector_Scale(Hypre_ParCSRVector this, double a) {
} /* end impl_Hypre_ParCSRVectorScale */

/* ********************************************************
 * impl_Hypre_ParCSRVectorDot
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_ParCSRVector_Dot(Hypre_ParCSRVector this, Hypre_Vector x, double* d) {
} /* end impl_Hypre_ParCSRVectorDot */

/* ********************************************************
 * impl_Hypre_ParCSRVectorAxpy
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_ParCSRVector_Axpy(Hypre_ParCSRVector this, double a, Hypre_Vector x) {
} /* end impl_Hypre_ParCSRVectorAxpy */

