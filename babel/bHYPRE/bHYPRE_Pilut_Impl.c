/*
 * File:          bHYPRE_Pilut_Impl.c
 * Symbol:        bHYPRE.Pilut-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Description:   Server-side implementation for bHYPRE.Pilut
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.12
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "bHYPRE.Pilut" (version 1.0.0)
 * 
 * Objects of this type can be cast to Solver objects using the
 * {\tt \_\_cast} methods.
 * 
 * RDF: Documentation goes here.
 * 
 * Pilut has not been implemented yet.
 * 
 * 
 */

#include "bHYPRE_Pilut_Impl.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut._includes) */
/* Put additional includes or other arbitrary code here... */

/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team <hypre-users@llnl.gov>, UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer and the GNU Lesser General Public License.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/

/* DO-NOT-DELETE splicer.end(bHYPRE.Pilut._includes) */

/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_Pilut__load(
  void)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut._load) */
  /* Insert-Code-Here {bHYPRE.Pilut._load} (static class initializer method) */
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut._load) */
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_Pilut__ctor(
  /* in */ bHYPRE_Pilut self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut._ctor) */
  /* Insert the implementation of the constructor method here... */
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_Pilut__dtor(
  /* in */ bHYPRE_Pilut self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut._dtor) */
  /* Insert the implementation of the destructor method here... */
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut._dtor) */
}

/*
 * Method:  Create[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_Create"

#ifdef __cplusplus
extern "C"
#endif
bHYPRE_Pilut
impl_bHYPRE_Pilut_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_Operator A)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.Create) */
  /* Insert-Code-Here {bHYPRE.Pilut.Create} (Create method) */

   /* int ierr = 0;*/
   bHYPRE_Pilut solver = bHYPRE_Pilut__create(); /* assumed to call HYPRE_PilutCreate()*/
   /* >>> requires data to be set up...
      struct bHYPRE_Pilut__data * data = bHYPRE_Pilut__get_data( self );

      data -> comm = (MPI_Comm *) mpi_comm;
   */

   return solver;

  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.Create) */
}

/*
 * Set the MPI Communicator.
 * DEPRECATED, use Create:
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_SetCommunicator"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Pilut_SetCommunicator(
  /* in */ bHYPRE_Pilut self,
  /* in */ bHYPRE_MPICommunicator mpi_comm)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.SetCommunicator) */
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_SetIntParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Pilut_SetIntParameter(
  /* in */ bHYPRE_Pilut self,
  /* in */ const char* name,
  /* in */ int32_t value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.SetIntParameter) */
  /* Insert the implementation of the SetIntParameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.SetIntParameter) */
}

/*
 * Set the double parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_SetDoubleParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Pilut_SetDoubleParameter(
  /* in */ bHYPRE_Pilut self,
  /* in */ const char* name,
  /* in */ double value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.SetDoubleParameter) */
  /* Insert the implementation of the SetDoubleParameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.SetDoubleParameter) */
}

/*
 * Set the string parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_SetStringParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Pilut_SetStringParameter(
  /* in */ bHYPRE_Pilut self,
  /* in */ const char* name,
  /* in */ const char* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.SetStringParameter) */
  /* Insert the implementation of the SetStringParameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.SetStringParameter) */
}

/*
 * Set the int 1-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_SetIntArray1Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Pilut_SetIntArray1Parameter(
  /* in */ bHYPRE_Pilut self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ int32_t* value,
  /* in */ int32_t nvalues)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.SetIntArray1Parameter) */
  /* Insert the implementation of the SetIntArray1Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.SetIntArray1Parameter) */
}

/*
 * Set the int 2-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_SetIntArray2Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Pilut_SetIntArray2Parameter(
  /* in */ bHYPRE_Pilut self,
  /* in */ const char* name,
  /* in array<int,2,column-major> */ struct sidl_int__array* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.SetIntArray2Parameter) */
  /* Insert the implementation of the SetIntArray2Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.SetIntArray2Parameter) */
}

/*
 * Set the double 1-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_SetDoubleArray1Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Pilut_SetDoubleArray1Parameter(
  /* in */ bHYPRE_Pilut self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ double* value,
  /* in */ int32_t nvalues)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.SetDoubleArray1Parameter) */
  /* Insert the implementation of the SetDoubleArray1Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.SetDoubleArray1Parameter) */
}

/*
 * Set the double 2-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_SetDoubleArray2Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Pilut_SetDoubleArray2Parameter(
  /* in */ bHYPRE_Pilut self,
  /* in */ const char* name,
  /* in array<double,2,column-major> */ struct sidl_double__array* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.SetDoubleArray2Parameter) */
  /* Insert the implementation of the SetDoubleArray2Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.SetDoubleArray2Parameter) */
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_GetIntValue"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Pilut_GetIntValue(
  /* in */ bHYPRE_Pilut self,
  /* in */ const char* name,
  /* out */ int32_t* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.GetIntValue) */
  /* Insert the implementation of the GetIntValue method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.GetIntValue) */
}

/*
 * Get the double parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_GetDoubleValue"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Pilut_GetDoubleValue(
  /* in */ bHYPRE_Pilut self,
  /* in */ const char* name,
  /* out */ double* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.GetDoubleValue) */
  /* Insert the implementation of the GetDoubleValue method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.GetDoubleValue) */
}

/*
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_Setup"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Pilut_Setup(
  /* in */ bHYPRE_Pilut self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.Setup) */
  /* Insert the implementation of the Setup method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.Setup) */
}

/*
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_Apply"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Pilut_Apply(
  /* in */ bHYPRE_Pilut self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.Apply) */
  /* Insert the implementation of the Apply method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.Apply) */
}

/*
 * Apply the adjoint of the operator to {\tt b}, returning {\tt x}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_ApplyAdjoint"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Pilut_ApplyAdjoint(
  /* in */ bHYPRE_Pilut self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.ApplyAdjoint) */
  /* Insert-Code-Here {bHYPRE.Pilut.ApplyAdjoint} (ApplyAdjoint method) */

   return 1; /* not implemented */

  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.ApplyAdjoint) */
}

/*
 * Set the operator for the linear system being solved.
 * DEPRECATED.  use Create
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_SetOperator"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Pilut_SetOperator(
  /* in */ bHYPRE_Pilut self,
  /* in */ bHYPRE_Operator A)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.SetOperator) */
  /* Insert the implementation of the SetOperator method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.SetOperator) */
}

/*
 * (Optional) Set the convergence tolerance.
 * DEPRECATED.  use SetDoubleParameter
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_SetTolerance"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Pilut_SetTolerance(
  /* in */ bHYPRE_Pilut self,
  /* in */ double tolerance)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.SetTolerance) */
  /* Insert the implementation of the SetTolerance method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.SetTolerance) */
}

/*
 * (Optional) Set maximum number of iterations.
 * DEPRECATED   use SetIntParameter
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_SetMaxIterations"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Pilut_SetMaxIterations(
  /* in */ bHYPRE_Pilut self,
  /* in */ int32_t max_iterations)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.SetMaxIterations) */
  /* Insert the implementation of the SetMaxIterations method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.SetMaxIterations) */
}

/*
 * (Optional) Set the {\it logging level}, specifying the degree
 * of additional informational data to be accumulated.  Does
 * nothing by default (level = 0).  Other levels (if any) are
 * implementation-specific.  Must be called before {\tt Setup}
 * and {\tt Apply}.
 * DEPRECATED   use SetIntParameter
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_SetLogging"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Pilut_SetLogging(
  /* in */ bHYPRE_Pilut self,
  /* in */ int32_t level)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.SetLogging) */
  /* Insert the implementation of the SetLogging method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.SetLogging) */
}

/*
 * (Optional) Set the {\it print level}, specifying the degree
 * of informational data to be printed either to the screen or
 * to a file.  Does nothing by default (level=0).  Other levels
 * (if any) are implementation-specific.  Must be called before
 * {\tt Setup} and {\tt Apply}.
 * DEPRECATED   use SetIntParameter
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_SetPrintLevel"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Pilut_SetPrintLevel(
  /* in */ bHYPRE_Pilut self,
  /* in */ int32_t level)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.SetPrintLevel) */
  /* Insert the implementation of the SetPrintLevel method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.SetPrintLevel) */
}

/*
 * (Optional) Return the number of iterations taken.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_GetNumIterations"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Pilut_GetNumIterations(
  /* in */ bHYPRE_Pilut self,
  /* out */ int32_t* num_iterations)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.GetNumIterations) */
  /* Insert the implementation of the GetNumIterations method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.GetNumIterations) */
}

/*
 * (Optional) Return the norm of the relative residual.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_GetRelResidualNorm"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Pilut_GetRelResidualNorm(
  /* in */ bHYPRE_Pilut self,
  /* out */ double* norm)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.GetRelResidualNorm) */
  /* Insert the implementation of the GetRelResidualNorm method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.GetRelResidualNorm) */
}
/* Babel internal methods, Users should not edit below this line. */
struct bHYPRE_Solver__object* impl_bHYPRE_Pilut_fconnect_bHYPRE_Solver(char* 
  url, sidl_BaseInterface *_ex) {
  return bHYPRE_Solver__connect(url, _ex);
}
char * impl_bHYPRE_Pilut_fgetURL_bHYPRE_Solver(struct bHYPRE_Solver__object* 
  obj) {
  return bHYPRE_Solver__getURL(obj);
}
struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_Pilut_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_MPICommunicator__connect(url, _ex);
}
char * impl_bHYPRE_Pilut_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj) {
  return bHYPRE_MPICommunicator__getURL(obj);
}
struct bHYPRE_Operator__object* 
  impl_bHYPRE_Pilut_fconnect_bHYPRE_Operator(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_Operator__connect(url, _ex);
}
char * impl_bHYPRE_Pilut_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj) {
  return bHYPRE_Operator__getURL(obj);
}
struct sidl_ClassInfo__object* impl_bHYPRE_Pilut_fconnect_sidl_ClassInfo(char* 
  url, sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connect(url, _ex);
}
char * impl_bHYPRE_Pilut_fgetURL_sidl_ClassInfo(struct sidl_ClassInfo__object* 
  obj) {
  return sidl_ClassInfo__getURL(obj);
}
struct bHYPRE_Pilut__object* impl_bHYPRE_Pilut_fconnect_bHYPRE_Pilut(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_Pilut__connect(url, _ex);
}
char * impl_bHYPRE_Pilut_fgetURL_bHYPRE_Pilut(struct bHYPRE_Pilut__object* obj) 
  {
  return bHYPRE_Pilut__getURL(obj);
}
struct bHYPRE_Vector__object* impl_bHYPRE_Pilut_fconnect_bHYPRE_Vector(char* 
  url, sidl_BaseInterface *_ex) {
  return bHYPRE_Vector__connect(url, _ex);
}
char * impl_bHYPRE_Pilut_fgetURL_bHYPRE_Vector(struct bHYPRE_Vector__object* 
  obj) {
  return bHYPRE_Vector__getURL(obj);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_Pilut_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connect(url, _ex);
}
char * impl_bHYPRE_Pilut_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) {
  return sidl_BaseInterface__getURL(obj);
}
struct sidl_BaseClass__object* impl_bHYPRE_Pilut_fconnect_sidl_BaseClass(char* 
  url, sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connect(url, _ex);
}
char * impl_bHYPRE_Pilut_fgetURL_sidl_BaseClass(struct sidl_BaseClass__object* 
  obj) {
  return sidl_BaseClass__getURL(obj);
}
