/*
 * File:          Hypre_ParCSRMatrix_Skel.c
 * Symbol:        Hypre.ParCSRMatrix-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.6.1
 * SIDL Created:  20020104 15:27:10 PST
 * Generated:     20020104 15:27:19 PST
 * Description:   Server-side glue code for Hypre.ParCSRMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include "Hypre_ParCSRMatrix_IOR.h"
#include "Hypre_ParCSRMatrix.h"
#include <stddef.h>

extern void
impl_Hypre_ParCSRMatrix__ctor(
  Hypre_ParCSRMatrix);

extern void
impl_Hypre_ParCSRMatrix__dtor(
  Hypre_ParCSRMatrix);

extern int32_t
impl_Hypre_ParCSRMatrix_AddToValues(
  Hypre_ParCSRMatrix,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_ParCSRMatrix_Apply(
  Hypre_ParCSRMatrix,
  Hypre_Vector,
  Hypre_Vector*);

extern int32_t
impl_Hypre_ParCSRMatrix_Assemble(
  Hypre_ParCSRMatrix);

extern int32_t
impl_Hypre_ParCSRMatrix_Create(
  Hypre_ParCSRMatrix,
  int32_t,
  int32_t,
  int32_t,
  int32_t);

extern int32_t
impl_Hypre_ParCSRMatrix_GetObject(
  Hypre_ParCSRMatrix,
  SIDL_BaseInterface*);

extern int32_t
impl_Hypre_ParCSRMatrix_GetRow(
  Hypre_ParCSRMatrix,
  int32_t,
  int32_t*,
  struct SIDL_int__array**,
  struct SIDL_double__array**);

extern int32_t
impl_Hypre_ParCSRMatrix_Initialize(
  Hypre_ParCSRMatrix);

extern int32_t
impl_Hypre_ParCSRMatrix_Print(
  Hypre_ParCSRMatrix,
  const char*);

extern int32_t
impl_Hypre_ParCSRMatrix_Read(
  Hypre_ParCSRMatrix,
  const char*,
  void*);

extern int32_t
impl_Hypre_ParCSRMatrix_SetCommunicator(
  Hypre_ParCSRMatrix,
  void*);

extern int32_t
impl_Hypre_ParCSRMatrix_SetDiagOffdSizes(
  Hypre_ParCSRMatrix,
  struct SIDL_int__array*,
  struct SIDL_int__array*);

extern int32_t
impl_Hypre_ParCSRMatrix_SetParameter(
  Hypre_ParCSRMatrix,
  const char*,
  double);

extern int32_t
impl_Hypre_ParCSRMatrix_SetRowSizes(
  Hypre_ParCSRMatrix,
  struct SIDL_int__array*);

extern int32_t
impl_Hypre_ParCSRMatrix_SetValues(
  Hypre_ParCSRMatrix,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_ParCSRMatrix_Setup(
  Hypre_ParCSRMatrix);

void
Hypre_ParCSRMatrix__set_epv(struct Hypre_ParCSRMatrix__epv *epv)
{
  epv->f__ctor = impl_Hypre_ParCSRMatrix__ctor;
  epv->f__dtor = impl_Hypre_ParCSRMatrix__dtor;
  epv->f_SetRowSizes = impl_Hypre_ParCSRMatrix_SetRowSizes;
  epv->f_AddToValues = impl_Hypre_ParCSRMatrix_AddToValues;
  epv->f_SetParameter = impl_Hypre_ParCSRMatrix_SetParameter;
  epv->f_Setup = impl_Hypre_ParCSRMatrix_Setup;
  epv->f_Initialize = impl_Hypre_ParCSRMatrix_Initialize;
  epv->f_Apply = impl_Hypre_ParCSRMatrix_Apply;
  epv->f_SetCommunicator = impl_Hypre_ParCSRMatrix_SetCommunicator;
  epv->f_GetRow = impl_Hypre_ParCSRMatrix_GetRow;
  epv->f_Create = impl_Hypre_ParCSRMatrix_Create;
  epv->f_Read = impl_Hypre_ParCSRMatrix_Read;
  epv->f_Assemble = impl_Hypre_ParCSRMatrix_Assemble;
  epv->f_Print = impl_Hypre_ParCSRMatrix_Print;
  epv->f_SetDiagOffdSizes = impl_Hypre_ParCSRMatrix_SetDiagOffdSizes;
  epv->f_SetValues = impl_Hypre_ParCSRMatrix_SetValues;
  epv->f_GetObject = impl_Hypre_ParCSRMatrix_GetObject;
}

struct Hypre_ParCSRMatrix__data*
Hypre_ParCSRMatrix__get_data(Hypre_ParCSRMatrix self)
{
  return (struct Hypre_ParCSRMatrix__data*)(self ? self->d_data : NULL);
}

void Hypre_ParCSRMatrix__set_data(
  Hypre_ParCSRMatrix self,
  struct Hypre_ParCSRMatrix__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
