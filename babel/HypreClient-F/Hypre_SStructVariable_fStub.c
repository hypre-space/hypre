/*
 * File:          Hypre_SStructVariable_fStub.c
 * Symbol:        Hypre.SStructVariable-v0.1.7
 * Symbol Type:   enumeration
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:23 PST
 * Generated:     20030306 17:05:27 PST
 * Description:   Client-side glue code for Hypre.SStructVariable
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 898
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_SIDL_int_IOR_h
#include "SIDL_int_IOR.h"
#endif
#ifndef included_SIDLfortran_h
#include "SIDLfortran.h"
#endif
#include <stddef.h>
void
SIDLFortran77Symbol(hypre_sstructvariable__array_createcol_f,
                  HYPRE_SSTRUCTVARIABLE__ARRAY_CREATECOL_F,
                  Hypre_SStructVariable__array_createCol_f)
  (int32_t *dimen,
   int32_t lower[],
   int32_t upper[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    SIDL_int__array_createCol(*dimen, lower, upper);
}

void
SIDLFortran77Symbol(hypre_sstructvariable__array_createrow_f,
                  HYPRE_SSTRUCTVARIABLE__ARRAY_CREATEROW_F,
                  Hypre_SStructVariable__array_createRow_f)
  (int32_t *dimen,
   int32_t lower[],
   int32_t upper[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    SIDL_int__array_createRow(*dimen, lower, upper);
}

void
SIDLFortran77Symbol(hypre_sstructvariable__array_create1d_f,
                  HYPRE_SSTRUCTVARIABLE__ARRAY_CREATE1D_F,
                  Hypre_SStructVariable__array_create1d_f)
  (int32_t *len, int64_t *result)
{
  *result = (ptrdiff_t)SIDL_int__array_create1d(*len);
}

void
SIDLFortran77Symbol(hypre_sstructvariable__array_create2dcol_f,
                  HYPRE_SSTRUCTVARIABLE__ARRAY_CREATE2DCOL_F,
                  Hypre_SStructVariable__array_create2dCol_f)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)SIDL_int__array_create2dCol(*m, *n);
}

void
SIDLFortran77Symbol(hypre_sstructvariable__array_create2drow_f,
                  HYPRE_SSTRUCTVARIABLE__ARRAY_CREATE2DROW_F,
                  Hypre_SStructVariable__array_create2dRow_f)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)SIDL_int__array_create2dRow(*m, *n);
}

void
SIDLFortran77Symbol(hypre_sstructvariable__array_addref_f,
                  HYPRE_SSTRUCTVARIABLE__ARRAY_ADDREF_F,
                  Hypre_SStructVariable__array_addRef_f)
  (int64_t *array)
{
  SIDL_int__array_addRef((struct SIDL_int__array *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(hypre_sstructvariable__array_deleteref_f,
                  HYPRE_SSTRUCTVARIABLE__ARRAY_DELETEREF_F,
                  Hypre_SStructVariable__array_deleteRef_f)
  (int64_t *array)
{
  SIDL_int__array_deleteRef((struct SIDL_int__array *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(hypre_sstructvariable__array_get1_f,
                  HYPRE_SSTRUCTVARIABLE__ARRAY_GET1_F,
                  Hypre_SStructVariable__array_get1_f)
  (int64_t *array, 
   int32_t *i1, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    SIDL_int__array_get1((const struct SIDL_int__array *)(ptrdiff_t)*array
    , *i1);
}

void
SIDLFortran77Symbol(hypre_sstructvariable__array_get2_f,
                  HYPRE_SSTRUCTVARIABLE__ARRAY_GET2_F,
                  Hypre_SStructVariable__array_get2_f)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    SIDL_int__array_get2((const struct SIDL_int__array *)(ptrdiff_t)*array
    , *i1, *i2);
}

void
SIDLFortran77Symbol(hypre_sstructvariable__array_get3_f,
                  HYPRE_SSTRUCTVARIABLE__ARRAY_GET3_F,
                  Hypre_SStructVariable__array_get3_f)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int32_t *i3, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    SIDL_int__array_get3((const struct SIDL_int__array *)(ptrdiff_t)*array
    , *i1, *i2, *i3);
}

void
SIDLFortran77Symbol(hypre_sstructvariable__array_get4_f,
                  HYPRE_SSTRUCTVARIABLE__ARRAY_GET4_F,
                  Hypre_SStructVariable__array_get4_f)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int32_t *i3, 
   int32_t *i4, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    SIDL_int__array_get4((const struct SIDL_int__array *)(ptrdiff_t)*array
    , *i1, *i2, *i3, *i4);
}

void
SIDLFortran77Symbol(hypre_sstructvariable__array_get_f,
                  HYPRE_SSTRUCTVARIABLE__ARRAY_GET_F,
                  Hypre_SStructVariable__array_get_f)
  (int64_t *array,
   int32_t indices[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    SIDL_int__array_get((const struct SIDL_int__array *)(ptrdiff_t)*array,
      indices);
}

void
SIDLFortran77Symbol(hypre_sstructvariable__array_set1_f,
                  HYPRE_SSTRUCTVARIABLE__ARRAY_SET1_F,
                  Hypre_SStructVariable__array_set1_f)
  (int64_t *array,
   int32_t *i1,
   int64_t *value)
{
  SIDL_int__array_set1((struct SIDL_int__array *)(ptrdiff_t)*array
  , *i1, (int32_t)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(hypre_sstructvariable__array_set2_f,
                  HYPRE_SSTRUCTVARIABLE__ARRAY_SET2_F,
                  Hypre_SStructVariable__array_set2_f)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int64_t *value)
{
  SIDL_int__array_set2((struct SIDL_int__array *)(ptrdiff_t)*array
  , *i1, *i2, (int32_t)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(hypre_sstructvariable__array_set3_f,
                  HYPRE_SSTRUCTVARIABLE__ARRAY_SET3_F,
                  Hypre_SStructVariable__array_set3_f)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int32_t *i3,
   int64_t *value)
{
  SIDL_int__array_set3((struct SIDL_int__array *)(ptrdiff_t)*array
  , *i1, *i2, *i3, (int32_t)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(hypre_sstructvariable__array_set4_f,
                  HYPRE_SSTRUCTVARIABLE__ARRAY_SET4_F,
                  Hypre_SStructVariable__array_set4_f)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int32_t *i3,
   int32_t *i4,
   int64_t *value)
{
  SIDL_int__array_set4((struct SIDL_int__array *)(ptrdiff_t)*array
  , *i1, *i2, *i3, *i4, (int32_t)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(hypre_sstructvariable__array_set_f,
                  HYPRE_SSTRUCTVARIABLE__ARRAY_SET_F,
                  Hypre_SStructVariable__array_set_f)
  (int64_t *array,
  int32_t indices[],
  int64_t *value)
{
  SIDL_int__array_set((struct SIDL_int__array *)(ptrdiff_t)*array, indices,
    (int32_t)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(hypre_sstructvariable__array_dimen_f,
                  HYPRE_SSTRUCTVARIABLE__ARRAY_DIMEN_F,
                  Hypre_SStructVariable__array_dimen_f)
  (int64_t *array, int32_t *result)
{
  *result =
    SIDL_int__array_dimen((struct SIDL_int__array *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(hypre_sstructvariable__array_lower_f,
                  HYPRE_SSTRUCTVARIABLE__ARRAY_LOWER_F,
                  Hypre_SStructVariable__array_lower_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    SIDL_int__array_lower((struct SIDL_int__array *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(hypre_sstructvariable__array_upper_f,
                  HYPRE_SSTRUCTVARIABLE__ARRAY_UPPER_F,
                  Hypre_SStructVariable__array_upper_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    SIDL_int__array_upper((struct SIDL_int__array *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(hypre_sstructvariable__array_stride_f,
                  HYPRE_SSTRUCTVARIABLE__ARRAY_STRIDE_F,
                  Hypre_SStructVariable__array_stride_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    SIDL_int__array_stride((struct SIDL_int__array *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(hypre_sstructvariable__array_iscolumnorder_f,
                  HYPRE_SSTRUCTVARIABLE__ARRAY_ISCOLUMNORDER_F,
                  Hypre_SStructVariable__array_isColumnOrder_f)
  (int64_t *array,
   SIDL_F77_Bool *result)
{
  *result = SIDL_int__array_isColumnOrder((struct SIDL_int__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(hypre_sstructvariable__array_isroworder_f,
                  HYPRE_SSTRUCTVARIABLE__ARRAY_ISROWORDER_F,
                  Hypre_SStructVariable__array_isRowOrder_f)
  (int64_t *array,
   SIDL_F77_Bool *result)
{
  *result = SIDL_int__array_isRowOrder((struct SIDL_int__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(hypre_sstructvariable__array_copy_f,
                  HYPRE_SSTRUCTVARIABLE__ARRAY_COPY_F,
                  Hypre_SStructVariable__array_copy_f)
  (int64_t *src,
   int64_t *dest)
{
  SIDL_int__array_copy((const struct SIDL_int__array *)(ptrdiff_t)*src,
                       (struct SIDL_int__array *)(ptrdiff_t)*dest);
}

void
SIDLFortran77Symbol(hypre_sstructvariable__array_smartcopy_f,
                  HYPRE_SSTRUCTVARIABLE__ARRAY_SMARTCOPY_F,
                  Hypre_SStructVariable__array_smartCopy_f)
  (int64_t *src)
{
  SIDL_int__array_smartCopy((struct SIDL_int__array *)(ptrdiff_t)*src);
}

void
SIDLFortran77Symbol(hypre_sstructvariable__array_ensure_f,
                  HYPRE_SSTRUCTVARIABLE__ARRAY_ENSURE_F,
                  Hypre_SStructVariable__array_ensure_f)
  (int64_t *src,
   int32_t *dimen,
   int     *ordering,
   int64_t *result)
{
  *result = (ptrdiff_t)
    SIDL_int__array_ensure((struct SIDL_int__array *)(ptrdiff_t)*src,
    *dimen, *ordering);
}

