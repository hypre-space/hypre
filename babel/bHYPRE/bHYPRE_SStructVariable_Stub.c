/*
 * File:          bHYPRE_SStructVariable_Stub.c
 * Symbol:        bHYPRE.SStructVariable-v1.0.0
 * Symbol Type:   enumeration
 * Babel Version: 0.9.8
 * sidl Created:  20050208 15:29:05 PST
 * Generated:     20050208 15:29:07 PST
 * Description:   Client-side glue code for bHYPRE.SStructVariable
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.9.8
 * source-line   = 888
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#include "bHYPRE_SStructVariable.h"
#ifndef included_sidl_int_IOR_h
#include "sidl_int_IOR.h"
#endif
#include <stddef.h>

struct bHYPRE_SStructVariable__array*
bHYPRE_SStructVariable__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[])
{
  return (struct bHYPRE_SStructVariable__array*)sidl_int__array_createCol(dimen,
    lower, upper);
}

struct bHYPRE_SStructVariable__array*
bHYPRE_SStructVariable__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[])
{
  return (struct bHYPRE_SStructVariable__array*)sidl_int__array_createRow(dimen,
    lower, upper);
}

struct bHYPRE_SStructVariable__array*
bHYPRE_SStructVariable__array_create1d(int32_t len)
{
  return (struct bHYPRE_SStructVariable__array*)sidl_int__array_create1d(len);
}

struct bHYPRE_SStructVariable__array*
bHYPRE_SStructVariable__array_create1dInit(
  int32_t len, 
  int32_t* data)
{
  return (struct 
    bHYPRE_SStructVariable__array*)sidl_int__array_create1dInit(len,
    (int32_t*)data);
}

struct bHYPRE_SStructVariable__array*
bHYPRE_SStructVariable__array_create2dCol(int32_t m, int32_t n)
{
  return (struct bHYPRE_SStructVariable__array*)sidl_int__array_create2dCol(m,
    n);
}

struct bHYPRE_SStructVariable__array*
bHYPRE_SStructVariable__array_create2dRow(int32_t m, int32_t n)
{
  return (struct bHYPRE_SStructVariable__array*)sidl_int__array_create2dRow(m,
    n);
}

struct bHYPRE_SStructVariable__array*
bHYPRE_SStructVariable__array_borrow(
  int32_t * firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[])
{
  return (struct bHYPRE_SStructVariable__array*)sidl_int__array_borrow(
    firstElement, dimen, lower, upper, stride);
}

struct bHYPRE_SStructVariable__array*
bHYPRE_SStructVariable__array_smartCopy(
  struct bHYPRE_SStructVariable__array *array)
{
  return (struct bHYPRE_SStructVariable__array*)
    sidl_int__array_smartCopy((struct sidl_int__array *)array);
}

void
bHYPRE_SStructVariable__array_addRef(
  struct bHYPRE_SStructVariable__array* array)
{
  sidl_int__array_addRef((struct sidl_int__array *)array);
}

void
bHYPRE_SStructVariable__array_deleteRef(
  struct bHYPRE_SStructVariable__array* array)
{
  sidl_int__array_deleteRef((struct sidl_int__array *)array);
}

enum bHYPRE_SStructVariable__enum
bHYPRE_SStructVariable__array_get1(
  const struct bHYPRE_SStructVariable__array* array,
  const int32_t i1)
{
  return (enum bHYPRE_SStructVariable__enum)
    sidl_int__array_get1((const struct sidl_int__array *)array
    , i1);
}

enum bHYPRE_SStructVariable__enum
bHYPRE_SStructVariable__array_get2(
  const struct bHYPRE_SStructVariable__array* array,
  const int32_t i1,
  const int32_t i2)
{
  return (enum bHYPRE_SStructVariable__enum)
    sidl_int__array_get2((const struct sidl_int__array *)array
    , i1, i2);
}

enum bHYPRE_SStructVariable__enum
bHYPRE_SStructVariable__array_get3(
  const struct bHYPRE_SStructVariable__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3)
{
  return (enum bHYPRE_SStructVariable__enum)
    sidl_int__array_get3((const struct sidl_int__array *)array
    , i1, i2, i3);
}

enum bHYPRE_SStructVariable__enum
bHYPRE_SStructVariable__array_get4(
  const struct bHYPRE_SStructVariable__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4)
{
  return (enum bHYPRE_SStructVariable__enum)
    sidl_int__array_get4((const struct sidl_int__array *)array
    , i1, i2, i3, i4);
}

enum bHYPRE_SStructVariable__enum
bHYPRE_SStructVariable__array_get5(
  const struct bHYPRE_SStructVariable__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5)
{
  return (enum bHYPRE_SStructVariable__enum)
    sidl_int__array_get5((const struct sidl_int__array *)array
    , i1, i2, i3, i4, i5);
}

enum bHYPRE_SStructVariable__enum
bHYPRE_SStructVariable__array_get6(
  const struct bHYPRE_SStructVariable__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6)
{
  return (enum bHYPRE_SStructVariable__enum)
    sidl_int__array_get6((const struct sidl_int__array *)array
    , i1, i2, i3, i4, i5, i6);
}

enum bHYPRE_SStructVariable__enum
bHYPRE_SStructVariable__array_get7(
  const struct bHYPRE_SStructVariable__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7)
{
  return (enum bHYPRE_SStructVariable__enum)
    sidl_int__array_get7((const struct sidl_int__array *)array
    , i1, i2, i3, i4, i5, i6, i7);
}

enum bHYPRE_SStructVariable__enum
bHYPRE_SStructVariable__array_get(
  const struct bHYPRE_SStructVariable__array* array,
  const int32_t indices[])
{
  return (enum bHYPRE_SStructVariable__enum)
    sidl_int__array_get((const struct sidl_int__array *)array, indices);
}

void
bHYPRE_SStructVariable__array_set1(
  struct bHYPRE_SStructVariable__array* array,
  const int32_t i1,
  enum bHYPRE_SStructVariable__enum const value)
{
  sidl_int__array_set1((struct sidl_int__array *)array
  , i1, (int32_t)value);
}

void
bHYPRE_SStructVariable__array_set2(
  struct bHYPRE_SStructVariable__array* array,
  const int32_t i1,
  const int32_t i2,
  enum bHYPRE_SStructVariable__enum const value)
{
  sidl_int__array_set2((struct sidl_int__array *)array
  , i1, i2, (int32_t)value);
}

void
bHYPRE_SStructVariable__array_set3(
  struct bHYPRE_SStructVariable__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  enum bHYPRE_SStructVariable__enum const value)
{
  sidl_int__array_set3((struct sidl_int__array *)array
  , i1, i2, i3, (int32_t)value);
}

void
bHYPRE_SStructVariable__array_set4(
  struct bHYPRE_SStructVariable__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  enum bHYPRE_SStructVariable__enum const value)
{
  sidl_int__array_set4((struct sidl_int__array *)array
  , i1, i2, i3, i4, (int32_t)value);
}

void
bHYPRE_SStructVariable__array_set5(
  struct bHYPRE_SStructVariable__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  enum bHYPRE_SStructVariable__enum const value)
{
  sidl_int__array_set5((struct sidl_int__array *)array
  , i1, i2, i3, i4, i5, (int32_t)value);
}

void
bHYPRE_SStructVariable__array_set6(
  struct bHYPRE_SStructVariable__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  enum bHYPRE_SStructVariable__enum const value)
{
  sidl_int__array_set6((struct sidl_int__array *)array
  , i1, i2, i3, i4, i5, i6, (int32_t)value);
}

void
bHYPRE_SStructVariable__array_set7(
  struct bHYPRE_SStructVariable__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  enum bHYPRE_SStructVariable__enum const value)
{
  sidl_int__array_set7((struct sidl_int__array *)array
  , i1, i2, i3, i4, i5, i6, i7, (int32_t)value);
}

void
bHYPRE_SStructVariable__array_set(
  struct bHYPRE_SStructVariable__array* array,
  const int32_t indices[],
  enum bHYPRE_SStructVariable__enum const value)
{
  sidl_int__array_set((struct sidl_int__array *)array, indices, (int32_t)value);
}

int32_t
bHYPRE_SStructVariable__array_dimen(
  const struct bHYPRE_SStructVariable__array* array)
{
  return sidl_int__array_dimen((struct sidl_int__array *)array);
}

int32_t
bHYPRE_SStructVariable__array_lower(
  const struct bHYPRE_SStructVariable__array* array,
  const int32_t ind)
{
  return sidl_int__array_lower((struct sidl_int__array *)array, ind);
}

int32_t
bHYPRE_SStructVariable__array_upper(
  const struct bHYPRE_SStructVariable__array* array,
  const int32_t ind)
{
  return sidl_int__array_upper((struct sidl_int__array *)array, ind);
}

int32_t
bHYPRE_SStructVariable__array_length(
  const struct bHYPRE_SStructVariable__array* array,
  const int32_t ind)
{
  return sidl_int__array_length((struct sidl_int__array *)array, ind);
}

int32_t
bHYPRE_SStructVariable__array_stride(
  const struct bHYPRE_SStructVariable__array* array,
  const int32_t ind)
{
  return sidl_int__array_stride((struct sidl_int__array *)array, ind);
}

int
bHYPRE_SStructVariable__array_isColumnOrder(
  const struct bHYPRE_SStructVariable__array* array)
{
  return sidl_int__array_isColumnOrder((struct sidl_int__array *)array);
}

int
bHYPRE_SStructVariable__array_isRowOrder(
  const struct bHYPRE_SStructVariable__array* array)
{
  return sidl_int__array_isRowOrder((struct sidl_int__array *)array);
}

void
bHYPRE_SStructVariable__array_copy(
  const struct bHYPRE_SStructVariable__array* src,
  struct bHYPRE_SStructVariable__array* dest)
{
  sidl_int__array_copy((const struct sidl_int__array *)src,
                       (struct sidl_int__array *)dest);
}

struct bHYPRE_SStructVariable__array*
bHYPRE_SStructVariable__array_slice(
  struct bHYPRE_SStructVariable__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart)
{
  return (struct bHYPRE_SStructVariable__array*)
    sidl_int__array_slice((struct sidl_int__array *)src,
                          dimen, numElem, srcStart, srcStride, newStart);
}

struct bHYPRE_SStructVariable__array*
bHYPRE_SStructVariable__array_ensure(
  struct bHYPRE_SStructVariable__array* src,
  int32_t dimen,
  int     ordering)
{
  return (struct bHYPRE_SStructVariable__array*)
    sidl_int__array_ensure((struct sidl_int__array *)src, dimen, ordering);
}

