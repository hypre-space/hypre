/*
 * File:          bHYPRE_SStructVariable_Stub.c
 * Symbol:        bHYPRE.SStructVariable-v1.0.0
 * Symbol Type:   enumeration
 * Babel Version: 0.8.2
 * SIDL Created:  20030401 14:47:20 PST
 * Generated:     20030401 14:47:27 PST
 * Description:   Client-side glue code for bHYPRE.SStructVariable
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.2
 * source-line   = 888
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#include "bHYPRE_SStructVariable.h"
#ifndef included_SIDL_int_IOR_h
#include "SIDL_int_IOR.h"
#endif
#include <stddef.h>

struct bHYPRE_SStructVariable__array*
bHYPRE_SStructVariable__array_createCol(int32_t        dimen,
                                        const int32_t lower[],
                                        const int32_t upper[])
{
  return (struct bHYPRE_SStructVariable__array*)SIDL_int__array_createCol(dimen,
    lower, upper);
}

struct bHYPRE_SStructVariable__array*
bHYPRE_SStructVariable__array_createRow(int32_t        dimen,
                                        const int32_t lower[],
                                        const int32_t upper[])
{
  return (struct bHYPRE_SStructVariable__array*)SIDL_int__array_createRow(dimen,
    lower, upper);
}

struct bHYPRE_SStructVariable__array*
bHYPRE_SStructVariable__array_create1d(int32_t len)
{
  return (struct bHYPRE_SStructVariable__array*)SIDL_int__array_create1d(len);
}

struct bHYPRE_SStructVariable__array*
bHYPRE_SStructVariable__array_create2dCol(int32_t m, int32_t n)
{
  return (struct bHYPRE_SStructVariable__array*)SIDL_int__array_create2dCol(m,
    n);
}

struct bHYPRE_SStructVariable__array*
bHYPRE_SStructVariable__array_create2dRow(int32_t m, int32_t n)
{
  return (struct bHYPRE_SStructVariable__array*)SIDL_int__array_create2dRow(m,
    n);
}

struct bHYPRE_SStructVariable__array*
bHYPRE_SStructVariable__array_borrow(int32_t *firstElement,
                                     int32_t       dimen,
const int32_t lower[],
const int32_t upper[],
const int32_t stride[])
{
  return (struct bHYPRE_SStructVariable__array*)SIDL_int__array_borrow(
    firstElement, dimen, lower, upper, stride);
}

struct bHYPRE_SStructVariable__array*
bHYPRE_SStructVariable__array_smartCopy(struct bHYPRE_SStructVariable__array 
  *array)
{
  return (struct bHYPRE_SStructVariable__array*)
    SIDL_int__array_smartCopy((struct SIDL_int__array *)array);
}

void
bHYPRE_SStructVariable__array_addRef(struct bHYPRE_SStructVariable__array* 
  array)
{
  SIDL_int__array_addRef((struct SIDL_int__array *)array);
}

void
bHYPRE_SStructVariable__array_deleteRef(struct bHYPRE_SStructVariable__array* 
  array)
{
  SIDL_int__array_deleteRef((struct SIDL_int__array *)array);
}

enum bHYPRE_SStructVariable__enum
bHYPRE_SStructVariable__array_get1(const struct bHYPRE_SStructVariable__array* 
  array,
                                   const int32_t i1)
{
  return (enum bHYPRE_SStructVariable__enum)
    SIDL_int__array_get1((const struct SIDL_int__array *)array
    , i1);
}

enum bHYPRE_SStructVariable__enum
bHYPRE_SStructVariable__array_get2(const struct bHYPRE_SStructVariable__array* 
  array,
                                   const int32_t i1,
                                   const int32_t i2)
{
  return (enum bHYPRE_SStructVariable__enum)
    SIDL_int__array_get2((const struct SIDL_int__array *)array
    , i1, i2);
}

enum bHYPRE_SStructVariable__enum
bHYPRE_SStructVariable__array_get3(const struct bHYPRE_SStructVariable__array* 
  array,
                                   const int32_t i1,
                                   const int32_t i2,
                                   const int32_t i3)
{
  return (enum bHYPRE_SStructVariable__enum)
    SIDL_int__array_get3((const struct SIDL_int__array *)array
    , i1, i2, i3);
}

enum bHYPRE_SStructVariable__enum
bHYPRE_SStructVariable__array_get4(const struct bHYPRE_SStructVariable__array* 
  array,
                                   const int32_t i1,
                                   const int32_t i2,
                                   const int32_t i3,
                                   const int32_t i4)
{
  return (enum bHYPRE_SStructVariable__enum)
    SIDL_int__array_get4((const struct SIDL_int__array *)array
    , i1, i2, i3, i4);
}

enum bHYPRE_SStructVariable__enum
bHYPRE_SStructVariable__array_get(const struct bHYPRE_SStructVariable__array* 
  array,
                                  const int32_t indices[])
{
  return (enum bHYPRE_SStructVariable__enum)
    SIDL_int__array_get((const struct SIDL_int__array *)array, indices);
}

void
bHYPRE_SStructVariable__array_set1(struct bHYPRE_SStructVariable__array* array,
                                   const int32_t i1,
                                   enum bHYPRE_SStructVariable__enum const 
  value)
{
  SIDL_int__array_set1((struct SIDL_int__array *)array
  , i1, (int32_t)value);
}

void
bHYPRE_SStructVariable__array_set2(struct bHYPRE_SStructVariable__array* array,
                                   const int32_t i1,
                                   const int32_t i2,
                                   enum bHYPRE_SStructVariable__enum const 
  value)
{
  SIDL_int__array_set2((struct SIDL_int__array *)array
  , i1, i2, (int32_t)value);
}

void
bHYPRE_SStructVariable__array_set3(struct bHYPRE_SStructVariable__array* array,
                                   const int32_t i1,
                                   const int32_t i2,
                                   const int32_t i3,
                                   enum bHYPRE_SStructVariable__enum const 
  value)
{
  SIDL_int__array_set3((struct SIDL_int__array *)array
  , i1, i2, i3, (int32_t)value);
}

void
bHYPRE_SStructVariable__array_set4(struct bHYPRE_SStructVariable__array* array,
                                   const int32_t i1,
                                   const int32_t i2,
                                   const int32_t i3,
                                   const int32_t i4,
                                   enum bHYPRE_SStructVariable__enum const 
  value)
{
  SIDL_int__array_set4((struct SIDL_int__array *)array
  , i1, i2, i3, i4, (int32_t)value);
}

void
bHYPRE_SStructVariable__array_set(struct bHYPRE_SStructVariable__array* array,
                                  const int32_t indices[],
                                  enum bHYPRE_SStructVariable__enum const value)
{
  SIDL_int__array_set((struct SIDL_int__array *)array, indices, (int32_t)value);
}

int32_t
bHYPRE_SStructVariable__array_dimen(const struct bHYPRE_SStructVariable__array* 
  array)
{
  return SIDL_int__array_dimen((struct SIDL_int__array *)array);
}

int32_t
bHYPRE_SStructVariable__array_lower(const struct bHYPRE_SStructVariable__array* 
  array,
                                    const int32_t ind)
{
  return SIDL_int__array_lower((struct SIDL_int__array *)array, ind);
}

int32_t
bHYPRE_SStructVariable__array_upper(const struct bHYPRE_SStructVariable__array* 
  array,
                                    const int32_t ind)
{
  return SIDL_int__array_upper((struct SIDL_int__array *)array, ind);
}

int32_t
bHYPRE_SStructVariable__array_stride(const struct 
  bHYPRE_SStructVariable__array* array,
                                     const int32_t ind)
{
  return SIDL_int__array_stride((struct SIDL_int__array *)array, ind);
}

int
bHYPRE_SStructVariable__array_isColumnOrder(const struct 
  bHYPRE_SStructVariable__array* array)
{
  return SIDL_int__array_isColumnOrder((struct SIDL_int__array *)array);
}

int
bHYPRE_SStructVariable__array_isRowOrder(const struct 
  bHYPRE_SStructVariable__array* array)
{
  return SIDL_int__array_isRowOrder((struct SIDL_int__array *)array);
}

void
bHYPRE_SStructVariable__array_copy(const struct bHYPRE_SStructVariable__array* 
  src,
                                         struct bHYPRE_SStructVariable__array* 
  dest)
{
  SIDL_int__array_copy((const struct SIDL_int__array *)src,
                       (struct SIDL_int__array *)dest);
}

struct bHYPRE_SStructVariable__array*
bHYPRE_SStructVariable__array_ensure(struct bHYPRE_SStructVariable__array* src,
                                     int32_t dimen,
                                     int     ordering)
{
  return (struct bHYPRE_SStructVariable__array*)
    SIDL_int__array_ensure((struct SIDL_int__array *)src, dimen, ordering);
}

