/*
 * File:          bHYPRE_ErrorCode.h
 * Symbol:        bHYPRE.ErrorCode-v1.0.0
 * Symbol Type:   enumeration
 * Babel Version: 1.0.0
 * Description:   Client-side glue code for bHYPRE.ErrorCode
 * 
 * WARNING: Automatically generated; changes will be lost
 */

#ifndef included_bHYPRE_ErrorCode_h
#define included_bHYPRE_ErrorCode_h

#ifndef included_bHYPRE_ErrorCode_IOR_h
#include "bHYPRE_ErrorCode_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct bHYPRE_ErrorCode__array*
bHYPRE_ErrorCode__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_ErrorCode__array*
bHYPRE_ErrorCode__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_ErrorCode__array*
bHYPRE_ErrorCode__array_create1d(int32_t len);

struct bHYPRE_ErrorCode__array*
bHYPRE_ErrorCode__array_create1dInit(
  int32_t len, 
  int32_t* data);

struct bHYPRE_ErrorCode__array*
bHYPRE_ErrorCode__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_ErrorCode__array*
bHYPRE_ErrorCode__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_ErrorCode__array*
bHYPRE_ErrorCode__array_borrow(
  int32_t * firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[]);

struct bHYPRE_ErrorCode__array*
bHYPRE_ErrorCode__array_smartCopy(
  struct bHYPRE_ErrorCode__array *array);

void
bHYPRE_ErrorCode__array_addRef(
  struct bHYPRE_ErrorCode__array* array);

void
bHYPRE_ErrorCode__array_deleteRef(
  struct bHYPRE_ErrorCode__array* array);

enum bHYPRE_ErrorCode__enum
bHYPRE_ErrorCode__array_get1(
  const struct bHYPRE_ErrorCode__array* array,
  const int32_t i1);

enum bHYPRE_ErrorCode__enum
bHYPRE_ErrorCode__array_get2(
  const struct bHYPRE_ErrorCode__array* array,
  const int32_t i1,
  const int32_t i2);

enum bHYPRE_ErrorCode__enum
bHYPRE_ErrorCode__array_get3(
  const struct bHYPRE_ErrorCode__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

enum bHYPRE_ErrorCode__enum
bHYPRE_ErrorCode__array_get4(
  const struct bHYPRE_ErrorCode__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

enum bHYPRE_ErrorCode__enum
bHYPRE_ErrorCode__array_get5(
  const struct bHYPRE_ErrorCode__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

enum bHYPRE_ErrorCode__enum
bHYPRE_ErrorCode__array_get6(
  const struct bHYPRE_ErrorCode__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

enum bHYPRE_ErrorCode__enum
bHYPRE_ErrorCode__array_get7(
  const struct bHYPRE_ErrorCode__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

enum bHYPRE_ErrorCode__enum
bHYPRE_ErrorCode__array_get(
  const struct bHYPRE_ErrorCode__array* array,
  const int32_t indices[]);

void
bHYPRE_ErrorCode__array_set1(
  struct bHYPRE_ErrorCode__array* array,
  const int32_t i1,
  enum bHYPRE_ErrorCode__enum const value);

void
bHYPRE_ErrorCode__array_set2(
  struct bHYPRE_ErrorCode__array* array,
  const int32_t i1,
  const int32_t i2,
  enum bHYPRE_ErrorCode__enum const value);

void
bHYPRE_ErrorCode__array_set3(
  struct bHYPRE_ErrorCode__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  enum bHYPRE_ErrorCode__enum const value);

void
bHYPRE_ErrorCode__array_set4(
  struct bHYPRE_ErrorCode__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  enum bHYPRE_ErrorCode__enum const value);

void
bHYPRE_ErrorCode__array_set5(
  struct bHYPRE_ErrorCode__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  enum bHYPRE_ErrorCode__enum const value);

void
bHYPRE_ErrorCode__array_set6(
  struct bHYPRE_ErrorCode__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  enum bHYPRE_ErrorCode__enum const value);

void
bHYPRE_ErrorCode__array_set7(
  struct bHYPRE_ErrorCode__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  enum bHYPRE_ErrorCode__enum const value);

void
bHYPRE_ErrorCode__array_set(
  struct bHYPRE_ErrorCode__array* array,
  const int32_t indices[],
  enum bHYPRE_ErrorCode__enum const value);

int32_t
bHYPRE_ErrorCode__array_dimen(
  const struct bHYPRE_ErrorCode__array* array);

int32_t
bHYPRE_ErrorCode__array_lower(
  const struct bHYPRE_ErrorCode__array* array,
  const int32_t ind);

int32_t
bHYPRE_ErrorCode__array_upper(
  const struct bHYPRE_ErrorCode__array* array,
  const int32_t ind);

int32_t
bHYPRE_ErrorCode__array_length(
  const struct bHYPRE_ErrorCode__array* array,
  const int32_t ind);

int32_t
bHYPRE_ErrorCode__array_stride(
  const struct bHYPRE_ErrorCode__array* array,
  const int32_t ind);

int
bHYPRE_ErrorCode__array_isColumnOrder(
  const struct bHYPRE_ErrorCode__array* array);

int
bHYPRE_ErrorCode__array_isRowOrder(
  const struct bHYPRE_ErrorCode__array* array);

struct bHYPRE_ErrorCode__array*
bHYPRE_ErrorCode__array_slice(
  struct bHYPRE_ErrorCode__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

void
bHYPRE_ErrorCode__array_copy(
  const struct bHYPRE_ErrorCode__array* src,
  struct bHYPRE_ErrorCode__array* dest);

struct bHYPRE_ErrorCode__array*
bHYPRE_ErrorCode__array_ensure(
  struct bHYPRE_ErrorCode__array* src,
  int32_t dimen,
  int     ordering);

#ifdef __cplusplus
}
#endif
#endif
