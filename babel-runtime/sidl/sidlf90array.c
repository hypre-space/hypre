/*
 * File:        sidlf90array.c
 * Copyright:   (c) 2003 The Regents of the University of California
 * Revision:    @(#) $Revision$
 * Date:        $Date$
 * Description: Functions to convert sidl arrays into F90 derived types
 *
 */

#include "sidlf90array.h"

#if defined(SIDL_MAX_F90_DESCRIPTOR) && !defined(FORTRAN90_DISABLED)

#include <stddef.h>
#include <stdio.h>
#include "CompilerCharacteristics.h"
#include "F90Compiler.h"

#ifndef included_sidl_double_IOR_h
#include "sidl_double_IOR.h"
#endif
#ifndef included_sidl_dcomplex_IOR_h
#include "sidl_dcomplex_IOR.h"
#endif
#ifndef included_sidl_fcomplex_IOR_h
#include "sidl_fcomplex_IOR.h"
#endif
#ifndef included_sidl_float_IOR_h
#include "sidl_float_IOR.h"
#endif
#ifndef included_sidl_int_IOR_h
#include "sidl_int_IOR.h"
#endif
#ifndef included_sidl_long_IOR_h
#include "sidl_long_IOR.h"
#endif

static F90_CompilerCharacteristics s_cc;

static int
getCompilerCharacteristics(void)
{
  static int s_notInitialized = 1;
  if (s_notInitialized) {
    s_notInitialized = F90_SetCompilerCharacteristics(&s_cc, FORTRAN_COMPILER);
    if (s_notInitialized) {
      fprintf(stderr,
              "Cannot determine F90 compiler characteristics for %s\n",
              FORTRAN_COMPILER);
    }
  }
  return s_notInitialized;
}

static int
genericConvert(void *ior,
               void *first,
               const int32_t lower[],
               const int32_t upper[],
               const int32_t stride[],
               const int dimen,
               const F90_ArrayDataType data_type,
               const size_t elem_size,
               struct sidl_fortran_array *dest)
{
  long unsigned extent[7];
#if (SIZEOF_LONG == 4)
  long * const low = (long *)lower;
#else
  long low[7];
#endif
  long chasmStride[7];
  int i;
  if (getCompilerCharacteristics()) return 1;
  dest->d_ior = (ptrdiff_t)ior;
  for(i = 0; i < dimen; ++i){
#if (SIZEOF_LONG != 4)
    low[i] = (long)lower[i];
#endif
    extent[i] = (upper[i] >= (lower[i]-1)) ? (1 + upper[i] - lower[i]) : 0;
    chasmStride[i] = stride[i]*elem_size;
  }
  return (s_cc.setArrayDesc)((void *)dest->d_descriptor, first, dimen,
                             F90_ArrayPointerInDerived,
                             data_type, elem_size,
                             low, extent, chasmStride);
}

static int 
genericNullify(const int                   dimen,
               F90_ArrayDataType           data_type,
               size_t                     elem_size,
               struct sidl_fortran_array *dest)
{
  static const long lower[7] = { 1, 1, 1, 1, 1, 1, 1};
  static const unsigned long extent[7] =  {1, 1, 1, 1, 1, 1, 1};
  long stride[7];
  struct sidl_dcomplex junk;
  int i;
  if (getCompilerCharacteristics()) return 1;
  dest->d_ior = 0;
  for(i = 0; i < dimen; ++i) {
    stride[i] = elem_size;
  }
  if ((s_cc.setArrayDesc)(dest->d_descriptor, (void *)&junk, 
                          dimen, F90_ArrayPointerInDerived,
                          data_type, elem_size,
                          lower, extent, stride)) return 1;
  (s_cc.nullifyArrayDesc)(dest->d_descriptor, dimen);
  return 0;
}

int
sidl_dcomplex__array_convert2f90(const struct sidl_dcomplex__array *src,
                                 const int src_dimen,
                                 struct sidl_fortran_array *dest)
{
  return src 
    ? genericConvert((void *)src,
                     (void *)src->d_firstElement,
                     src->d_metadata.d_lower,
                     src->d_metadata.d_upper,
                     src->d_metadata.d_stride,
                     src->d_metadata.d_dimen,
                     F90_DComplex,
                     sizeof(struct sidl_dcomplex),
                     dest)
    : genericNullify(src_dimen, F90_DComplex,
                     sizeof(struct sidl_dcomplex), dest);
}

int
sidl_double__array_convert2f90(const struct sidl_double__array *src,
                               const int src_dimen,
                               struct sidl_fortran_array *dest)
{
  return src 
    ? genericConvert((void *)src,
                     (void *)src->d_firstElement,
                     src->d_metadata.d_lower,
                     src->d_metadata.d_upper,
                     src->d_metadata.d_stride,
                     src_dimen,
                     F90_Double,
                     sizeof(double),
                     dest)
    : genericNullify(src_dimen, F90_Double,
                     sizeof(double), dest);
}

int
sidl_fcomplex__array_convert2f90(const struct sidl_fcomplex__array *src,
                                 const int src_dimen,
                                 struct sidl_fortran_array *dest)
{
  return src 
    ? genericConvert((void *)src,
                     (void *)src->d_firstElement,
                     src->d_metadata.d_lower,
                     src->d_metadata.d_upper,
                     src->d_metadata.d_stride,
                     src_dimen,
                     F90_Complex,
                     sizeof(struct sidl_fcomplex),
                     dest)
    : genericNullify(src_dimen, F90_Complex,
                     sizeof(struct sidl_fcomplex), dest);
}

int
sidl_float__array_convert2f90(const struct sidl_float__array *src,
                              const int src_dimen,
                              struct sidl_fortran_array *dest)
{
  return src 
    ? genericConvert((void *)src,
                     (void *)src->d_firstElement,
                     src->d_metadata.d_lower,
                     src->d_metadata.d_upper,
                     src->d_metadata.d_stride,
                     src_dimen,
                     F90_Real,
                     sizeof(float),
                     dest)
    : genericNullify(src_dimen, F90_Real,
                     sizeof(float), dest);
}

int
sidl_int__array_convert2f90(const struct sidl_int__array *src,
                            const int src_dimen,
                            struct sidl_fortran_array *dest)
{
  return src 
    ? genericConvert((void *)src,
                     (void *)src->d_firstElement,
                     src->d_metadata.d_lower,
                     src->d_metadata.d_upper,
                     src->d_metadata.d_stride,
                     src_dimen,
                     F90_Integer4,
                     sizeof(int32_t),
                     dest)
    : genericNullify(src_dimen, F90_Integer4,
                     sizeof(int32_t), dest);
}


int
sidl_long__array_convert2f90(const struct sidl_long__array *src,
                             const int src_dimen,
                             struct sidl_fortran_array *dest)
{
  return src 
    ? genericConvert((void *)src,
                     (void *)src->d_firstElement,
                     src->d_metadata.d_lower,
                     src->d_metadata.d_upper,
                     src->d_metadata.d_stride,
                     src_dimen,
                     F90_Integer8,
                     sizeof(int64_t),
                     dest)
    : genericNullify(src_dimen, F90_Integer8,
                     sizeof(int64_t), dest);
}
#endif /* defined(SIDL_MAX_F90_DESCRIPTOR) && !defined(FORTRAN90_DISABLED) */
