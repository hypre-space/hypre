/*
 * File:        sidlArray.h
 * Revision:    @(#) $Revision$
 * Date:        $Date$
 * Description: Generic array data types & convenience macros
 *
 * Copyright (c) 2000-2004, The Regents of the University of Calfornia.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the Components Team <components@llnl.gov>
 * UCRL-CODE-2002-054
 * All rights reserved.
 * 
 * This file is part of Babel. For more information, see
 * http://www.llnl.gov/CASC/components/. Please read the COPYRIGHT file
 * for Our Notice and the LICENSE file for the GNU Lesser General Public
 * License.
 * 
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License (as published by
 * the Free Software Foundation) version 2.1 dated February 1999.
 * 
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms and
 * conditions of the GNU Lesser General Public License for more details.
 * 
 * You should have recieved a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */

#ifndef included_sidlArray_h
#define included_sidlArray_h

#ifndef included_sidlType_h
/* need int32_t */
#include "sidlType.h"
#endif /* included_sidlType_h */

/*
 * The maximum dimension for a SIDL array is defined here.
 * It should probably be 7 because Fortran arrays are limited to 7
 * dimensions.
 */
#define SIDL_MAX_ARRAY_DIMENSION 7

enum sidl_array_ordering {
  sidl_general_order=0, /* this must be zero (i.e. a false value) */
  sidl_column_major_order=1,
  sidl_row_major_order=2
};

enum sidl_array_type {
  /* these values must match values used in F77 & F90 too */
  sidl_bool_array      = 1,
  sidl_char_array      = 2,
  sidl_dcomplex_array  = 3,
  sidl_double_array    = 4,
  sidl_fcomplex_array  = 5,
  sidl_float_array     = 6,
  sidl_int_array       = 7,
  sidl_long_array      = 8,
  sidl_opaque_array    = 9,
  sidl_string_array    = 10,
  sidl_interface_array = 11 /* an array of sidl.BaseInterface's */
};

struct sidl__array; /* forward declaration */

#ifdef __cplusplus
extern "C" { /*}*/
#endif
/**
 * The virtual function table for the multi-dimensional arrays of
 * any type.
 */
struct sidl__array_vtable {

  /*
   * This function should release resources associates with the array
   * passed in.  It is called when the reference count goes to zero.
   */
  void (*d_destroy)(struct sidl__array *);

  /*
   * If this array controls its own data (i.e. owns the memory), this
   * can simply increment the reference count of the argument and
   * return it.  If the data is borrowed (e.g. a borrowed array), this
   * should make a new array of the same size and copy data from the
   * passed in array to the new array.
   */
  struct sidl__array *(*d_smartcopy)(struct sidl__array *);

  /*
   * Return the type of the array. The type is an integer value
   * that should match one of the values in enum sidl_array_type.
   */
  int32_t (*d_arraytype)(void);
};
#ifdef __cplusplus
}
#endif

struct sidl__array {
  int32_t                         *d_lower;
  int32_t                         *d_upper;
  int32_t                         *d_stride;
  const struct sidl__array_vtable *d_vtable;
  int32_t                          d_dimen;
  int32_t                          d_refcount;
};

/**
 * The relation operators available in the built-in quantifier operators.
 */
#define RELATION_OP_EQUAL         0
#define RELATION_OP_NOT_EQUAL     1
#define RELATION_OP_LESS_THAN     2
#define RELATION_OP_LESS_EQUAL    3
#define RELATION_OP_GREATER_THAN  4
#define RELATION_OP_GREATER_EQUAL 5

/**
 * Return the dimension of the array.
 */
#define sidlArrayDim(array) (((const struct sidl__array *)(array))->d_dimen)

/**
 * Macro to return the lower bound on the index for dimension ind of array.
 * A valid index for dimension ind must be greater than or equal to
 * sidlLower(array,ind). 
 */
#define sidlLower(array,ind) (((const struct sidl__array *)(array))->d_lower[(ind)])

/**
 * Macro to return the upper bound on the index for dimension ind of array.
 * A valid index for dimension ind must be less than or equal to
 * sidlUpper(array,ind). 
 */
#define sidlUpper(array,ind) (((const struct sidl__array *)(array))->d_upper[(ind)])

/**
 * Macro to return the number of elements in dimension ind of an array.
 */
#define sidlLength(array,ind) (sidlUpper((array),(ind)) - \
                               sidlLower((array),(ind)) + 1)

/**
 * Macro to return the stride between elements in a particular dimension.
 * To move from the address of element i to element i + 1 in the dimension
 * ind, add sidlStride(array,ind).
 */
#define sidlStride(array,ind) (((const struct sidl__array *)(array))->d_stride[(ind)])

/**
 * Helper macro for calculating the offset in a particular dimension.
 * This macro makes multiple references to array and ind, so you should
 * not use ++ or -- on arguments to this macro.
 */
#define sidlArrayDimCalc(array, ind, var) \
  (sidlStride(array,ind)*((var) - sidlLower(array,ind)))


/**
 * Return the address of an element in a one dimensional array.
 * This macro may make multiple references to array and ind1, so do not
 * use ++ or -- when using this macro.
 */
#define sidlArrayAddr1(array, ind1) \
  ((array)->d_firstElement + sidlArrayDimCalc(array, 0, ind1))

/**
 * Macro to return an element of a one dimensional array as an LVALUE
 * (i.e. it can appear on the left hand side of an assignment operator or it
 * can appear in a right hand side). This macro may make multiple references
 * to array and ind1, so do not use ++ or -- when using this macro.
 */
#define sidlArrayElem1(array, ind1) \
  (*(sidlArrayAddr1(array,ind1)))


/**
 * Return the address of an element in a two dimensional array.
 * This macro may make multiple references to array, ind1 & ind2; so do not
 * use ++ or -- when using this macro.
 */
#define sidlArrayAddr2(array, ind1, ind2) \
  (sidlArrayAddr1(array, ind1) + sidlArrayDimCalc(array, 1, ind2))

/**
 * Macro to return an element of a two dimensional array as an LVALUE
 * (i.e. it can appear on the left hand side of an assignment operator or it
 * can appear in a right hand side). This macro may make  multiple 
 * references to array, ind1 and ind2; so do not use ++ or -- when using
 * this macro. 
 */
#define sidlArrayElem2(array, ind1, ind2) \
  (*(sidlArrayAddr2(array, ind1, ind2)))


/**
 * Return the address of an element in a three dimensional array.
 * This macro may make multiple references to array, ind1, ind2 & ind3; so
 * do not use ++ or -- when using this macro.
 */
#define sidlArrayAddr3(array, ind1, ind2, ind3) \
  (sidlArrayAddr2(array, ind1, ind2) + sidlArrayDimCalc(array, 2, ind3))

/**
 * Macro to return an element of a three dimensional array as an LVALUE
 * (i.e. it can appear on the left hand side of an assignment operator or it
 * can appear in a right hand side). This macro may make multiple references
 * to array, ind1, ind2 & ind3; so do  not use ++ or -- when using this
 * macro. 
 */
#define sidlArrayElem3(array, ind1, ind2, ind3) \
  (*(sidlArrayAddr3(array, ind1, ind2, ind3)))


/**
 * Return the address of an element in a four dimensional array.
 * This macro may make multiple references to array, ind1, ind2, ind3 &
 * ind4; so do not use ++ or -- when using this macro.
 */
#define sidlArrayAddr4(array, ind1, ind2, ind3, ind4) \
  (sidlArrayAddr3(array, ind1, ind2, ind3) + sidlArrayDimCalc(array, 3, ind4))

/**
 * Macro to return an element of a four dimensional array as an LVALUE
 * (i.e. it can appear on the left hand side of an assignment operator or it
 * can appear in a right hand side).  This macro may make multiple
 * references to array, ind1, ind2, ind3 & ind4; so do not use ++ or -- when
 * using this macro. 
 */
#define sidlArrayElem4(array, ind1, ind2, ind3, ind4) \
  (*(sidlArrayAddr4(array, ind1, ind2, ind3, ind4)))

/**
 * Return the address of an element in a five dimensional array.
 * This macro may make multiple references to array, ind1, ind2, ind3,
 * ind4 & ind5; so do not use ++ or -- when using this macro.
 */
#define sidlArrayAddr5(array, ind1, ind2, ind3, ind4, ind5) \
  (sidlArrayAddr4(array, ind1, ind2, ind3, ind4) + \
   sidlArrayDimCalc(array, 4, ind5))

/**
 * Macro to return an element of a five dimensional array as an LVALUE
 * (i.e. it can appear on the left hand side of an assignment operator or it
 * can appear in a right hand side).  This macro may make multiple
 * references to array, ind1, ind2, ind3, ind4 & ind5; so do not use ++ or
 * -- when using this macro.
 */
#define sidlArrayElem5(array, ind1, ind2, ind3, ind4, ind5) \
  (*(sidlArrayAddr5(array, ind1, ind2, ind3, ind4, ind5)))

/**
 * Return the address of an element in a six dimensional array.
 * This macro may make multiple references to array, ind1, ind2, ind3,
 * ind4, ind5 & ind6; so do not use ++ or -- when using this macro.
 */
#define sidlArrayAddr6(array, ind1, ind2, ind3, ind4, ind5, ind6) \
  (sidlArrayAddr5(array, ind1, ind2, ind3, ind4, ind5) + \
   sidlArrayDimCalc(array, 5, ind6))

/**
 * Macro to return an element of a six dimensional array as an LVALUE
 * (i.e. it can appear on the left hand side of an assignment operator or it
 * can appear in a right hand side).  This macro may make multiple
 * references to array, ind1, ind2, ind3, ind4, ind5 & ind6; so do not use
 * ++ or -- when using this macro.
 */
#define sidlArrayElem6(array, ind1, ind2, ind3, ind4, ind5, ind6) \
  (*(sidlArrayAddr6(array, ind1, ind2, ind3, ind4, ind5, ind6)))

/**
 * Return the address of an element in a seven dimensional array.
 * This macro may make multiple references to array, ind1, ind2, ind3,
 * ind4, ind5, ind6 & ind7; so do not use ++ or -- when using this macro.
 */
#define sidlArrayAddr7(array, ind1, ind2, ind3, ind4, ind5, ind6, ind7) \
  (sidlArrayAddr6(array, ind1, ind2, ind3, ind4, ind5, ind6) + \
   sidlArrayDimCalc(array, 6, ind7))

/**
 * Macro to return an element of a seven dimensional array as an LVALUE
 * (i.e. it can appear on the left hand side of an assignment operator or it
 * can appear in a right hand side).  This macro may make multiple
 * references to array, ind1, ind2, ind3, ind4, ind5, ind6 & ind7; so do not
 * use ++ or -- when using this macro.
 */
#define sidlArrayElem7(array, ind1, ind2, ind3, ind4, ind5, ind6, ind7) \
  (*(sidlArrayAddr7(array, ind1, ind2, ind3, ind4, ind5, ind6, ind7)))

/**
 * Macro to return an address of a one dimensional rarray as an LVALUE
 * (i.e. it can appear on the left hand side of an assignment operator or it
 * can appear in a right hand side).  (Rarrays are just native arrays, but
 * they are in column order, so these macros may be useful in C.)
 */
#define RarrayAddr1(array, ind1) \
  ((array)+(ind1))

/**
 * Macro to return an element of a one dimensional rarray as an LVALUE
 * (i.e. it can appear on the left hand side of an assignment operator or it
 * can appear in a right hand side).  (Rarrays are just native arrays, but
 * they are in column order, so these macros may be useful in C.)
 */
#define RarrayElem1(array, ind1) \
  (*(RarrayAddr1(array, ind1)))

/**
 * Macro to return an address of a two dimensional rarray as an LVALUE
 * (i.e. it can appear on the left hand side of an assignment operator or it
 * can appear in a right hand side).  (Rarrays are just native arrays, but
 * they are in column order, so these macros may be useful in C.)
 * @param ind? is the element you wish to reference in dimension ?.
 * @param len? is the length of the dimension ?. 
 */
#define RarrayAddr2(array, ind1, ind2, len1)		\
  ((array)+(ind1)+((ind2)*(len1)))

/**
 * Macro to return an element of a two dimensional rarray as an LVALUE
 * (i.e. it can appear on the left hand side of an assignment operator or it
 * can appear in a right hand side).  (Rarrays are just native arrays, but
 * they are in column order, so these macros may be useful in C.)
 * @param ind? is the element you wish to reference in dimension ?.
 * @param len? is the length of the dimension ?. 
 */
#define RarrayElem2(array, ind1, ind2, len1)		\
  (*(RarrayAddr2(array, ind1, ind2, len1)))

/**
 * Macro to return an address of a three dimensional rarray as an LVALUE
 * (i.e. it can appear on the left hand side of an assignment operator or it
 * can appear in a right hand side).  (Rarrays are just native arrays, but
 * they are in column order, so these macros may be useful in C.)
 * @param ind? is the element you wish to reference in dimension ?.
 * @param len? is the length of the dimension ?. 
 */
#define RarrayAddr3(array, ind1, ind2, ind3, len1, len2)	\
  ((array)+(ind1)+((ind2)*(len1))+((ind3)*(len1)*(len2)))

/**
 * Macro to return an element of a three dimensional rarray as an LVALUE
 * (i.e. it can appear on the left hand side of an assignment operator or it
 * can appear in a right hand side).  (Rarrays are just native arrays, but
 * they are in column order, so these macros may be useful in C.)
 * @param ind? is the element you wish to reference in dimension ?.
 * @param len? is the length of the dimension ?. 
 */
#define RarrayElem3(array, ind1, ind2, ind3, len1, len2)	\
  (*(RarrayAddr3(array, ind1, ind2, ind3, len1, len2)))

/**
 * Macro to return an address of a four dimensional rarray as an LVALUE
 * (i.e. it can appear on the left hand side of an assignment operator or it
 * can appear in a right hand side).  (Rarrays are just native arrays, but
 * they are in column order, so these macros may be useful in C.)
 * @param ind? is the element you wish to reference in dimension ?.
 * @param len? is the length of the dimension ?. 
 */
#define RarrayAddr4(array, ind1, ind2, ind3, ind4, len1, len2, len3)	\
  ((array)+(ind1)+((ind2)*(len1))+((ind3)*(len1)*(len2))+((ind4)*(len1)*(len2)*(len3)))

/**
 * Macro to return an element of a four dimensional rarray as an LVALUE
 * (i.e. it can appear on the left hand side of an assignment operator or it
 * can appear in a right hand side).  (Rarrays are just native arrays, but
 * they are in column order, so these macros may be useful in C.)
 * @param ind? is the element you wish to reference in dimension ?.
 * @param len? is the length of the dimension ?. 
 */
#define RarrayElem4(array, ind1, ind2, ind3, ind4, len1, len2, len3)	\
  (*(RarrayAddr4(array, ind1, ind2, ind3, ind4, len1, len2, len3)))

/**
 * Macro to return an address of a five dimensional rarray as an LVALUE
 * (i.e. it can appear on the left hand side of an assignment operator or it
 * can appear in a right hand side).  (Rarrays are just native arrays, but
 * they are in column order, so these macros may be useful in C.)
 * @param ind? is the element you wish to reference in dimension ?.
 * @param len? is the length of the dimension ?. 
 */
#define RarrayAddr5(array, ind1, ind2, ind3, ind4, ind5, len1, len2, len3, len4) \
  ((array)+(ind1)+((ind2)*(len1))+((ind3)*(len1)*(len2))+((ind4)*(len1)*(len2)*(len3))+((ind5)*(len1)*(len2)*(len3)*(len4)))

/**
 * Macro to return an element of a five dimensional rarray as an LVALUE
 * (i.e. it can appear on the left hand side of an assignment operator or it
 * can appear in a right hand side).  (Rarrays are just native arrays, but
 * they are in column order, so these macros may be useful in C.)
 * @param ind? is the element you wish to reference in dimension ?.
 * @param len? is the length of the dimension ?. 
 */
#define RarrayElem5(array, ind1, ind2, ind3, ind4, ind5, len1, len2, len3, len4) \
  (*(RarrayAddr5(array, ind1, ind2, ind3, ind4, ind5, len1, len2, len3, len4)))

/**
 * Macro to return an address of a six dimensional rarray as an LVALUE
 * (i.e. it can appear on the left hand side of an assignment operator or it
 * can appear in a right hand side).  (Rarrays are just native arrays, but
 * they are in column order, so these macros may be useful in C.)
 * @param ind? is the element you wish to reference in dimension ?.
 * @param len? is the length of the dimension ?. 
 */
#define RarrayAddr6(array, ind1, ind2, ind3, ind4, ind5, ind6, len1, len2, len3, len4, len5) \
  ((array)+(ind1)+((ind2)*(len1))+((ind3)*(len1)*(len2))+((ind4)*(len1)*(len2)*(len3))+((ind5)*(len1)*(len2)*(len3)*(len4)) +\
   ((ind6)*(len1)*(len2)*(len3)*(len4)*(len5)))

/**
 * Macro to return an element of a six dimensional rarray as an LVALUE
 * (i.e. it can appear on the left hand side of an assignment operator or it
 * can appear in a right hand side).  (Rarrays are just native arrays, but
 * they are in column order, so these macros may be useful in C.)
 * @param ind? is the element you wish to reference in dimension ?.
 * @param len? is the length of the dimension ?. 
 */
#define RarrayElem6(array, ind1, ind2, ind3, ind4, ind5, ind6, len1, len2, len3, len4, len5) \
  (*(RarrayAddr6(array, ind1, ind2, ind3, ind4, ind5, ind6, len1, len2, len3, len4, len5)))

/**
 * Macro to return an address of a seven dimensional rarray as an LVALUE
 * (i.e. it can appear on the left hand side of an assignment operator or it
 * can appear in a right hand side).  (Rarrays are just native arrays, but
 * they are in column order, so these macros may be useful in C.)
 * @param ind? is the element you wish to reference in dimension ?.
 * @param len? is the length of the dimension ?. 
 */
#define RarrayAddr7(array, ind1, ind2, ind3, ind4, ind5, ind6, ind7, len1, len2, len3, len4, len5, len6) \
  ((array)+(ind1)+((ind2)*(len1))+((ind3)*(len1)*(len2))+((ind4)*(len1)*(len2)*(len3))+((ind5)*(len1)*(len2)*(len3)*(len4)) +\
   ((ind6)*(len1)*(len2)*(len3)*(len4)*(len5)) + ((ind7)*(len1)*(len2)*(len3)*(len4)*(len5)*(len6)))

/**
 * Macro to return an element of a seven dimensional rarray as an LVALUE
 * (i.e. it can appear on the left hand side of an assignment operator or it
 * can appear in a right hand side).  (Rarrays are just native arrays, but
 * they are in column order, so these macros may be useful in C.)
 * @param ind? is the element you wish to reference in dimension ?.
 * @param len? is the length of the dimension ?. 
 */
#define RarrayElem7(array, ind1, ind2, ind3, ind4, ind5, ind6, ind7, len1, len2, len3, len4, len5, len6) \
  (*(RarrayAddr7(array, ind1, ind2, ind3, ind4, ind5, ind6, ind7, len1, len2, len3, len4, len5, len6)))


#ifdef __cplusplus
extern "C" { /*}*/
#endif

/**
 * Increment the arrays internal reference count by one. To make a
 * persistent copy (i.e. that lives longer than the current method
 * call) use smartCopy.
 */
void
sidl__array_addRef(struct sidl__array* array);

/**
 * If array is borrowed, allocate a new self-sufficient array and copy
 * the borrowed array into the new array; otherwise, increment the
 * reference count and return the array passed in. Use this whenever
 * you want to make a copy of a method argument because arrays passed
 * into methods aren't guaranteed to exist after the method call.
 */
struct sidl__array *
sidl__array_smartCopy(struct sidl__array *array);

/**
 * Decrement the arrays internal reference count by one. If the reference
 * count goes to zero, destroy the array.
 * Return true iff the array is destroyed
 */
void
sidl__array_deleteRef(struct sidl__array* array);

/**
 * Return the dimension of array. If the array pointer is NULL,
 * zero is returned.
 */
int32_t
sidl__array_dimen(const struct sidl__array* array);

/**
 * Return the lower index bound on dimension ind. If ind is not a valid
 * dimension, zero is returned. The valid range for ind is 0 to dimen-1.
 */
int32_t
sidl__array_lower(const struct sidl__array* array,
                  const int32_t ind);

/**
 * Return the upper index bound on dimension ind. If ind is not a valid
 * dimension, negative one is returned. The valid range for ind is 0 to
 * dimen-1.
 */
int32_t
sidl__array_upper(const struct sidl__array* array,
                  const int32_t ind);

/**
 * Return the number of element in dimension ind. If ind is not a valid
 * dimension, negative one is returned. The valid range for ind is 0 to
 * dimen-1.
 */
int32_t
sidl__array_length(const struct sidl__array* array,
                   const int32_t ind);

/**
 * Return the stride of dimension ind. If ind is not a valid
 * dimension, zero is returned. The valid range for ind is 0 to
 * dimen-1.
 */
int32_t
sidl__array_stride(const struct sidl__array* array,
                   const int32_t ind);

/**
 * Return a true value iff the array is a contiguous column-major ordered
 * array.  A NULL array argument causes 0 to be returned.
 */
sidl_bool
sidl__array_isColumnOrder(const struct sidl__array* array);

/**
 * Return a true value iff the array is a contiguous row-major ordered
 * array.  A NULL array argument causes 0 to be returned.
 */
sidl_bool
sidl__array_isRowOrder(const struct sidl__array* array);

/**
 * Return an integer indicating the type of elements held by the
 * array. Zero is returned if array is NULL.
 */
int32_t
sidl__array_type(const struct sidl__array* array);

/**
 * The following two functions are used for low level array reference
 * count debugging. They are not intended for Babel end-users.
 */
void
sidl__array_add(struct sidl__array * const array);

void 
sidl__array_remove(struct sidl__array * const array);
#ifdef __cplusplus
}
#endif
#endif /* included_sidlArray_h */
