/*
 * File:        sidlArray.h
 * Copyright:   (c) 2001 The Regents of the University of California
 * Release:     $Name$
 * Revision:    @(#) $Revision$
 * Date:        $Date$
 * Description: Convenience macros for accessing array elements
 *
 * Copyright (c) 2000-2001, The Regents of the University of Calfornia.
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

enum sidl_array_ordering {
  sidl_general_order=0, /* this must be zero (i.e. a false value) */
  sidl_column_major_order,
  sidl_row_major_order
};

/**
 * Return the dimension of the array.
 */
#define sidlArrayDim(array) ((array)->d_dimen)

/**
 * Macro to return the lower bound on the index for dimension ind of array.
 * A valid index for dimension ind must be greater than or equal to
 * sidlLower(array,ind). 
 */
#define sidlLower(array,ind) ((array)->d_lower[(ind)])

/**
 * Macro to return the upper bound on the index for dimension ind of array.
 * A valid index for dimension ind must be less than or equal to
 * sidlUpper(array,ind). 
 */
#define sidlUpper(array,ind) ((array)->d_upper[(ind)])

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
#define sidlStride(array,ind) ((array)->d_stride[(ind)])

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

#endif /* included_sidlArray_h */
