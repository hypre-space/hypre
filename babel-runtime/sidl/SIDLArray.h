/*
 * File:        SIDLArray.h
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

#ifndef included_SIDLArray_h
#define included_SIDLArray_h

enum SIDL_array_ordering {
  SIDL_general_order=0, /* this must be zero (i.e. a false value) */
  SIDL_column_major_order,
  SIDL_row_major_order
};

/**
 * Return the dimension of the array.
 */
#define SIDLArrayDim(array) ((array)->d_dimen)

/**
 * Macro to return the lower bound on the index for dimension ind of array.
 * A valid index for dimension ind must be greater than or equal to
 * SIDLLower(array,ind). 
 */
#define SIDLLower(array,ind) ((array)->d_lower[(ind)])

/**
 * Macro to return the upper bound on the index for dimension ind of array.
 * A valid index for dimension ind must be less than or equal to
 * SIDLUpper(array,ind). 
 */
#define SIDLUpper(array,ind) ((array)->d_upper[(ind)])

/**
 * Macro to return the stride between elements in a particular dimension.
 * To move from the address of element i to element i + 1 in the dimension
 * ind, add SIDLStride(array,ind).
 */
#define SIDLStride(array,ind) ((array)->d_stride[(ind)])

/**
 * Helper macro for calculating the offset in a particular dimension.
 * This macro makes multiple references to array and ind, so you should
 * not use ++ or -- on arguments to this macro.
 */
#define SIDLArrayDimCalc(array, ind, var) \
  (SIDLStride(array,ind)*((var) - SIDLLower(array,ind)))


/**
 * Return the address of an element in a one dimensional array.
 * This macro may make multiple references to array and ind1, so do not
 * use ++ or -- when using this macro.
 */
#define SIDLArrayAddr1(array, ind1) \
  (array)->d_firstElement + SIDLArrayDimCalc(array, 0, ind1)

/**
 * Macro to return an element of a one dimensional array as an LVALUE
 * (i.e. it can appear on the left hand side of an assignment operator or it
 * can appear in a right hand side). This macro may make multiple references
 * to array and ind1, so do not use ++ or -- when using this macro.
 */
#define SIDLArrayElem1(array, ind1) \
  (*(SIDLArrayAddr1(array,ind1)))


/**
 * Return the address of an element in a two dimensional array.
 * This macro may make multiple references to array, ind1 & ind2; so do not
 * use ++ or -- when using this macro.
 */
#define SIDLArrayAddr2(array, ind1, ind2) \
  SIDLArrayAddr1(array, ind1) + SIDLArrayDimCalc(array, 1, ind2)

/**
 * Macro to return an element of a two dimensional array as an LVALUE
 * (i.e. it can appear on the left hand side of an assignment operator or it
 * can appear in a right hand side). This macro may make  multiple 
 * references to array, ind1 and ind2; so do not use ++ or -- when using
 * this macro. 
 */
#define SIDLArrayElem2(array, ind1, ind2) \
  (*(SIDLArrayAddr2(array, ind1, ind2)))


/**
 * Return the address of an element in a three dimensional array.
 * This macro may make multiple references to array, ind1, ind2 & ind3; so
 * do not use ++ or -- when using this macro.
 */
#define SIDLArrayAddr3(array, ind1, ind2, ind3) \
  SIDLArrayAddr2(array, ind1, ind2) + SIDLArrayDimCalc(array, 2, ind3)

/**
 * Macro to return an element of a three dimensional array as an LVALUE
 * (i.e. it can appear on the left hand side of an assignment operator or it
 * can appear in a right hand side). This macro may make multiple references
 * to array, ind1, ind2 & ind3; so do  not use ++ or -- when using this
 * macro. 
 */
#define SIDLArrayElem3(array, ind1, ind2, ind3) \
  (*(SIDLArrayAddr3(array, ind1, ind2, ind3)))


/**
 * Return the address of an element in a four dimensional array.
 * This macro may make multiple references to array, ind1, ind2, ind3 &
 * ind4; so do not use ++ or -- when using this macro.
 */
#define SIDLArrayAddr4(array, ind1, ind2, ind3, ind4) \
  SIDLArrayAddr3(array, ind1, ind2, ind3) + SIDLArrayDimCalc(array, 3, ind4)

/**
 * Macro to return an element of a four dimensional array as an LVALUE
 * (i.e. it can appear on the left hand side of an assignment operator or it
 * can appear in a right hand side).  This macro may make multiple
 * references to array, ind1, ind2, ind3 & ind4; so do not use ++ or -- when
 * using this macro. 
 */
#define SIDLArrayElem4(array, ind1, ind2, ind3, ind4) \
  (*(SIDLArrayAddr4(array, ind1, ind2, ind3, ind4)))

#endif /* included_SIDLArray_h */
