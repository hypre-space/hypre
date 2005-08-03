/*
 * File:          sidl_Scope_fStub.c
 * Symbol:        sidl.Scope-v0.9.3
 * Symbol Type:   enumeration
 * Babel Version: 0.10.8
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Client-side glue code for sidl.Scope
 * 
 * Copyright (c) 2000-2002, The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the Components Team <components@llnl.gov>
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
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.8
 * xml-url       = /home/painter/babel/share/babel-0.10.8/repository/sidl.Scope-v0.9.3.xml
 */

#ifndef included_sidl_int_IOR_h
#include "sidl_int_IOR.h"
#endif
#ifndef included_sidlfortran_h
#include "sidlfortran.h"
#endif
#include <stddef.h>
void
SIDLFortran77Symbol(sidl_scope__array_createcol_f,
                  SIDL_SCOPE__ARRAY_CREATECOL_F,
                  sidl_Scope__array_createCol_f)
  (int32_t *dimen,
   int32_t lower[],
   int32_t upper[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_int__array_createCol(*dimen, lower, upper);
}

void
SIDLFortran77Symbol(sidl_scope__array_createrow_f,
                  SIDL_SCOPE__ARRAY_CREATEROW_F,
                  sidl_Scope__array_createRow_f)
  (int32_t *dimen,
   int32_t lower[],
   int32_t upper[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_int__array_createRow(*dimen, lower, upper);
}

void
SIDLFortran77Symbol(sidl_scope__array_create1d_f,
                  SIDL_SCOPE__ARRAY_CREATE1D_F,
                  sidl_Scope__array_create1d_f)
  (int32_t *len, int64_t *result)
{
  *result = (ptrdiff_t)sidl_int__array_create1d(*len);
}

void
SIDLFortran77Symbol(sidl_scope__array_create2dcol_f,
                  SIDL_SCOPE__ARRAY_CREATE2DCOL_F,
                  sidl_Scope__array_create2dCol_f)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)sidl_int__array_create2dCol(*m, *n);
}

void
SIDLFortran77Symbol(sidl_scope__array_create2drow_f,
                  SIDL_SCOPE__ARRAY_CREATE2DROW_F,
                  sidl_Scope__array_create2dRow_f)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)sidl_int__array_create2dRow(*m, *n);
}

void
SIDLFortran77Symbol(sidl_scope__array_addref_f,
                  SIDL_SCOPE__ARRAY_ADDREF_F,
                  sidl_Scope__array_addRef_f)
  (int64_t *array)
{
  sidl_int__array_addRef((struct sidl_int__array *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(sidl_scope__array_deleteref_f,
                  SIDL_SCOPE__ARRAY_DELETEREF_F,
                  sidl_Scope__array_deleteRef_f)
  (int64_t *array)
{
  sidl_int__array_deleteRef((struct sidl_int__array *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(sidl_scope__array_get1_f,
                  SIDL_SCOPE__ARRAY_GET1_F,
                  sidl_Scope__array_get1_f)
  (int64_t *array, 
   int32_t *i1, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_int__array_get1((const struct sidl_int__array *)(ptrdiff_t)*array
    , *i1);
}

void
SIDLFortran77Symbol(sidl_scope__array_get2_f,
                  SIDL_SCOPE__ARRAY_GET2_F,
                  sidl_Scope__array_get2_f)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_int__array_get2((const struct sidl_int__array *)(ptrdiff_t)*array
    , *i1, *i2);
}

void
SIDLFortran77Symbol(sidl_scope__array_get3_f,
                  SIDL_SCOPE__ARRAY_GET3_F,
                  sidl_Scope__array_get3_f)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int32_t *i3, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_int__array_get3((const struct sidl_int__array *)(ptrdiff_t)*array
    , *i1, *i2, *i3);
}

void
SIDLFortran77Symbol(sidl_scope__array_get4_f,
                  SIDL_SCOPE__ARRAY_GET4_F,
                  sidl_Scope__array_get4_f)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int32_t *i3, 
   int32_t *i4, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_int__array_get4((const struct sidl_int__array *)(ptrdiff_t)*array
    , *i1, *i2, *i3, *i4);
}

void
SIDLFortran77Symbol(sidl_scope__array_get5_f,
                  SIDL_SCOPE__ARRAY_GET5_F,
                  sidl_Scope__array_get5_f)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int32_t *i3, 
   int32_t *i4, 
   int32_t *i5, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_int__array_get5((const struct sidl_int__array *)(ptrdiff_t)*array
    , *i1, *i2, *i3, *i4, *i5);
}

void
SIDLFortran77Symbol(sidl_scope__array_get6_f,
                  SIDL_SCOPE__ARRAY_GET6_F,
                  sidl_Scope__array_get6_f)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int32_t *i3, 
   int32_t *i4, 
   int32_t *i5, 
   int32_t *i6, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_int__array_get6((const struct sidl_int__array *)(ptrdiff_t)*array
    , *i1, *i2, *i3, *i4, *i5, *i6);
}

void
SIDLFortran77Symbol(sidl_scope__array_get7_f,
                  SIDL_SCOPE__ARRAY_GET7_F,
                  sidl_Scope__array_get7_f)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int32_t *i3, 
   int32_t *i4, 
   int32_t *i5, 
   int32_t *i6, 
   int32_t *i7, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_int__array_get7((const struct sidl_int__array *)(ptrdiff_t)*array
    , *i1, *i2, *i3, *i4, *i5, *i6, *i7);
}

void
SIDLFortran77Symbol(sidl_scope__array_get_f,
                  SIDL_SCOPE__ARRAY_GET_F,
                  sidl_Scope__array_get_f)
  (int64_t *array,
   int32_t indices[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_int__array_get((const struct sidl_int__array *)(ptrdiff_t)*array,
      indices);
}

void
SIDLFortran77Symbol(sidl_scope__array_set1_f,
                  SIDL_SCOPE__ARRAY_SET1_F,
                  sidl_Scope__array_set1_f)
  (int64_t *array,
   int32_t *i1,
   int64_t *value)
{
  sidl_int__array_set1((struct sidl_int__array *)(ptrdiff_t)*array
  , *i1, (int32_t)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(sidl_scope__array_set2_f,
                  SIDL_SCOPE__ARRAY_SET2_F,
                  sidl_Scope__array_set2_f)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int64_t *value)
{
  sidl_int__array_set2((struct sidl_int__array *)(ptrdiff_t)*array
  , *i1, *i2, (int32_t)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(sidl_scope__array_set3_f,
                  SIDL_SCOPE__ARRAY_SET3_F,
                  sidl_Scope__array_set3_f)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int32_t *i3,
   int64_t *value)
{
  sidl_int__array_set3((struct sidl_int__array *)(ptrdiff_t)*array
  , *i1, *i2, *i3, (int32_t)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(sidl_scope__array_set4_f,
                  SIDL_SCOPE__ARRAY_SET4_F,
                  sidl_Scope__array_set4_f)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int32_t *i3,
   int32_t *i4,
   int64_t *value)
{
  sidl_int__array_set4((struct sidl_int__array *)(ptrdiff_t)*array
  , *i1, *i2, *i3, *i4, (int32_t)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(sidl_scope__array_set5_f,
                  SIDL_SCOPE__ARRAY_SET5_F,
                  sidl_Scope__array_set5_f)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int32_t *i3,
   int32_t *i4,
   int32_t *i5,
   int64_t *value)
{
  sidl_int__array_set5((struct sidl_int__array *)(ptrdiff_t)*array
  , *i1, *i2, *i3, *i4, *i5, (int32_t)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(sidl_scope__array_set6_f,
                  SIDL_SCOPE__ARRAY_SET6_F,
                  sidl_Scope__array_set6_f)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int32_t *i3,
   int32_t *i4,
   int32_t *i5,
   int32_t *i6,
   int64_t *value)
{
  sidl_int__array_set6((struct sidl_int__array *)(ptrdiff_t)*array
  , *i1, *i2, *i3, *i4, *i5, *i6, (int32_t)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(sidl_scope__array_set7_f,
                  SIDL_SCOPE__ARRAY_SET7_F,
                  sidl_Scope__array_set7_f)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int32_t *i3,
   int32_t *i4,
   int32_t *i5,
   int32_t *i6,
   int32_t *i7,
   int64_t *value)
{
  sidl_int__array_set7((struct sidl_int__array *)(ptrdiff_t)*array
  , *i1, *i2, *i3, *i4, *i5, *i6, *i7, (int32_t)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(sidl_scope__array_set_f,
                  SIDL_SCOPE__ARRAY_SET_F,
                  sidl_Scope__array_set_f)
  (int64_t *array,
  int32_t indices[],
  int64_t *value)
{
  sidl_int__array_set((struct sidl_int__array *)(ptrdiff_t)*array, indices,
    (int32_t)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(sidl_scope__array_dimen_f,
                  SIDL_SCOPE__ARRAY_DIMEN_F,
                  sidl_Scope__array_dimen_f)
  (int64_t *array, int32_t *result)
{
  *result =
    sidl_int__array_dimen((struct sidl_int__array *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(sidl_scope__array_lower_f,
                  SIDL_SCOPE__ARRAY_LOWER_F,
                  sidl_Scope__array_lower_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_int__array_lower((struct sidl_int__array *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(sidl_scope__array_upper_f,
                  SIDL_SCOPE__ARRAY_UPPER_F,
                  sidl_Scope__array_upper_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_int__array_upper((struct sidl_int__array *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(sidl_scope__array_length_f,
                  SIDL_SCOPE__ARRAY_LENGTH_F,
                  sidl_Scope__array_length_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_int__array_length((struct sidl_int__array *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(sidl_scope__array_stride_f,
                  SIDL_SCOPE__ARRAY_STRIDE_F,
                  sidl_Scope__array_stride_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_int__array_stride((struct sidl_int__array *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(sidl_scope__array_iscolumnorder_f,
                  SIDL_SCOPE__ARRAY_ISCOLUMNORDER_F,
                  sidl_Scope__array_isColumnOrder_f)
  (int64_t *array,
   SIDL_F77_Bool *result)
{
  *result = sidl_int__array_isColumnOrder((struct sidl_int__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(sidl_scope__array_isroworder_f,
                  SIDL_SCOPE__ARRAY_ISROWORDER_F,
                  sidl_Scope__array_isRowOrder_f)
  (int64_t *array,
   SIDL_F77_Bool *result)
{
  *result = sidl_int__array_isRowOrder((struct sidl_int__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(sidl_scope__array_copy_f,
                  SIDL_SCOPE__ARRAY_COPY_F,
                  sidl_Scope__array_copy_f)
  (int64_t *src,
   int64_t *dest)
{
  sidl_int__array_copy((const struct sidl_int__array *)(ptrdiff_t)*src,
                       (struct sidl_int__array *)(ptrdiff_t)*dest);
}

void
SIDLFortran77Symbol(sidl_scope__array_smartcopy_f,
                  SIDL_SCOPE__ARRAY_SMARTCOPY_F,
                  sidl_Scope__array_smartCopy_f)
  (int64_t *src)
{
  sidl_int__array_smartCopy((struct sidl_int__array *)(ptrdiff_t)*src);
}

void
SIDLFortran77Symbol(sidl_scope__array_slice_f,
                  SIDL_SCOPE__ARRAY_SLICE_F,
                  sidl_Scope__array_slice_f)
  (int64_t *src,
   int32_t *dimen,
   int32_t numElem[],
   int32_t srcStart[],
   int32_t srcStride[],
   int32_t newStart[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_int__array_slice((struct sidl_int__array *)(ptrdiff_t)*src,
      *dimen, numElem, srcStart, srcStride, newStart);
}

void
SIDLFortran77Symbol(sidl_scope__array_ensure_f,
                  SIDL_SCOPE__ARRAY_ENSURE_F,
                  sidl_Scope__array_ensure_f)
  (int64_t *src,
   int32_t *dimen,
   int     *ordering,
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_int__array_ensure((struct sidl_int__array *)(ptrdiff_t)*src,
    *dimen, *ordering);
}

