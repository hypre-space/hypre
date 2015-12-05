/*
 * File:          sidl_Resolve_fStub.c
 * Symbol:        sidl.Resolve-v0.9.15
 * Symbol Type:   enumeration
 * Babel Version: 1.0.0
 * Release:       $Name: V2-2-0b $
 * Revision:      @(#) $Id: sidl_Resolve_fStub.c,v 1.1 2007/02/06 01:23:08 painter Exp $
 * Description:   Client-side glue code for sidl.Resolve
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
 */

#ifndef included_sidl_int_IOR_h
#include "sidl_int_IOR.h"
#endif
#ifndef included_sidlfortran_h
#include "sidlfortran.h"
#endif
#include <stddef.h>
#include "sidl_Resolve_fAbbrev.h"
void
SIDLFortran90Symbol(sidl_resolve__array_createcol_m,
                  SIDL_RESOLVE__ARRAY_CREATECOL_M,
                  sidl_Resolve__array_createCol_m)
  (int32_t *dimen,
   int32_t lower[],
   int32_t upper[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_int__array_createCol(*dimen, lower, upper);
}

void
SIDLFortran90Symbol(sidl_resolve__array_createrow_m,
                  SIDL_RESOLVE__ARRAY_CREATEROW_M,
                  sidl_Resolve__array_createRow_m)
  (int32_t *dimen,
   int32_t lower[],
   int32_t upper[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_int__array_createRow(*dimen, lower, upper);
}

void
SIDLFortran90Symbol(sidl_resolve__array_create1d_m,
                  SIDL_RESOLVE__ARRAY_CREATE1D_M,
                  sidl_Resolve__array_create1d_m)
  (int32_t *len, int64_t *result)
{
  *result = (ptrdiff_t)sidl_int__array_create1d(*len);
}

void
SIDLFortran90Symbol(sidl_resolve__array_create2dcol_m,
                  SIDL_RESOLVE__ARRAY_CREATE2DCOL_M,
                  sidl_Resolve__array_create2dCol_m)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)sidl_int__array_create2dCol(*m, *n);
}

void
SIDLFortran90Symbol(sidl_resolve__array_create2drow_m,
                  SIDL_RESOLVE__ARRAY_CREATE2DROW_M,
                  sidl_Resolve__array_create2dRow_m)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)sidl_int__array_create2dRow(*m, *n);
}

void
SIDLFortran90Symbol(sidl_resolve__array_addref_m,
                  SIDL_RESOLVE__ARRAY_ADDREF_M,
                  sidl_Resolve__array_addRef_m)
  (int64_t *array)
{
  sidl_int__array_addRef((struct sidl_int__array *)(ptrdiff_t)*array);
}

void
SIDLFortran90Symbol(sidl_resolve__array_deleteref_m,
                  SIDL_RESOLVE__ARRAY_DELETEREF_M,
                  sidl_Resolve__array_deleteRef_m)
  (int64_t *array)
{
  sidl_int__array_deleteRef((struct sidl_int__array *)(ptrdiff_t)*array);
}

void
SIDLFortran90Symbol(sidl_resolve__array_get1_m,
                  SIDL_RESOLVE__ARRAY_GET1_M,
                  sidl_Resolve__array_get1_m)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *result)
{
  *result = 
    sidl_int__array_get1((const struct sidl_int__array *)(ptrdiff_t)*array
    , *i1);
}

void
SIDLFortran90Symbol(sidl_resolve__array_get2_m,
                  SIDL_RESOLVE__ARRAY_GET2_M,
                  sidl_Resolve__array_get2_m)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int32_t *result)
{
  *result = 
    sidl_int__array_get2((const struct sidl_int__array *)(ptrdiff_t)*array
    , *i1, *i2);
}

void
SIDLFortran90Symbol(sidl_resolve__array_get3_m,
                  SIDL_RESOLVE__ARRAY_GET3_M,
                  sidl_Resolve__array_get3_m)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int32_t *i3, 
   int32_t *result)
{
  *result = 
    sidl_int__array_get3((const struct sidl_int__array *)(ptrdiff_t)*array
    , *i1, *i2, *i3);
}

void
SIDLFortran90Symbol(sidl_resolve__array_get4_m,
                  SIDL_RESOLVE__ARRAY_GET4_M,
                  sidl_Resolve__array_get4_m)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int32_t *i3, 
   int32_t *i4, 
   int32_t *result)
{
  *result = 
    sidl_int__array_get4((const struct sidl_int__array *)(ptrdiff_t)*array
    , *i1, *i2, *i3, *i4);
}

void
SIDLFortran90Symbol(sidl_resolve__array_get5_m,
                  SIDL_RESOLVE__ARRAY_GET5_M,
                  sidl_Resolve__array_get5_m)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int32_t *i3, 
   int32_t *i4, 
   int32_t *i5, 
   int32_t *result)
{
  *result = 
    sidl_int__array_get5((const struct sidl_int__array *)(ptrdiff_t)*array
    , *i1, *i2, *i3, *i4, *i5);
}

void
SIDLFortran90Symbol(sidl_resolve__array_get6_m,
                  SIDL_RESOLVE__ARRAY_GET6_M,
                  sidl_Resolve__array_get6_m)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int32_t *i3, 
   int32_t *i4, 
   int32_t *i5, 
   int32_t *i6, 
   int32_t *result)
{
  *result = 
    sidl_int__array_get6((const struct sidl_int__array *)(ptrdiff_t)*array
    , *i1, *i2, *i3, *i4, *i5, *i6);
}

void
SIDLFortran90Symbol(sidl_resolve__array_get7_m,
                  SIDL_RESOLVE__ARRAY_GET7_M,
                  sidl_Resolve__array_get7_m)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int32_t *i3, 
   int32_t *i4, 
   int32_t *i5, 
   int32_t *i6, 
   int32_t *i7, 
   int32_t *result)
{
  *result = 
    sidl_int__array_get7((const struct sidl_int__array *)(ptrdiff_t)*array
    , *i1, *i2, *i3, *i4, *i5, *i6, *i7);
}

void
SIDLFortran90Symbol(sidl_resolve__array_get_m,
                  SIDL_RESOLVE__ARRAY_GET_M,
                  sidl_Resolve__array_get_m)
  (int64_t *array,
   int32_t indices[],
   int32_t *result)
{
  *result = 
    sidl_int__array_get((const struct sidl_int__array *)(ptrdiff_t)*array,
      indices);
}

void
SIDLFortran90Symbol(sidl_resolve__array_set1_m,
                  SIDL_RESOLVE__ARRAY_SET1_M,
                  sidl_Resolve__array_set1_m)
  (int64_t *array,
   int32_t *i1,
   int32_t *value)
{
  sidl_int__array_set1((struct sidl_int__array *)(ptrdiff_t)*array
  , *i1, *value);
}

void
SIDLFortran90Symbol(sidl_resolve__array_set2_m,
                  SIDL_RESOLVE__ARRAY_SET2_M,
                  sidl_Resolve__array_set2_m)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int32_t *value)
{
  sidl_int__array_set2((struct sidl_int__array *)(ptrdiff_t)*array
  , *i1, *i2, *value);
}

void
SIDLFortran90Symbol(sidl_resolve__array_set3_m,
                  SIDL_RESOLVE__ARRAY_SET3_M,
                  sidl_Resolve__array_set3_m)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int32_t *i3,
   int32_t *value)
{
  sidl_int__array_set3((struct sidl_int__array *)(ptrdiff_t)*array
  , *i1, *i2, *i3, *value);
}

void
SIDLFortran90Symbol(sidl_resolve__array_set4_m,
                  SIDL_RESOLVE__ARRAY_SET4_M,
                  sidl_Resolve__array_set4_m)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int32_t *i3,
   int32_t *i4,
   int32_t *value)
{
  sidl_int__array_set4((struct sidl_int__array *)(ptrdiff_t)*array
  , *i1, *i2, *i3, *i4, *value);
}

void
SIDLFortran90Symbol(sidl_resolve__array_set5_m,
                  SIDL_RESOLVE__ARRAY_SET5_M,
                  sidl_Resolve__array_set5_m)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int32_t *i3,
   int32_t *i4,
   int32_t *i5,
   int32_t *value)
{
  sidl_int__array_set5((struct sidl_int__array *)(ptrdiff_t)*array
  , *i1, *i2, *i3, *i4, *i5, *value);
}

void
SIDLFortran90Symbol(sidl_resolve__array_set6_m,
                  SIDL_RESOLVE__ARRAY_SET6_M,
                  sidl_Resolve__array_set6_m)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int32_t *i3,
   int32_t *i4,
   int32_t *i5,
   int32_t *i6,
   int32_t *value)
{
  sidl_int__array_set6((struct sidl_int__array *)(ptrdiff_t)*array
  , *i1, *i2, *i3, *i4, *i5, *i6, *value);
}

void
SIDLFortran90Symbol(sidl_resolve__array_set7_m,
                  SIDL_RESOLVE__ARRAY_SET7_M,
                  sidl_Resolve__array_set7_m)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int32_t *i3,
   int32_t *i4,
   int32_t *i5,
   int32_t *i6,
   int32_t *i7,
   int32_t *value)
{
  sidl_int__array_set7((struct sidl_int__array *)(ptrdiff_t)*array
  , *i1, *i2, *i3, *i4, *i5, *i6, *i7, *value);
}

void
SIDLFortran90Symbol(sidl_resolve__array_set_m,
                  SIDL_RESOLVE__ARRAY_SET_M,
                  sidl_Resolve__array_set_m)
  (int64_t *array,
  int32_t indices[],
  int32_t *value)
{
  sidl_int__array_set((struct sidl_int__array *)(ptrdiff_t)*array, indices,
    *value);
}

void
SIDLFortran90Symbol(sidl_resolve__array_dimen_m,
                  SIDL_RESOLVE__ARRAY_DIMEN_M,
                  sidl_Resolve__array_dimen_m)
  (int64_t *array, int32_t *result)
{
  *result =
    sidl_int__array_dimen((struct sidl_int__array *)(ptrdiff_t)*array);
}

void
SIDLFortran90Symbol(sidl_resolve__array_lower_m,
                  SIDL_RESOLVE__ARRAY_LOWER_M,
                  sidl_Resolve__array_lower_m)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_int__array_lower((struct sidl_int__array *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran90Symbol(sidl_resolve__array_upper_m,
                  SIDL_RESOLVE__ARRAY_UPPER_M,
                  sidl_Resolve__array_upper_m)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_int__array_upper((struct sidl_int__array *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran90Symbol(sidl_resolve__array_length_m,
                  SIDL_RESOLVE__ARRAY_LENGTH_M,
                  sidl_Resolve__array_length_m)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_int__array_length((struct sidl_int__array *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran90Symbol(sidl_resolve__array_stride_m,
                  SIDL_RESOLVE__ARRAY_STRIDE_M,
                  sidl_Resolve__array_stride_m)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_int__array_stride((struct sidl_int__array *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran90Symbol(sidl_resolve__array_iscolumnorder_m,
                  SIDL_RESOLVE__ARRAY_ISCOLUMNORDER_M,
                  sidl_Resolve__array_isColumnOrder_m)
  (int64_t *array,
   SIDL_F90_Bool *result)
{
  *result = sidl_int__array_isColumnOrder((struct sidl_int__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran90Symbol(sidl_resolve__array_isroworder_m,
                  SIDL_RESOLVE__ARRAY_ISROWORDER_M,
                  sidl_Resolve__array_isRowOrder_m)
  (int64_t *array,
   SIDL_F90_Bool *result)
{
  *result = sidl_int__array_isRowOrder((struct sidl_int__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran90Symbol(sidl_resolve__array_copy_m,
                  SIDL_RESOLVE__ARRAY_COPY_M,
                  sidl_Resolve__array_copy_m)
  (int64_t *src,
   int64_t *dest)
{
  sidl_int__array_copy((const struct sidl_int__array *)(ptrdiff_t)*src,
                       (struct sidl_int__array *)(ptrdiff_t)*dest);
}

void
SIDLFortran90Symbol(sidl_resolve__array_smartcopy_m,
                  SIDL_RESOLVE__ARRAY_SMARTCOPY_M,
                  sidl_Resolve__array_smartCopy_m)
  (int64_t *src)
{
  sidl_int__array_smartCopy((struct sidl_int__array *)(ptrdiff_t)*src);
}

void
SIDLFortran90Symbol(sidl_resolve__array_slice_m,
                  SIDL_RESOLVE__ARRAY_SLICE_M,
                  sidl_Resolve__array_slice_m)
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
SIDLFortran90Symbol(sidl_resolve__array_ensure_m,
                  SIDL_RESOLVE__ARRAY_ENSURE_M,
                  sidl_Resolve__array_ensure_m)
  (int64_t *src,
   int32_t *dimen,
   int     *ordering,
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_int__array_ensure((struct sidl_int__array *)(ptrdiff_t)*src,
    *dimen, *ordering);
}

