/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.6 $
 ***********************************************************************EHEADER*/




/* This is similar to the Hash_i_dh class (woe, for a lack
   of templates); this this class is for hashing data
   consisting of single, non-negative integers.
*/


#ifndef HASH_I_DH
#define HASH_I_DH

#include "euclid_common.h"

                                 
/*
    class methods 
    note: all parameters are inputs; the only output 
          is the "HYPRE_Int" returned by Hash_i_dhLookup.
*/
extern void Hash_i_dhCreate(Hash_i_dh *h, HYPRE_Int size);
  /* For proper operation, "size," which is the minimal
     size of the hash table, must be a power of 2.
     Or, pass "-1" to use the default.
   */


extern void Hash_i_dhDestroy(Hash_i_dh h);
extern void Hash_i_dhReset(Hash_i_dh h);

extern void Hash_i_dhInsert(Hash_i_dh h, HYPRE_Int key, HYPRE_Int data);
  /* throws error if <data, data> is already inserted;
     grows hash table if out of space.
   */

extern HYPRE_Int  Hash_i_dhLookup(Hash_i_dh h, HYPRE_Int key);
    /* returns "data" associated with "key,"
       or -1 if "key" is not found.
     */

#endif
