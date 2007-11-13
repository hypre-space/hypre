/*BHEADER**********************************************************************
 * Copyright (c) 2007, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
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
          is the "int" returned by Hash_i_dhLookup.
*/
extern void Hash_i_dhCreate(Hash_i_dh *h, int size);
  /* For proper operation, "size," which is the minimal
     size of the hash table, must be a power of 2.
     Or, pass "-1" to use the default.
   */


extern void Hash_i_dhDestroy(Hash_i_dh h);
extern void Hash_i_dhReset(Hash_i_dh h);

extern void Hash_i_dhInsert(Hash_i_dh h, int key, int data);
  /* throws error if <data, data> is already inserted;
     grows hash table if out of space.
   */

extern int  Hash_i_dhLookup(Hash_i_dh h, int key);
    /* returns "data" associated with "key,"
       or -1 if "key" is not found.
     */

#endif
