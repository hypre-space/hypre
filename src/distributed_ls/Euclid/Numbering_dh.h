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




#ifndef NUMBERING_DH_H
#define NUMBERING_DH_H


/* code and algorithms in this class adopted from Edmond Chow's
   ParaSails
*/


#include "euclid_common.h"

struct _numbering_dh {
  HYPRE_Int   size;    /* max number of indices that can be stored;
                    (length of idx_ext[]) 
                  */
  HYPRE_Int   first;   /* global number of 1st local index (row) */
  HYPRE_Int   m;       /* number of local indices (number of local rows in mat) */
  HYPRE_Int   *idx_ext;   /* sorted list of external indices */
  HYPRE_Int   *idx_extLo; /* sorted list of external indices that are < first */
  HYPRE_Int   *idx_extHi; /* sorted list of external indices that are >= first+m */
  HYPRE_Int   num_ext; /* number of external (non-local) indices = num_extLo+num_extHi */
  HYPRE_Int   num_extLo; /* number of external indices < first */
  HYPRE_Int   num_extHi; /* number of external indices >= first+num_loc */
  Hash_i_dh global_to_local;

  bool debug;
};

extern void Numbering_dhCreate(Numbering_dh *numb);
extern void Numbering_dhDestroy(Numbering_dh numb);

  /* must be called before calling Numbering_dhGlobalToLocal() or
     Numbering_dhLocalToGlobal().
   */
extern void Numbering_dhSetup(Numbering_dh numb, Mat_dh mat);


  /* input: global_in[len], which contains global row numbers.
     output: local_out[len], containing corresponding local numbers.
     note: global_in[] and local_out[] may be identical.
   */
extern void Numbering_dhGlobalToLocal(Numbering_dh numb, HYPRE_Int len, 
                                      HYPRE_Int *global_in, HYPRE_Int *local_out);

#endif
