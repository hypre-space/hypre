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




#ifndef VEC_DH_H
#define VEC_DH_H

#include "euclid_common.h"

struct _vec_dh {
  HYPRE_Int n;
  double *vals;
};

extern void Vec_dhCreate(Vec_dh *v);
extern void Vec_dhDestroy(Vec_dh v);
extern void Vec_dhInit(Vec_dh v, HYPRE_Int size);
        /* allocates storage, but does not initialize values */

extern void Vec_dhDuplicate(Vec_dh v, Vec_dh *out);
        /* creates vec and allocates storage, but neither
         * initializes nor copies values 
         */

extern void Vec_dhCopy(Vec_dh x, Vec_dh y);
        /* copies values from x to y;
         * y must have proper storage allocated,
         * e.g, through previous call to Vec_dhDuplicate,
         * or Vec_dhCreate and Vec_dhInit.
         */

extern void Vec_dhSet(Vec_dh v, double value);
extern void Vec_dhSetRand(Vec_dh v);

extern void Vec_dhRead(Vec_dh *v, HYPRE_Int ignore, char *filename);
extern void Vec_dhReadBIN(Vec_dh *v, char *filename);
extern void Vec_dhPrint(Vec_dh v, SubdomainGraph_dh sg, char *filename);
extern void Vec_dhPrintBIN(Vec_dh v, SubdomainGraph_dh sg, char *filename); 
#endif
