/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.5 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_Log2:
 *   This routine returns the integer, floor(log_2(p)).
 *   If p <= 0, it returns a -1.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_Log2(HYPRE_Int p)
{
   HYPRE_Int  e;

   if (p <= 0)
      return -1;

   e = 0;
   while (p > 1)
   {
      e += 1;
      p /= 2;
   }
 
   return  e;
}
