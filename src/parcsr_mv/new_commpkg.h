/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.9 $
 ***********************************************************************EHEADER*/




#ifndef hypre_NEW_COMMPKG
#define hypre_NEW_COMMPKG


typedef struct
{
   int                   length;
   int                   storage_length; 
   int                   *id;
   int                   *vec_starts;
   int                   element_storage_length; 
   int                   *elements;
   double                *d_elements;
   void                  *v_elements;
   
}  hypre_ProcListElements;   




#endif /* hypre_NEW_COMMPKG */

