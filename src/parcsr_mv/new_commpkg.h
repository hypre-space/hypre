/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.10 $
 ***********************************************************************EHEADER*/




#ifndef hypre_NEW_COMMPKG
#define hypre_NEW_COMMPKG


typedef struct
{
   HYPRE_Int                   length;
   HYPRE_Int                   storage_length; 
   HYPRE_Int                   *id;
   HYPRE_Int                   *vec_starts;
   HYPRE_Int                   element_storage_length; 
   HYPRE_Int                   *elements;
   double                *d_elements;
   void                  *v_elements;
   
}  hypre_ProcListElements;   




#endif /* hypre_NEW_COMMPKG */

