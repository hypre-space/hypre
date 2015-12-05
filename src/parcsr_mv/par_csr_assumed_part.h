/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.4 $
 ***********************************************************************EHEADER*/

#ifndef hypre_PARCSR_ASSUMED_PART
#define  hypre_PARCSR_ASSUMED_PART

typedef struct
{
   int                   length;
   int                   row_start;
   int                   row_end;
   int                   storage_length;
   int                   *proc_list;
   int		         *row_start_list;
   int                   *row_end_list;  
  int                    *sort_index;
} hypre_IJAssumedPart;




#endif /* hypre_PARCSR_ASSUMED_PART */

