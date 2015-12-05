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
 * Header file link lists
 *
 *****************************************************************************/

#ifndef HYPRE_LINKLIST_HEADER
#define HYPRE_LINKLIST_HEADER

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LIST_HEAD -1
#define LIST_TAIL -2

struct double_linked_list
{
       HYPRE_Int                        data;
       struct double_linked_list *next_elt;
       struct double_linked_list *prev_elt;
       HYPRE_Int                        head;
       HYPRE_Int                        tail;
};

typedef struct double_linked_list hypre_ListElement;
typedef hypre_ListElement  *hypre_LinkList;  

#ifdef __cplusplus
}
#endif

#endif
