/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

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

struct double_linked_list
{
   HYPRE_Int                  data;
   struct double_linked_list *next_elt;
   struct double_linked_list *prev_elt;
   HYPRE_Int                  head;
   HYPRE_Int                  tail;
};

typedef struct double_linked_list hypre_ListElement;
typedef hypre_ListElement *hypre_LinkList;

#ifdef __cplusplus
}
#endif

#endif

