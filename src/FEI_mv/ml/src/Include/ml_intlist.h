/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.5 $
 ***********************************************************************EHEADER*/




/* ******************************************************************** */
/* See the file COPYRIGHT for a complete copyright notice, contact      */
/* person and disclaimer.                                               */        
/* ******************************************************************** */

/* ******************************************************************** */
/* Data structure to hold multiple lists of integers (used to hold the  */
/*  element to node lists in this context)                              */
/* ******************************************************************** */
/* Author        : Charles Tong (LLNL)                                  */
/* Date          : April, 1997                                          */
/* ******************************************************************** */

#ifndef _MLINTLIST_
#define _MLINTLIST_

#include <stdio.h>
#include "ml_common.h"
#include "ml_defs.h"
#include "ml_memory.h"

/* ******************************************************************** */
/*  length  : number of sub-lists                                       */
/*  start   : the range of locations pointed to by start[i] and         */
/*            in members hold the node information for element [i].     */
/*  members : an one-dimensional integer array to store the node lists  */
/* -------------------------------------------------------------------- */

typedef struct ML_IntList_Struct
{
   int ML_id;
   int cur_mem_leng;
   int length;
   int *start;
   int *members;

} ML_IntList;

/* ******************************************************************** */
/* functions to manipulate the Int_lists structures                     */
/* -------------------------------------------------------------------- */

#ifndef ML_CPP
#ifdef __cplusplus
extern "C" 
{
#endif
#endif

extern int ML_IntList_Create(ML_IntList **, int, int);
extern int ML_IntList_Load_Sublist(ML_IntList *, int, int *);
extern int ML_IntList_Get_Sublist(ML_IntList *, int, int *, int *);
extern int ML_IntList_Destroy(ML_IntList **);
extern int ML_IntList_Print(ML_IntList *);

#ifndef ML_CPP
#ifdef __cplusplus
}
#endif
#endif

#endif

