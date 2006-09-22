/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
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



#ifndef __SUPERLU_UTIL /* allow multiple inclusions */
#define __SUPERLU_UTIL

#include "HYPRE_utilities.h"
#include <stdio.h>
#if HAVE_STDLIB_H
#include <stdlib.h>
#endif
#if HAVE_STRING_H
#include <string.h>
#endif
#if HAVE_MALLOC_H
#include <malloc.h>
#endif
#include <assert.h>

#include "fortran.h"

/* Macros */
#ifndef USER_ABORT
#define USER_ABORT(msg) superlu_abort_and_exit(msg)
#endif

#define ABORT(err_msg) \
 { char msg[256];\
   sprintf(msg,"%s at line %d in file %s\n",err_msg,__LINE__, __FILE__);\
   USER_ABORT(msg); }


#ifndef USER_MALLOC
#define USER_MALLOC(size) superlu_malloc(size)
#endif

#define SUPERLU_MALLOC(size) USER_MALLOC(size)

#ifndef USER_FREE
#define USER_FREE(addr) superlu_free(addr)
#endif

#define SUPERLU_FREE(addr) USER_FREE(addr)


#define MAX(x, y) 	( (x) > (y) ? (x) : (y) )
#define MIN(x, y) 	( (x) < (y) ? (x) : (y) )

/* 
 * Constants 
 */
#define EMPTY	(-1)
#define NO	(-1)
#define FALSE	0
#define TRUE	1

/*
 * Type definitions
 */
typedef float    flops_t;
typedef unsigned char Logical;

/* 
 * The following enumerate type is used by the statistics variable 
 * SuperLUStat, to keep track of flop count and time spent at various stages.
 *
 * Note that not all of the fields are disjoint.
 */
typedef enum {
    COLPERM, /* find a column ordering that minimizes fills */
    RELAX,   /* find artificial supernodes */
    ETREE,   /* compute column etree */
    EQUIL,   /* equilibrate the original matrix */
    FACT,    /* perform LU factorization */
    RCOND,   /* estimate reciprocal condition number */
    SOLVE,   /* forward and back solves */
    REFINE,  /* perform iterative refinement */
    FLOAT,   /* time spent in floating-point operations */
    TRSV,    /* fraction of FACT spent in xTRSV */
    GEMV,    /* fraction of FACT spent in xGEMV */
    FERR,    /* estimate error bounds after iterative refinement */
    NPHASES  /* total number of phases */
} PhaseType;

typedef struct {
    int     *panel_histo; /* histogram of panel size distribution */
    double  *utime;       /* running time at various phases */
    flops_t *ops;         /* operation count at various phases */
} SuperLUStat_t;

/* Macros */
#define FIRSTCOL_OF_SNODE(i)	(xsup[i])


#ifdef __cplusplus
extern "C" {
#endif

extern void    superlu_abort_and_exit(char*);
extern void    *superlu_malloc (int);
extern int     *intMalloc (int);
extern int     *intCalloc (int);
extern void    superlu_free (void*);
extern void    SetIWork (int, int, int, int *, int **, int **, int **,
                         int **, int **, int **, int **);
extern void    StatInit(int, int);
extern void    StatFree();
extern int     sp_coletree (int *, int *, int *, int, int, int *);
extern void    relax_snode  (int, int *, int, int *, int *);
extern void    resetrep_col (const int, const int *, int *);
extern int     spcoletree (int *, int *, int *, int, int, int *);
extern int     *TreePostorder (int, int *);
extern double  SuperLU_timer_ ();
extern int     sp_ienv (int);
extern int     superlu_lsame (char *, char *);
extern int     superlu_xerbla (char *, int *);
extern void    ifill (int *, int, int);
extern void    snode_profile (int, int *);
extern void    super_stats (int, int *);
extern void    PrintSumm (char *, int, int, int);
extern void    PrintStat (SuperLUStat_t *);
extern void    print_panel_seg(int, int, int, int, int *, int *);
extern void    check_repfnz(int, int, int, int *);

#ifdef __cplusplus
  }
#endif

#endif /* __SUPERLU_UTIL */
