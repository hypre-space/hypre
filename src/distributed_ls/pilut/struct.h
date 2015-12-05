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




#ifndef STRUCT_H
#define STRUCT_H

/*
 * struct.h
 *
 * This file contains data structures for ILU routines.
 *
 * Started 9/26/95
 * George
 *
 * 7/8
 *  - change to generic HYPRE_Int and double (in all files) and verified
 *  - added rrowlen to rmat and verified
 * 7/9
 *  - add recv info to the LDU communication struct TriSolveCommType
 * 7/29
 *  - add maxntogo and remove unused out and address buffers from cinfo
 *  - rearranged all structures to have ptrs first, then ints, ints, structs.
 *    This is under the assumption that that is the most likely order
 *    for things to be natural word length, so reduces padding.
 *
 * $Id: struct.h,v 2.5 2010/12/20 19:27:34 falgout Exp $
 */

#ifndef true
# define true  1
# define false 0
#endif

#ifndef bool
# ifdef Boolean
   typedef Boolean bool;
# else
   typedef unsigned char bool;
# endif
#endif
 
/*************************************************************************
* This data structure holds the data distribution
**************************************************************************/
struct distdef {
  HYPRE_Int ddist_nrows;		/* The order of the distributed matrix */
  HYPRE_Int ddist_lnrows;           /* The local number of rows */
  HYPRE_Int *ddist_rowdist;	/* How the rows are distributed among processors */
};

typedef struct distdef DataDistType;

#define DataDistTypeNrows(data_dist)      ((data_dist)->    ddist_nrows)
#define DataDistTypeLnrows(data_dist)     ((data_dist)->   ddist_lnrows)
#define DataDistTypeRowdist(data_dist)    ((data_dist)->  ddist_rowdist)

/*************************************************************************
* The following data structure stores info for a communication phase during
* the triangular solvers.
**************************************************************************/
struct cphasedef {
  double **raddr;	/* A rnbrpes+1 list of addresses to recv data into */

  HYPRE_Int *spes;	/* A snbrpes    list of PEs to send data */
  HYPRE_Int *sptr;	/* An snbrpes+1 list indexing sindex for each spes[i] */
  HYPRE_Int *sindex;	/* The packets to send per PE */
  HYPRE_Int *auxsptr;	/* Auxiliary send ptr, used at intermediate points */

  HYPRE_Int *rpes;	/* A rnbrpes   list of PEs to recv data */
  HYPRE_Int *rdone;	/* A rnbrpes   list of # elements recv'd in this hypre_LDUSolve */
  HYPRE_Int *rnum;        /* A nlevels x npes array of the number of elements to recieve */

  HYPRE_Int snbrpes;		/* The total number of neighboring PEs (to send to)   */
  HYPRE_Int rnbrpes;		/* The total number of neighboring PEs (to recv from) */
};

typedef struct cphasedef TriSolveCommType;


/*************************************************************************
* This data structure holds the factored matrix
**************************************************************************/
struct factormatdef {
  HYPRE_Int *lsrowptr;	/* Pointers to the locally stored rows start */
  HYPRE_Int *lerowptr;	/* Pointers to the locally stored rows end */
  HYPRE_Int *lcolind;	/* Array of column indices of lnrows */
   double *lvalues;	/* Array of locally stored values */
  HYPRE_Int *lrowptr;

  HYPRE_Int *usrowptr;	/* Pointers to the locally stored rows start */
  HYPRE_Int *uerowptr;	/* Pointers to the locally stored rows end */
  HYPRE_Int *ucolind;	/* Array of column indices of lnrows */
   double *uvalues;	/* Array of locally stored values */
  HYPRE_Int *urowptr;

  double *dvalues;	/* Diagonal values */

  double *nrm2s;	/* Array of the 2-norms of the rows for tolerance testing */

  HYPRE_Int *perm;		/* perm and invperm arrays for factorization */
  HYPRE_Int *iperm;

  /* Communication info for triangular system solution */
  double *gatherbuf;            /* maxsend*snbrpes buffer for sends */

  double *lx;
  double *ux;
  HYPRE_Int lxlen, uxlen;

  HYPRE_Int nlevels;			/* The number of reductions performed */
  HYPRE_Int nnodes[MAXNLEVEL];	/* The number of nodes at each reduction level */

  TriSolveCommType lcomm;	/* Communication info during the Lx=y solve */
  TriSolveCommType ucomm;	/* Communication info during the Ux=y solve */
};

typedef struct factormatdef FactorMatType;


/*************************************************************************
* This data structure holds the reduced matrix
**************************************************************************/
struct reducematdef {
  HYPRE_Int *rmat_rnz;		/* Pointers to the locally stored rows */
  HYPRE_Int *rmat_rrowlen;	/* Length allocated for each row */
  HYPRE_Int **rmat_rcolind;	/* Array of column indices of lnrows */
   double **rmat_rvalues;	/* Array of locally stored values */

  HYPRE_Int rmat_ndone;	     /* The number of vertices factored so far */
  HYPRE_Int rmat_ntogo;  /* The number of vertices not factored. This is the size of rmat */
  HYPRE_Int rmat_nlevel;	     /* The number of reductions performed so far */
};

typedef struct reducematdef ReduceMatType;



/*************************************************************************
* This data structure stores information about the send in each phase 
* of parallel hypre_ILUT
**************************************************************************/
struct comminfodef {
  double *gatherbuf;	/* Assembly buffer for sending colind & values */

  HYPRE_Int *incolind;	/* Receive buffer for colind */
   double *invalues;	/* Receive buffer for values */

  HYPRE_Int *rnbrind;	/* The neighbor processors */
  HYPRE_Int *rrowind;	/* The indices that are received */
  HYPRE_Int *rnbrptr;	/* Array of size rnnbr+1 into rrowind */

  HYPRE_Int *snbrind;	/* The neighbor processors */
  HYPRE_Int *srowind;	/* The indices that are sent */
  HYPRE_Int *snbrptr;	/* Array of size snnbr+1 into srowind */

  HYPRE_Int maxnsend;		/* The maximum number of rows being sent */
  HYPRE_Int maxnrecv;		/* The maximum number of rows being received */
  HYPRE_Int maxntogo;         /* The maximum number of rows left on any PE */

  HYPRE_Int rnnbr;		/* Number of neighbor processors */
  HYPRE_Int snnbr;		/* Number of neighbor processors */
};

typedef struct comminfodef CommInfoType;


/*************************************************************************
* The following data structure stores communication info for mat-vec
**************************************************************************/
struct mvcommdef {
  HYPRE_Int *spes;	/* Array of PE numbers */
  HYPRE_Int *sptr;	/* Array of send indices */
  HYPRE_Int *sindex;	/* Array that stores the actual indices */

  HYPRE_Int *rpes;
  double **raddr;

  double *bsec;		/* Stores the actual b vector */
  double *gatherbuf;	/* Used to gather the outgoing packets */
  HYPRE_Int *perm;	/* Used to map the LIND back to GIND */

  HYPRE_Int snpes;		/* Number of send PE's */
  HYPRE_Int rnpes;
};

typedef struct mvcommdef MatVecCommType;


/*************************************************************************
* The following data structure stores key-value pair
**************************************************************************/
struct KeyValueType {
  HYPRE_Int key;
  HYPRE_Int val;
};

typedef struct KeyValueType KeyValueType;


#endif
