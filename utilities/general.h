/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * General structures and values
 *
 *****************************************************************************/

#ifndef hypre_GENERAL_HEADER
#define hypre_GENERAL_HEADER

/*--------------------------------------------------------------------------
 * Define various functions
 *--------------------------------------------------------------------------*/

#ifndef hypre_max
#define hypre_max(a,b)  (((a)<(b)) ? (b) : (a))
#endif
#ifndef hypre_min
#define hypre_min(a,b)  (((a)<(b)) ? (a) : (b))
#endif

#ifndef hypre_round
#define hypre_round(x)  ( ((x) < 0.0) ? ((int)(x - 0.5)) : ((int)(x + 0.5)) )
#endif

#endif
