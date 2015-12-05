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
 * $Revision: 2.4 $
 ***********************************************************************EHEADER*/


/******************************************************************************
 *
 * Header info for the hypre_BoxNeighbors structures
 *
 *****************************************************************************/

#ifndef hypre_BOX_NEIGHBORS_HEADER
#define hypre_BOX_NEIGHBORS_HEADER

/*--------------------------------------------------------------------------
 * hypre_RankLink:
 *--------------------------------------------------------------------------*/

typedef struct hypre_RankLink_struct
{
   int                           rank;
   int                           prank;
   struct hypre_RankLink_struct *next;

} hypre_RankLink;

/*--------------------------------------------------------------------------
 * hypre_BoxNeighbors:
 *--------------------------------------------------------------------------*/

typedef struct hypre_BoxNeighbors_struct
{
   hypre_BoxArray      *boxes;            /* boxes in the neighborhood */
   int                 *procs;            /* procs for 'boxes' */
   int                 *boxnums;          /* local boxnums for 'boxes' */
   int                 *ids;              /* ids for 'boxes' */
   int                  first_local;      /* first local box address */
   int                  num_local;        /* number of local boxes */

   hypre_Index          periodic;         /* directions of periodicity */
   int                  id_period;        /* period used for box ids */
   int                  num_periods;      /* number of box set periods */
   hypre_Index         *pshifts;          /* shifts of periodicity */

   hypre_RankLink     **rank_links;       /* neighbors of local boxes */

} hypre_BoxNeighbors;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_RankLink
 *--------------------------------------------------------------------------*/

#define hypre_RankLinkRank(link)  ((link) -> rank)
#define hypre_RankLinkPRank(link) ((link) -> prank)
#define hypre_RankLinkNext(link)  ((link) -> next)

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_BoxNeighbors
 *--------------------------------------------------------------------------*/

#define hypre_BoxNeighborsBoxes(neighbors)       ((neighbors) -> boxes)
#define hypre_BoxNeighborsProcs(neighbors)       ((neighbors) -> procs)
#define hypre_BoxNeighborsBoxnums(neighbors)     ((neighbors) -> boxnums)
#define hypre_BoxNeighborsIDs(neighbors)         ((neighbors) -> ids)
#define hypre_BoxNeighborsFirstLocal(neighbors)  ((neighbors) -> first_local)
#define hypre_BoxNeighborsNumLocal(neighbors)    ((neighbors) -> num_local)
#define hypre_BoxNeighborsPeriodic(neighbors)    ((neighbors) -> periodic)
#define hypre_BoxNeighborsIDPeriod(neighbors)    ((neighbors) -> id_period)
#define hypre_BoxNeighborsNumPeriods(neighbors)  ((neighbors) -> num_periods)
#define hypre_BoxNeighborsPShifts(neighbors)     ((neighbors) -> pshifts)
#define hypre_BoxNeighborsPShift(neighbors, i)   ((neighbors) -> pshifts[i])
#define hypre_BoxNeighborsRankLinks(neighbors)   ((neighbors) -> rank_links)

#define hypre_BoxNeighborsNumBoxes(neighbors) \
(hypre_BoxArraySize(hypre_BoxNeighborsBoxes(neighbors)))
#define hypre_BoxNeighborsRankLink(neighbors, b) \
(hypre_BoxNeighborsRankLinks(neighbors)[b])

/*--------------------------------------------------------------------------
 * Looping macros:
 *--------------------------------------------------------------------------*/
 
#define hypre_BeginBoxNeighborsLoop(n, neighbors, b)\
{\
   hypre_RankLink *hypre__rank_link;\
   int             hypre__num_boxes;\
\
   hypre__num_boxes = hypre_BoxNeighborsNumBoxes(neighbors) / \
      hypre_BoxNeighborsNumPeriods(neighbors);\
\
   hypre__rank_link = hypre_BoxNeighborsRankLink(neighbors, b);\
   while (hypre__rank_link)\
   {\
      n = hypre_RankLinkRank(hypre__rank_link) +\
          hypre_RankLinkPRank(hypre__rank_link)*hypre__num_boxes;

#define hypre_EndBoxNeighborsLoop\
      hypre__rank_link = hypre_RankLinkNext(hypre__rank_link);\
   }\
}

#endif
