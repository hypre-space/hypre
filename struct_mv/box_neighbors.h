/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
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

typedef struct rank_link
{
   int               rank;
   struct rank_link *next;

} hypre_RankLink;

/*--------------------------------------------------------------------------
 * hypre_BoxNeighbors:
 *--------------------------------------------------------------------------*/

typedef struct
{
   hypre_BoxArray  *boxes;
   int              box_rank;
   int              max_distance;         /* in infinity norm */

   hypre_RankLink  *rank_links[3][3][3];  /* neighbors of box `box_rank' */

} hypre_BoxNeighbors;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_RankLink
 *--------------------------------------------------------------------------*/

#define hypre_RankLinkRank(link)      ((link) -> rank)
#define hypre_RankLinkDistance(link)  ((link) -> distance)
#define hypre_RankLinkNext(link)      ((link) -> next)

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_BoxNeighbors
 *--------------------------------------------------------------------------*/

#define hypre_BoxNeighborsBoxes(neighbors)        ((neighbors) -> boxes)
#define hypre_BoxNeighborsBoxRank(neighbors)      ((neighbors) -> box_rank)
#define hypre_BoxNeighborsMaxDistance(neighbors)  ((neighbors) -> max_distance)
#define hypre_BoxNeighborsRankLinks(neighbors)    ((neighbors) -> rank_links)

#define hypre_BoxNeighborsRankLink(neighbors, i, j, k) \
(hypre_BoxNeighborsRankLinks(neighbors)[i+1][j+1][k+1])

/*--------------------------------------------------------------------------
 * Looping macros:
 *--------------------------------------------------------------------------*/
 
#define hypre_BeginBoxNeighborsLoop(rank, neighbors, distance_index)\
{\
   int             hypre__istart = 0;\
   int             hypre__jstart = 0;\
   int             hypre__kstart = 0;\
   int             hypre__istop  = 0;\
   int             hypre__jstop  = 0;\
   int             hypre__kstop  = 0;\
   hypre_RankLink *hypre__rank_link;\
   int             hypre__i, hypre__j, hypre__k;\
\
   hypre__i = hypre_IndexX(distance_index);\
   if (hypre__i < 0)\
      hypre__istart = -1;\
   else if (hypre__i > 0)\
      hypre__istop = 1;\
\
   hypre__j = hypre_IndexY(distance_index);\
   if (hypre__j < 0)\
      hypre__jstart = -1;\
   else if (hypre__j > 0)\
      hypre__jstop = 1;\
\
   hypre__k = hypre_IndexZ(distance_index);\
   if (hypre__k < 0)\
      hypre__kstart = -1;\
   else if (hypre__k > 0)\
      hypre__kstop = 1;\
\
   for (hypre__k = hypre__kstart; hypre__k <= hypre__kstop; hypre__k++)\
   {\
      for (hypre__j = hypre__jstart; hypre__j <= hypre__jstop; hypre__j++)\
      {\
         for (hypre__i = hypre__istart; hypre__i <= hypre__istop; hypre__i++)\
         {\
            hypre__rank_link = hypre_BoxNeighborsRankLink(neighbors,\
                                                          hypre__i,\
                                                          hypre__j,\
                                                          hypre__k);\
            while (hypre__rank_link)\
            {\
               rank = hypre_RankLinkRank(hypre__rank_link);

#define hypre_EndBoxNeighborsLoop\
               hypre__rank_link = hypre_RankLinkNext(hypre__rank_link);\
            }\
         }\
      }\
   }\
}

#endif
