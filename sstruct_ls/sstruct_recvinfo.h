/*--------------------------------------------------------------------------
 * hypre_SStructRecvInfo data structure
 *--------------------------------------------------------------------------*/
#ifndef hypre_RECVINFODATA_HEADER
#define hypre_RECVINFODATA_HEADER


typedef struct 
{
   int                   size;

   hypre_BoxArrayArray  *recv_boxes;
   int                 **recv_procs;

} hypre_SStructRecvInfoData;

#endif
