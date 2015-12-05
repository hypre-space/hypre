/*--------------------------------------------------------------------------
 * hypre_SStructSendInfo data structure
 *--------------------------------------------------------------------------*/
#ifndef hypre_SENDINFODATA_HEADER
#define hypre_SENDINFODATA_HEADER


typedef struct 
{
   int                   size;

   hypre_BoxArrayArray  *send_boxes;
   int                 **send_procs;
   int                 **send_remote_boxnums;

} hypre_SStructSendInfoData;

#endif
