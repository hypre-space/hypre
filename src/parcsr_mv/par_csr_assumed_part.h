
#ifndef hypre_PARCSR_ASSUMED_PART
#define  hypre_PARCSR_ASSUMED_PART

typedef struct
{
   int                   length;
   int                   row_start;
   int                   row_end;
   int                   storage_length;
   int                   *proc_list;
   int		         *row_start_list;
   int                   *row_end_list;  
  int                    *sort_index;
} hypre_IJAssumedPart;




#endif /* hypre_PARCSR_ASSUMED_PART */

