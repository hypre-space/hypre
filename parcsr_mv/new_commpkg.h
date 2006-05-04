
#ifndef hypre_NEW_COMMPKG
#define hypre_NEW_COMMPKG

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

typedef struct
{
  int                   length;
  int                   storage_length; 
  int                   *id;
  int                   *vec_starts;
  int                   element_storage_length; 
  int                   *elements;
   double                *d_elements;
}  hypre_ProcListElements;   




#endif /* hypre_NEW_COMMPKG */

