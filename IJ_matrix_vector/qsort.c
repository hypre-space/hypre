/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void qsort0( int *v,
             int  left,
             int  right )
{
   int i, last;

   if (left >= right)
      return;
   swap( v, left, (left+right)/2);
   last = left;
   for (i = left+1; i <= right; i++)
      if (v[i] < v[left])
      {
         swap(v, ++last, i);
      }
   swap(v, left, last);
   qsort0(v, left, last-1);
   qsort0(v, last+1, right);
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void qsort1( int *v,
	     double *w,
             int  left,
             int  right )
{
   int i, last;

   if (left >= right)
      return;
   swap2( v, w, left, (left+right)/2);
   last = left;
   for (i = left+1; i <= right; i++)
      if (v[i] < v[left])
      {
         swap2(v, w, ++last, i);
      }
   swap2(v, w, left, last);
   qsort1(v, w, left, last-1);
   qsort1(v, w, last+1, right);
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void swap( int *v,
           int  i,
           int  j )
{
   int temp;

   temp = v[i];
   v[i] = v[j];
   v[j] = temp;
}
/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void swap2(int     *v,
           double  *w,
           int      i,
           int      j )
{
   int temp;
   double temp2;

   temp = v[i];
   v[i] = v[j];
   v[j] = temp;
   temp2 = w[i];
   w[i] = w[j];
   w[j] = temp2;
}
