/* When called, o2n is (points to) an array of length n,
   and <n, rp, cval> comprise a compressed sparse row
   representation of a (possibly unsymmetric) graph.
   On return, o2n[i] indicates that node i is ordered o2n[i].
*/
#undef __FUNC__
#define __FUNC__ "orderSubdomainGraph_private"
void orderSubdomainGraph_private(int n, int *rp, int *cval, int *o2n)
{
  START_FUNC_DH
  int i, j, *marker, *color, thisNodesColor, *colorCounter;
  int *color = o2n;

  marker = (int*)MALLOC_DH(n*sizeof(int));
  color = (int*)MALLOC_DH(n*sizeof(int));
  colorCounter = (int*)MALLOC_DH(n*sizeof(int));
  for (i=0; i<n; ++i) {
    marker[i] = -1;
    colorCounter[i] = 0;
  }
  colorCounter[0] = 0;

/*------------------------------------------------------------------
 * color the nodes 
 *------------------------------------------------------------------*/
  for (i=0; i<n; ++i) {
    /* mark colors of all nabors as unavailable */
    for (j=rp[i]; j<rp[i+1]; ++j) {
      int nabor = cval[j];
      if (nabor != i) {  /* Speak not of thyself, avoid self loops */
        int naborsColor = color[nabor];
        marker[naborsColor] = i;
      }
    }

    /* assign vertex i the "smallest" possible color */
    thisNodesColor = -1;
    for (j=0; j<n; ++j) {
      if (marker[j] != i) {
        thisNodesColor = j;
        break;
      }
    }
    color[i] = thisNodesColor;
    colorCounter[1+thisNodesColor] += 1;
  }

/*------------------------------------------------------------------
 * build ordering vector; if two nodes are similarly colored,
 * they will have the same realtive ordering as before.
 *------------------------------------------------------------------*/
  /* prefix-sum to find lowest-numbered node for each color */
  for (i=1; i<n; ++i) {
    if (colorCounter[i] == 0) break;
    colorCounter[i] += colorCounter[i-1];
  }

  for (i=0; i<n; ++i) {
    o2n[i] = colorCounter[color[i]];
    colorCounter[color[i]] += 1;
  }


  if (Parser_dhHasSwitch(parser_dh, "-debugSubdomainOrdering") && myid_dh == 1) {
    fprintf(stderr, "\nNode coloring vector (number or nodes with color[i]): ");
    for (j=0; j<n; ++j) {
      if (color[j]) {
        fprintf(stderr, "  color= %i  count= %i\n", j, color[j]);
      }
    }
  }

  FREE_DH(marker); CHECK_V_ERROR;
  FREE_DH(color); CHECK_V_ERROR;
  FREE_DH(colorCounter); CHECK_V_ERROR;
  END_FUNC_DH
}
