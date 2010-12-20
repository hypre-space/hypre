#include "headers.h"

/*--------------------------------------------------------------------------
 * Test driver prepares files
 *      for testing unstructured matrix topology interface (csr storage)
 *--------------------------------------------------------------------------*/
 
main( HYPRE_Int   argc,
      char *argv[] )
{
  FILE *f;

  HYPRE_Int ierr;

  HYPRE_Int ind;

  HYPRE_Int i,j,k,l,m;
  HYPRE_Int i_dof, j_dof;

  HYPRE_Int level;                 /* number of actual levels used in V--cycle: */


  HYPRE_Int num_elements, num_nodes, num_dofs;

  HYPRE_Int Max_level = 25; 

  HYPRE_Int max_level;

  /* Schwarz smoother: --------------------------------------------------- */
  /* internal format: ---------------------------------------------------- */

  HYPRE_Int **i_domain_dof,  **j_domain_dof;
  double **domain_matrixinverse;


  /* Interpolation P and stiffness matrices Matrix; ---------------------- */
  hypre_CSRMatrix     **P;
  hypre_CSRMatrix     **Matrix;

  hypre_AMGeMatrixTopology **A;


  /* Dirichlet boundary conditions information: -------------------------- */
  HYPRE_Int *i_node_on_boundary;
  HYPRE_Int *i_dof_on_boundary;

  /* auxiliary arrays for enforcing Dirichlet boundary conditions: -------- */
  HYPRE_Int *i_dof_dof_a, *j_dof_dof_a;
  double *a_dof_dof;



  /* dimensions of elements, dofs: -------------------------------- */
  HYPRE_Int  *Num_elements, *Num_dofs;

  /* initial graphs: element_node, boundarysurface_node: ------------------ */
  HYPRE_Int *i_element_node_0, *j_element_node_0;

  /* not used: num_boundarysurfaces = 0; ---------------------------------- */
  HYPRE_Int *i_boundarysurface_node, *j_boundarysurface_node;
  HYPRE_Int num_boundarysurfaces;




  /* PDEsystem information: ----------------------------------------------- */
  HYPRE_Int system_size;


  HYPRE_Int num_functions;



  HYPRE_Int *i_element_dof_0, *j_element_dof_0;
  double *element_data;




  /* element matrices information: ---------------------------------------- */
  HYPRE_Int *i_element_chord_0, *j_element_chord_0;
  double *a_element_chord_0;

  HYPRE_Int *i_chord_dof_0, *j_chord_dof_0;
  HYPRE_Int *Num_chords;

  /* ---------------------------------------------------------------------- */



  /* node coordinates: ---------------------------------------------------- */  
  double *x_coord, *y_coord;


  /* ---------------------------------------------------------------------- */
  /*  PCG & V_cycle arrays:                                                 */
  /* ---------------------------------------------------------------------- */

  double *x, *rhs, *r, *v, *g, **w, **d,
    *v_coarse, *w_coarse, *d_coarse, *v_fine, *w_fine, *d_fine, *aux_fine;
  HYPRE_Int max_iter = 1000;
  HYPRE_Int coarse_level; 
  HYPRE_Int nu = 1;  

  double reduction_factor;



  /*-----------------------------------------------------------
   * Set defaults
   *-----------------------------------------------------------*/
  HYPRE_Int  arg_index;
  HYPRE_Int time_index;
  HYPRE_Int  print_usage;
  char  *element_node_file = NULL, 
    *element_matrix_file = NULL, *coordinates_file=NULL,
    *node_on_boundary_file=NULL;
 
  HYPRE_Int element_node_file_type = 0;
  HYPRE_Int element_matrix_file_type = 0;
  HYPRE_Int coordinates_file_type = 0;
  HYPRE_Int node_on_boundary_file_type = 0;


  Max_level = 25;


  element_node_file = "/home/panayot/linear_solvers/seq_ls/amge/charles/element_node_small";
  element_matrix_file = "/home/panayot/linear_solvers/seq_ls/amge/charles/element_chord_small";
  /* coordinates_file = "coordinates"; */
  node_on_boundary_file = "/home/panayot/linear_solvers/seq_ls/amge/charles/node_on_boundary_small";


  /*-----------------------------------------------------------
   * Parse command line
   *-----------------------------------------------------------*/
 
   print_usage = 0;
   arg_index = 1;


  while (arg_index < argc && (!print_usage))
    {
      if ( strcmp(argv[arg_index], "-fromfile") == 0 )
	{
	  arg_index++;
	  element_node_file_type = 1;


	  element_node_file = argv[arg_index];
	  hypre_printf("Element_node graph from file: %s\n", element_node_file);
	}
      else if ( strcmp(argv[arg_index], "-elmmat") == 0 )
	{
	  arg_index++;
	  element_matrix_file_type = 1;


	  element_matrix_file = argv[arg_index];
	  hypre_printf("Element matrices from file: %s\n", element_matrix_file);
	}
      else if ( strcmp(argv[arg_index], "-bc") == 0 )
	{
	  arg_index++;
	  node_on_boundary_file_type = 1;


	  node_on_boundary_file = argv[arg_index];
	  hypre_printf("Dirichlet b.c. nodes from file: %s\n", node_on_boundary_file);
	}
      else if ( strcmp(argv[arg_index], "-graphics") == 0 )
	{
	  arg_index++;
	  coordinates_file_type = 1;


	  coordinates_file = argv[arg_index];
	  hypre_printf("Node coordinates from file: %s\n", coordinates_file);
	}
      else if ( strcmp(argv[arg_index], "-help") == 0 )
	{
	  print_usage = 1;
	}
      else
	{
	  arg_index++;
	}
    }


  if (element_node_file)
    hypre_printf("element_node from file: %s\n", element_node_file);
  if (element_matrix_file)
    hypre_printf("Element matrices from file: %s\n", element_matrix_file);
  if (coordinates_file)
    hypre_printf("Node coordinates from file: %s\n", coordinates_file);
  if (node_on_boundary_file)
    hypre_printf("Dirichlet b.c. nodes from file: %s\n", node_on_boundary_file);



   /*-----------------------------------------------------------
    * Print usage info
    *-----------------------------------------------------------*/
 
  if (print_usage)
   {
      hypre_printf("\n");
      hypre_printf("Usage: %s [<options>]\n", argv[0]);
      hypre_printf("\n");
      hypre_printf("  -fromfile <filename>   : build graph element_node from file\n");
      hypre_printf("\n");
      hypre_printf("  -elmmat   <filename>   : read fine-grid element matrices from file\n");
      hypre_printf("\n");
      hypre_printf("  -graphics <filename>   : read x_coord, y_coord from file\n");
      hypre_printf("\n");
      hypre_printf("  -bc       <filename>   : Dirichlet b.c. nodes from file\n");
      hypre_printf("\n");

      exit(1);
   }
  /*------------------------------------------------------------------------
   * initializing
   *------------------------------------------------------------------------*/


  Num_chords =  hypre_CTAlloc(HYPRE_Int, Max_level);
  Num_elements =  hypre_CTAlloc(HYPRE_Int, Max_level);

  Num_dofs =  hypre_CTAlloc(HYPRE_Int, Max_level);



  for (l=0; l < Max_level; l++)
    {
      Num_dofs[l] = 0;
      Num_elements[l] = 0;
      Num_chords[l] = 0;
    }



  /* builds initial graph(s): element_node(_0) (and boundarysurface_node):  */

  ierr = hypre_AMGeInitialGraphs(&i_element_node_0,
				 &j_element_node_0,

				 &i_boundarysurface_node, 
				 &j_boundarysurface_node,


				 &num_elements,
				 &num_nodes,
				 &num_boundarysurfaces,

				 &i_node_on_boundary, 

				 element_node_file);


  ind = 0;
  for (i=0; i < num_elements; i++)
    for (j=i_element_node_0[i]; j < i_element_node_0[i+1]; j++)
      if (j_element_node_0[j] < 0)
	{
	  ind = -1;
	  break;
	}

  if (ind == -1)
    for (i=0; i < num_elements; i++)
      for (j=i_element_node_0[i]; j < i_element_node_0[i+1]; j++)
	j_element_node_0[j]++;



  hypre_TFree(i_boundarysurface_node);
  hypre_TFree(j_boundarysurface_node);

  num_boundarysurfaces = 0;
  i_boundarysurface_node = NULL;
  j_boundarysurface_node = NULL;

  f = fopen(node_on_boundary_file, "r");
  for (i=0; i<num_nodes; i++)
    hypre_fscanf(f, "%ld\n", &i_node_on_boundary[i]);

  fclose(f);  



  Num_elements[0] = num_elements;


  /* ELEMENT MATRICES READ: ============================================ */
  system_size = 1;

  num_dofs = num_nodes;

  Num_dofs[0] = num_dofs;

  i_dof_on_boundary = i_node_on_boundary;

  i_element_dof_0 = i_element_node_0;
  j_element_dof_0 = j_element_node_0;

  ierr = hypre_AMGeElmMatRead(&element_data, 

			      i_element_dof_0,
			      j_element_dof_0,
			      Num_elements[0],

			      element_matrix_file);


  hypre_printf("store element matrices in element_chord format: ----------------\n");

  ierr = hypre_AMGeElementMatrixDof(i_element_dof_0, j_element_dof_0,

				    element_data,

				    &i_element_chord_0,
				    &j_element_chord_0,
				    &a_element_chord_0,

				    &i_chord_dof_0,
				    &j_chord_dof_0,

				    &Num_chords[0],

				    Num_elements[0], Num_dofs[0]);
  /*
  for (i=0; i < Num_elements[0]; i++)
    for (j=i_element_chord_0[i]; j < i_element_chord_0[i+1]; j++)
      {
	k = j_element_chord_0[j];
	i_dof = j_chord_dof_0[i_chord_dof_0[k]];
	j_dof = j_chord_dof_0[i_chord_dof_0[k]+1];

	if (i_dof == j_dof) hypre_printf("diagonal entry %d: %e\n", i_dof,
				   a_element_chord_0[j]);
      }
      */

  /* AMGeAGSetup: ---------------------------------------------------- */

  ierr = hypre_AMGeAGSetup(&P,

			   &Matrix,

			   &A,

			   &level,

			   /* ------ fine-grid element matrices ----- */
			   i_element_chord_0,
			   j_element_chord_0,
			   a_element_chord_0,

			   i_chord_dof_0,
			   j_chord_dof_0,

			   i_element_dof_0,
			   j_element_dof_0,

			   /* nnz: of the assembled matrices -------*/
			   Num_chords,


			   /* --------- Dirichlet b.c. ----------- */
			   i_dof_on_boundary, 

			   /* --------------------------------------- */

			   Num_elements,
			   Num_dofs,

			   Max_level);

  if (coordinates_file != NULL)
    {
      ierr = hypre_AMGe2dGraphics(A,
				  level, 

				  i_element_node_0, 
				  j_element_node_0,

				  Num_elements,

				  coordinates_file);

    }

  


  max_level = level;

  hypre_printf("END AMGeAGSetup; ============================================\n");

  hypre_TFree(i_dof_on_boundary);



  hypre_printf("END AMGeInterpolationSetup; =================================\n");


  ierr = hypre_AMGeAGSchwarzSmootherSetup(&i_domain_dof,
					  &j_domain_dof,
					  &domain_matrixinverse,

					  A,

					  Matrix,

					  &level,


					  Num_elements,
					  Num_dofs); 

  hypre_printf("END AMGeSchwarzSmootherSetup; =====================================\n");


  /* ========================================================================== */
  /* ======================== S O L U T I O N   P A R T: ====================== */
  /* ========================================================================== */

  /* ========================================================================== */
  /* one V(1,1) --cycle as preconditioner in PCG: ============================= */
  /* symmetric multiplicative Schwarz pre--smoothing, the same post--smoothing; */
  /* ========================================================================== */


  w = hypre_CTAlloc(double*, level+1); 
  d = hypre_CTAlloc(double*, level+1);


  for (l=0; l < level+1; l++)
    {
      if (Num_dofs[l] > 0)
	{
	  w[l] = hypre_CTAlloc(double, Num_dofs[l]);
	  d[l] = hypre_CTAlloc(double, Num_dofs[l]);
	}
      else
	{
	  level = l-1;
	  break;
	}
    }


  num_dofs = Num_dofs[0];

  x = hypre_CTAlloc(double, num_dofs); 
  rhs = hypre_CTAlloc(double, num_dofs);

  r = hypre_CTAlloc(double, num_dofs); 

  g = hypre_CTAlloc(double, num_dofs);

  v_fine = hypre_CTAlloc(double, num_dofs);
  w_fine = hypre_CTAlloc(double, num_dofs);
  d_fine = hypre_CTAlloc(double, num_dofs);
  aux_fine = hypre_CTAlloc(double, num_dofs);

  coarse_level = level;
  v_coarse = hypre_CTAlloc(double, Num_dofs[coarse_level]);
  w_coarse = hypre_CTAlloc(double, Num_dofs[coarse_level]);
  d_coarse = hypre_CTAlloc(double, Num_dofs[coarse_level]);

  for (l=0; l < level; l++)
    {
      hypre_printf("\n\n=========================================================\n");
      hypre_printf("         Testing level[%d] Schwarz PCG solve:              \n",l);
      hypre_printf("===========================================================\n");
 
      for (i=0; i < Num_dofs[l]; i++)
	x[i] = 0.e0;

      for (i=0; i < Num_dofs[l]; i++)
	rhs[i] = rand();

      i_dof_dof_a = hypre_CSRMatrixI(Matrix[l]);
      j_dof_dof_a = hypre_CSRMatrixJ(Matrix[l]);
      a_dof_dof   = hypre_CSRMatrixData(Matrix[l]);


      ierr = hypre_SchwarzSolve(x, rhs, g, 

				i_dof_dof_a, j_dof_dof_a, a_dof_dof,

				i_domain_dof[l], j_domain_dof[l],
				/* Num_elements[l+1], */
				Num_elements[l],
				domain_matrixinverse[l], 

				Num_dofs[l]);


      ierr = hypre_Schwarzpcg(x, rhs,
			      a_dof_dof,
			      i_dof_dof_a, j_dof_dof_a,

			      i_domain_dof[l], j_domain_dof[l],
			      domain_matrixinverse[l], 
			      /* Num_elements[l+1], */
			      Num_elements[l],

			      v_fine, w_fine, d_fine, aux_fine,

			      max_iter, 

			      Num_dofs[l]);


      hypre_printf("\n\n=======================================================\n");
      hypre_printf("             END test PCG solve:                           \n");
      hypre_printf("===========================================================\n");
 
    }

  hypre_printf("\n\n===============================================================\n");
  hypre_printf(" ------- V_cycle & symmetric multiplicative Schwarz smoothing: --------\n");
  hypre_printf("================================================================\n");

  num_dofs = Num_dofs[0];

  for (i=0; i < num_dofs; i++)
    rhs[i] = hypre_Rand();
  
  
  ierr = hypre_VcycleSchwarzpcg(x, rhs,
				w, d,

				&reduction_factor,
			       
				Matrix,
				i_domain_dof,
				j_domain_dof,
				domain_matrixinverse,
				Num_elements,

				P,

				aux_fine, r, g, 
				v_fine, w_fine, d_fine, 

				max_iter, 

				v_coarse, w_coarse, d_coarse, 

				nu, 
				level, coarse_level, 
				Num_dofs);




  hypre_TFree(x);
  hypre_TFree(rhs);

  hypre_TFree(g);
  hypre_TFree(r);

  for (l=0; l < level+1; l++)
    if (Num_dofs[l] > 0)
      {
	ind = 0;
	/* for (i=0; i < Num_elements[l+1]; i++) */
	for (i=0; i < Num_elements[l]; i++)
	  ind += (i_domain_dof[l][i+1] - i_domain_dof[l][i])
	       * (i_domain_dof[l][i+1] - i_domain_dof[l][i]);

	hypre_printf("\n-------------------- level: %d -----------------------\n", l);
	hypre_printf("num_dofs: %d, num_elements: %d, nnz: %d, Schwarz_nnz: %d\n",
	       Num_dofs[l], Num_elements[l], Num_chords[l], ind);
      }

  for (l=0; l < level+1; l++)
    if (Num_dofs[l] > 0)
      {
	hypre_TFree(w[l]);
	hypre_TFree(d[l]);

	hypre_CSRMatrixDestroy(Matrix[l]);

      }

  for (l=0; l < max_level; l++)
    {

      if (system_size == 1 &&Num_dofs[l+1] > 0)
	{
	  hypre_CSRMatrixI(P[l]) = NULL;
	  hypre_CSRMatrixJ(P[l]) = NULL;
	}
  
    }

  for (l=0; l < level; l++)
    {

      hypre_TFree(i_domain_dof[l]);
      hypre_TFree(j_domain_dof[l]);
      hypre_TFree(domain_matrixinverse[l]);

      hypre_CSRMatrixDestroy(P[l]);

    }


  hypre_TFree(aux_fine);
  hypre_TFree(v_fine);
  hypre_TFree(w_fine);
  hypre_TFree(d_fine);
  hypre_TFree(w);
  hypre_TFree(d);


  hypre_TFree(v_coarse);
  hypre_TFree(w_coarse);
  hypre_TFree(d_coarse);

  for (l=0; l < max_level+1; l++)
    hypre_DestroyAMGeMatrixTopology(A[l]);


  hypre_TFree(Num_elements);
  hypre_TFree(Num_dofs);
  hypre_TFree(Num_chords);

  hypre_TFree(i_chord_dof_0);
  hypre_TFree(j_chord_dof_0);

  hypre_TFree(i_element_chord_0);
  hypre_TFree(j_element_chord_0);
  hypre_TFree(a_element_chord_0);


  hypre_TFree(P);
  hypre_TFree(Matrix);
  hypre_TFree(A);

  hypre_TFree(i_domain_dof);
  hypre_TFree(j_domain_dof);
  hypre_TFree(domain_matrixinverse);



}
