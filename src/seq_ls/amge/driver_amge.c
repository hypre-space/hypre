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
  HYPRE_Int i,j,k,l,m;

  HYPRE_Int level;            /* number of actual levels used in V--cycle: */


  HYPRE_Int num_elements, num_nodes, num_dofs;

  HYPRE_Int Max_level = 25; 

  HYPRE_Int max_level;

  /* nested dissection ILU(1) smoother: ---------------------------------- */
  /* internal format: ---------------------------------------------------- */

  HYPRE_Int **i_ILUdof_to_dof;
  HYPRE_Int **i_ILUdof_ILUdof_t, **j_ILUdof_ILUdof_t,
    **i_ILUdof_ILUdof, **j_ILUdof_ILUdof;
  HYPRE_Real **LD_data, **U_data;


  /* Interpolation P and stiffness matrices Matrix; ---------------------- */
  hypre_CSRMatrix     **P;
  hypre_CSRMatrix     **Matrix;

  hypre_AMGeMatrixTopology **A;


  /* Dirichlet boundary conditions information: -------------------------- */
  HYPRE_Int *i_node_on_boundary;
  HYPRE_Int *i_dof_on_boundary;

  /* auxiliary arrays for enforcing Dirichlet boundary conditions: -------- */
  HYPRE_Int *i_dof_dof_a, *j_dof_dof_a;
  HYPRE_Real *a_dof_dof;


  /* coarsenode information and coarsenode neighborhood information; ----- */
  HYPRE_Int **i_node_coarsenode, **j_node_coarsenode;
  HYPRE_Int **i_node_neighbor_coarsenode, **j_node_neighbor_coarsenode;


  /* dimensions of nodes, elements, dofs: -------------------------------- */
  HYPRE_Int *Num_nodes, *Num_elements, *Num_dofs;

  /* initial graphs: element_node, boundarysurface_node: ------------------ */
  HYPRE_Int *i_element_node_0, *j_element_node_0;

  /* not used: num_boundarysurfaces = 0; ---------------------------------- */
  HYPRE_Int *i_boundarysurface_node, *j_boundarysurface_node;
  HYPRE_Int num_boundarysurfaces;


  /* nested dissection blocks: -------------------------------------------- */
  HYPRE_Int **i_block_node, **j_block_node;
  HYPRE_Int *Num_blocks;


  /* PDEsystem information: ----------------------------------------------- */
  HYPRE_Int system_size;


  HYPRE_Int num_functions;


  HYPRE_Int *i_dof_node_0, *j_dof_node_0;
  HYPRE_Int *i_node_dof_0, *j_node_dof_0;

  HYPRE_Int *i_element_dof_0, *j_element_dof_0;
  HYPRE_Real *element_data;


  HYPRE_Int **i_node_dof, **j_node_dof;


  /* element matrices information: ---------------------------------------- */
  HYPRE_Int *i_element_chord_0, *j_element_chord_0;
  HYPRE_Real *a_element_chord_0;

  HYPRE_Int *i_chord_dof_0, *j_chord_dof_0;
  HYPRE_Int *Num_chords;

  /* ---------------------------------------------------------------------- */


  /* counters: ------------------------------------------------------------ */  
  HYPRE_Int dof_coarsedof_counter, dof_neighbor_coarsedof_counter;


  /* node coordinates: ---------------------------------------------------- */  
  HYPRE_Real *x_coord, *y_coord;


  /* ---------------------------------------------------------------------- */
  /*  PCG & V_cycle arrays:                                                 */
  /* ---------------------------------------------------------------------- */

  HYPRE_Real *x, *rhs, *r, *v, **w, **d,
    *aux, *v_coarse, *w_coarse, *d_coarse, *v_fine, *w_fine, *d_fine;
  HYPRE_Int max_iter = 1000;
  HYPRE_Int coarse_level; 
  HYPRE_Int nu = 2;  

  HYPRE_Real reduction_factor;



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


  /*

  element_node_file = hypre_CTAlloc(char, 1);
  element_matrix_file = hypre_CTAlloc(char, 1);
  coordinates_file = hypre_CTAlloc(char, 1);
  node_on_boundary_file = hypre_CTAlloc(char, 1);
  */

  element_node_file = "/home/panayot/linear_solvers/seq_linear_solvers/amge/charles/element_node";
  element_matrix_file = "/home/panayot/linear_solvers/seq_linear_solvers/amge/charles/element_chord";
  /* coordinates_file = "coordinates"; */
  node_on_boundary_file = "/home/panayot/linear_solvers/seq_linear_solvers/amge/charles/node_on_boundary";


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
  Num_nodes =  hypre_CTAlloc(HYPRE_Int, Max_level);
  Num_dofs =  hypre_CTAlloc(HYPRE_Int, Max_level);

  Num_blocks = hypre_CTAlloc(HYPRE_Int, Max_level);

  for (l=0; l < Max_level; l++)
    {
      Num_dofs[l] = 0;
      Num_elements[l] = 0;
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


  Num_nodes[0] = num_nodes;
  Num_elements[0] = num_elements;

  /* builds AMGeMatrixTopology: ------------------------------------------ */






  ierr = hypre_AMGeMatrixTopologySetup(&A,
				       &level, 

				       i_element_node_0,
				       j_element_node_0,

				       num_elements, num_nodes, 

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

  hypre_printf("END AMGeMatrixTopologySetup; =================================\n");






  ierr = hypre_AMGeCoarsenodeSetup(A,
				   &level,

				   &i_node_neighbor_coarsenode,
				   &j_node_neighbor_coarsenode,

				   &i_node_coarsenode,
				   &j_node_coarsenode,

				   &i_block_node,
				   &j_block_node,

				   Num_blocks, 
				   Num_elements,
				   Num_nodes);

  hypre_printf("END AMGeCoarsenodeSetup; =================================\n");

  

  /* ELEMENT MATRICES READ: ============================================ */
  system_size = 1;

  num_functions = 3; 

  ierr = compute_dof_node(&i_dof_node_0, &j_dof_node_0,
			  Num_nodes[0], system_size, &num_dofs);

  Num_dofs[0] = num_dofs;

  if (system_size == 1)
    i_dof_on_boundary = i_node_on_boundary;
  else
    {
      ierr = compute_dof_on_boundary(&i_dof_on_boundary,
				     i_node_on_boundary,
				     
				     Num_nodes[0], system_size);
      free(i_node_on_boundary);
    }


  ierr = transpose_matrix_create(&i_node_dof_0,
				 &j_node_dof_0,

				 i_dof_node_0, j_dof_node_0,

				 Num_dofs[0], Num_nodes[0]);


  if (system_size == 1)
    {
      i_element_dof_0 = i_element_node_0;
      j_element_dof_0 = j_element_node_0;
    }
  else
    ierr = matrix_matrix_product(&i_element_dof_0, &j_element_dof_0, 

				 i_element_node_0, j_element_node_0,
				 i_node_dof_0, j_node_dof_0,

				 Num_elements[0], Num_nodes[0], Num_dofs[0]);


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



  ierr = hypre_AMGeInterpolationSetup(&P,

				      &Matrix,

				      A,

				      &level,

				 /* ------ fine-grid element matrices ----- */
				      i_element_chord_0,
				      j_element_chord_0,
				      a_element_chord_0,

				      i_chord_dof_0,
				      j_chord_dof_0,

				 /* nnz: of the assembled matrices -------*/
				      Num_chords,

				 /* ----- coarse node information  ------ */
				      i_node_neighbor_coarsenode,
				      j_node_neighbor_coarsenode,

				      i_node_coarsenode,
				      j_node_coarsenode,


				 /* --------- Dirichlet b.c. ----------- */
				      i_dof_on_boundary, 

				 /* -------- PDEsystem information -------- */
				      system_size,
				      num_functions,

				      i_dof_node_0, j_dof_node_0,
				      i_node_dof_0, j_node_dof_0,

				      &i_node_dof,
				      &j_node_dof,

				 /* --------------------------------------- */

				      Num_elements,
				      Num_nodes,
				      Num_dofs);


  hypre_TFree(i_dof_on_boundary);


  hypre_TFree(i_dof_node_0);
  hypre_TFree(j_dof_node_0);

  hypre_printf("END AMGeInterpolationSetup; =================================\n");


  ierr = hypre_AMGeSmootherSetup(&i_ILUdof_to_dof,

				 &i_ILUdof_ILUdof,
				 &j_ILUdof_ILUdof,
				 &LD_data,
			     
				 &i_ILUdof_ILUdof_t,
				 &j_ILUdof_ILUdof_t,
				 &U_data,


				 Matrix,

				 &level,

				 i_block_node, j_block_node,

				 i_node_dof, j_node_dof,

				 Num_blocks,
				 Num_nodes,
				 Num_dofs); 

  hypre_printf("END AMGeSmootherSetup; =================================\n");

  hypre_TFree(i_node_dof_0);
  hypre_TFree(j_node_dof_0);



  for (l=0; l < level+1; l++)
    {
      hypre_TFree(i_block_node[l]);
      hypre_TFree(j_block_node[l]);
    }



  for (l=1; l < level+1; l++)
    {
      hypre_TFree(i_node_dof[l]);
      hypre_TFree(j_node_dof[l]);
    }

  hypre_TFree(i_node_dof);
  hypre_TFree(j_node_dof);

  hypre_TFree(i_block_node);
  hypre_TFree(j_block_node);




  /* ========================================================================== */
  /* ======================== S O L U T I O N   P A R T: ====================== */
  /* ========================================================================== */

  /* one V(1,1) --cycle as preconditioner in PCG: ============================== */
  /* ILU solve pre--smoothing, ILU solve post--smoothing; ===================== */



  w = hypre_CTAlloc(HYPRE_Real*, level+1); 
  d = hypre_CTAlloc(HYPRE_Real*, level+1);

  for (l=0; l < level+1; l++)
    {
      Num_dofs[l] = Num_nodes[l] * system_size;
      if (Num_dofs[l] > 0)
	{
	  w[l] = hypre_CTAlloc(HYPRE_Real, Num_dofs[l]);
	  d[l] = hypre_CTAlloc(HYPRE_Real, Num_dofs[l]);
	}
      else
	{
	  level = l-1;
	  break;
	}
    }


  num_dofs = Num_dofs[0];

  x = hypre_CTAlloc(HYPRE_Real, num_dofs); 
  rhs = hypre_CTAlloc(HYPRE_Real, num_dofs);

  r = hypre_CTAlloc(HYPRE_Real, num_dofs); 
  aux = hypre_CTAlloc(HYPRE_Real, num_dofs);
  v_fine = hypre_CTAlloc(HYPRE_Real, num_dofs);
  w_fine = hypre_CTAlloc(HYPRE_Real, num_dofs);
  d_fine = hypre_CTAlloc(HYPRE_Real, num_dofs);

  coarse_level = level;
  v_coarse = hypre_CTAlloc(HYPRE_Real, Num_dofs[coarse_level]);
  w_coarse = hypre_CTAlloc(HYPRE_Real, Num_dofs[coarse_level]);
  d_coarse = hypre_CTAlloc(HYPRE_Real, Num_dofs[coarse_level]);

  for (l=0; l < level; l++)
    {
      hypre_printf("\n\n=======================================================\n");
      hypre_printf("             Testing level[%d] PCG solve:                  \n",l);
      hypre_printf("===========================================================\n");
 
      for (i=0; i < Num_dofs[l]; i++)
	x[i] = 0.e0;

      for (i=0; i < Num_dofs[l]; i++)
	rhs[i] = rand();

      i_dof_dof_a = hypre_CSRMatrixI(Matrix[l]);
      j_dof_dof_a = hypre_CSRMatrixJ(Matrix[l]);
      a_dof_dof   = hypre_CSRMatrixData(Matrix[l]);


      ierr = hypre_ILUsolve(x,

			    i_ILUdof_to_dof[l],
					 
			    i_ILUdof_ILUdof[l],
			    j_ILUdof_ILUdof[l],
			    LD_data[l],

			    i_ILUdof_ILUdof_t[l],
			    j_ILUdof_ILUdof_t[l],
			    U_data[l],

			    rhs, 

			    Num_dofs[l]);


      ierr = hypre_ILUpcg(x, rhs,
			  a_dof_dof,
			  i_dof_dof_a, j_dof_dof_a,

			  i_ILUdof_to_dof[l],

			  i_ILUdof_ILUdof[l], 
			  j_ILUdof_ILUdof[l],
			  LD_data[l],

			  i_ILUdof_ILUdof_t[l], 
			  j_ILUdof_ILUdof_t[l],
			  U_data[l],

			  v_fine, w_fine, d_fine, max_iter, 

			  Num_dofs[l]);


      hypre_printf("\n\n=======================================================\n");
      hypre_printf("             END test PCG solve:                           \n");
      hypre_printf("===========================================================\n");
 
    }

  hypre_printf("\n\n===============================================================\n");
  hypre_printf(" ------- V_cycle & nested dissection ILU(1) smoothing: --------\n");
  hypre_printf("================================================================\n");

  num_dofs = Num_dofs[0];

  for (i=0; i < num_dofs; i++)
    rhs[i] = rand();
  
  
  ierr = hypre_VcycleILUpcg(x, rhs,
			    w, d,

			    &reduction_factor,
			       
			    Matrix,
			    i_ILUdof_to_dof,

			    i_ILUdof_ILUdof, 
			    j_ILUdof_ILUdof,
			    LD_data,

			    i_ILUdof_ILUdof_t, 
			    j_ILUdof_ILUdof_t,
			    U_data,

			    P,

			    aux, r, 
			    v_fine, w_fine, d_fine, max_iter, 

			    v_coarse, w_coarse, d_coarse, 

			    nu, 
			    level, coarse_level, 
			    Num_dofs);




  hypre_TFree(x);
  hypre_TFree(rhs);

  hypre_TFree(r);
  hypre_TFree(aux);
  for (l=0; l < level+1; l++)
    if (Num_dofs[l] > 0)
      {
	hypre_printf("\n------------ level: %d --------------\n", level);
	hypre_printf("num_dofs: %d, num_elements: %d, nnz: %d\n",
	       Num_dofs[l], Num_elements[l], Num_chords[l]);
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

      hypre_TFree(i_node_coarsenode[l]);
      hypre_TFree(j_node_coarsenode[l]);

      hypre_TFree(i_node_neighbor_coarsenode[l]);
      hypre_TFree(j_node_neighbor_coarsenode[l]);

      if (system_size == 1 &&Num_dofs[l+1] > 0)
	{
	  hypre_CSRMatrixI(P[l]) = NULL;
	  hypre_CSRMatrixJ(P[l]) = NULL;
	}
  
    }
  for (l=0; l < level; l++)
    {

      hypre_TFree(i_ILUdof_to_dof[l]);
      hypre_TFree(i_ILUdof_ILUdof[l]);
      hypre_TFree(j_ILUdof_ILUdof[l]);
      hypre_TFree(LD_data[l]);


      hypre_TFree(i_ILUdof_ILUdof_t[l]);
      hypre_TFree(j_ILUdof_ILUdof_t[l]);
	
      hypre_TFree(U_data[l]);

      hypre_CSRMatrixDestroy(P[l]);

    }



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


  hypre_TFree(Num_nodes);
  hypre_TFree(Num_elements);
  hypre_TFree(Num_dofs);
  hypre_TFree(Num_blocks);
  hypre_TFree(Num_chords);

  hypre_TFree(i_chord_dof_0);
  hypre_TFree(j_chord_dof_0);

  hypre_TFree(i_element_chord_0);
  hypre_TFree(j_element_chord_0);
  hypre_TFree(a_element_chord_0);


  hypre_TFree(P);
  hypre_TFree(Matrix);
  hypre_TFree(A);

  hypre_TFree(i_ILUdof_to_dof);
  hypre_TFree(i_ILUdof_ILUdof);
  hypre_TFree(j_ILUdof_ILUdof);
  hypre_TFree(LD_data);

  hypre_TFree(i_ILUdof_ILUdof_t);
  hypre_TFree(j_ILUdof_ILUdof_t);
  hypre_TFree(U_data);

  hypre_TFree(i_node_coarsenode);
  hypre_TFree(j_node_coarsenode);

  hypre_TFree(i_node_neighbor_coarsenode);
  hypre_TFree(j_node_neighbor_coarsenode);

}
