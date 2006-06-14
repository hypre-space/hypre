typedef struct
{
    hypre_IJMatrix    *Face_iedge;
    hypre_IJMatrix    *Element_iedge;
    hypre_IJMatrix    *Edge_iedge;
                                                                                                                            
    hypre_IJMatrix    *Element_Face;
    hypre_IJMatrix    *Element_Edge;
                                                                                                                            
} hypre_PTopology;

