#ifndef SUBDOMAIN
#define SUBDOMAIN

#include "Method.h"
#include "definitions.h"
#include "Packet.h"

class Subdomain : public Method{

  private:

    int SN;                // Subdomain number (starting from 0)
    
    int NPackets;
    Packet Pa[PACKETS];    // Pointer to the packets (give information about
                           // the bdr with the rest of the subdomains).
				
    // double Bdr[6*MAX_PACKET];		
    double *Bdr;

  public:

    Subdomain(int sn, Method *m, int *tr);

    void Solve();
    void inprod(int l, double *v1, double *v2, double &res);
    void PCG(int l, int num_of_iter,  double *y, double *b, Matrix *A);
    void V_cycle_MG(int l,double *w,double *v);
    void V_cycle_HB(int l,double *w, double *v);
    void CG(int l, int num_of_iter,  double *y, double *b, Matrix *A );
    void gmres(int n,int &nit,double *x,double *b, Matrix *A);

    #if HYPRE_SOLVE == ON
      void Initialize_FEGrid(void *grid);
      void Solve_HYPRE(int, int, int, char *);
      void Solve_HYPRE(double *solution, char *);
    #endif  
    
    void InitializeDirichletBdr();
    void Update(reals x);
    void Update_Slave_Values(reals x);
    void Update_Slave_Values(int  *x);
    void Update_Slave_Face_Values(int *x);
    void Zero_Slave_Values(reals x);
    void Print();

    //== Local refinement procedures =========================================
    void add_face_to_packet(int subd_number, int face_number);
    void add_edge_to_packet(int node1, int node2, set<int, less<int> > &subdomains);
    void Update_packets(int *, vector<queue> &, vector<face> &, int **);
    int  find(int node, int *faces, int &number_faces);
    void ReorderFaceNodes(int packet_number, double *coord);
    // void Update_Interpolation_Bdr(int *V, double *A);

    void Refine(int num_levels, int Refinement);
    void LocalRefine(int *r_array);
    void MarkFaces(double *dBuf);
    void get_face_splittings(int, int &, double3 &, vector<queue> &,
			     vector<face> &);
    //========================================================================

    void CreateGlobalNodeIndex(int *g_index, int &nslaves);
    void CreateGlobalElemIndex(int *g_index);
    void CreateGlobalFaceIndex(int *g_index, int &nslaves);
};

#endif   // SUBDOMAIN
