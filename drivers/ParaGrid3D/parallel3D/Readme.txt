1. In Problems we give what kind of meshes we have produced.

2. For debugging : 
  In Main.cpp include file degug.h, compile with
  "CC -g Main.cpp -lm" and debug a.out. The debugger should be started from
  emacs opened from grendel.

3. To start a problem in "functions.cpp" the file with the Problem's functions
  should be included. The functions should be, for example,  as given in file
  pressure_bioscreen.cpp. According to the mesh one should adjust the permia-
  bilities according to the mark of the layer. The used functions should be 
  declared in file "functions.h". The user shuld also set the constants Con-
  vection, Exact, Problem. There are some more constants concerning the 
  solvers that have to be adjusted. These are in "definitions.h".

4. Domain decomposition problem we start with 
  make -f makefile_dd dd
  (first setting the necessary constants, as explained above). One should see
  what options for the compiler are used in order to start mpi. We run the
  program with (for example with 4 processors):
  mpirun -np 4 dd
  There are also definitions in dd_definitions.h.

5. For Parallel computations using compiler derivatives one should unmark the
  comments concerning the parallel computations in file Main.cpp and compile
  with "make -f makefile_grendel programa" (the compilation should have at the
  end "-mp" included) - files ulocks.h and task.h have to be included.

  Using compiler directives:

#include <stdio.h>
main(){
  #pragma parallel
  {
    printf("Hello\n");
  }
}

  Set the number of threads with 
  setenv MP_SET_NUMTHREADS num
  Compile with :
  CC ....cpp -mp
