--------------------------------------------------------------
System interface:
See the file doc.txt for documentation of the system interface
between lobpcg and hypre.



--------------------------------------------------------------
LLNL comments: 

The LLNL file system is shared, so make sure you untar hypre 
into different directories for every cluster. You can use only one directory 
for tera and gps because they are binary compatible.  
To compile hypre, just 
cd hypre-1.6.0/src
and simply run configure and make. 
configure is tuned to work on every LLNL cluster without any special options.

To compile lobcpg, create a new directory hypre-1.6.0/src/lobpcg
uncompress/untar the lobpcg distribution file and run 
make -f Makefile.cluster
where Makefile.cluster is the corresponding make file:
Makefile.blue for Blue,
Makefile.dec for Tera/GPS,
Makefile.linux for LX.  

To run the lobpcg eigensolver on each machine:

On blue, use PSUB/POE, IBM's analog of PBS - follow a comment in blue.1
You'll need to edit a line in blue.1 that points to the right directory.
If you want to learn more, check
http://www.llnl.gov/asci/platforms/bluepac/psub.example.html
http://www.llnl.gov/asci/platforms/bluepac/
http://www.llnl.gov/computing/tutorials/workshops/workshop/poe/MAIN.html

On tera, gps and lx, our shell script IJ_eigen_solver.sh works, so just run it.
On LX, you first need to enable passwordless ssh between the nodes,
by exectuting from the command line:
kinit -f
It will ask you for a password again.

To use tera, gps and lx for larger problem, one needs to use PBS. 
Read some stuff at
http://www.llnl.gov/icc/lc/OCF_resources.html
in particular things from
http://www.llnl.gov/computing/training/#tutorials



--------------------------------------------------------------
CU-Denver MATH Beowulf comments: 

There are several options to compile hypre, see 
corresponging comments in:  

Makefile.beowulf_gcc_scali   (gnu)
Makefile.beowulf_pgcc_scali  (pgi)
Makefile.beowulf_mpich  

and then to complile lobpcg use the right Makefile. 
To run the lobpcg eigensolver, the procedure depends 
on if scali or mpich is used for compilation. 

a) For scali: 
Examples (background job):
scasub -mpimon -np 6 -npn 2 IJ_eigen_solver -n 50 50 50

b) For mpich: 
Examples (interactive job):
mpirun -np 5 IJ_eigen_solver -solver 12 -itr 20
mpimon IJ_eigen_solver -- node1 2 node2 2
