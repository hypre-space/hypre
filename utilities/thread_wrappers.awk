#BHEADER***********************************************************************
# (c) 1998   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision$
#EHEADER***********************************************************************

#===========================================================================
# To use, do:
#
# /usr/xpg4/bin/awk -f {this file} < {input file} > {output file}
#
#===========================================================================

BEGIN {
  special_types = "\
HYPRE_StructGrid \
HYPRE_StructMatrix \
HYPRE_StructStencil \
HYPRE_StructVector \
HYPRE_StructSolver"
}

/ P\(\(/ {
  ####################################################
  # parse prototype and define various variables
  ####################################################

  split($0, b, "[\ \t]*P\(\([\ \t]*");
  routine_string = b[1];
  m = match(b[2], "[\ \t]*));");
  arg_string = substr(b[2], 1, m-1);

  n = split(routine_string, a, "[^A-Za-z_0-9]");
  routine = a[n];
  m = match(routine_string, routine);
  routine_type = substr(routine_string, 1, m-1);
  routine_args = routine"Args";
  routine_push = routine"Push";
  routine_vptr = routine"VoidPtr";

  num_args = split(arg_string, arg_array, "[\ \t]*,[\ \t]*");
  for (i = 1; i <= num_args; i++)
    {
      n = split(arg_array[i], a, "[^A-Za-z_0-9]");
      arg[i] = a[n];
      m = match(arg_array[i], arg[i]);
      arg_type[i] = substr(arg_array[i], 1, m-1);
    }

  ####################################################
  # write the wrapper routine for this prototype
  ####################################################

  print "";
  print "/*----------------------------------------------------------------";
  print " * "routine" thread wrappers";
  print " *----------------------------------------------------------------*/";
  print "";
  print "typedef struct {";
  for (i = 1; i <= num_args; i++)
    {
      print "   "arg_type[i] arg[i]";";
    }
  print "   "routine_type" returnvalue[hypre_MAX_THREADS];";
  print "} "routine_args";";
  print "";
  print "void";
  print routine_vptr"( void *argptr )";
  print "{";
  print "   int threadid = hypre_GetThreadID();";
  print "";
  print "   "routine_args" *localargs =";
  print "      ("routine_args" *) argptr;";
  print "";
  print "   (localargs -> returnvalue[threadid]) =";
  print "      "routine"(";
  endline = ",";
  for (i = 1; i <= num_args; i++)
    {
      if (i == num_args)
	{
	  endline = " );";
	}
      m = match(arg_type[i], "[^A-Za-z_0-9]");
      base_type = substr(arg_type[i], 1, m-1);
      if (special_types ~ base_type)
	{
	  base_pointer = substr(arg_type[i], m);
	  if (base_pointer ~ "\*")
	    {
	      print "         &(*(localargs -> "arg[i]"))[threadid]"endline;
	    }
	  else
	    {
	      print "         (localargs -> "arg[i]")[threadid]"endline;
	    }
	}
      else
	{
	  print "         localargs -> "arg[i] endline;
	}
    }
  print "}";
  print "";
  print routine_type;
  print routine_push"(";
  for (i = 1; i <= num_args-1; i++)
    {
      print "   "arg_type[i] arg[i]",";
    }
  print "   "arg_type[num_args] arg[num_args]" )";
  print "{";
  print "   "routine_args" pushargs;";
  print "   int i;";
  print "   "routine_type" returnvalue;";
  print "";
  for (i = 1; i <= num_args; i++)
    {
      print "   pushargs."arg[i]" = "arg[i]";";
    }
  print "   for (i = 0; i < NUM_THREADS; i++)";
  print "      hypre_work_put( "routine_vptr", (void *)&pushargs );";
  print "";
  print "   hypre_work_wait();";
  print "";
  print "   returnvalue = pushargs.returnvalue[0];";
  print "";
  print "   return returnvalue;";
  print "}";
}


