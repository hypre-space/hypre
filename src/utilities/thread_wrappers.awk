# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

#===========================================================================
# To use, do:
#
# /usr/xpg4/bin/awk -f {this file} < {input file} > {output file}
#
#===========================================================================

BEGIN {
  prefix = "HYPRE_"
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
      m = match(arg_type[i], "[^A-Za-z_0-9]");
      if (arg_type[i] ~ prefix)
	{
	  base_type = substr(arg_type[i], 1, m-1);
	  print "   "base_type"Array *" arg[i]";";
	}
      else
	{
	  print "   "arg_type[i] arg[i]";";
	}
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
      if (arg_type[i] ~ prefix)
	{
	  base_pointer = substr(arg_type[i], m);
	  if (base_pointer ~ "\*")
	    {
	      print "         &(*(localargs -> "arg[i]"))[threadid]"endline;
	    }
	  else
	    {
	      print "         (*(localargs -> "arg[i]"))[threadid]"endline;
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
  endline = ",";
  for (i = 1; i <= num_args; i++)
    {
      if (i == num_args)
	{
	  endline = " )";
	}
      m = match(arg_type[i], "[^A-Za-z_0-9]");
      if (arg_type[i] ~ prefix)
	{
	  base_type = substr(arg_type[i], 1, m-1);
	  base_pointer = substr(arg_type[i], m);
	  if (base_pointer ~ "\*")
	    {
	      print "   "base_type"Array *" arg[i] endline;
	    }
	  else
	    {
	      print "   "base_type"Array "arg[i] endline;
	    }
	}
      else
	{
	  print "   "arg_type[i] arg[i] endline;
	}
    }
  print "{";
  print "   "routine_args" pushargs;";
  print "   int i;";
  print "   "routine_type" returnvalue;";
  print "";
  for (i = 1; i <= num_args; i++)
    {
      m = match(arg_type[i], "[^A-Za-z_0-9]");
      if (arg_type[i] ~ prefix)
	{
	  base_type = substr(arg_type[i], 1, m-1);
	  base_pointer = substr(arg_type[i], m);
	  print "   pushargs."arg[i]" = ("base_type"Array *)"arg[i]";";
	}
      else
	{
	  print "   pushargs."arg[i]" = "arg[i]";";
	}
    }
  print "   for (i = 0; i < hypre_NumThreads; i++)";
  print "      hypre_work_put( "routine_vptr", (void *)&pushargs );";
  print "";
  print "   hypre_work_wait();";
  print "";
  print "   returnvalue = pushargs.returnvalue[0];";
  print "";
  print "   return returnvalue;";
  print "}";
}


