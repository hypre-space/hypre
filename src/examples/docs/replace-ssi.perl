#!/usr/bin/perl

# This replaces the server-side includes.

use IO::File;

while (<>) {
    &process_or_include($_);
}

sub process_or_include {
    local $_ = shift;
     if (/^<!--#include file = "(\S+)"-->/) {
         &include($1);
    } else {
        &process($_);
    }
}

sub include {
    my $name = shift;
    my $F = IO::File->new($name)
        or die "Cannot open $name: $!";
    while (<$F>) {
        &process_or_include($_);
    }
}

sub process {
    my $line = shift;
    print "$line";
}
