# HTML.PERL by Nikos Drakos <nikos@cbl.leeds.ac.uk> 2-DEC-93
# Computer Based Learning Unit, University of Leeds.
#
# Extension to LaTeX2HTML to translate hypermedia extensions
# to LaTeX defined in html.sty to equivalent HTML commands.
#
#
#
# Modifications:
#
#  nd = Nikos Drakos <nikos@cbl.leeds.ac.uk>
#  jz = Jelle van Zeijl <jvzeijl@isou17.estec.esa.nl>

# nd 18-AUG-94 - Added do_cmd_htmladdtonavigation
# nd 26-JUL-94 - Moved do_env_latexonly from main script and added support
#                for do_cmd_latexonly
# jz 22-APR-94 - Added command htmlref
# nd 15-APR-94 - Added command htmladdnormallinkfoot
# nd  2-DEC-93 - Created

package main; 

sub do_cmd_htmladdnormallink{
    local($_) = @_;
    local($text, $url);
    s/$next_pair_pr_rx/$text = $2; ''/eo;
    s/$next_pair_pr_rx/$url = $2; ''/eo;
    join('',&make_href($url,$text),$_);
}

sub do_cmd_htmladdnormallinkfoot{
    &do_cmd_htmladdnormallink;
}

sub do_cmd_htmladdimg{
    local($_) = @_;
    local($url);
    local($alt, $dummy) = &get_next_optional_argument;
    $alt =~ s/["`']//g;
    s/$next_pair_pr_rx/$url = $2; ''/eo;
    $url = &revert_to_raw_tex($url);
    join('',&embed_image($url,"external",0,"$alt",0),$_);
}

sub do_cmd_externallabels{
    local($_) = @_;
    local($URL,$labelfile);
    s/$next_pair_pr_rx/$URL = $2; ''/eo;
    s/$next_pair_pr_rx/$labelfile = $2; ''/eo;
#
	local($dir,$nosave) = ('','');
#
    if (-f "$labelfile") {
	require($labelfile)}
    else {
	$global{'warnings'} .= 
	    "Could not find the external label file: $labelfile\n" ;
    }
    $_;
}

sub do_cmd_htmlhead {
    local($_) = @_;
    local(@tmp, $section_number, $sec_id);
    s/$next_pair_pr_rx//o; $curr_sec = $2;
    s/$next_pair_pr_rx//o; $TITLE = $2;
    $curr_sec =~ s/\*$/star/;
    $current_depth = $section_commands{$curr_sec};
    @tmp = split(/$;/, $encoded_section_number{&encode_title($TITLE)});
    $section_number = shift(@tmp);
    $TITLE = "$section_number " . $TITLE if $section_number;
    @tmp = @curr_sec_id;
#   $tmp[$current_depth] = 0;
    $toc_section_info{join(' ', @tmp)} =
      "$current_depth$delim$CURRENT_FILE$delim$TITLE";
    join('', '<P>' , &make_section_heading($TITLE, "H2"), $_);
}

sub do_cmd_segment {
    local($_) = @_;
    local($ctr, $index);
    s/$next_pair_pr_rx//o;	# Ditch file
    s/$next_pair_pr_rx//o; $ctr = $2;
    s/$next_pair_pr_rx//o;	# Ditch heading
    $segment_sec_id[$index] += 1 if ($index = $section_commands{$ctr});
    $SEGMENTED = 1;
    $_;
}

sub do_cmd_segmentstar {
    local($_);
    &do_cmd_segment($_);
}

sub do_cmd_bodytext {
    local($_) = @_;
    s/$next_pair_pr_rx//o; $BODYTEXT = &revert_to_raw_tex($2);
    $_;
    }

sub do_cmd_internal{
    local($_) = @_;
    local($type, $prefix, $file, $var, $buf);
    $type = "internals";
    s/$optional_arg_rx/$type = $1; ''/eo;
    s/$next_pair_pr_rx/$prefix = $2; ''/eo;
    $file = "${prefix}$type.pl";
    return ($_) unless (-f $file);
    local($dir,$nosave) = ('',1); 
    local($tmpdir,$rest) = ('',''); 
    ($tmpdir, $rest) = split("/", $file, 2); 
    while ($rest) { $dir .= $tmpdir . "/";
	($tmpdir, $rest) =  split("/", $rest, 2);
    }
    if (! ($type =~ /(figure|table)/)) {
print "\nLoading segment data from $file ";
	require ($file);
	return ($_);
	}
    open (CAPTIONS, $file);
    $buf = join('', <CAPTIONS>);
    if ($type =~ /figure/ ) {
	if (defined $figure_captions) {$figure_captions .= $buf;}
	else {$figure_captions = $buf;}
	}
    else {
	if (defined $table_captions) {$table_captions .= $buf;}
	else {$table_captions = $buf;}
	}
    close (CAPTIONS);
    $_;
    }
	
sub do_cmd_externalref{
    local($_) = @_;
    &process_ref($external_ref_mark,$external_ref_mark);
}

sub do_cmd_hyperref {
    local($_) = @_;
    local($text);
    s/$next_pair_pr_rx/$text = $2; ''/eo;
    s/$next_pair_pr_rx//o; # Throw this away ...
    s/$next_pair_pr_rx//o; # ... and this
    &process_ref($cross_ref_mark,$cross_ref_mark,$text);
}

sub do_cmd_htmlref {
    local($_) = @_;
    local($text);
    s/$next_pair_pr_rx/$text = $2; ''/eo;
    &process_ref($cross_ref_mark,$cross_ref_mark,$text);
}

# IGNORE the contents of this environment 
sub do_env_latexonly {
    "";
}

# IGNORE the argument of this command
sub do_cmd_latexonly {
    local($_) = @_;
    s/$next_pair_pr_rx//o;
    $_;
}

# IGNORE the argument of this command
sub do_cmd_htmlimage {
    local($_) = @_;
    s/$next_pair_pr_rx//o;
    $global{'warnings'} .= "\nThe command \"htmlimage\" is only effective inside an
environment which generates an image (eg \"figure\")\n";
    $_;
}

sub do_cmd_htmladdtonavigation {
    local($_) = @_;
    s/$next_pair_pr_rx//o;
    $CUSTOM_BUTTONS = $2;
    $_;
}

1;				# This must be the last line



