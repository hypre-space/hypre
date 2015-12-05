
######################################################################
#
# The perl scripts below define how `latex2html' should handle the
# definitions in `hypre.sty'.
#
# (C) 2007 Lawrence Livermore National Security, LLC.
#
# $Revision: 2.3 $
#
######################################################################

#---------------------------------------------------------------------
# Various verbatim-like text commands:
#
# See hypre.sty file for usage details.
#
# Note (HACK): To get the `~' character to always print, an empty
# HTML comment was added after the \~ command.
#
#---------------------------------------------------------------------

sub do_cmd_code{
    local($_) = @_;
    local($text);
    s/$next_pair_pr_rx/$text = $&; ''/eo;
    $text =~ s/~/\\~<!---->/g;                 # print `~' characters
    join('',"<code>",$text,"</code>",$_);
}

sub do_cmd_file{
    local($_) = @_;
    local($text);
    s/$next_pair_pr_rx/$text = $&; ''/eo;
    $text =~ s/~/\\~<!---->/g;                 # print `~' characters
    join('',"<code>",$text,"</code>",$_);
}

sub do_cmd_kbd{
    local($_) = @_;
    local($text);
    s/$next_pair_pr_rx/$text = $&; ''/eo;
    $text =~ s/~/\\~<!---->/g;                 # print `~' characters
    join('',"<kbd>",$text,"</kbd>",$_);
}

#---------------------------------------------------------------------
# Commands: hypre
#---------------------------------------------------------------------

sub do_cmd_hypre{
    local($_) = @_;
    join('',"HYPRE",$_);
}

######################################################################
#
# Title Page and Copyright Page definitions:
#
######################################################################

#---------------------------------------------------------------------
# Environment: TitlePage
#   Does nothing.
#---------------------------------------------------------------------

sub do_env_TitlePage{
    local($_) = @_;
    $_ = &translate_environments($_);
    join('',"<DIV ALIGN=\"LEFT\">\n$_</DIV>","\n");
}

#---------------------------------------------------------------------
# Command: Title
#---------------------------------------------------------------------

sub do_cmd_Title{
    local($_) = @_;
    s/$next_pair_pr_rx/$text = $&; ''/eo;
    $TITLE = $text;
    join('',"<H1>$text</H1>\n<HR SIZE=4 ALIGN=\"CENTER\" NOSHADE>",$_);
}

#---------------------------------------------------------------------
# Command: SubTitle
#---------------------------------------------------------------------

sub do_cmd_SubTitle{
    local($_) = @_;
    s/$next_pair_pr_rx/$text = $&; ''/eo;
    join('',"<H3 ALIGN=\"RIGHT\">$text</H3>\n",$_);
}

#---------------------------------------------------------------------
# Command: Author
#---------------------------------------------------------------------

sub do_cmd_Author{
    local($_) = @_;
    s/$next_pair_pr_rx/$text = $&; ''/eo;
    join('',"<H3>$text</H3>\n<HR SIZE=2 ALIGN=\"CENTER\" NOSHADE>",$_);
}

#---------------------------------------------------------------------
# Environment: CopyrightPage
#   Does nothing.
#---------------------------------------------------------------------

sub do_env_CopyrightPage{
    local($_) = @_;
    $_ = &translate_environments($_);
    $_;
}

######################################################################
#
# Miscellaneous commands and environments:
#
######################################################################

#---------------------------------------------------------------------
# Environment: display
#---------------------------------------------------------------------

sub do_env_display{
    local($_) = @_;
    join('',"<blockquote>",$_,"</blockquote>");
}

######################################################################
# This next line must be the last line
######################################################################

1;

