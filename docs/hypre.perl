
######################################################################
#
# The perl scripts below define how `latex2html' should handle the
# definitions in `hypre.sty'.
#
# (C) 1998 Regents of the University of California.
#
# $Revision$
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
    $_;
}

#---------------------------------------------------------------------
# Command: Title
#---------------------------------------------------------------------

sub do_cmd_Title{
    local($_) = @_;
    s/$next_pair_pr_rx/$text = $&; ''/eo;
    $TITLE = $text;
    join('',"<H1>$text</H1>\n",$_);
}

#---------------------------------------------------------------------
# Command: SubTitle
#---------------------------------------------------------------------

sub do_cmd_SubTitle{
    local($_) = @_;
    s/$next_pair_pr_rx/$text = $&; ''/eo;
    join('',"<strong>$text</strong>\n",$_);
}

#---------------------------------------------------------------------
# Command: Author
#---------------------------------------------------------------------

sub do_cmd_Author{
    local($_) = @_;
    s/$next_pair_pr_rx/$text = $&; ''/eo;
    join('',"<strong>$text</strong>\n",$_);
}

#---------------------------------------------------------------------
# Environment: CopyrightPage
#   Does nothing.
#---------------------------------------------------------------------

sub do_env_CopyrightPage{
    local($_) = @_;
    $_;
}

#---------------------------------------------------------------------
# Command: InsertGraphics
#---------------------------------------------------------------------

sub do_cmd_InsertGraphics{
    local($_) = @_;
    s/$next_pair_pr_rx/$text = $&; ''/eo;
    s/$next_pair_pr_rx/; ''/eo;
    join('',"<img src=\"$text.gif\">\n",$_);
}

######################################################################
# This next line must be the last line
######################################################################

1;

