####
# $Revision$
# $Id$
# yes or no input processor definitions for including in sh scripts
# uses global variables:
#	 response, prompt, question, padding, default
# some usages:
#	yes_or_no "prompt string" default_answer pad
#	yes_or_no "prompt string" default_answer ""
# Original author unknown. Swiped from anonymous sources
# for ISIS++ by Ben Allan
####

#####
#	tab-->blank, multiple blanks-->single blank, leading/trailing blanks wiped
#####
min_white()
{
	minimized=`echo "$1" | tr -s '\011' ' ' | sed -e 's/^ *//' -e 's/ *$//'`
}

prt_prompt()
{
	awk 'END {printf "   %s ", "'"$1"'"}' </dev/null
}

#####
#	Get an input.  Lots of control over formatting of questions.
#
#	Is there a default value?					yes/no
#	What character indicates end of question?	"?" vs ":"
#	Do we pad questions to minimum size?		yes/no
#
#	Examples:
#
#	    Enter name                       : <answer>
#		Enter expiration [2-july-1993]   : <answer>
#		Do you have licenses? <answer>
#		Install man pages? [no] <answer>
#####
prompt_and_read_raw()
{
	prompt="$1"
	question="$2"
	padding="$3"
	default="$4"

	response=""

	# Presentation form of default has [] around it
	if test ! -z "$default"
	then
		view_def=" [$default]"
	else
		view_def=""
	fi

	if test -z "$padding"
	then
		# No padding, default to right of question mark
		true_prompt="$prompt$question$view_def"
	else
		# Padded, so default to left of question mark
		format="%${padding}s$question"
		true_prompt=`
			awk 'END {printf "'"$format"'", "'"$prompt$view_def"'"}' </dev/null`
	fi

	#	Insist we get a response
	while test -z "$response"
	do
		prt_prompt "$true_prompt"
		if read new_response
		then
			min_white "$new_response"
			new_response="$minimized"
		else
			echo "Exiting"
			cleanup_and_exit 1
		fi

		if test -z "$new_response"
		then
			response="$default"
		else
			response="$new_response"
		fi
	done
}

################################################################################
#
#	print_padded_response_raw() - This routine is used to simulate
#		the output seen by prompt_and_read_raw().
#
################################################################################

print_padded_response_raw() {

	# Retrieve paramters
	prompt="$1"
	question="$2"
	padding="$3"
	response="$4"

	if test -z "$padding"
	then
		# No padding, default to right of question mark
		echo "   $prompt$question $response"
	else
		# Padded, so default to left of question mark
		format="   %${padding}s$question $response\n"
		awk 'END {printf "'"$format"'", "'"$prompt"'"}' </dev/null
	fi
}

#####
#	Basic Q&A functions
#####
prompt_and_read()
{
	prompt="$1"

	prompt_and_read_raw "$prompt" "?" "" ""
}

prompt_and_read_with_default()
{
	prompt="$1"
	default="$2"

	prompt_and_read_raw "$prompt" "?" "" "$default"
}

#####
#	Padded versions (keep padding the same, share it with print_values)
#####
pretty_pad="-39"
prompt_and_read_padded()
{
	prompt="$1"

	prompt_and_read_raw "$prompt" ":" "$pretty_pad" ""
}

print_padded_response()
{
	print_padded_response_raw "$1" ":" "$pretty_pad" "$2"
}

prompt_and_read_with_default_padded()
{
	prompt="$1"
	default="$2"

	prompt_and_read_raw "$prompt" ":" "$pretty_pad" "$default"
}
# try to get a yes/no answer from someone
yes_or_no()
{
	query="$1"
	default="$2"
	padding="$3"
	response=""
	until test ! -z "$response"
	do
		if test -z "$padding"
		then
			prompt_and_read_with_default "$query" "$default"
		else
			prompt_and_read_with_default_padded "$query" "$default"
		fi
		case "$response" in
			y|Y|yes|Yes|YES)	response=y;;
			n|N|no|No|NO)		response=n;;
			*)					response="";
								echo "Please answer either 'yes' or 'no'.";;
		esac
	done
}
