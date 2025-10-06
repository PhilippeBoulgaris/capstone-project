function cd() {
if [[ $# -eq 0 ]]
then
	builtin cd /files
else
	builtin cd "$*"
fi
}

if [ -f /files/.bashrc ];
    then . /files/.bashrc
fi
