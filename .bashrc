__nuvolos_welcome="Welcome to Nuvolos, the Collaborative Computational Workspace!

/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/

About your environment:
   * You can paste the content of your clipboard with Ctrl+Shift+V (Windows) 
     and command+V (macOS).
   * Use conda to install additional software packages.
   * A conda environment called 'base' is active by default, hit 'conda info' 
     and 'conda list' to learn more.
   * The terminal is scrollable, you can use your mouse wheel or the inner 
     scrollbar to scroll.
   * To modify your bashrc, edit /files/.bashrc
   * For even more info, hit up our documentation at https://docs.nuvolos.cloud

/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/"

echo "$__nuvolos_welcome"

PS1="\t - \[\033[1;34m\]\u\[\033[00m\]:\[\033[1;36m\]\w\$\[\033[00m\] "

if [ -f ${CONDA_DIR}/bin/activate ]; then
    source ${CONDA_DIR}/bin/activate
    conda activate
fi