#!/usr/bin/env zsh
for p in ../recalc_Ecm_JackFiles/*; do 
    echo `basename $p` `calc $p | awk '{print $2, $3}'` >> energies.dat
done
