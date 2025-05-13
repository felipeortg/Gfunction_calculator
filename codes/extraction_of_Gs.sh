#!/usr/bin/env zsh

while read linei; do
    while read linef; do
    echo ----------
    echo Computing transition $(echo $linei | awk '{print $1}') to $(echo $linef | awk '{print $1}')

        energyi=$(echo $linei | awk '{print $2}')
        momentumi=$(echo $linei | awk '{print $1}' | awk -F"_" '{print $2}' | awk '{print substr($0,2,1), substr($0,3,1), substr($0,4,1)}')
        energyf=$(echo $linef | awk '{print $2}')
        momentumf=$(echo $linef | awk '{print $1}' | awk -F"_" '{print $2}' | awk '{print substr($0,2,1), substr($0,3,1), substr($0,4,1)}')

        sed "s/thisisPi/${momentumi}/
        s/thisisPf/${momentumf}/
        s/thisisEimin/${energyi}/
        s/thisisEfmin/${energyf}/" config_files/config_template.txt > temp.config

        zsh calcG.sh temp.config

    done < energies.dat

done < energies.dat

rm temp.config
