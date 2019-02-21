configfile=$1
energyfile="Etemp.npy"


python En_setup.py $configfile $energyfile

python IN_calc.py $configfile $energyfile &

python IA_calc.py $configfile $energyfile &

python Sum_calc.py $configfile $energyfile &

wait

python Gsave.py $configfile $energyfile

rm $energyfile



