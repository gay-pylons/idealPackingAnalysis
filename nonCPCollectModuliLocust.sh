for n in 2048;	
do echo $n;
python3 collectModuliLocust.py ../idealPackingLibrary/$n/posMinPressureSweep2/posMin 10;
python3 collectModuliLocust.py ../idealPackingLibrary/$n/radMinPressureSweep2/radMin 10;
done;
