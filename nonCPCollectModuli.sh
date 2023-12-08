for n in 64 128 256 512 1024 2048 4096;	
do echo $n;
python3 collectModuli.py ../idealPackingLibrary/$n/posMinPressureSweep2/posMin 10;
python3 collectModuli.py ../idealPackingLibrary/$n/radMinPressureSweep2/radMin 10;
done;
