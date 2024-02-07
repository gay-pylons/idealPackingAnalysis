for n in 64 128 256 512 1024 2048 4096;	
do echo $n;
python3 2-collectModuli.py ../idealPackingLibrary/$n/radCPPressureSweep/idealPack$n 10;
python3 2-collectModuli.py ../idealPackingLibrary/$n/posCPPressureSweep/posMin 10;
done;
