for n in 64 128 256 512 1024 2048 4096;	
do echo $n;
python3 2-collectModuli.py ../idealPackingLibrary/$n/jumblePressureSweep/idealPack$n 10;
done;
