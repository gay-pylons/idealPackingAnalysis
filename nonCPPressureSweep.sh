for n in 64 128 256 512 1024 2048 4096;
do echo $n;
python3 posMinPressureSweep.py $n posMin 10;
python3 radMinPressureSweep.py $n radMin 10;
done;
