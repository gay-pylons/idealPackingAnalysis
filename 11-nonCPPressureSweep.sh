#for n in 64 128 256 512 1024 2048 4096;
for n in 8192;
do echo $n;
python3 posMinPressureSweep.py $n posMin 10 8;
#python3 radMinPressureSweepLocust.py $n radMin 10;
done;
