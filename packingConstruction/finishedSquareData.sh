for n in 64 128 256 512 1024 2048 4096 8192;
do python3 makeTriangulationFromPacking.py $n idealPack$n posRad .915 10 0;
python3 makeTriangulationFromPacking.py $n idealPack$n posRad .915 20 10;
python3 makeTriangulationFromPacking.py $n idealPack$n posRad .915 30 20;
python3 camLMPacking.py $n
done
