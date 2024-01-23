for phi in .915 .920 .925 .930 .935 .940 .945 .950 .955 .960 .965;
do echo $phi;
python3 cpReadyPacking.py 512 posRadMin $phi 10;
done;
