for phi in .850 .855 .860 .865 .870 .875 .880 .885 .890 .895 .900 .905 .910 .915 .920 .925;
do python3 00-cpPosMinPacking.py 512 posMin $phi 30;
python3 01-cpPosRadMinPacking.py 512 radMin $phi 30;
done
