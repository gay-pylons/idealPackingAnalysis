for n in 64 128 256 512 1024 2048 4096 8192;
	do
		for i in $(seq 0 9);
		do python3 13-JumblePressureSweep.py $n idealPack$n-$i;
		done
	done
