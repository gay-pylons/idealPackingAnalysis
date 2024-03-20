for n in 64 128 256 512 1024 2048 4096 8192;
	do sbatch 131-JumblePressureSweep.srun $n;
	done;
