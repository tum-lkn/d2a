sudo sysctl -A | grep "sched" | grep -v "domain"
gcc test_sched.c -lpthread -o test_sched
./test_sched 2 &
./test_sched 3 &
./test_sched 1 &
sleep 15s
tar -czvf rate-cost-cfs-no-yield-schedlat-10ms.tar.gz stats-long-worker-3.csv stats-short-worker-3.csv stats-second-short-worker-3.csv *.log
rm *.csv *.log
scp rate-cost-cfs-no-yield-schedlat-10ms.tar.gz sim@pk-vnf2cpu02.forschung.lkn.ei.tum.de:/home/sim/git-repos/rl-bin-packing/data/rate-cost-cfs-no-yield-schedlat-10ms
