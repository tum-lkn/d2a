#define _GNU_SOURCE
#include <stdio.h>
#include <sched.h>
#include <stdlib.h>
#include <inttypes.h>
#include <pthread.h>
#include <time.h>
#include <sys/resource.h>
#include <unistd.h>


uint64_t rdtsc(){
    unsigned int lo,hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((uint64_t)hi << 32) | lo;
}


static void *copy_file(uint32_t pid) {
  FILE *f_in, *f_ot;
  char filename_in[32];
  char filename_ot[32];
  char c;
  sprintf(filename_in, "/proc/%" PRIu32 "/sched", pid);
  sprintf(filename_ot, "/home/nfv/proc-%" PRIu32 "-sched.log", pid);
  printf("Copy %s to %s\n", filename_in, filename_ot);
  f_in = fopen(filename_in, "r");
  if(f_in == NULL) { printf("Could not open input file\n"); return NULL; } 
  f_ot = fopen(filename_ot, "w");
  if(f_ot == NULL) { printf("Could not open output file\n"); fclose(f_in); return NULL; }
  while((c = fgetc(f_in)) != EOF){
    fputc(c, f_ot);
  }
  fclose(f_in);
  fclose(f_ot);
  return NULL;
}


static void *worker(double yield_after_s, FILE * stats) {
  uint32_t duration_s = 15;
  int bsize = duration_s * 1000 * 10;
  uint64_t ticks[bsize];
  uint64_t yielded_ts[bsize];
  uint64_t resumed[bsize];
  double td;
  uint32_t idx = 0;
  uint32_t idx_yielded = 0;
  uint32_t idx_resumed = 0;
  uint64_t yield_after_ns = (uint64_t)(yield_after_s * 2200000000)*100000000000;
  // uint64_t yield_after_ns = (uint64_t)(yield_after_s * 2200000000);
  int yielded = 1;
  sleep(1);

  for(int i = 0; i < bsize; i++) {
    ticks[i] = 0;
    yielded_ts[i] = 0;
    resumed[i] = 0;
  }
  
  uint64_t start = rdtsc();
  uint64_t start_tick = rdtsc();
  uint64_t start_period;
  do {
    // Check if the thread yielded. If so clear the flag, start a new scheduling
    // period and capture the time the thread resumed execution.
    if(yielded == 1) {
      yielded = 0;
      start_period = rdtsc();
      if(idx_resumed < bsize - 1) {
        resumed[idx_resumed] = rdtsc();
      }
      idx_resumed++;
    }
    // Measure the execution time.
    if(rdtsc() - start_tick >= 220000) {
      start_tick = rdtsc();
      if(idx < bsize - 1) {
        ticks[idx] = rdtsc();
        idx++;
      }
    }
    // If the thread executed for more than yield_after_s seconds then
    // voluntarily yield the CPU and note the time this happened.
    if(rdtsc() - start_period >= yield_after_ns) {
      yielded = 1;
      if(idx_yielded < bsize - 1) {
        yielded_ts[idx_yielded] = rdtsc();
      }
      idx_yielded++;
      sched_yield();
    }
  } while(rdtsc() - start < 2200000000 * 10);

  printf("Copy scheduling stats\n");
  copy_file(getpid());
  printf("Store results\n");
  fprintf(stats, "ticks,yielded,resumed\n");
  for(int i = 0; i < bsize; i++) {
    fprintf(stats, "%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n", ticks[i], yielded_ts[i], resumed[i]);
  }
  fclose(stats);
  return NULL;
}

static void * worker_long(void *val) {
  if(setpriority(PRIO_PROCESS, 0, 0) == -1) {
      printf("Failed to set nice value of long worker\n");
      return NULL;
  }
  FILE * stats = fopen("stats-long-worker-3.csv", "w");
  return worker(0.01, stats);
}

static void * worker_short(void *val) {
  if(setpriority(PRIO_PROCESS, 0, 8) == -1) {
      printf("Failed to set nice value of short worker\n");
      return NULL;
  }
  FILE * stats = fopen("stats-short-worker-3.csv", "w");
  return worker(0.001, stats);
}

static void * worker_short2(void *val) {
  if(setpriority(PRIO_PROCESS, 0, 8) == -1) {
      printf("Failed to set nice value of second short worker\n");
      return NULL;
  }
  FILE * stats = fopen("stats-second-short-worker-3.csv", "w");
  return worker(0.001, stats);
}

int main(int argc, char *argv[]) {
  cpu_set_t  mask;
  CPU_ZERO(&mask);
  CPU_SET(20, &mask);
  int rc1;
  rc1 = sched_setaffinity(0, sizeof(mask), &mask);
  if(rc1 != 0) {
    printf("Could not set task affinity of thread 1\n");
    return -1;
  }

  struct sched_param param;
  // param.sched_priority = 4;
  // rc1 = sched_setscheduler(0, SCHED_RR, &param);
  rc1 =0; 
  if(rc1 != 0) {
    printf("Could not set scheduler, error code %i\n", rc1);
    return -1;
  }

  if(strtol(argv[1], NULL, 10) == 1) {
    printf("Start long worker\n");
    worker_long(NULL);
    printf("Stop worker\n");
  } 
  if(strtol(argv[1], NULL, 10) == 2) {
    printf("Start short worker\n");
    worker_short(NULL);
  }
  if(strtol(argv[1], NULL, 10) == 3) {
    printf("Start second short worker\n");
    worker_short2(NULL);
  }
  return 0;
}
