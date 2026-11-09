#!/bin/sh
# report-perf.sh -- join the FULL accuracy report and the board benchmark
# report into one markdown table (and the README's performance section).
#
# Usage: scripts/report-perf.sh ACC_REPORT BENCH_REPORT [OUT]
#
# ACC_REPORT    markdown written by the runner (--report)
# BENCH_REPORT  markdown written by scripts/bench-board.sh
#
# Prints one row per entry: max_ulp, speedup, status.  Exit status 1 if any
# entry has max_ulp > 1.0, any fail/flag/sweep count, or speedup <= 1.0.
set -e

ACC=${1:?usage: report-perf.sh ACC_REPORT BENCH_REPORT [OUT]}
BENCH=${2:?usage: report-perf.sh ACC_REPORT BENCH_REPORT [OUT]}
OUT=${3:-}

awk -v acc="$ACC" -v bench="$BENCH" -v out="$OUT" '
  BEGIN {
    n = 0;
    while ((getline line < acc) > 0) {
      if (line ~ /^\| [a-z0-9_]+ \|/ && line !~ /function/) {
        split(line, c, "|");
        nm = c[2]; gsub(/ /, "", nm);
        if (nm == "") continue;
        ulp = c[4] + 0; failv = c[7] + 0; sw = c[8] + 0; fl = c[9] + 0;
        st = c[10]; gsub(/ /, "", st);
        ulpof[nm] = ulp; failof[nm] = failv; swof[nm] = sw; flof[nm] = fl;
        passof[nm] = (st == "PASS");
        if (!(nm in seenacc)) { seenacc[nm] = 1; order[++n] = nm }
      }
    }
    close(acc);
    while ((getline line < bench) > 0) {
      if (line ~ /^\| [a-z0-9_]+ \|/ && line !~ /function/) {
        split(line, c, "|");
        nm = c[2]; gsub(/ /, "", nm);
        if (nm == "") continue;
        sp = c[5]; gsub(/ /, "", sp); sub(/x$/, "", sp);
        spof[nm] = sp + 0;
        if (!(nm in ulpof)) { order[++n] = nm }
      }
    }
    close(bench);
    fmt = "%-10s %9s %9s  %s\n";
    if (out != "") { printf "| function | max_ulp | speedup | status |\n|---|---|---|---|\n" > out }
    printf "%-10s %9s %9s  %s\n", "function", "max_ulp", "speedup", "status";
    for (i = 1; i <= n; i++) {
      nm = order[i];
      if (seen[nm]) continue;
      seen[nm] = 1;
      bad = "";
      if (!passof[nm]) bad = bad " accFAIL";
      if (ulpof[nm] > 1.0) bad = bad " ulp>";
      if (failof[nm] > 0) bad = bad " >1ulp";
      if (swof[nm] > 0) bad = bad " sweep";
      if (flof[nm] > 0) bad = bad " flags";
      if (!(nm in spof)) bad = bad " nobench";
      else if (spof[nm] <= 1.0) bad = bad " slow";
      st = (bad == "") ? "OK" : "FAIL:" bad;
      printf fmt, nm, ulpof[nm], spof[nm], st;
      if (out != "") printf "| %s | %.4f | %.2fx | %s |\n", nm, ulpof[nm], spof[nm], st > out;
      tot++; if (bad != "") nbad++;
    }
    printf "\n%d entries, %d failing\n", tot, nbad;
    if (out != "") printf "\n%d entries, %d failing\n", tot, nbad >> out;
    exit (nbad == 0 ? 0 : 1);
  }
'
