#!/bin/bash
# every case in a fresh process, sequentially; results appended to results.txt, full logs in logs/
cd "$(dirname "$0")"
mkdir -p logs
run() {
  local name=$(echo "$*" | tr ' ' '_')
  [ -f "logs/$name.log" ] && grep -qa "^RES" "logs/$name.log" && return   # done in an earlier run
  timeout 3000 julia --startup-file=no --project=. bench.jl "$@" > "logs/$name.log" 2>&1
  local rc=$?
  if grep -qa "^RES" "logs/$name.log"; then grep -a "^RES" "logs/$name.log" >> results.txt
  else echo "RES | $* | CRASH rc=$rc $(grep -a -m1 'ERROR\|error' logs/$name.log | cut -c1-120)" >> results.txt; fi
}
# 1. Host, DAX underlying (dividend curve), Float64
for s in slnm vg heston; do
  for b in FD EnzymeF EnzymeR Zygote Mooncake; do run $s dax64 A $b; done
  run $s dax64 B FD FD NM
  for c in "FD FD" "EnzymeF FD" "EnzymeR FD" "Zygote FD" "Mooncake FD" "FD EnzymeF" "FD EnzymeR" "FD Zygote" "FD Mooncake" "EnzymeF EnzymeF"; do run $s dax64 B $c LBFGS; done
  for c in "FD FD FD FD" "Zygote FD FD FD" "Mooncake FD FD FD" "EnzymeR FD FD FD" "EnzymeF FD FD FD" \
           "Zygote FD Mooncake FD" "Mooncake FD Mooncake FD" "Zygote FD EnzymeR FD" "Zygote FD FD EnzymeF" "EnzymeR FD FD EnzymeF"; do run $s dax64 C $c; done
  for b in FD Zygote Mooncake EnzymeR EnzymeF; do run $s dax64 J $b; done
done
# 2. Arrays: host and device, Float64 and Float32 (flat dividend)
for s in slnm vg heston; do
  for a in host64 host32 dev64 dev32; do
    for b in FD Zygote Mooncake EnzymeR; do run $s $a A $b; done
    run $s $a B FD FD NM
    run $s $a B FD FD LBFGS
    for o in Zygote Mooncake; do run $s $a C $o FD FD FD; done
    run $s $a J Zygote
  done
  run $s host64 C FD FD FD FD
  run $s host32 C FD FD FD FD
  run $s dev32 C EnzymeR FD FD FD
done
# 3. A large surface: Heston/Cos, 4 maturities × 500 strikes
for a in host64 host32 dev64 dev32; do
  for b in FD Zygote; do run big $a A $b; done
  run big $a B FD FD NM
  run big $a C Zygote FD FD FD
  run big $a J Zygote
done
echo DONE >> results.txt
