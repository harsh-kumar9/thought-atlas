#!/bin/bash
# Submit the equal-dose order-swap run (sched1) as 8 single-GPU jobs. Run from the repo root on ada.
# J0 generates the unsteered baseline (the schedule arms need its lengths) and judges it; the sustained arms
# start at once; the scheduled arms wait for J0. Each job steers one condition and judges only that condition.
set -euo pipefail
RUN=${RUN:-sched1}; L=19; FIX=${FIX:-1}; RND=${RND:-1}; SUS=${SUS:-1}   # FIX=0 skips the fixed-switch arms; RND=0 the random arm; SUS=0 the sustained arms
SB="sbatch --exclude=grace --time=20:00:00"; SC=mode_control/scripts/mode.sbatch
j0=$($SB --job-name=${RUN}-base $SC pilot $RUN schedule baseline | awk '{print $NF}'); echo "baseline: $j0"
ids=$j0
SUSARMS="sus_idea_L$L sus_math_L$L"; [ "$SUS" = 1 ] || SUSARMS=""
for c in $SUSARMS; do
  j=$($SB --job-name=${RUN}-${c%_L$L} $SC pilot $RUN schedule $c | awk '{print $NF}'); ids=$ids:$j
done
SCHED="idea_then_math_L$L math_then_idea_L$L"; [ "$RND" = 1 ] && SCHED="$SCHED random_then_opp_L$L"; [ "$FIX" = 1 ] && SCHED="$SCHED idea_then_math_fix_L$L math_then_idea_fix_L$L"
for c in $SCHED; do
  j=$($SB --dependency=afterok:$j0 --job-name=${RUN}-${c%_L$L} $SC pilot $RUN schedule $c | awk '{print $NF}'); ids=$ids:$j
done
# final answer-quality judge (Idea originality/feasibility) and a clean analysis pass once every arm is judged
$SB --dependency=afterok:$ids --job-name=${RUN}-quality $SC quality $RUN
squeue -u "$USER" -h -o "%i %j %T %r"
