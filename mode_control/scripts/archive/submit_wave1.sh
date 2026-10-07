#!/bin/bash
# D-48..D-53 wave 1 (2026-10-06): independent jobs only, ashton_low. Idempotent: vgen/extract resume.
set -euo pipefail
cd /ada1/u/harsh/thought-atlas
S="sbatch --qos=ashton_low --exclude=grace"
M=mode_control/scripts/mode.sbatch
# moral readout corpus
for s in a b; do $S -J vgen_corpus1_moral_$s --time=12:00:00 $M vgen corpus1_moral_$s; done
# quality judge on the idea corpus (H2a slopes)
for s in a b; do $S -J qual_corpus1_idea_$s --time=6:00:00 $M quality corpus1_idea_$s; done
# free-text grades for the math corpus (H2a slopes)
mkdir -p mode_control/data/extract_in3
for s in a b; do ln -sfn /ada1/u/harsh/thought-atlas/mode_control/data/runs/corpus1_math_$s/traces_baseline.parquet mode_control/data/extract_in3/traces_corpus1_math_${s}__baseline.parquet; done
$S -J freex3 --time=6:00:00 $M freexg mode_control/data/extract_in3 mode_control/data/extract_free3 0 1
# Countdown crossover (D-51)
for c in baseline ol_math_A2 conv_up conv_down ol_random; do $S -J vgen_cd_$c --time=16:00:00 $M vgen cd_countdown $c; done
for c in conv_up conv_down; do $S -J vgen_cdm_$c --time=16:00:00 $M vgen cd_math $c; done
# transplant arms that do not need the signature vectors (D-52, D-53)
$S -J vgen_txi_base --time=20:00:00 $M vgen tx_idea baseline,ol_random
$S -J vgen_txi_port --time=20:00:00 $M vgen tx_idea prompt_math,prompt_idea
$S -J vgen_txo_base --time=20:00:00 $M vgen tx_moral baseline,ol_random
$S -J vgen_txo_port --time=20:00:00 $M vgen tx_moral prompt_math,prompt_idea
$S -J vgen_txm_port --time=20:00:00 $M vgen tx_math prompt_math,prompt_idea
$S -J vgen_txg_port --time=20:00:00 $M vgen tx_gpqa prompt_math,prompt_idea
$S -J vgen_txg_f3 --time=20:00:00 $M vgen tx_gpqa sealdir_matched,rl_zeroT_s60
$S -J vgen_txp_port --time=20:00:00 $M vgen tx_planning prompt_math,prompt_idea
$S -J vgen_txp_f3 --time=20:00:00 $M vgen tx_planning sealdir_matched,rl_zeroT_s60
squeue -u harsh -o "%i %j %q %T %R" | grep -E "vgen|qual|freex" 
