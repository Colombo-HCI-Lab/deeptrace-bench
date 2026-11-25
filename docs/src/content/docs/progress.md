---
title: "Progress"
description: "Where the benchmark stands, what each round of work added, and what comes next"
---

## Where things stand

As of 2026-11-25:

- **Models.** 35 detectors have adapters and weights, and all 35 pass parity against upstream code ([Models](models.md#parity)). AltFreezing and TALL are blocked on weights ([Models](models.md#status)).
- **Data.** Builders cover every open South Asian set, the reproduction sets and HiDF. Request-only sets (FakeAVCeleb, InDeepFake, DeePhy, DF-Platter, FF++, Celeb-DF v2 and others) wait on requests ([datasets.md](datasets.md#requests-to-send)).
- **Runs.** All 21 video models with weights are scored on HiDF, MAVOS-DD Hindi and the MAVOS-DD German language control, the four audio-visual models on MAVOS-DD Hindi, and every audio model on its reproductions and on every South Asian and cross-corpus set the guard allows ([Results](results.md)). Every video model has its HiDF quick share and every audio-visual model its MAVOS-DD Hindi one. Of the 42 pairs of the 21 video models on the audio-visual MAVOS-DD sets, 28 are scored and the rest are ordered on SPARK (waves `s8*`); after them, the empty cells left wait on data access or weights. The shortcut probe separates every cross-corpus audio set by trivial features, so which of those numbers count waits on the team ([Results](results.md#the-shortcut-probe)). Every audio model has a settled score direction; no video model does yet ([Models](models.md#status)).
- **Compute.** The DESI lab's SPARK cluster runs the harness and scores any pair or a whole wave of them, ordered with one command; the same wave stage runs on any other GPU too ([Where to run](spark.md)). The two 24 GB GPUs outside SPARK that ran most of the round are switched off; SPARK scores the rest of it.
- **Rules.** The evaluation setup is still `proposed` (decision 5, below), so every number can still move.

## Timeline

### 2026-11-25: video models on the audio-visual sets, ten more

Wave `s8c_video_on_av`, run 45329263353 in [deeptrace-experiments](https://github.com/UNIL-DESI/deeptrace-experiments) (commit `5b418d6`, harness pinned at `ab7ebcb`).

- Xception, FFD, UCF, SRM and GenConViT on `mavos_dd_hi_av_quick`, and RECCE, SPSL, EfficientNet-B4, FakeSTormer and UnivFD on `mavos_dd_hi_av`, landed. All but GenConViT score below the same model on the matching `mavos_dd_hi_video` set, by 0.021 to 0.051 AUC (UnivFD 0.67 against 0.71); GenConViT is 0.537 on both, about chance either way.
- RealForensics on `mavos_dd_hi_av` slowed to 0.14 items/s and was interrupted at 6642 of 7513 items, and the DFDC winner on that set was skipped for budget. Both, with LipForensics' remainder, are ordered as wave `s8e` after `s8d`.

### 2026-11-24: video models on the audio-visual sets, nine more

Wave `s8b_video_on_av`, run 45118735814 in [deeptrace-experiments](https://github.com/UNIL-DESI/deeptrace-experiments) (commit `9a0debf`, harness pinned at `ab7ebcb`).

- SPSL, SBI, FTCN, FakeSTormer and Effort on `mavos_dd_hi_av_quick`, and Xception, UCF, GenD and SBI on `mavos_dd_hi_av`, landed. Each again scores below the same model on the matching `mavos_dd_hi_video` set, by 0.008 (Effort) to 0.063 (SBI) AUC. The models that do best on video lose the most: FTCN 0.77 against 0.83, GenD 0.69 against 0.74, SBI 0.71 against 0.78, since a detector that only looks at faces cannot catch the voice-only fakes the audio-visual labels add.
- LipForensics on `mavos_dd_hi_av` scored 3807 of 7513 items in 11.4 hours (0.09 items/s) before the job's budget ran out. Scoring resumes, so a later wave (`s8e`) finishes the rest.

### 2026-11-23: video models on the audio-visual sets, first nine

Waves `s8a2_video_on_av` and `s8a3_video_on_av`, runs 45033291487 and 45117560182 in [deeptrace-experiments](https://github.com/UNIL-DESI/deeptrace-experiments) (commits `f087992` and `e1ce372`, harness pinned at `ab7ebcb`).

- F3Net, CORE and RealForensics on `mavos_dd_hi_av_quick`, and CORE, F3Net, SRM, Effort, FFD and GenConViT on `mavos_dd_hi_av`, landed. Each scores below the same model on the matching `mavos_dd_hi_video` set: the DeepfakeBench models and GenConViT by 0.012 to 0.037 AUC (0.48 to 0.59 here), RealForensics 0.69 against 0.73 on the quick share. That is the expected direction when voice-only fakes count as fake.
- The first try, wave `s8a` (run 44913508610), queued 19.7 hours for SPARK's GPU worker and was cut by GitHub's 24-hour job limit 4 hours into scoring, before it could write a report. Its container on SPARK ran on and scored all nine pairs, so the two follow-up waves only collected them from the worker's store. Experiments commit `1063eac` now counts the queue in the job's time budget, so a long queue ends with the unreached pairs skipped and a report written.

### 2026-11-21: the last quick shares

Wave `s7b_quick_shares_rest`, run 44153757744 in [deeptrace-experiments](https://github.com/UNIL-DESI/deeptrace-experiments) (commit `584d33f`, harness pinned at `b16fb4b`).

- SPARK's GPU worker was back on 2026-11-20, so the fourth attempt of the rerun ran: FakeStormer on `hidf_video_quick`, and LipFD and AVAD on `mavos_dd_hi_av_quick`, landed.
- AVH-Align failed at once: it cuts its mouth crops with LipForensics' preprocessing code, and nothing cloned LipForensics' checkout unless a LipForensics run had already done so in the same container. Commit `a30d955` makes model setup clone a borrowed checkout too, and wave `s7c_avh_align_quick` (run 44857645283, harness pinned at `a30d955`) scored it.
- The video models on the audio-visual MAVOS-DD sets, the last 42 cells the data on hand allows, are ordered on SPARK. Their labels count a voice-only fake as fake, which a video model can't see, so these numbers sit beside the same models' `mavos_dd_hi_video` ones as a measure of what a frames-only detector misses on a platform that also gets voice fakes.

### 2026-11-18: the quick shares for the full-set models

Wave `s7_quick_shares`, run 44021999405 in [deeptrace-experiments](https://github.com/UNIL-DESI/deeptrace-experiments) (commit `0213fcd`, harness pinned at `b16fb4b`).

- Eight video models had been scored only on full HiDF, and three audio-visual models only on full MAVOS-DD Hindi, so their quick-subset cells in the Results grid were empty. This wave orders those eleven quick shares.
- Seven landed: GenConViT, RealForensics, FTCN, LAA-Net, Effort, UnivFD and DFDC B7 on `hidf_video_quick`.
- After eleven hours the worker's GPU dropped out ("CUDA unknown error" at CUDA init), so the last four failed within seconds of starting: FakeStormer on `hidf_video_quick`, and AVH-Align, LipFD and AVAD on `mavos_dd_hi_av_quick`. Their rerun, `s7b_quick_shares_rest` (run 44153757744, commit `584d33f`), could not start a GPU container on the same worker until it was fixed; it ran on 2026-11-20 (below).
- The video-only models on the audio-visual evalsets were left unordered on purpose; they were ordered on 2026-11-21 (above).

### 2026-11-15: one home for every number

Docs and site only; no new runs.

- Results has one grid: every countable run, quick subsets beside their full sets, with TPR at 5% FPR on hover. `results.md` no longer copies run numbers: it holds the findings and the breakdowns the grid can't show, grouped by experiment instead of by round.
- Coverage merged into the Results grid, which showed the same AUCs: the grid now has every detector against every evalset, with pairs waiting for data access or an adapter marked and the GPU time of each unscored pair on hover, under a tally of the pairs in each state. The old address forwards there.
- The site's data was rebuilt from the published runs, which added the last twelve language-control pairs and cleared the "no parity check" flag from the eight models that passed since 2026-11-11.
- Every doc states a fact once and links to it elsewhere. [Models](models.md) has one status table, one table of reference environments and the same fact table per model; [Datasets](datasets.md#at-a-glance) lists datasets by how they are obtained.

### 2026-11-13: every adapter passes parity

Commit `b16fb4b` here, and parity run 43495906113 in [deeptrace-experiments](https://github.com/UNIL-DESI/deeptrace-experiments).

- XLSR-Mamba (against mamba-ssm's own CUDA kernels on SPARK's RTX 3090), LipForensics and HAVIC pass, so all 35 adapters do ([Models](models.md#parity)).

### 2026-11-13: the language control finished, parity and the shortcut probe

Commits `d1c3034` to `3245781` here, and the `matrix`, `parity` and `shortcut` runs after 42492073467, up to 43470925926, in [deeptrace-experiments](https://github.com/UNIL-DESI/deeptrace-experiments).

- Waves `s6b_language_control_rest`, `s6c_language_control_hi` and `s6d_effort` scored the twelve pairs `s6_language_control` left, so all 21 video models with weights have both MAVOS-DD German and Hindi's quick share ([Results](results.md#the-language-control-mavos-dd-german)).
- Effort's checkpoint, which Google Drive refuses to scripts, now comes from a private Hugging Face mirror under Oshan's account that SPARK's token can read (`809479f`). Moving a weight's host no longer drops a model's earlier runs from the site: weights count by name in the config hash, and `weights.lock.yaml` still pins each file.
- AASIST3, XLS-R + SLS, GenD, GenD PE-L and Effort pass parity on SPARK ([Models](models.md#parity)), after a stage fix that mapped each model to its reference script and Effort's reference filling CLIP's `position_ids` buffer.
- The shortcut probe ran on all nine cross-corpus audio sets and two same-corpus baselines ([Results](results.md#the-shortcut-probe)); MLAAD Hindi hit Hugging Face's rate limit the first time and passed in a job of its own.

### 2026-11-09: the language control lands

Run 42492073467 in [deeptrace-experiments](https://github.com/UNIL-DESI/deeptrace-experiments), at `ed4bab6` here.

- Wave `s6_language_control` scored 17 video models on MAVOS-DD German's quick share in its 24 hours, after a 73-minute download of the German subset (11 tries against Hugging Face's rate limit) ([Results](results.md#the-language-control-mavos-dd-german)). Every model scores lower on German than on Hindi, so the drop from HiDF to MAVOS-DD is the set's, not Hindi's.
- Twelve pairs were left: FakeSTormer interrupted at the budget, the DFDC winner and LipForensics on German and the eight Hindi quick-share pairs skipped, and Effort's weights couldn't be fetched (Google Drive refuses gdown).

### 2026-11-08: the generator check, the shortcut probe and the language control

Commits `ed4bab6` to `e1bf16f` here.

- The generator-level contamination check of DF Arena and AntiDeepfake is written up ([Results](results.md#generator-level-contamination-check)). Few exact generator matches; voice-conversion systems found in no training corpus are caught as well as XTTS-v2, so familiarity alone doesn't explain their ceiling.
- The [shortcut probe](evaluation.md#shortcut-probe) is built, with a `shortcut` stage on SPARK; it waits for the worker.
- MAVOS-DD German joins as a language control for the video drop on Hindi, with a new `control` role; wave `s6_language_control` is scoring it on SPARK.
- Parity references for AASIST3, XLS-R + SLS, GenD, GenD PE-L and Effort are written, with their stacks in the SPARK `parity` stage; the job waits for the worker.

### 2026-11-08: LipFD scored, the new adapters done

Run 42455272481 in [deeptrace-experiments](https://github.com/UNIL-DESI/deeptrace-experiments), at `e10dfe3` here.

- Wave `s5c_lipfd` scored LipFD on all 7,513 MAVOS-DD Hindi videos in 6.4 hours, 0.32 a second, well inside its budget ([Results](results.md#faces-and-talking-heads-mavos-dd-hindi)). All eight pairs of the five adapters added on 2026-11-03 are now scored.

### 2026-11-08: seven of the eight new pairs scored

Run 42106621881 in [deeptrace-experiments](https://github.com/UNIL-DESI/deeptrace-experiments), at `e10dfe3` here.

- Wave `s5b_new_adapters_rest` scored FakeSTormer, LAA-Net and UnivFD on HiDF and LAA-Net and UnivFD on MAVOS-DD Hindi ([Results](results.md#faces-and-talking-heads-mavos-dd-hindi)). LAA-Net is the strongest of the five on HiDF; UnivFD, the generic AI-image detector, is the only one of the 21 models scored on both sets to do better on MAVOS-DD than on HiDF.
- LipFD ran out of budget at 4,811 of 7,513 videos, at 0.12 a second, and runs again in a wave of its own.

### 2026-11-06: the new adapters pass parity and start scoring

Commits `ee96d7e` onward here, and the `parity` and `matrix` runs from 41776938824 in [deeptrace-experiments](https://github.com/UNIL-DESI/deeptrace-experiments).

- LipFD, AVAD and LAA-Net pass parity on SPARK ([Models](models.md#parity)). It took three jobs: the references first failed on a wrong-shaped stand-in in AVAD's reference, a missing matplotlib in LAA-Net's stack, and the container's library path, which made each stack load the image's own torch.
- Wave `s5_new_adapters` ran out its 24 hours with two pairs done, AVAD and FakeSTormer on MAVOS-DD Hindi, AVAD taking 21 of them ([Results](results.md#faces-and-talking-heads-mavos-dd-hindi)). LipFD failed to load (`e10dfe3` fixes it), and the other six pairs went to `s5b_new_adapters_rest`, HiDF first and LipFD last.

### 2026-11-03: the last five adapters

Commits `acb58df` onward here, and the `parity` stage in [deeptrace-experiments](https://github.com/UNIL-DESI/deeptrace-experiments).

- UnivFD, LipFD, AVAD, LAA-Net and FakeSTormer now have adapters, the talking-head and lip-sync focus being the reason to want LipFD and AVAD most ([Models](models.md#per-model)). TALL and AltFreezing remain blocked on weights.
- UnivFD and FakeSTormer pass parity. A new `parity` stage checks the other three on SPARK, building each upstream stack on the worker. Its first job was running when this entry was written.
- Wave `s5_new_adapters` scores the eight new pairs once parity lands: all five on MAVOS-DD Hindi, the three frame and clip models on HiDF too.

### 2026-10-31: SpeechFake done, the audio round finished

Commits after `ea65acb` here, and `52303c4` to `1ffcd09` in [deeptrace-experiments](https://github.com/UNIL-DESI/deeptrace-experiments).

- `s3_speechfake` scored 49 of its 50 pairs in 20 hours, about 1.1 million clips and 4.6 kWh; the split-zip fetch held and deleted its parts ([Results](results.md#cross-corpus-audio)). It is cross-corpus, so none of it counts until the team decides after the shortcut probe.
- AASIST on Bengali lost the GPU at load in `s3_speechfake` and again in a wave of its own, both times as the first process of its job (`CUDA driver initialization failed`), and passed on the next try. The wave stage now retries once a score that dies within a minute.
- `s4_indicsynth_hi`, the last three pairs, ran the same day with SpeechFake's data cleared off the disk, and that ended the round: every runnable audio pair is scored.

### 2026-10-30: BD-GRF6 done, SpeechFake unblocked

Commits after `e6c05c0` here, and `db4650c` to `a97f3e6` in [deeptrace-experiments](https://github.com/UNIL-DESI/deeptrace-experiments).

- `s2_cross_rest` scored the last five BD-GRF6 pairs, so all ten audio models have it ([Results](results.md#cross-corpus-audio)). AASIST3 is the only model clear of chance there; the set is cross-corpus, so it waits on the shortcut probe.
- SpeechFake's 50 pairs didn't run. Its 116 GB of multilingual fakes downloaded in 10 hours, but they are one zip split in three parts, which Python's `zipfile` can't open, and the fetcher had only ever read them by range requests for samples. A full fetch now opens the parts with the same split-zip reader, extracts only what the builder labels (the 8 GB of South Asian clips, 93,741 of them) and deletes the parts. The parts' checksums are pinned in the dataset config.
- `s3_speechfake` (the 50 pairs) started the same evening from the parts already on the worker. IndicSynth Hindi's three pairs follow in a wave of their own, since its 115 GB can't share the disk with SpeechFake.

### 2026-10-30: the first SPARK wave lands

Commits after `2282567` here, and `acbc9f9` in [deeptrace-experiments](https://github.com/UNIL-DESI/deeptrace-experiments).

- `s1_audio_same_rest` scored 26 of its 29 pairs in 21 hours on SPARK's RTX 3090 (877K clips, 4.6 kWh). AASIST3, XLS-R + SLS and XLSR-Mamba now have their reproductions ([Reproductions](results.md#reproductions)) and five more South Asian sets each, and the 11 retries filled the gaps from earlier failures.
- The seven first-round models scored on IndicTTS again matched their 2026-10-12 runs within 0.001, a second cross-machine reproduction after Urdu CSALT.
- IndicSynth Hindi didn't fit: the worker's shared disk filled at 73 of its 115 GB, so its three pairs wait until `s2_cross_rest` frees the space. s2 (BD-GRF6 and SpeechFake, 55 pairs) started straight after.

### 2026-10-29: audio round paused, then moved to SPARK

Commits after `622bff9` here, and `c64c9f4` and `fb2e779` in [deeptrace-experiments](https://github.com/UNIL-DESI/deeptrace-experiments).

- IndicSynth Bengali and Urdu are scored for all ten audio models; the three models without parity split them, XLS-R + SLS best ([Results](results.md#audio-south-asian-speech)).
- The first cross-corpus wave scored MLAAD Bengali, Hindi and Sinhala and most of BD-GRF6. The guard refused 15 MLAAD pairs for the five models trained on it. These numbers wait on the shortcut probe.
- The round was paused with 84 audio pairs left: the three models without parity on the six reproduction sets, 11 retries of earlier failures, the rest of BD-GRF6 and all 50 SpeechFake pairs. A rerun of each wave skips what's done and resumes the interrupted pair from its saved parts. The two GPUs outside SPARK were switched off.
- The 84 pairs went to SPARK the same afternoon as two waves: `s1_audio_same_rest` (29 pairs: AASIST3, XLS-R + SLS and XLSR-Mamba on the six reproduction and South Asian sets, and the 11 retries) and `s2_cross_rest` (55: the rest of BD-GRF6, then all of SpeechFake). s1 started at once; s2 follows it.
- [Coverage](https://colombo-hci-lab.github.io/deeptrace-bench/coverage/) is now one grid: a column per model, a row per evalset, AUC in bands of 0.1. The site had been dropping 28 audio runs scored before the evalset config gained its `subset` field on 2026-10-27; unset, the field changes nothing a run measured, so they count again. Ten audio runs from 2026-10-12 (seven on IndicTTS, three reproductions) had dropped out the same way after a notes-only edit to their models' configs on 2026-10-19; a run now also counts while its model config differs only in descriptive fields, so they are back too, and every one of the 11 retries has a counted run.
- Every page of this site has one address. Results, the Timeline and How it works are now the only pages of their docs, and a section without a layout of its own shows there as prose, which put the 2026-10-28 round's write-up on Results. Every other doc moved from `/docs/<slug>/` to `/<slug>/`, and old addresses forward. The build now fails on a link to a missing page, to a forwarding address or to a missing anchor.

### 2026-10-29: five more video adapters scored

Commits `cefce3a` to `18a08d5`.

- FTCN, GenConViT (its ED branch), RealForensics, the DFDC winner's seven-model ensemble and AVH-Align have adapters, each matching upstream in upstream's own environment within 1e-5. AVH-Align runs on an AV-HuBERT rebuilt without fairseq.
- All five were scored on full HiDF and MAVOS-DD Hindi (AVH-Align on MAVOS-DD's audio and video). The temporal models hold up best on MAVOS-DD; the DFDC winner falls below chance there ([Results](results.md#faces-and-talking-heads-mavos-dd-hindi)).
- A power cut on 2026-10-28 stopped every machine mid-wave; runs resumed from their saved parts, so two RealForensics runs record only the resumed session's compute.
- `evaluate.py` now writes each result through a temporary file and a rename: over the workers' SMB mount, a shorter file copied over a longer one had kept the old last byte.

### 2026-10-28: the first full benchmark round

The waves and their ledgers are in [deeptrace-experiments](https://github.com/UNIL-DESI/deeptrace-experiments) (`waves/`); the runs are in `results/` and `results/ledger.csv` here.

- A `matrix` stage scores a whole wave of pairs, measuring each one (time per phase and per item, GPU memory and utilisation, energy), and runs on SPARK or any other GPU. Summaries now carry accuracy, balanced accuracy, average precision and TPR at 1% and 5% FPR.
- All 15 video models with weights, Effort among them, were scored on full HiDF and full MAVOS-DD Hindi. MAVOS-DD is far harder than HiDF for every one of them.
- Re-runs on other GPUs matched the 2026-10-12 round (Urdu CSALT audio, HiDF video), and Effort agreed with itself across two GPUs within 1e-4.
- FTCN, GenConViT and RealForensics passed parity and are queued on both video sets. The rest of the audio grid follows: IndicSynth Bengali and Urdu, MAVOS-DD Hindi audio, and the cross-corpus sets.

### 2026-10-28: a wider video roster

Config and docs only; no new runs.

- Eleven video and audio-visual models join the registry with weights confirmed reachable, none adapted yet ([Models](models.md#why-these-models)): FTCN, RealForensics, AVAD, AVH-Align, GenConViT, the DFDC winner, LipFD, LAA-Net, FakeSTormer, UnivFD, and TALL (blocked, no weights).
- Ten datasets join it ([Datasets](datasets.md#at-a-glance)). Four are South Asian: HAV-DF (Hindi), DeepFakeX (Urdu), ILLUSION (Indic languages) and INDIFACE, all on request. Counterfactual-Deepfake-Bench is open and has an explicit Indian cell. TalkingHeadBench, KoDF, Celeb-DF++, AV-Deepfake1M and DFDC are anchors and contrasts.
- Video comes before audio in the work queue from here on.

### 2026-10-19: any pair on SPARK, one command to order it

The changes are in [deeptrace-experiments](https://github.com/UNIL-DESI/deeptrace-experiments) (`abeac55` to `32d361c`); here, only [Where to run](spark.md) changed.

- SPARK's `score` stage now takes any pair. It reads an evalset's datasets from this repo's registry, and it still refuses to score before a download is proved complete. A reference run is optional.
- A helper, `scripts/exp.py`, places an order (it looks up the published reference itself), starts the run, waits for it and prints the report. It also lists every past run and repairs the reports a failed run leaves behind.
- Two test runs passed. AASIST-L on Urdu CSALT matched the 2026-10-12 round within 1e-3. XLS-R + SLS on Urdu CSALT ran with no reference; its number waits on parity.

### 2026-10-16: running on SPARK

Commits `6ae9dd1` to `5327e2f` here; the runs themselves are in [deeptrace-experiments](https://github.com/UNIL-DESI/deeptrace-experiments).

- Runs on the DESI lab's SPARK cluster go through `UNIL-DESI/deeptrace-experiments`, which pins a commit of this repo ([Where to run](spark.md)).
- On its x86 worker (an RTX 3090), the full test suite and the GenD and AASIST smoke tests passed in this repo's locked environment.
- The first scoring run there repeated AASIST on Urdu CSALT and matched the 2026-10-12 round: same manifest, same 6,794 items and 19 failures, AUC within 1.2e-5 and EER within 2.0e-4. The small EER shift fits the eval config's move from TF32 to fp32 on CUDA since then.
- Nine setup failures came before that, all fixed on our side; deeptrace-experiments' `CLAUDE.md` lists them.

### 2026-10-12: first zero-shot round

Commits `7928960` to `4ab78ad`, then the site (`429fb2d`, `7a79c22`).

- Four more adapters pass parity, making 17: XLS-R + AASIST (against upstream's torch 1.8.1 and vendored fairseq), both AntiDeepfake models (fairseq 0.12.2) and SBI.
- New reproductions of XLS-R + AASIST and both AntiDeepfake models, and the first zero-shot round: seven audio models on the five open South Asian sets, three video models on HiDF ([Results](results.md)).
- IndicSynth's evalsets switched to a seeded sample of 5,000 fakes per generator.
- The status page flags runs whose model lacks a parity or score-direction check. Docs and status page became one Astro site, published from `main` by GitHub Actions.

### 2026-10-05: every model in the kickoff Doc

Commits `bf2ae26` to `0424f48`, then the first status page (`d17c65c` to `4d77c1d`).

- 25 adapters: SBI, LipForensics, HAVIC, GenD's PE-L backbone, the XLS-R family (run without fairseq or mamba-ssm), AntiDeepfake, AASIST3 and four more DeepfakeBench detectors.
- New builders: IndicSynth with IndicSUPERB, the IndicTTS challenge set, BD-GRF6, Bangla voices, HiDF, SpeechFake.
- A pipeline check runs every working adapter on two real and two fake items of each open evalset: 195 pairs pass.
- RECCE, SRM, CORE and FFD pass parity.

### 2026-10-05: ten adapters and the first full runs

Commits `446714f` to `e524148`, then `d949b74` to `af30749`.

- AASIST, AASIST-L, DF Arena 500M and 1B, and five DeepfakeBench detectors, all passing parity.
- Full reproductions of AASIST and AASIST-L on ASVspoof 2019 LA eval and both DF Arena models on In-the-Wild ([Results](results.md#reproductions)).
- cuDNN's default TF32 convolutions broke GPU parity, so CUDA runs use full fp32 (`compute.tf32`).
- The first South Asian runs: AASIST and DF Arena on Urdu CSALT and BanglaFake.

### 2026-10-04: the pipeline runs end to end

Commits `1d0c0ab` to `30587a5`.

- The smoke namespace, dataset sampling, the shared SCRFD face pipeline, the GenD adapter and the smoke test ([Getting started](getting_started.md)).

### 2026-10-03: the scaffold

Commits `15a2e2c` and `d3a6d03`.

- Configs for 13 models and 16 datasets, checked live that day; setup scripts, the contamination guard, metrics and resumable scoring; a pre-commit hook that keeps data out of git.

## Next

In order:

1. **Settle the evaluation setup** (decision 5). The audio question now has a concrete case: average 4-second windows or score each utterance whole, which may explain MMS-300M's In-the-Wild gap ([Models](models.md#per-model)).
2. **FF++ or Celeb-DF v2**, to settle every video model's score direction.
3. **Generators they haven't seen.** Per-generator numbers on MLAAD Hindi's 2026 systems, once the team agrees the guard carve-out below, or on SEA-Spoof once its request is granted. The desk check is done ([Results](results.md#generator-level-contamination-check)).
4. **Decide which cross-corpus audio numbers count** (E5). The shortcut probe separates every cross-corpus set by trivial features; the options are in [Results](results.md#the-shortcut-probe).
5. **Fine-tuned baselines** (E6) on [identity-safe splits](evaluation.md#fine-tuning-splits). Waits on the team choosing the models and training data (Oshan, 2026-11-14); no training code exists yet.
6. **deeptrace's own data** (E7), once collection starts.

## Open decisions that shape the results

- **Decision 4**, the rule for keeping an attribute. As proposed it uses AUC only. On HiDF one group's AUC barely moves while its false-positive rate doubles ([Results](results.md#video-hidf)), so a false-positive-rate rule at a fixed threshold may be needed too ([the rule](evaluation.md#metrics)).
- **Decision 5**, the evaluation setup in [`configs/eval/default.yaml`](/configs/eval/default.yaml): due 2026-10-09, still open. The settings in question are in [Evaluation](evaluation.md#settings-still-open).
- **`group_gap`**, the headline fairness number, is still undefined ([evaluation.md](evaluation.md#metrics)).
- **A per-generator guard carve-out** (proposed 2026-11-08): let a model trained on MLAAD be scored on MLAAD systems released after its training (Voxtral, OmniVoice and VoxCPM2 in Hindi), as a diagnostic outside the headline numbers. The guard refuses the whole dataset today ([Results](results.md#generator-level-contamination-check)).
- **Cut candidates** (proposed 2026-10-26): AASIST-L, AASIST3, DF Arena 500M, EfficientNet-B4 and Deepfake-Eval-2024 add little the others don't show for South Asia. Each is marked in the [Model catalog](catalog.md#models) and [Dataset catalog](catalog.md#datasets) with its reason; the team decides whether to drop them.
