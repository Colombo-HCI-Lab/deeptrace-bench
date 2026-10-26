---
title: "Datasets"
description: "Every dataset: what we hold, how to get the rest and who has to ask, and what changes how its numbers read"
---

Access checked on the live pages 2026-10-05; "In hand" as of 2026-10-12. Terms change, so recheck before depending on one. The configs (`configs/datasets/<id>.yaml`) are the source of truth, and `uv run scripts/setup_datasets.py --list` prints the current state. Contents, languages and sizes are in the [Dataset catalog](catalog.md#datasets); licence clauses that bind us beyond the licence name are in [Data policy](data_policy.md#licence-clauses-that-bind-us). Why each dataset was picked, and whether it is a cut candidate, is on its page in the [Dataset catalog](catalog.md#datasets). Models are scored on evalsets built from these ([Evaluation](evaluation.md#evalsets-not-datasets)).

## At a glance

"In hand": a full copy is downloaded and has a manifest. "Sample": only the few items the pipeline check uses.

| Dataset                 | Modality    | Role (config) | In hand    | To get it (who)                               | Licence                 |
| ----------------------- | ----------- | ------------ | ---------- | --------------------------------------------- | ----------------------- |
| Urdu (CSALT)            | audio       | South Asian  | yes        | open                                          | CC BY-NC 4.0            |
| BanglaFake              | audio       | South Asian  | yes        | open                                          | none declared           |
| Bangla voices           | audio       | South Asian  | yes        | open                                          | CC BY 4.0               |
| IndicTTS challenge      | audio       | South Asian  | yes        | open                                          | CC BY 4.0               |
| IndicSynth              | audio       | South Asian  | Hindi only | open                                          | CC BY-NC 4.0            |
| IndicSUPERB             | audio       | real reference | yes      | open                                          | CC BY 4.0               |
| OpenSLR Sinhala         | audio       | real reference | yes (1 of 16 zips) | open                                  | CC BY-SA 4.0            |
| HiDF                    | video       | South Asian (race labels) | yes | open                                          | CC BY-NC 4.0            |
| In-the-Wild             | audio       | reproduction | yes        | open                                          | CC BY-SA 4.0            |
| ASVspoof 2019 LA        | audio       | reproduction | yes        | open (slow server)                            | DataShare licence text  |
| BD-GRF6                 | audio       | South Asian  | sample     | open                                          | CC BY 4.0               |
| SpeechFake              | audio       | South Asian  | sample     | open (ModelScope)                             | CC BY 4.0               |
| LRLspoof                | audio       | South Asian  | deferred   | open, one 462 GB tarball                      | MIT                     |
| MAVOS-DD (Hindi)        | audio-video | South Asian  | no         | accept Hugging Face terms (anyone)            | CC BY-NC-SA 4.0         |
| MLAAD                   | audio       | South Asian  | no         | accept Hugging Face terms (anyone)            | CC BY-NC 4.0            |
| Svarah                  | audio       | real reference | no       | accept Hugging Face terms (anyone)            | none declared           |
| SEA-Spoof               | audio       | South Asian  | no         | email authors, then a form (academic)         | non-commercial academic |
| InDeepFake              | audio-video | South Asian  | no         | emailed form naming the PI                    | custom                  |
| FakeAVCeleb             | audio-video | South Asian  | no         | form, needs ethics approval                   | custom, non-commercial  |
| DeePhy                  | video       | South Asian  | no         | signed by UCSC Director or Registrar          | custom                  |
| DF-Platter              | video       | South Asian  | no         | signed by UCSC Director or Registrar          | custom                  |
| Deepfake-Eval-2024      | audio-video | in the wild  | no         | Hugging Face request (anyone), often rejected | evaluation only         |
| Casual Conversations v2 | video       | real reference | no       | form + Meta licence (anyone)                  | custom                  |
| IndieFake               | audio       | South Asian  | no         | form (anyone)                                 | CC BY 4.0               |
| FF++                    | video       | reproduction | no         | form (anyone)                                 | research only           |
| Celeb-DF v2             | video       | reproduction | no         | form (anyone)                                 | research only           |

Dropped: IAV-DF (its only sample link is dead) and Indic-CodecFake (no download anywhere), both rechecked 2026-10-05. The two pipeline-test datasets are in [Getting started](getting_started.md#the-pipeline-test-datasets).

Gaps worth knowing:

- **South Asian video:** only MAVOS-DD's Hindi subset needs no institutional request. Every other South Asian video set is Indian and needs one.
- **Sinhala:** the only deepfake audio is MLAAD's 1,000 Edge-TTS clips, paired with OpenSLR's real speech: cross-corpus, so diagnostic only.
- **Nepali:** the only labelled real and fake is in the IndicTTS challenge set. LRLspoof has more Nepali fakes, but only inside its 462 GB tarball.

## Getting the open ones

```bash
uv run scripts/setup_datasets.py urdu_csalt banglafake bd_grf6 bangla_voices indictts_challenge hidf asvspoof2019_la in_the_wild openslr_sinhala indicsuperb
uv run scripts/setup_datasets.py indicsynth --languages hi bn ur
```

| Dataset | Catch |
| --- | --- |
| MLAAD, MAVOS-DD, Svarah | accept the terms at <https://huggingface.co/datasets/mueller91/MLAAD>, <https://huggingface.co/datasets/unibuc-cs/MAVOS-DD> and <https://huggingface.co/datasets/ai4bharat/Svarah>, put `HF_TOKEN` in `.env`, then run `setup_datasets.py mlaad` (or `mavos_dd_hi`). Svarah's builder is a stub until its schema can be read |
| ASVspoof 2019 LA | DataShare is slow (about 0.3 MB/s on 2026-10-05). Fetch with a resumable `curl -C -` into `datasets/asvspoof2019_la/LA.zip`; the setup script then checks and extracts it |
| SpeechFake | only samples are practical; a full copy is the whole 143 GB |
| IndicSUPERB | can only be sampled from a full copy |

To try a dataset before downloading it, `--sample N` fetches N real and N fake items into the smoke namespace ([Adding a dataset](adding_a_dataset.md#3-sample-it-then-build-the-manifest)).

## Requests to send

These need a person. Send them early: replies take days to months. On 2026-10-04 the FF++, InDeepFake and FakeAVCeleb forms and both IAB Rubric pages were live.

| Dataset                 | What to do                                                                                                       | Who                                                         |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| FF++                    | Google Form linked from <https://github.com/ondyari/FaceForensics>; they email a download script                 | any team member                                             |
| InDeepFake              | Fill in the access request form (PDF in the repo README) and email it; academic institutions only               | applicant, naming the PI                                    |
| FakeAVCeleb             | Request form on <https://sites.google.com/view/fakeavcelebdash-lab/>                                             | applicant with ethics approval or a faculty sponsor         |
| DeePhy, DF-Platter      | Sign the IAB Rubric licence and email it to databases@iab-rubric.org, one email per dataset                     | someone with legal authority: UCSC's Director or Registrar  |
| Deepfake-Eval-2024      | Request on its Hugging Face page with a link to prior deepfake-detection work                                    | any team member                                             |
| IndieFake               | Form on <https://indie-fake-dataset.netlify.app/>                                                                | any team member                                             |
| Celeb-DF v2             | Request form linked from <https://github.com/yuezunli/celeb-deepfakeforensics>                                   | any team member                                             |
| Casual Conversations v2 | Request on <https://ai.meta.com/datasets/casual-conversations-v2-dataset/> and accept Meta's licence             | any team member                                             |
| SEA-Spoof               | Email the authors (wu_jinyang@a-star.edu.sg, imcc.sg@gmail.com), then submit the form on <https://huggingface.co/datasets/Jack-ppkdczgx/SEA-Spoof> | an applicant at an academic institution |

- **FakeAVCeleb:** requests without ethics (IRB) approval are rejected; complete ones got access within days, says a maintainer (GitHub issue 13, 2024-12-18), though others report months. Don't use the unofficial Hugging Face mirrors.
- **FF++:** an issue from 2025-10-05 said its download server was down. It answered again on 2026-10-04, but no download was tested.
- **DF-Platter and DeePhy:** read [Data policy](data_policy.md#sharing-across-institutions) first: the licence covers only the signing institution, and the team spans more than one.

When a copy arrives, put it in the shared store and link it in:

```bash
uv run scripts/setup_datasets.py fakeavceleb --from /path/to/FakeAVCeleb_v1.2
```

## Notes per dataset

Only what changes how a number should be read; configs hold the rest, and [Results](results.md#the-datasets) gives the real and fake counts of every scored set.

| Dataset | Note |
| --- | --- |
| Urdu (CSALT) | 17 speakers. The team Doc's 35k clips is wrong for this release |
| BanglaFake | Counts differ from the README's. The evalset keeps the sust and mozilla sub-corpora, where every fake has a real twin, and drops news (1,000 real, no fakes). Mozilla is Common Voice, which XLS-R saw in pretraining, so those models get `source_overlap` |
| BD-GRF6 | 13,566 clips; no generator or speakers named. Third-gender real clips are MP3 and female fakes WAV, so file format may give the label away. Treated as cross-corpus until every class is checked on the full download |
| IndicTTS challenge | One female and one male voice per language, generator unnamed. Its English rows include LJSpeech and LibriSpeech |
| IndicSynth | Made from IndicSUPERB speakers (with XTTS-v2, VITS and FreeVC24), so its evalsets pair the two by speaker as same-corpus. Scored as a seeded sample of fakes since 2026-10-12 |
| Cross-corpus pairs | MLAAD (fake only) takes its real speech from IndicSUPERB, or OpenSLR Sinhala for `mlaad_si`; SpeechFake's Bengali, Tamil, Marathi and Malayalam do the same, and its Hindi fakes share no recordings with its Common Voice reals. They wait on the [shortcut probe](evaluation.md#shortcut-probe-planned) |
| HiDF | Clips of about 3 s, too short for HAVIC's 3.2 s windows; "Indian" covers 90 base videos. They are 512 px face crops, found only with the detector's padded retry ([Evaluation](evaluation.md#preprocessing)) |
| MAVOS-DD (Hindi) | 3,632 real and 3,881 fake videos from eight generators (face swaps, audio-driven talking heads, reenactment, and knnvc voice conversion with the video untouched). Labels are per track: `mavos_dd_hi_audio` scores only real and knnvc items until the metadata says whether the other fakes keep their audio |
| The requests | FakeAVCeleb: South Asian against four other ethnic groups. DF-Platter and DeePhy: skin tone, gender and age labels (DF-Platter scored as a stratified sample). SEA-Spoof: ten named generators, commercial ones included. Casual Conversations v2 and Svarah: false-positive rates on real Indian faces and voices |

## Checked but not added

Searched 2026-10-05:

| Why not | Datasets |
| --- | --- |
| Fake only or unknown provenance | `satwc-reddy/indian-language-deepfake-speech` (Telugu, Tamil, Malayalam, Konkani; worth adding with IndicSUPERB reals if Telugu or Konkani voice conversion matters); a Nepali set with no README or licence |
| Not downloadable | MLADDC and IndicFake (TMLR 2025) aren't released; HAV-DF's Kaggle page is gone |
| Real references only | FairFace ("Indian" faces), FLEURS (AntiDeepfake trained on it), IndicVoices-R (gated, about 1 TB) |
| No South Asian class | AI-Face, DeepSpeak |
