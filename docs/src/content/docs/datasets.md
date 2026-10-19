---
title: "Datasets"
description: "Every dataset: what we have, what waits on a request, who has to ask or sign, and the licence catch"
group: benchmark
order: 3
---

Which datasets exist, which we hold, which wait on a request and who must send it, and each licence's catch. Read it before planning a run or sending a request.

Access checked on the live pages 2026-10-05; "In hand" as of 2026-10-12. Terms change, so recheck before depending on one. Source of truth: `configs/datasets/<id>.yaml`; `uv run scripts/setup_datasets.py --list` prints the current state. Licence clauses that bind us are in [data_policy.md](data_policy.md#licence-clauses-that-bind-us). To add a dataset, follow [adding_a_dataset.md](adding_a_dataset.md).

## At a glance

Rows hold real and fake unless marked. "In hand": a full copy is downloaded and has a manifest. "Sample": only the few items the pipeline check uses. Models are scored on evalsets built from these ([evaluation.md](evaluation.md#evalsets-not-datasets)).

| Dataset                 | Modality    | Content, size                                      | In hand    | To get it (who)                               | Licence, catch                            |
| ----------------------- | ----------- | -------------------------------------------------- | ---------- | --------------------------------------------- | ----------------------------------------- |
| Urdu (CSALT)            | audio       | Urdu, 2 GB                                         | yes        | open                                          | CC BY-NC 4.0                              |
| BanglaFake              | audio       | Bangla, 5.6 GB                                     | yes        | open                                          | none declared: ask before publishing      |
| Bangla voices           | audio       | Bangla, 75 speakers, paired, 0.5 GB                | yes        | open                                          | CC BY 4.0                                 |
| IndicTTS challenge      | audio       | 16 Indian languages incl. Nepali, 17.4 GB          | yes        | open                                          | CC BY 4.0                                 |
| IndicSynth              | audio       | 12 Indic languages, fake only, 128 GB for hi+bn+ur | Hindi only | open                                          | CC BY-NC 4.0                              |
| IndicSUPERB             | audio       | real only, the real side of IndicSynth, 5.8 GB     | yes        | open                                          | CC BY 4.0                                 |
| OpenSLR Sinhala         | audio       | Sinhala, real only, 0.9 GB (1 of 16 zips)          | yes        | open                                          | CC BY-SA 4.0                              |
| HiDF                    | video       | per-face race labels incl. "Indian", 1.6 GB        | yes        | open                                          | CC BY-NC 4.0                              |
| In-the-Wild             | audio       | English, reproduction, 8.2 GB                      | yes        | open                                          | CC BY-SA 4.0                              |
| ASVspoof 2019 LA        | audio       | English, reproduction, 7.6 GB                      | yes        | open (slow server)                            | DataShare licence text                    |
| BD-GRF6                 | audio       | Bangla by gender incl. third gender, 3 GB          | sample     | open                                          | CC BY 4.0                                 |
| SpeechFake              | audio       | Hindi, Bengali, Tamil, Marathi, Malayalam, 143 GB  | sample     | open (ModelScope)                             | CC BY 4.0                                 |
| LRLspoof                | audio       | fake only, Nepali among 66 languages, 462 GB       | deferred   | open, one tarball                             | MIT                                       |
| MAVOS-DD (Hindi)        | audio-video | Hindi, the only open South Asian video, 35.7 GB    | no         | accept Hugging Face terms (anyone)            | CC BY-NC-SA 4.0                           |
| MLAAD                   | audio       | fake only, 8 South Asian languages, 9.5 GB         | no         | accept Hugging Face terms (anyone)            | CC BY-NC 4.0                              |
| Svarah                  | audio       | real Indian-accented English, 1.1 GB               | no         | accept Hugging Face terms (anyone)            | none declared: nothing derived published  |
| SEA-Spoof               | audio       | Hindi and Tamil, 81.5 GB                           | no         | email authors, then a form (academic)         | non-commercial academic only              |
| InDeepFake              | audio-video | 7 Indian languages, 31 GB                          | no         | emailed form naming the PI                    | no sharing of data or derived data        |
| FakeAVCeleb             | audio-video | 100 real South Asian videos, size unknown          | no         | form, needs ethics approval                   | non-commercial; two versions of the terms |
| DeePhy                  | video       | Indian subjects, 30 GB                             | no         | signed by UCSC Director or Registrar          | assume DF-Platter's terms                 |
| DF-Platter              | video       | Indian subjects, 133,260 videos, 500 GB            | no         | signed by UCSC Director or Registrar          | signing institution only                  |
| Deepfake-Eval-2024      | audio-video | in the wild, about 5% South Asian, 20.3 GB         | no         | Hugging Face request (anyone), often rejected | evaluation only, no training              |
| Casual Conversations v2 | video       | real only, India the largest group, size unknown   | no         | form + Meta licence (anyone)                  | some labels evaluation only               |
| IndieFake               | audio       | Indian-accented English, size unknown              | no         | form (anyone)                                 | CC BY 4.0                                 |
| FF++                    | video       | reproduction, about 40 GB (c23)                    | no         | form (anyone)                                 | research only                             |
| Celeb-DF v2             | video       | reproduction, about 10 GB                          | no         | form (anyone)                                 | research only                             |

Dropped: IAV-DF (its only sample link is dead) and Indic-CodecFake (no download anywhere), both rechecked 2026-10-05.

Gaps worth knowing:

- **South Asian video:** only MAVOS-DD's Hindi subset needs no institutional request. Every other South Asian video set is Indian and needs one.
- **Sinhala:** the only deepfake audio is MLAAD's 1,000 Edge-TTS clips, paired with OpenSLR's real speech: cross-corpus, so diagnostic only (see below).
- **Nepali:** the only labelled real and fake is in the IndicTTS challenge set. LRLspoof has more Nepali fakes, but only inside one 462 GB tarball.

## Getting the open ones

```bash
uv run scripts/setup_datasets.py urdu_csalt banglafake bd_grf6 bangla_voices indictts_challenge hidf asvspoof2019_la in_the_wild openslr_sinhala indicsuperb
uv run scripts/setup_datasets.py indicsynth --languages hi bn ur
```

- **MLAAD, MAVOS-DD, Svarah:** accept the terms at <https://huggingface.co/datasets/mueller91/MLAAD>, <https://huggingface.co/datasets/unibuc-cs/MAVOS-DD> and <https://huggingface.co/datasets/ai4bharat/Svarah>, put `HF_TOKEN` in `.env`, then run `uv run scripts/setup_datasets.py mlaad` (or `mavos_dd_hi`). Svarah's builder is a stub until its schema can be read.
- **ASVspoof 2019 LA:** DataShare is slow (about 0.3 MB/s on 2026-10-05). Fetch with a resumable `curl -C -` into `datasets/asvspoof2019_la/LA.zip`; the setup script then checks and extracts it.
- **SpeechFake:** only samples are practical; a full copy is the whole 143 GB.
- **Try before downloading:** `--sample N` fetches N real and N fake items into the smoke namespace, kept apart and never reported ([adding_a_dataset.md](adding_a_dataset.md)). IndicSUPERB can only be sampled from a full copy.

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
- **DF-Platter and DeePhy:** read [data_policy.md](data_policy.md#sharing-across-institutions) first. The licence limits use to research colleagues at the signing institution, and the team spans more than one.

When a copy arrives, put it in the shared store and link it in:

```bash
uv run scripts/setup_datasets.py fakeavceleb --from /path/to/FakeAVCeleb_v1.2
```

## Notes per dataset

Only what changes how a number should be read; configs hold the rest.

- **Urdu (CSALT).** 6,794 WAV files of 17 speakers (3,398 real, 3,396 VITS and Tacotron fakes). The team Doc's 35k clips is wrong for this release.
- **BanglaFake.** 13,796 real and 12,796 VITS fakes, not the README's numbers. The evalset keeps the sust (9,999 real, 9,999 fake) and mozilla (2,797 pairs) sub-corpora, where every fake has a real twin, and drops news (1,000 real, no fakes). Mozilla is Common Voice, which XLS-R saw in pretraining, so those models get `source_overlap` ([evaluation.md](evaluation.md#contamination-verdicts)).
- **BD-GRF6.** 13,566 clips; no generator or speakers named. Third-gender real clips are MP3 and female fakes WAV, so file format may give the label away. Treated as cross-corpus until every class is checked on the full download.
- **IndicTTS challenge.** 31,102 labelled clips, one female and one male voice per language, generator unnamed. Its English rows include LJSpeech and LibriSpeech.
- **IndicSynth** (XTTS-v2, VITS, FreeVC24) was made from IndicSUPERB speakers, so its evalsets pair the two by speaker as same-corpus. Fakes outnumber the reals about 40 to 1 (Hindi: 205,938 against 5,059), so since 2026-10-12 the evalsets score a seeded sample of 5,000 per language and generator.
- **Cross-corpus pairs.** MLAAD (fake only) takes its real speech from IndicSUPERB, or OpenSLR Sinhala for `mlaad_si`; SpeechFake's Bengali, Tamil, Marathi and Malayalam do the same, and its Hindi fakes share no recordings with its Common Voice reals. A detector can score well on these by telling the recordings apart, so they wait on the [shortcut probe](evaluation.md#shortcut-probe-planned) and never feed headline numbers.
- **HiDF.** 4,361 real and 4,361 face-swapped videos of about 3 s, too short for HAVIC's 3.2 s windows; "Indian" covers 90 base videos. They are 512 px face crops, found only with the detector's padded retry ([evaluation.md](evaluation.md#preprocessing)).
- **MAVOS-DD (Hindi).** 3,632 real and 3,881 fake videos from eight generators (face swaps, audio-driven talking heads, reenactment, and knnvc voice conversion with the video untouched). Labels are per track: `mavos_dd_hi_audio` scores only real and knnvc items until the metadata says whether the other fakes keep their audio.
- **What the requests add.** FakeAVCeleb: South Asian against four other ethnic groups. DF-Platter and DeePhy: skin tone, gender and age labels (DF-Platter scored as a stratified sample). SEA-Spoof: ten named generators, commercial ones included. Casual Conversations v2 and Svarah: false-positive rates on real Indian faces and voices.

## Checked but not added

Searched 2026-10-05:

- **Fake only or unknown provenance:** `satwc-reddy/indian-language-deepfake-speech` (Telugu, Tamil, Malayalam, Konkani; worth adding with IndicSUPERB reals if Telugu or Konkani voice conversion matters) and a Nepali set with no README or licence.
- **Not downloadable:** MLADDC and IndicFake (TMLR 2025) aren't released; HAV-DF's Kaggle page is gone.
- **Real references:** FairFace ("Indian" faces), FLEURS (AntiDeepfake trained on it), IndicVoices-R (gated, about 1 TB).
- **No South Asian class:** AI-Face, DeepSpeak.

## Pipeline-test datasets

Two small open sets prove the code works and are never reported ([smoke_test.md](smoke_test.md)). Checked 2026-10-04.

| Dataset               | Source                                                                  | Licence         | Contents                                                       |
| --------------------- | ----------------------------------------------------------------------- | --------------- | -------------------------------------------------------------- |
| `mendeley_roop_akool` | <https://data.mendeley.com/datasets/pdcp9mjy3z/3>, one 4.1 GB zip       | CC BY 4.0       | face-crop frames (not videos), Bangladeshi and Indian subjects |
| `unidatapro_videos`   | <https://huggingface.co/datasets/UniDataPro/deepfake-videos-dataset>    | CC BY-NC-ND 4.0 | 5 real phone videos and 5 face-swap fakes made from them, with sound |
