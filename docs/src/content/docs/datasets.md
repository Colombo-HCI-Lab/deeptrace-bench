---
title: "Datasets"
description: "Every dataset by how it is obtained: what one command fetches, what has a catch, which requests to send and who has to ask, and what changes how its numbers read"
---

Access checked on the live pages 2026-10-05. Terms change, so recheck before depending on one. The configs (`configs/datasets/<id>.yaml`) are the source of truth, and `uv run scripts/setup_datasets.py --list` prints the current state. What each dataset contains, its languages, size and licence, what we hold of it, why it was picked and whether it is a cut candidate are in the [Dataset catalog](catalog.md#datasets); licence clauses that bind us beyond the licence name are in [Data policy](data_policy.md#licence-clauses-that-bind-us). Models are scored on evalsets built from these ([Evaluation](evaluation.md#evalsets-not-datasets)).

## At a glance

| Access | Datasets | How |
| --- | --- | --- |
| Open, one command | Urdu (CSALT), BanglaFake, Bangla voices, IndicTTS challenge, IndicSynth, IndicSUPERB, OpenSLR Sinhala, HiDF, In-the-Wild, BD-GRF6 | [Getting the open ones](#getting-the-open-ones) |
| Open, with a catch | ASVspoof 2019 LA (slow server), SpeechFake (ModelScope, 125 GB at its peak), LRLspoof (one 462 GB tarball, deferred) | the table under [Getting the open ones](#getting-the-open-ones) |
| Open on Hugging Face, not fetched yet | Counterfactual-Deepfake-Bench, TalkingHeadBench | as the open ones |
| Accept the terms on Hugging Face (anyone) | MAVOS-DD, MLAAD, Svarah | the same table |
| On request (a person has to ask) | FF++, Celeb-DF v2, Celeb-DF++, KoDF, DFDC, AV-Deepfake1M, FakeAVCeleb, InDeepFake, DeePhy, DF-Platter, ILLUSION, INDIFACE, HAV-DF, DeepFakeX, Deepfake-Eval-2024, Casual Conversations v2, IndieFake, SEA-Spoof | [Requests to send](#requests-to-send) |
| Dropped (rechecked 2026-10-05) | IAV-DF (its only sample link is dead), Indic-CodecFake (no download anywhere) | |
| Pipeline tests | Mendeley Roop/Akool, UniDataPro | [Getting started](getting_started.md#the-pipeline-test-datasets) |

Gaps worth knowing:

- **South Asian video:** only MAVOS-DD's Hindi subset needs no request; it is MAVOS-DD's only South Asian language. Counterfactual-Deepfake-Bench is open but its "Indian" people are synthetic. Every other South Asian video set needs a request: HAV-DF (Hindi), DeepFakeX (Urdu, the only Urdu video found), ILLUSION (seven Indic languages in its real-world set) and the Indian sets. No Bangla, Sinhala, Nepali or Tamil face-video deepfake set was found (searched 2026-10-28).
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
| SpeechFake | the multilingual fakes are one 116 GB zip split in three parts. A full fetch downloads them, keeps only the South Asian 8 GB and deletes the parts, then adds Common Voice's 21 GB zip, so it needs about 125 GB free at its peak. The full set was fetched on SPARK for its 50 pairs and cleared on 2026-10-31. `--sample N` reads single clips by range requests instead |
| IndicSUPERB | can only be sampled from a full copy |
| OpenSLR Sinhala | one of its 16 zips is fetched |
| LRLspoof | one 462 GB tarball, so deferred |

To try a dataset before downloading it, `--sample N` fetches N real and N fake items into the smoke namespace ([Adding a dataset](adding_a_dataset.md#3-sample-it-then-build-the-manifest)).

## Requests to send

These need a person. Send them early: replies take days to months. On 2026-10-04 the FF++, InDeepFake and FakeAVCeleb forms and both IAB Rubric pages were live. On 2026-11-26 the FF++, Celeb-DF v2 and FakeAVCeleb forms were submitted and the signed InDeepFake form was emailed, all for the project (applicant Oshan, PI Dr. Dilrukshi Gamage).

| Dataset                 | What to do                                                                                                       | Who                                                         |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| FF++                    | [Google Form](https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform) (linked from <https://github.com/ondyari/FaceForensics>); they email a download script | any team member                                             |
| InDeepFake              | Fill in and sign the [access request form](https://github.com/user-attachments/files/21651935/InDeepFake_Access_Request_Form.pdf) (PDF linked from the repo README) and email it to ruchira@it.iiests.ac.in; academic institutions only | applicant, naming the PI                                    |
| FakeAVCeleb             | [Google Form](https://docs.google.com/forms/d/e/1FAIpQLSfPDd3oV0auqmmWEgCSaTEQ6CGpFeB-ozQJ35x-B_0Xjd93bw/viewform) (linked from <https://sites.google.com/view/fakeavcelebdash-lab/>); access comes as a Google Drive share | applicant with ethics approval or a faculty sponsor         |
| DeePhy, DF-Platter      | Sign the IAB Rubric licence and email it to databases@iab-rubric.org, one email per dataset                     | someone with legal authority: UCSC's Director or Registrar  |
| Deepfake-Eval-2024      | Request on its Hugging Face page with a link to prior deepfake-detection work                                    | any team member                                             |
| IndieFake               | Form on <https://indie-fake-dataset.netlify.app/>                                                                | any team member                                             |
| Celeb-DF v2             | [Google Form](https://docs.google.com/forms/d/e/1FAIpQLScoXint8ndZXyJi2Rcy4MvDHkkZLyBFKN43lTeyiG88wrG0rA/viewform) (linked from <https://github.com/yuezunli/celeb-deepfakeforensics>) | any team member                                             |
| Casual Conversations v2 | Request on <https://ai.meta.com/datasets/casual-conversations-v2-dataset/> and accept Meta's licence             | any team member                                             |
| HAV-DF                  | Email the BML Munjal University authors (addresses in the paper, arXiv 2411.15457); ask for the release behind the splits later papers report | any team member |
| DeepFakeX               | Request access on <https://zenodo.org/records/14202454> with a research purpose                                  | any team member                                             |
| ILLUSION                | Signed licence to IAB Rubric, the same lab as DeePhy and DF-Platter; ask whether Set D (real-world, Indic languages) can come alone | someone with legal authority, as for DeePhy |
| INDIFACE                | Email the authors (Kuckreja, Hoque, Dhall, Das; FG 2024); no hosting link found                                  | any team member                                             |
| KoDF                    | Request form linked from <https://moneybrain-research.github.io/kodf/>                                           | any team member                                             |
| Celeb-DF++              | Request form linked from <https://github.com/OUC-VAS/Celeb-DF-PP>                                                | any team member                                             |
| AV-Deepfake1M           | Register on the challenge site and email the signed EULA (<https://github.com/ControlNet/AV-Deepfake1M>)          | any team member                                             |
| DFDC                    | Download from AWS (needs an AWS account) or from Kaggle (accept the competition rules)                           | any team member                                             |
| SEA-Spoof               | Email the authors (wu_jinyang@a-star.edu.sg, imcc.sg@gmail.com), then submit the form on <https://huggingface.co/datasets/Jack-ppkdczgx/SEA-Spoof> | an applicant at an academic institution |

- **FakeAVCeleb:** requests without ethics (IRB) approval are rejected; complete ones got access within days, says a maintainer (GitHub issue 13, 2024-12-18), though others report months. The form (checked 2026-11-26) has no ethics field, so state the approval, or the faculty sponsor, in the purpose. It asks for an official email, the supervisor's name and email, a Gmail address to share the Google Drive folder with, and a purpose of at least 100 words. The Drive access expires after one week, so download as soon as it arrives. Don't use the unofficial Hugging Face mirrors.
- **FF++:** an issue from 2025-10-05 said its download server was down. It answered again on 2026-10-04, but no download was tested. The form asks for an email, name, affiliation, the supervisor's name and email, and a purpose, and the applicant ticks agreement to the terms of use.
- **Celeb-DF v2:** the form has no email field and records the Google account it is filled in from, so sign in with the account that should get the reply, or put the academic email in the purpose. It also asks for city, country, the dataset version and a typed signature. Submitting it shows the Google Drive and Baidu download links at once, with no review; keep those links out of this repo. The Drive file often hits Google's "too many users" download limit; downloading it through a signed-in account (`rclone backend copyid` with the file id) still worked. The project's copy arrived on 2026-11-26; SPARK fetches it from a private Hugging Face mirror with the project's `HF_TOKEN`.
- **InDeepFake:** the PDF moved from the repo to a GitHub attachment link (checked 2026-11-26). It asks for the applicant's and the PI's names and emails, the affiliation, a purpose, a ticked agreement, a date and a signature.
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
| IndicSynth | Made from IndicSUPERB speakers (with XTTS-v2, VITS and FreeVC24), so its evalsets pair the two by speaker as same-corpus. Scored as a seeded sample of fakes ([Results](results.md#the-datasets)) |
| MLAAD, SpeechFake | MLAAD is fake only and SpeechFake has no real speech in four of its five languages, so their evalsets take real speech from another corpus; SpeechFake's Hindi fakes share no recordings with its Common Voice reals. All are cross-corpus ([Results](results.md#the-datasets)), and the [shortcut probe](results.md#the-shortcut-probe) shows what that costs |
| HiDF | Clips of about 3 s; "Indian" covers 90 base videos. They are 512 px face crops, found only with the detector's padded retry ([Evaluation](evaluation.md#preprocessing)) |
| MAVOS-DD (Hindi) | Labels are per track, so its three evalsets count the same clips differently ([Results](results.md#the-datasets)) |
| The requests | FakeAVCeleb: South Asian against four other ethnic groups. DF-Platter and DeePhy: skin tone, gender and age labels (DF-Platter scored as a stratified sample). SEA-Spoof: ten named generators, commercial ones included. Casual Conversations v2 and Svarah: false-positive rates on real Indian faces and voices |

## Checked but not added

Searched 2026-10-05:

| Why not | Datasets |
| --- | --- |
| Fake only or unknown provenance | `satwc-reddy/indian-language-deepfake-speech` (Telugu, Tamil, Malayalam, Konkani; worth adding with IndicSUPERB reals if Telugu or Konkani voice conversion matters); a Nepali set with no README or licence |
| Not downloadable | MLADDC and IndicFake (TMLR 2025) aren't released; HAV-DF's Kaggle page is gone (it is now on request, above) |
| Real references only | FairFace ("Indian" faces), FLEURS (AntiDeepfake trained on it), IndicVoices-R (gated, about 1 TB) |
| No South Asian class | AI-Face, DeepSpeak (v2 is US residents only); PolyGlotFake (no Indic language) |
| Images only, or no demographics | FairFD and RWFS (images); DF40, WildDeepfake and RedFace (no demographic labels), searched 2026-10-28 |
