# Datasets

Access checked against the live pages on 2026-10-03; datasets added on 2026-10-05 were checked that day. The source of truth is `configs/datasets/<id>.yaml`; `uv run scripts/setup_datasets.py --list` prints the current table. Access terms change, so recheck a page before depending on it.

## At a glance

| Dataset            | Modality    | Access today                          | Real + fake | Size                 | Use                                      |
| ------------------ | ----------- | ------------------------------------- | ----------- | -------------------- | ---------------------------------------- |
| Urdu (CSALT)       | audio       | open (Hugging Face)                   | both        | 2 GB                 | first South Asian run                    |
| BanglaFake         | audio       | open (Hugging Face)                   | both        | 5.6 GB               | first; licence undeclared                |
| In-the-Wild        | audio       | open (Hugging Face)                   | both        | 8.2 GB               | reproduction for DF Arena; English       |
| OpenSLR Sinhala    | audio       | open (OpenSLR SLR52)                  | real only   | 0.9 GB (1 of 16 zips) | real side of MLAAD's Sinhala             |
| MAVOS-DD (Hindi)   | audio-video | accept terms on Hugging Face          | both        | 35.7 GB              | the only open South Asian video          |
| IndicSynth         | audio       | open, per language                    | fake only   | 128 GB (hi+bn+ur)    | paired with IndicSUPERB                  |
| IndicSUPERB        | audio       | open (object store)                   | real only   | 4 GB (test splits)   | real side of IndicSynth                  |
| MLAAD              | audio       | accept terms on Hugging Face, instant | fake only   | 9.5 GB (South Asian) | needs a real source; cross-corpus        |
| ASVspoof 2019 LA   | audio       | open (Edinburgh DataShare)            | both        | 7.6 GB               | reproduction, score direction            |
| FF++               | video       | request form                          | both        | ~40 GB (c23)         | reproduction                             |
| InDeepFake         | audio-video | emailed form naming the PI            | both        | 31 GB                | request first: lightest                  |
| FakeAVCeleb        | audio-video | form + ethics approval, slow          | both        | ?                    | request now; 100 real South Asian videos |
| DeePhy             | video       | institution-signed licence            | both        | 30 GB                | one signing round with DF-Platter        |
| DF-Platter         | video       | institution-signed licence            | both        | 500 GB               | stratified sample                        |
| Deepfake-Eval-2024 | mixed       | manual approval, often rejected       | both        | 20 GB                | evaluation only; ~5% South Asian         |
| Celeb-DF v2        | video       | request form                          | both        | ~10 GB               | reproduction for video models            |
| Casual Conversations v2 | video  | form + Meta licence                   | real only   | ?                    | false positives on real Indian faces     |
| IndieFake          | audio       | request form                          | both        | ?                    | low priority: English                    |
| SpeechFake         | audio       | open, ModelScope                      | fake        | 116 GB for Hindi     | deferred                                 |
| IAV-DF             | audio-video | sample link dead                      |             |                      | dropped                                  |
| Indic-CodecFake    | audio       | no download anywhere                  |             |                      | dropped                                  |

The only South Asian video that needs no institutional request is MAVOS-DD's Hindi subset (accept its terms on Hugging Face); every other South Asian video set is Indian and needs a request. The only Sinhala deepfake audio is MLAAD's 1,000 Edge-TTS clips, now paired with real Sinhala speech from OpenSLR (cross-corpus, so diagnostic only). The only Nepali is in LRLspoof (spoof only, one 452 GB tarball, not configured).

## Getting the open ones

```bash
uv run scripts/setup_datasets.py urdu_csalt banglafake asvspoof2019_la in_the_wild openslr_sinhala indicsuperb
uv run scripts/setup_datasets.py indicsynth --languages hi bn ur
```

MLAAD and MAVOS-DD: accept their terms at <https://huggingface.co/datasets/mueller91/MLAAD> and <https://huggingface.co/datasets/unibuc-cs/MAVOS-DD> with your Hugging Face account, put `HF_TOKEN` in `.env`, then `uv run scripts/setup_datasets.py mlaad` (or `mavos_dd_hi`). ASVspoof 2019 LA comes from Edinburgh DataShare, which is slow (about 0.3 MB/s on 2026-10-05, so hours for 7.6 GB); a resumable `curl -C -` into `datasets/asvspoof2019_la/LA.zip` survives interruptions, and the setup script then checks and extracts it.

To try a dataset before downloading all of it, `--sample N` fetches N real and N fake items into the smoke namespace (see [adding_a_dataset.md](adding_a_dataset.md)); it needs the dataset's builder to be written.

## Requests to send

These need a person. Send them early: replies take days to months. Rechecked 2026-10-04: the FF++, InDeepFake and FakeAVCeleb request forms and both IAB Rubric pages are live. A FakeAVCeleb maintainer wrote (GitHub issue 13, 2024-12-18) that requests missing IRB approval are rejected and complete ones get access within days; other applicants report waiting months. An FF++ issue from 2025-10-05 says its download server was down; it answered again on 2026-10-04, but a download wasn't tested.

| Dataset            | What to do                                                                                                          | Who                                                        |
| ------------------ | ------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| FF++               | Google Form linked from <https://github.com/ondyari/FaceForensics>; they email a download script                    | any team member                                            |
| InDeepFake         | Fill in the access request form (PDF in the repo README) and email it                                               | applicant, naming the PI                                   |
| FakeAVCeleb        | Request form on <https://sites.google.com/view/fakeavcelebdash-lab/>; requests without ethics approval are rejected | applicant with ethics approval or a faculty sponsor        |
| DeePhy, DF-Platter | Sign the IAB Rubric licence and email it to databases@iab-rubric.org, one email per dataset                         | someone with legal authority: UCSC's Director or Registrar |
| Deepfake-Eval-2024 | Request on its Hugging Face page with a link to prior deepfake-detection work                                       | any team member                                            |
| IndieFake          | Form on <https://indie-fake-dataset.netlify.app/>                                                                   | any team member                                            |
| Celeb-DF v2        | Request form linked from <https://github.com/yuezunli/celeb-deepfakeforensics>                                      | any team member                                            |
| Casual Conversations v2 | Request on <https://ai.meta.com/datasets/casual-conversations-v2-dataset/> and accept Meta's licence           | any team member                                            |

When a copy arrives, put it in the shared store and link it in:

```bash
uv run scripts/setup_datasets.py fakeavceleb --from /path/to/FakeAVCeleb_v1.2
```

**Before requesting DF-Platter**, read [data_policy.md](data_policy.md): its licence lets only research colleagues at the signing institution use it, and the team spans more than one institution.

## Notes per dataset

- **Urdu (CSALT).** `Bonafide/`, `Spoofed_TTS/` (VITS) and `Spoofed_Tacotron/`, 17 speakers each, 6,794 WAV files. Our Doc says 35k clips; check the count on download.
- **BanglaFake.** One `final_data.zip` with three sub-corpora (checked 2026-10-05): sust (9,999 real, 9,999 fake), mozilla (2,797 real-fake pairs of five Common Voice speakers) and news (1,000 real, no fakes). 13,796 real and 12,796 VITS fakes, not the README's numbers. The evalset keeps sust and mozilla, where every fake has a real counterpart. Common Voice makes XLS-R-based models `source_overlap` on it. No official split, no declared licence.
- **In-the-Wild.** 31,779 clips of 54 public figures (19,963 bona fide, 11,816 spoof), labels in `meta.csv`. Not South Asian: the reproduction set for models the guard refuses on ASVspoof 2019 LA eval, such as DF Arena.
- **OpenSLR Sinhala (SLR52).** Crowd-sourced read Sinhala, 185k utterances in 16 zips; only the first (11,550 clips, anonymised speaker ids) is fetched. The real side of `mlaad_si`.
- **MAVOS-DD (Hindi).** 3,632 real and 3,881 fake videos from eight generators: face swaps (inswapper, hififace, roop), audio-driven talking heads (echomimic, sonic, memo), reenactment (liveportrait) and voice conversion (knnvc, video untouched). Per-track labels drive two evalsets, `mavos_dd_hi_video` and `mavos_dd_hi_audio`; the audio of the non-VC fakes stays unlabelled until the dataset's metadata can be read. No speaker or demographic labels.
- **Celeb-DF v2.** 890 real and 5,639 fake videos of 59 celebrities, mostly Caucasian; for reproduction only.
- **Casual Conversations v2.** Real only: 5,567 consenting participants in seven countries, India the largest, with Fitzpatrick and Monk skin tone, age, gender and language labels.
- **IndicSynth.** Parquet shards per language with audio embedded; generators XTTS-v2, VITS and FreeVC24, with speaker and gender columns. Fake only; generated from IndicSUPERB speakers.
- **IndicSUPERB (Kathbath).** m4a audio in tar archives; converted to 16 kHz WAV at manifest time. Licence not yet checked.
- **MLAAD.** `fake/<lang>/<generator>/` with a `meta.csv` each, 1,000 clips per folder. Hindi (9 generators), Bengali (4), Kannada (2), and one generator each for Tamil, Marathi, Urdu, Sinhala and Malayalam. Its real counterpart, M-AILABS, has no South Asian language.
- **FakeAVCeleb.** 500 real and 19,500 fake videos over five ethnic groups (English-speaking celebrities); 100 real South Asian videos. Unofficial Hugging Face mirrors exist; don't use them.
- **DF-Platter.** 133,260 videos of Indian subjects, labelled for gender, age, Fitzpatrick skin tone and occlusion. Sets A (one face), B and C (several faces), c23 and c40.
- **DeePhy.** 100 real and 5,040 fake videos (FaceShifter, FaceSwap, FSGAN), 10 attributes per video including gender, age group and skin tone.
- **InDeepFake.** 389 real and 4,680 fake videos in seven Indian languages; face swaps and TTS plus Wav2Lip. Also on IEEE DataPort (subscription).
- **Deepfake-Eval-2024.** In the wild, 52 languages, no country field. Evaluation only.
## Pipeline-test datasets

Open data that proves the code works, used by the smoke test ([smoke_test.md](smoke_test.md)). Their role is `pipeline_test`: the registry keeps them out of real evalsets, and their numbers are never published or reported. Checked 2026-10-04.

| Dataset               | Modality | Access                                | Licence         | What's in it                                                                   |
| --------------------- | -------- | ------------------------------------- | --------------- | ------------------------------------------------------------------------------ |
| `mendeley_roop_akool` | image    | one 4.1 GB zip, sampled by HTTP range | CC BY 4.0       | face-crop frames; 30 real and 450 fake videos; Bangladeshi and Indian subjects |
| `unidatapro_videos`   | video    | Hugging Face, pinned revision         | CC BY-NC-ND 4.0 | 5 real phone videos and 5 face-swap fakes made from them                       |

- **Mendeley Roop/Akool** (<https://data.mendeley.com/datasets/pdcp9mjy3z/3>, Daffodil International University, collected with ethical approval). Not videos, as earlier notes said: 500 x 500 JPEG face crops. Its zip holds 3,744 real frames and 104,200 fake frames (49,997 `Tech-1`, 54,203 `Tech-2`; the page says 106,948). The page names Roop and Akool but not which is which, so `method` stays `tech_1` / `tech_2`. The crops are so tight that the face detector misses them until the image is padded (see [evaluation.md](evaluation.md)). File names of fakes carry the name of the person swapped in.
- **UniDataPro deepfake videos** (<https://huggingface.co/datasets/UniDataPro/deepfake-videos-dataset>). The free preview of a commercial set: pairs of a real phone video and a fake made from it with an AI-generated face, by one of three online services. 480p to 1080p, about 68 MB. The only open, licensed source of paired real and fake videos small enough to fetch on the fly.

## Adding a dataset

See [adding_a_dataset.md](adding_a_dataset.md): config, builder (with `label_from_path`, so it can be sampled), evalset, smoke test.
