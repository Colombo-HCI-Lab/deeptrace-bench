# Datasets

Access checked against the live pages on 2026-10-03. The source of truth is `configs/datasets/<id>.yaml`; `uv run scripts/setup_datasets.py --list` prints the current table. Access terms change, so recheck a page before depending on it.

## At a glance

| Dataset            | Modality    | Access today                          | Real + fake | Size                 | Use                                      |
| ------------------ | ----------- | ------------------------------------- | ----------- | -------------------- | ---------------------------------------- |
| Urdu (CSALT)       | audio       | open (Hugging Face)                   | both        | 2 GB                 | first South Asian run                    |
| BanglaFake         | audio       | open (Hugging Face)                   | both        | 5.6 GB               | first; licence undeclared                |
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
| IndieFake          | audio       | request form                          | both        | ?                    | low priority: English                    |
| SpeechFake         | audio       | open, ModelScope                      | fake        | 116 GB for Hindi     | deferred                                 |
| IAV-DF             | audio-video | sample link dead                      |             |                      | dropped                                  |
| Indic-CodecFake    | audio       | no download anywhere                  |             |                      | dropped                                  |

No South Asian video dataset can be downloaded without approval, so the video side starts with requests. There is no Nepali deepfake data anywhere, and the only Sinhala is MLAAD's 1,000 Edge-TTS clips.

## Getting the open ones

```bash
uv run scripts/setup_datasets.py urdu_csalt banglafake asvspoof2019_la indicsuperb
uv run scripts/setup_datasets.py indicsynth --languages hi bn ur
```

MLAAD: accept its terms at <https://huggingface.co/datasets/mueller91/MLAAD> with your Hugging Face account, put `HF_TOKEN` in `.env`, then `uv run scripts/setup_datasets.py mlaad`.

## Requests to send

These need a person. Send them early: replies take days to months.

| Dataset            | What to do                                                                                                          | Who                                                        |
| ------------------ | ------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| FF++               | Google Form linked from <https://github.com/ondyari/FaceForensics>; they email a download script                    | any team member                                            |
| InDeepFake         | Fill in the access request form (PDF in the repo README) and email it                                               | applicant, naming the PI                                   |
| FakeAVCeleb        | Request form on <https://sites.google.com/view/fakeavcelebdash-lab/>; requests without ethics approval are rejected | applicant with ethics approval or a faculty sponsor        |
| DeePhy, DF-Platter | Sign the IAB Rubric licence and email it to databases@iab-rubric.org, one email per dataset                         | someone with legal authority: UCSC's Director or Registrar |
| Deepfake-Eval-2024 | Request on its Hugging Face page with a link to prior deepfake-detection work                                       | any team member                                            |
| IndieFake          | Form on <https://indie-fake-dataset.netlify.app/>                                                                   | any team member                                            |

When a copy arrives, put it in the shared store and link it in:

```bash
uv run scripts/setup_datasets.py fakeavceleb --from /path/to/FakeAVCeleb_v1.2
```

**Before requesting DF-Platter**, read [data_policy.md](data_policy.md): its licence lets only research colleagues at the signing institution use it, and the team spans more than one institution.

## Notes per dataset

- **Urdu (CSALT).** `Bonafide/`, `Spoofed_TTS/` (VITS) and `Spoofed_Tacotron/`, 17 speakers each, 6,794 WAV files. Our Doc says 35k clips; check the count on download.
- **BanglaFake.** One `final_data.zip`. 12,260 real and 13,260 VITS fakes, 7 speakers, no official split, no declared licence.
- **IndicSynth.** Parquet shards per language with audio embedded; generators XTTS-v2, VITS and FreeVC24, with speaker and gender columns. Fake only; generated from IndicSUPERB speakers.
- **IndicSUPERB (Kathbath).** m4a audio in tar archives; converted to 16 kHz WAV at manifest time. Licence not yet checked.
- **MLAAD.** `fake/<lang>/<generator>/` with a `meta.csv` each, 1,000 clips per folder. Hindi (9 generators), Bengali (4), Kannada (2), and one generator each for Tamil, Marathi, Urdu, Sinhala and Malayalam. Its real counterpart, M-AILABS, has no South Asian language.
- **FakeAVCeleb.** 500 real and 19,500 fake videos over five ethnic groups (English-speaking celebrities); 100 real South Asian videos. Unofficial Hugging Face mirrors exist; don't use them.
- **DF-Platter.** 133,260 videos of Indian subjects, labelled for gender, age, Fitzpatrick skin tone and occlusion. Sets A (one face), B and C (several faces), c23 and c40.
- **DeePhy.** 100 real and 5,040 fake videos (FaceShifter, FaceSwap, FSGAN), 10 attributes per video including gender, age group and skin tone.
- **InDeepFake.** 389 real and 4,680 fake videos in seven Indian languages; face swaps and TTS plus Wav2Lip. Also on IEEE DataPort (subscription).
- **Deepfake-Eval-2024.** In the wild, 52 languages, no country field. Evaluation only.
- **Optional open video set.** A Bangladeshi set on Mendeley Data (480 videos, CC BY 4.0, <https://data.mendeley.com/datasets/pdcp9mjy3z/3>) is useful for testing the video pipeline on non-FF++ faces. Its subjects' ethnicity isn't documented, so don't report results on it.

## Adding a dataset

1. Add `configs/datasets/<id>.yaml` with access, licence terms, `contains`, and `derived_from` (any corpus its items came from, so the contamination guard can follow it).
2. Write `src/deeptrace_bench/datasets/<id>.py` against the real files: `build_manifest(root)` returning the schema in `manifest.py`, with group attributes as `g_<attr>` plus `g_<attr>_src`.
3. `uv run scripts/setup_datasets.py <id> --manifest`, then add an evalset in `configs/evalsets/`.
