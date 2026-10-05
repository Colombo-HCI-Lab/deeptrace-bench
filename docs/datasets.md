# Datasets

Access checked against the live pages on 2026-10-03, and every one again on 2026-10-05, when the datasets added that day were checked too. The source of truth is `configs/datasets/<id>.yaml`; `uv run scripts/setup_datasets.py --list` prints the current table. Access terms change, so recheck a page before depending on it.

## At a glance

| Dataset                 | Modality    | Access today                          | Real + fake | Size                  | Use                                                  |
| ----------------------- | ----------- | ------------------------------------- | ----------- | --------------------- | ---------------------------------------------------- |
| Urdu (CSALT)            | audio       | open (Hugging Face)                   | both        | 2 GB                  | first South Asian run                                |
| BanglaFake              | audio       | open (Hugging Face)                   | both        | 5.6 GB                | first; licence undeclared                            |
| BD-GRF6                 | audio       | open (Zenodo)                         | both        | 3 GB                  | Bangla by gender, including third gender             |
| Bangla voices           | audio       | open (Mendeley Data)                  | both        | 0.5 GB                | Bangla, 75 speakers, paired real and fake            |
| IndicTTS challenge      | audio       | open (Hugging Face)                   | both        | 17.4 GB               | 16 Indian languages, Nepali among them               |
| IndicSynth              | audio       | open, per language                    | fake only   | 128 GB (hi+bn+ur)     | paired with IndicSUPERB                              |
| IndicSUPERB             | audio       | open (object store)                   | real only   | 5.8 GB (test splits)  | real side of IndicSynth                              |
| OpenSLR Sinhala         | audio       | open (OpenSLR SLR52)                  | real only   | 0.9 GB (1 of 16 zips) | real side of MLAAD's Sinhala                         |
| HiDF                    | video       | open (Zenodo)                         | both        | 1.6 GB (videos)       | per-face race labels; "Indian" is the South Asian group |
| In-the-Wild             | audio       | open (Hugging Face)                   | both        | 8.2 GB                | reproduction for DF Arena, AntiDeepfake; English     |
| ASVspoof 2019 LA        | audio       | open (Edinburgh DataShare)            | both        | 7.6 GB                | reproduction, score direction                        |
| MAVOS-DD (Hindi)        | audio-video | accept terms on Hugging Face          | both        | 35.7 GB               | the only open South Asian video                      |
| MLAAD                   | audio       | accept terms on Hugging Face, instant | fake only   | 9.5 GB (South Asian)  | needs a real source; cross-corpus                    |
| Svarah                  | audio       | accept terms on Hugging Face          | real only   | 1.1 GB                | real Indian-accented English by state and language   |
| SEA-Spoof               | audio       | email the authors + manual approval   | both        | 81.5 GB               | the largest labelled Hindi and Tamil spoof set       |
| FF++                    | video       | request form                          | both        | ~40 GB (c23)          | reproduction                                         |
| InDeepFake              | audio-video | emailed form naming the PI            | both        | 31 GB                 | request first: lightest                              |
| FakeAVCeleb             | audio-video | form + ethics approval, slow          | both        | ?                     | request now; 100 real South Asian videos             |
| DeePhy                  | video       | institution-signed licence            | both        | 30 GB                 | one signing round with DF-Platter                    |
| DF-Platter              | video       | institution-signed licence            | both        | 500 GB                | stratified sample                                    |
| Deepfake-Eval-2024      | mixed       | manual approval, often rejected       | both        | 20 GB                 | evaluation only; ~5% South Asian                     |
| Celeb-DF v2             | video       | request form                          | both        | ~10 GB                | reproduction for video models                        |
| Casual Conversations v2 | video       | form + Meta licence                   | real only   | ?                     | false positives on real Indian faces                 |
| IndieFake               | audio       | request form                          | both        | ?                     | low priority: English                                |
| SpeechFake              | audio       | open, ModelScope                      | both        | 143 GB, a split zip   | Hindi TTS and VC, Bengali, Tamil, Marathi, Malayalam |
| LRLspoof                | audio       | open (Hugging Face)                   | fake        | 462 GB, one tarball   | deferred; Nepali                                     |
| IAV-DF                  | audio-video | sample link dead                      |             |                       | dropped                                              |
| Indic-CodecFake         | audio       | no download anywhere                  |             |                       | dropped                                              |

The only South Asian video that needs no institutional request is MAVOS-DD's Hindi subset (accept its terms on Hugging Face); every other South Asian video set is Indian and needs a request. The only Sinhala deepfake audio is MLAAD's 1,000 Edge-TTS clips, now paired with real Sinhala speech from OpenSLR (cross-corpus, so diagnostic only). The only labelled real and fake Nepali is in the IndicTTS challenge set; LRLspoof has more Nepali spoofs but sits in one 462 GB tarball. HiDF isn't South Asian: it is here because each face carries a race label, and "Indian" (90 base videos) can be compared with the other groups.

## Getting the open ones

```bash
uv run scripts/setup_datasets.py urdu_csalt banglafake bd_grf6 bangla_voices indictts_challenge hidf asvspoof2019_la in_the_wild openslr_sinhala indicsuperb
uv run scripts/setup_datasets.py indicsynth --languages hi bn ur
```

MLAAD, MAVOS-DD and Svarah: accept their terms at <https://huggingface.co/datasets/mueller91/MLAAD>, <https://huggingface.co/datasets/unibuc-cs/MAVOS-DD> and <https://huggingface.co/datasets/ai4bharat/Svarah> with your Hugging Face account, put `HF_TOKEN` in `.env`, then `uv run scripts/setup_datasets.py mlaad` (or `mavos_dd_hi`; Svarah's builder is still a stub until its schema can be read). ASVspoof 2019 LA comes from Edinburgh DataShare, which is slow (about 0.3 MB/s on 2026-10-05, so hours for 7.6 GB); a resumable `curl -C -` into `datasets/asvspoof2019_la/LA.zip` survives interruptions, and the setup script then checks and extracts it.

To try a dataset before downloading all of it, `--sample N` fetches N real and N fake items into the smoke namespace (see [adding_a_dataset.md](adding_a_dataset.md)); it needs the dataset's builder to be written. Rows of Hugging Face parquet shards (IndicSynth, the IndicTTS challenge set) are sampled by reading a few shards' label columns and one row group of audio each; zips inside zips (BD-GRF6) by fetching the smallest inner zip per label; tar-only sets (IndicSUPERB) only from a full local copy.

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
| SEA-Spoof          | Email the authors (wu_jinyang@a-star.edu.sg, imcc.sg@gmail.com), then submit the form on <https://huggingface.co/datasets/Jack-ppkdczgx/SEA-Spoof> | an applicant at an academic institution |

When a copy arrives, put it in the shared store and link it in:

```bash
uv run scripts/setup_datasets.py fakeavceleb --from /path/to/FakeAVCeleb_v1.2
```

**Before requesting DF-Platter**, read [data_policy.md](data_policy.md): its licence lets only research colleagues at the signing institution use it, and the team spans more than one institution.

## Notes per dataset

- **Urdu (CSALT).** `Bonafide/`, `Spoofed_TTS/` (VITS) and `Spoofed_Tacotron/`, 17 speakers each, 6,794 WAV files. Our Doc says 35k clips; check the count on download.
- **BanglaFake.** One `final_data.zip` with three sub-corpora (checked 2026-10-05): sust (9,999 real, 9,999 fake), mozilla (2,797 real-fake pairs of five Common Voice speakers) and news (1,000 real, no fakes). 13,796 real and 12,796 VITS fakes, not the README's numbers. The evalset keeps sust and mozilla, where every fake has a real counterpart. Common Voice makes XLS-R-based models `source_overlap` on it. No official split, no declared licence.
- **BD-GRF6** (Zenodo 18656689, 2026, CC BY 4.0). Bangla speech of Bangladeshi speakers in six classes, real and fake for male, female and third gender; 13,566 clips. `Real and Fake.zip` holds one zip per class (`Femail Fake.zip`, sic). No generator or speakers named. The third-gender real clips are MP3 and the female fakes WAV, so file format may give labels away: check every class on the full download before trusting a score. Taken as cross-corpus for that reason.
- **Bangla voices** (Mendeley Data 10.17632/4ftmwt86vr.1, University of Asia Pacific, 2024, CC BY 4.0). 75 speakers (`S<set><F|M><nn>`) in 15 sets, each reading 30 sentences once real and once faked: 4,500 clips of 2 to 5 s. Every fake has a real twin by the same speaker and sentence. Fetched as Mendeley's whole-dataset zip, whose signed S3 link serves range requests.
- **IndicTTS challenge** (`SherryT997/IndicTTS-Deepfake-Challenge-Data`, CC BY 4.0). 31,102 labelled clips (`is_tts`) in 16 Indian languages including Nepali, Assamese, Bodo, Dogri and Manipuri; the test split's labels are hidden. Generator and real source unnamed; ids carry a language code, gender and emotion (one female and one male voice per language), and the English rows include LJSpeech and LibriSpeech utterances. Audio embedded in parquet.
- **In-the-Wild.** 31,779 clips of 54 public figures (19,963 bona fide, 11,816 spoof), labels in `meta.csv`. Not South Asian: the reproduction set for models the guard refuses on ASVspoof 2019 LA eval, such as DF Arena.
- **OpenSLR Sinhala (SLR52).** Crowd-sourced read Sinhala, 185k utterances in 16 zips; only the first (11,550 clips, anonymised speaker ids) is fetched. The real side of `mlaad_si`.
- **MAVOS-DD (Hindi).** 3,632 real and 3,881 fake videos from eight generators: face swaps (inswapper, hififace, roop), audio-driven talking heads (echomimic, sonic, memo), reenactment (liveportrait) and voice conversion (knnvc, video untouched). Per-track labels drive two evalsets, `mavos_dd_hi_video` and `mavos_dd_hi_audio`; the audio of the non-VC fakes stays unlabelled until the dataset's metadata can be read. No speaker or demographic labels.
- **Celeb-DF v2.** 890 real and 5,639 fake videos of 59 celebrities, mostly Caucasian; for reproduction only.
- **Casual Conversations v2.** Real only: 5,567 consenting participants in seven countries, India the largest, with Fitzpatrick and Monk skin tone, age, gender and language labels.
- **IndicSynth.** Parquet shards per language (Hindi: 107 shards, 205,938 rows) with audio embedded, 100 rows per row group, shards sorted by generator; generators XTTS-v2, VITS and FreeVC24, with target and source speaker and gender columns. Fake only; generated from IndicSUPERB speakers, whose Kathbath numbering the speaker ids share.
- **IndicSUPERB (Kathbath).** m4a audio in plain tars (`kb_data_clean_m4a/<language>/<split>/audio/<utt>-<speaker>-<m|f>.m4a`), decoded by the audio loader. It is also the real side of SpeechFake's and MLAAD's Bengali, Hindi, Tamil, Marathi and Malayalam evalsets. Tars can't be read in place, so it is downloaded in full (5.8 GB, about 1.5 h at 1 MB/s) and sampled from that copy. CC BY 4.0 per the Hugging Face card of `ai4bharat/Kathbath` (gated there, and without the test splits).
- **MLAAD.** `fake/<lang>/<generator>/` with a `meta.csv` each, 1,000 clips per folder. Hindi (9 generators), Bengali (4), Kannada (2), and one generator each for Tamil, Marathi, Urdu, Sinhala and Malayalam. Its real counterpart, M-AILABS, has no South Asian language, so `mlaad_hi` and `mlaad_bn` take real speech from IndicSUPERB and `mlaad_si` from OpenSLR Sinhala (all cross-corpus).
- **HiDF** (KDD 2025, Zenodo 16140829, CC BY-NC 4.0). 4,361 real and 4,361 fake videos of about 3 s, each fake a real video with another face swapped in by a commercial tool; `metadata.csv` gives race, gender and age per face (image and video ids are separate namespaces; a fake's donor face is an image row). The videos are 512 px face crops (chin cut off), which the detector only finds with the padded retry (`video_pad_retry`). Too short for HAVIC's 3.2 s windows.
- **Svarah** (AI4Bharat). Real Indian-accented English, 117 speakers with native state, district and first language; gated, so its builder waits for the schema. No licence declared on the card.
- **SEA-Spoof** (A*STAR, 2025). Hindi 71,171 and Tamil 49,640 utterances among seven languages, ten named generators including commercial ones, speakers per row. Manual approval after emailing the authors; academic, non-commercial.
- **FakeAVCeleb.** 500 real and 19,500 fake videos over five ethnic groups (English-speaking celebrities); 100 real South Asian videos. Unofficial Hugging Face mirrors exist; don't use them.
- **DF-Platter.** 133,260 videos of Indian subjects, labelled for gender, age, Fitzpatrick skin tone and occlusion. Sets A (one face), B and C (several faces), c23 and c40.
- **DeePhy.** 100 real and 5,040 fake videos (FaceShifter, FaceSwap, FSGAN), 10 attributes per video including gender, age group and skin tone.
- **InDeepFake.** 389 real and 4,680 fake videos in seven Indian languages; face swaps and TTS plus Wav2Lip. Also on IEEE DataPort (subscription).
- **Deepfake-Eval-2024.** In the wild, 52 languages, no country field. Evaluation only.
- **SpeechFake** (ACL 2025, ModelScope, CC BY 4.0). Its multilingual part is one 123 GB split zip (`MD.z01`, `MD.z02`, `MD.zip`), read as one archive by range requests (`remote_zip.SplitZip`, which scans the 179 MB central directory, about 3 min). South Asian content: Hindi Edge-TTS (15,382) and SeedVC voice conversion (14,633, LibriTTS speech into Common Voice Hindi voices), Edge-TTS Bengali, Tamil (Singapore voices), Marathi (20,000 each) and Malayalam (3,726), and 4,630 real Common Voice Hindi clips in its real part. Hindi pairs within the dataset; the other languages take IndicSUPERB's real speech. Cross-corpus either way.
- **LRLspoof, IAV-DF, Indic-CodecFake** (rechecked 2026-10-05). LRLspoof (MIT) is spoof-only in 66 languages, Nepali among them, in one 462 GB tarball. IAV-DF's sample link is still dead. Indic-CodecFake's download buttons still say coming soon; its GitHub repo has a few demo clips and no licence.
## Checked but not added

Searched on 2026-10-05 for South Asian sets beyond the Doc's; these were left out, for now:

- **Indian-language fake speech** (`satwc-reddy/indian-language-deepfake-speech`, CC BY 4.0): Telugu, Tamil, Malayalam and Konkani fakes (MMS-TTS, SIGVC, RVC), fake only, a community upload with no review; its real side isn't shipped. Worth adding with IndicSUPERB reals if Telugu or Konkani voice conversion matters.
- **Nepali audio deepfake** (two identical Hugging Face copies, 2,064 real and 2,064 fake): no README, no licence, unknown provenance.
- **MLADDC** (20 languages from VoxLingua107, seven South Asian): not released yet. **IndicFake** (TMLR 2025, 17 Indian languages): no download link. **HAV-DF** (Hindi audio-video): its Kaggle page is gone.
- **Real references:** FairFace (real faces labelled "Indian", CC BY 4.0), FLEURS (South Asian read speech, no Sinhala; AntiDeepfake trained on it), IndicVoices-R (15 South Asian languages, gated, about 1 TB).
- **Demographic labels without a South Asian class:** AI-Face (Asian, White, Black, Others) and DeepSpeak (gated; ethnicity labels unverified).

## Pipeline-test datasets

Open data that proves the code works, used by the smoke test ([smoke_test.md](smoke_test.md)). Their role is `pipeline_test`: the registry keeps them out of real evalsets, and their numbers are never published or reported. Checked 2026-10-04.

| Dataset               | Modality | Access                                | Licence         | What's in it                                                                   |
| --------------------- | -------- | ------------------------------------- | --------------- | ------------------------------------------------------------------------------ |
| `mendeley_roop_akool` | image    | one 4.1 GB zip, sampled by HTTP range | CC BY 4.0       | face-crop frames; 30 real and 450 fake videos; Bangladeshi and Indian subjects |
| `unidatapro_videos`   | audio-video | Hugging Face, pinned revision      | CC BY-NC-ND 4.0 | 5 real phone videos and 5 face-swap fakes made from them, with sound           |

- **Mendeley Roop/Akool** (<https://data.mendeley.com/datasets/pdcp9mjy3z/3>, Daffodil International University, collected with ethical approval). Not videos, as earlier notes said: 500 x 500 JPEG face crops. Its zip holds 3,744 real frames and 104,200 fake frames (49,997 `Tech-1`, 54,203 `Tech-2`; the page says 106,948). The page names Roop and Akool but not which is which, so `method` stays `tech_1` / `tech_2`. The crops are so tight that the face detector misses them until the image is padded (see [evaluation.md](evaluation.md)). File names of fakes carry the name of the person swapped in.
- **UniDataPro deepfake videos** (<https://huggingface.co/datasets/UniDataPro/deepfake-videos-dataset>). The free preview of a commercial set: pairs of a real phone video and a fake made from it with an AI-generated face, by one of three online services. 480p to 1080p, about 68 MB. The only open, licensed source of paired real and fake videos small enough to fetch on the fly. Every file has an audio track, so the dataset is audio-video: `unidatapro_videos` tests the video path and `unidatapro_av` the audio-visual one (HAVIC).

## Adding a dataset

See [adding_a_dataset.md](adding_a_dataset.md): config, builder (with `label_from_path`, so it can be sampled), evalset, smoke test.
