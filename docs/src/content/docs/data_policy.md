---
title: "Data policy"
description: "What may enter git, licence clauses that bind us, sharing data across institutions"
---

## What may enter git

| Allowed in git                                               | Never in git (lives in `DTB_ROOT`)                     |
| ------------------------------------------------------------ | ------------------------------------------------------ |
| code, configs, docs, weight hashes                           | datasets, media, archives, model weights               |
| aggregate results: metrics per group, run summaries          | manifests (they name items)                            |
| manifest summaries: counts per label and group, content hash | per-item scores, per-subject tables, split assignments |
| synthetic test fixtures                                      | anything derived from a real item, even a tiny sample  |

The pre-commit hook (`git config core.hooksPath scripts/hooks`) refuses data, media, archive and weight files, anything over 5 MB, and any CSV or JSON with an `item_id` or `rel_path` column. `evaluate.py --publish` copies only aggregate tables, never per-subject ones.

Pipeline tests are item-level too: sampled datasets, face crops and smoke reports (which name items and embed crops) stay in `DTB_ROOT/smoke/`, where the [publish guard](method.md#why-a-number-counts) keeps them. Test fixtures use invented file names, since some datasets' file names carry real people's names.

## Licence clauses that bind us

| Dataset             | Clause                                                                                                  |
| ------------------- | ------------------------------------------------------------------------------------------------------- |
| DF-Platter          | only research colleagues at the signing institution may use it; no copying beyond backup                |
| DeePhy              | research and educational use; assume DF-Platter's terms until the licence text is read                  |
| InDeepFake          | no copying, publishing or distributing any part of the dataset or derived data; access revocable        |
| Deepfake-Eval-2024  | evaluation only: never train or fine-tune on it                                                         |
| FakeAVCeleb         | non-commercial research and education; the site and the GitHub README describe the terms differently    |
| BanglaFake          | no licence declared: ask the authors before publishing results                                          |
| Svarah              | no licence declared: publish nothing derived from it                                                    |
| SEA-Spoof           | non-commercial academic use only                                                                        |
| Casual Conversations v2 | some labels are for evaluation only                                                                 |
| UniDataPro preview  | CC BY-NC-ND 4.0: non-commercial, and no derivatives may be shared (crops and scores stay in DTB_ROOT)   |
| Mendeley Roop/Akool | CC BY 4.0 with attribution; the page warns third-party content in the source videos may need permission |
| most others         | CC BY-NC 4.0: non-commercial, with attribution                                                          |

Upstream code and weights:

| Upstream | Licence | So |
| --- | --- | --- |
| AASIST, GenD, HAVIC, LipForensics, XLSR-Mamba | MIT | may be copied into the repo |
| SBI | research only | imported from `third_party/`, never copied |
| DeepfakeBench | CC BY-NC | imported, never copied |
| Effort, XLS-R + SLS | none declared | imported, never copied |
| insightface's pretrained models, including the shared face detector (`det_10g.onnx`, from `buffalo_l`) | non-commercial research only | the detection code we port is MIT |

Every dataset whose terms we've read allows publishing aggregate results, with the required citation.

## Sharing across institutions

The team spans more than one institution, and the cluster belongs to a third. A licence signed by one institution doesn't automatically cover members of another. **Before anyone downloads DF-Platter or DeePhy**, settle who signs and whether each institution needs its own agreement; ask IAB Rubric if unsure.

The repo can't enforce licences on the cluster; filesystem permissions can. Give each restricted dataset its own Unix group, with access lists on `datasets/<id>/` and the matching `scores/` and `results/` folders (per-item scores carry item ids), so only people covered by the agreement can read them. Record who is covered in the dataset's config notes.

## Data collected through deeptrace

Participant media (real people's faces and voices) stays in deeptrace's own storage until all three hold: the consent form covers processing outside Sri Lanka, DCSR confirms it may sit on Curnagl rather than Urblauna (UNIL's cluster for sensitive data), and the ethics approval covers it. None of it ever enters this repo, test fixtures included.
