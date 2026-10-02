# Data policy

Almost every dataset here is licensed for non-commercial research only, several forbid passing on any derived data, and one restricts use to a single institution. This page is what the repo does about it and what it can't do on its own.

## What may enter git

| Allowed in git                                               | Never in git (lives in `DTB_ROOT`)                     |
| ------------------------------------------------------------ | ------------------------------------------------------ |
| code, configs, docs, weight hashes                           | datasets, media, archives, model weights               |
| aggregate results: metrics per group, run summaries          | manifests (they name items)                            |
| manifest summaries: counts per label and group, content hash | per-item scores, per-subject tables, split assignments |
| synthetic test fixtures                                      | anything derived from a real item, even a tiny sample  |

The pre-commit hook (`git config core.hooksPath scripts/hooks`) refuses data, media, archive and weight files, anything over 5 MB, and any CSV or JSON with an `item_id` or `rel_path` column. `evaluate.py --publish` copies only aggregate tables and leaves per-subject tables behind.

## Licence clauses that bind us

| Dataset            | Clause                                                                                               |
| ------------------ | ---------------------------------------------------------------------------------------------------- |
| DF-Platter         | only research colleagues at the signing institution may use it; no copying beyond backup             |
| DeePhy             | research and educational use; assume DF-Platter's terms until the licence text is read               |
| InDeepFake         | no copying, publishing or distributing any part of the dataset or derived data; access revocable     |
| Deepfake-Eval-2024 | evaluation only: never train or fine-tune on it                                                      |
| FakeAVCeleb        | non-commercial research and education; the site and the GitHub README describe the terms differently |
| BanglaFake         | no licence declared: ask the authors before publishing results                                       |
| most others        | CC BY-NC 4.0: non-commercial, with attribution                                                       |

Publishing aggregate results is allowed for every dataset we've read the terms of, with the required citation.

## Sharing across institutions

The team spans more than one institution, and the cluster belongs to a third. A licence signed by one institution doesn't automatically cover members of another. **Before anyone downloads DF-Platter or DeePhy**, settle who signs and whether each institution needs its own agreement, and ask IAB Rubric if unsure.

On the cluster, the repo can't enforce licences, but filesystem permissions can. Give each restricted dataset its own Unix group, with access lists on `datasets/<id>/`, and on the matching `scores/` and `results/` folders (per-item scores carry item ids), so only people covered by the agreement can read them. Record who is covered in the dataset's config notes.

## Data collected through deeptrace

Participant media (real people's faces and voices) stays in deeptrace's own storage until three things are true: the consent form covers processing outside Sri Lanka, DCSR confirms it may sit on Curnagl rather than Urblauna (UNIL's cluster for sensitive data), and the ethics approval covers it. None of it enters this repo, ever, including in test fixtures.
