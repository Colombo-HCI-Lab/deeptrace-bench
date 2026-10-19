# deeptrace-bench

Benchmark harness for deepfake detection in South Asia. It runs open video and audio detectors on South Asian deepfake datasets under one evaluation setup and reports results per group (gender, skin tone, language, generator), so numbers from different datasets and models can be compared and the gaps between groups measured. It starts with existing datasets and models, and later evaluates the data collected through [deeptrace](https://github.com/Colombo-HCI-Lab/deeptrace).

Where it stands and what the runs show so far: [Progress](https://colombo-hci-lab.github.io/deeptrace-bench/progress/) and [Results](https://colombo-hci-lab.github.io/deeptrace-bench/results/).

## Docs

Everything lives on the site, <https://colombo-hci-lab.github.io/deeptrace-bench/>, written in [`docs/src/content/docs/`](docs/src/content/docs).

| To | Read |
| --- | --- |
| understand the pipeline and the terms | [How it works](https://colombo-hci-lab.github.io/deeptrace-bench/method/) |
| install it and run the smoke test | [Getting started](https://colombo-hci-lab.github.io/deeptrace-bench/docs/getting_started/) |
| check a model or a dataset | [Models](https://colombo-hci-lab.github.io/deeptrace-bench/docs/models/), [Datasets](https://colombo-hci-lab.github.io/deeptrace-bench/docs/datasets/) |
| see how a number is made | [Evaluation](https://colombo-hci-lab.github.io/deeptrace-bench/docs/evaluation/) |
| run on a GPU | [Where to run](https://colombo-hci-lab.github.io/deeptrace-bench/docs/spark/) |
| add a model or a dataset | [Adding a model](https://colombo-hci-lab.github.io/deeptrace-bench/docs/adding_a_model/), [Adding a dataset](https://colombo-hci-lab.github.io/deeptrace-bench/docs/adding_a_dataset/) |
| browse every doc | [Docs](https://colombo-hci-lab.github.io/deeptrace-bench/docs/) |

## Data stays out of git

Datasets, weights, manifests and per-item scores live under `DTB_ROOT`, never in the repo, and the pre-commit hook enforces it ([Data policy](https://colombo-hci-lab.github.io/deeptrace-bench/docs/data_policy/)).

## Licence

Not yet chosen. The repo has been public since 2026-10-05, so GitHub Pages can serve the site. Upstream detector code keeps its own licence and is never copied in unless it is MIT.
