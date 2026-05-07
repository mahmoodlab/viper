# Datasheet for VIPER

This datasheet follows the structure proposed by Gebru et al. (*Datasheets for*  
*Datasets*, 2018). It complements the dataset card hosted at  
[https://huggingface.co/datasets/MahmoodLab/viper](https://huggingface.co/datasets/MahmoodLab/viper).

## Motivation

**For what purpose was the dataset created?**
VIPER (Vision-language In Preclinical Evaluation of Rodents) was created to
evaluate vision-language models on rodent toxicologic histopathology, a
domain that anchors preclinical drug-safety assessment but has been almost
entirely absent from the pathology-VLM evaluation ecosystem. Existing
pathology benchmarks are dominated by human clinical specimens (often with a
strong oncology focus) and do not fully capture morphologic patterns that toxicologic
pathologists routinely interpret, such as non-neoplastic findings,
and species-specific anatomy.

**Who created the dataset?**
The dataset was created by the Mahmood Lab at Harvard Medical School and
Brigham and Women's Hospital, in collaboration with veterinary pathology
partners at the University of Bern (COMPATH), UC Davis, University of
Augsburg, TU Dresden, UT MD Anderson Cancer Center, and the University of
Lausanne.

**Who funded it?**
This work was funded by NIH NIGMS R35GM138216.

## Composition

**What do the instances represent?**
Each instance is a (image, question, answer) triple plus metadata.
The image is a 1,024 × 1,024-pixel H&E-stained region of interest (ROI)
extracted from a rat preclinical toxicology study. Each question takes one of
three formats: multiple-choice (MCQ; 5 options), KPrim (4 true/false
statements), or free-text (open-ended).

**How many instances are there?**
1,251 questions across 419 unique H&E ROIs from 9 organ systems.


| Question type        | Count | Scoring                                                  |
| -------------------- | ----- | -------------------------------------------------------- |
| MCQ (5 options)      | 419   | mean accuracy across 5 cyclic-shift rotations            |
| KPrim (4 statements) | 414   | ETH half-point rule (4/4 → 1.0, 3/4 → 0.5, ≤2/4 → 0.0)   |
| Free-text            | 418   | LLM-as-judge: 0.7·diagnostic-accuracy + 0.3·completeness |



| Organ                    | n   |
| ------------------------ | --- |
| kidney                   | 267 |
| liver                    | 258 |
| thyroid                  | 168 |
| male_reproductive_system | 149 |
| urinary_bladder          | 141 |
| lung                     | 84  |
| heart                    | 66  |
| gastrointestinal_tract   | 61  |
| salivary_gland           | 57  |



| Paper category (7-class taxonomy)                         | n   |
| --------------------------------------------------------- | --- |
| identify_anatomy                                          | 362 |
| probe_over_reading (overcalling lesions on normal tissue) | 240 |
| localize_in_image                                         | 227 |
| identify_pathology                                        | 221 |
| characterize_feature                                      | 78  |
| identify_artifact                                         | 63  |
| quantify_feature                                          | 60  |


**Does the dataset contain all possible instances or is it a sample?**
It is a sample. Candidate ROIs were drawn at random per organ from larger
preclinical toxicology cohorts (~1,000 to ~5,000 ROIs per organ), embedded
with TRACE (Jaume et al. 2024), clustered into 20 morphologically diverse
bins per organ, and then sampled by a board-certified veterinary pathologist
to ensure coverage across a large range of histologic morphologies.

**What data does each instance consist of?**


| Field            | Type          | Description                                                                   |
| ---------------- | ------------- | ----------------------------------------------------------------------------- |
| `image`          | PNG bytes     | 1,024 × 1,024 H&E RGB ROI, EXIF-stripped                                      |
| `image_id`       | string        | Stable content-hash identifier                                                |
| `question`       | string        | The question text                                                             |
| `question_type`  | enum          | `mcq` / `kprim` / `free_text`                                                 |
| `answer`         | string        | MCQ: option letter; KPrim: JSON list of booleans; free-text: reference answer |
| `choices`        | list[string]  | 5 options for MCQ, 4 statements for KPrim, [] for free-text                   |
| `synonyms`       | string | null | JSON list of acceptable free-text synonyms                                    |
| `scoring_rubric` | string | null | Free-text grading rubric used by the LLM judge                                |
| `organ`          | string        | One of 9 paper-aligned organ slugs                                            |
| `category`       | string        | One of 7 paper-aligned categories                                             |
| `magnification`  | string        | `"2.5x"`, `"5x"`, or `"20x"`                                                  |
| `source`         | string        | `"TG-GATEs"` or `"MMO"`                                                       |


**Are there any missing data?**
Free-text questions optionally have `synonyms` and `scoring_rubric`. MCQ and
KPrim questions do not use those fields. All other fields are populated for
every instance.

**Is there a label or target?**
Yes. `answer`, `synonyms`, and `scoring_rubric` together encode the gold
standard. Authoritative scoring of model outputs is implemented in
`viper.scoring`.

## Collection process

**How was the data acquired?**
Image ROIs were drawn from two openly licensed preclinical-pathology
resources: Open TG-GATEs (Japan NIBIO; CC BY-SA 2.1 JP) and the MMO
(Citlalli et al., 2022; CC BY-NC 4.0). For each organ, ~1,000 to ~5,000
candidate ROIs were extracted; ROIs were embedded with TRACE (Jaume et al.,
2024) and K-means clustered into 20 morphologically diverse bins per organ.
A veterinary pathologist sampled across bins to ensure broad coverage.

**Who was involved in the data-collection process?**
Three ECVP-board-certified veterinary pathologists (one as benchmark author
and gold standard, two as external readers).

**Over what timeframe was the data collected?**
2025–2026 (preceding NeurIPS Datasets & Benchmarks submission, May 2026).

**How was the question-answer content generated?**
The benchmark author wrote one *seed* question per ROI anchored in visible
morphology (e.g. *"Where is the most pronounced (artefactual) atelectasis?"*).
Seed questions were then expanded into MCQ (5-option), KPrim (4-statement),
and free-text variants by GPT-5.4. Each MCQ and KPrim variant was
adversarially filtered: GPT-5.2 was queried at temperature 0 with the
question stem but no image, three trials with reshuffled MCQ option order.
Candidates correct on any image-free trial (MCQ) or with worst-case KPrim
≥ 3/4 were regenerated up to three times with explicit feedback before being
escalated to the pathologist for manual revision or removal. Free-text
variants were not adversarially filtered; each was paired with an
LLM-generated scoring rubric reviewed by the authoring pathologist. All
final questions and rubrics were reviewed by a veterinary pathologist who
manually approved, revised, or rejected each item.

**Has the data been validated for quality?**
Yes. A reader study with three ECVP-board-certified veterinary pathologists
(VP₁ = benchmark author = gold standard; VP₂ and VP₃ = external readers) on
a randomly sampled 100-question subset showed strong inter-rater concordance
(Krippendorff's α reported in the paper). A second 100-question reader-study
subset extended the study to a physician pathologist and a no-image baseline
to characterize human transferability and image dependence.

## Preprocessing applied for release

The published parquet differs from the lab-internal source in the following
ways. None of these changes alter scoring; they remove identifiers and
non-paper metadata.

- **Dropped lab-internal columns**: `image_url` (internal GCS URL), `study`
(CRO study identifier), `seed_question`, `seed_answer`,
`base_question_id`, `permutation_id`, `diagnostic_skill`,
`morphologic_domain`. The first six are internal provenance; the last two
are an earlier two-axis taxonomy that does not appear in the paper (paper
uses a single column with seven categories).
- **Regrouped `organ*`* to the paper's nine-bucket convention (heart and
lung split, male reproductive organs merged, all GI tissues merged).
- **Normalized `question` text** by replacing the literal phrasing
`(MMO study, …)` with `(rat preclinical study, …)` so questions read
study-agnostic. The dataset-level provenance is preserved in the `source`
column.
- **Re-minted `image_id`** from `sha256(image_bytes)[:12]` so identifiers
carry no internal study or tile-coordinate information.
- **Re-encoded every image** through PIL to drop EXIF and PNG textual
metadata.
- **Verified** that no string column matches a deny-list including
`tremont`, `JNJ`, internal study-code regex `\b[A-C]\d{3}\b`, `gs://`,
`googleapis`, `BWH`, `MGB`, `confidential`, `truman`, `sealsync`.

## Uses

**Intended uses.**
Evaluating vision-language models on rodent toxicologic pathology. The
benchmark is designed to probe visual grounding (does the model use the
image?), domain transfer (does a human-pathology model work on rat tissue?),
and generalization (does adding a 5th MCQ option and rotating its position
break the model?).

**Out-of-scope uses.**
VIPER scores on a single ROI do not characterize a slide-, organ-, or
study-level diagnostic system. VIPER does not capture neurotoxicity, the
full species spectrum used in preclinical safety, or longitudinal /
dose-response reasoning. VIPER must not be used as a clinical decision-support
benchmark.

## Distribution

**How will the dataset be distributed?**
Hosted on the Hugging Face Hub at
`[MahmoodLab/viper](https://huggingface.co/datasets/MahmoodLab/viper)`. Code
to evaluate models against it lives at
[https://github.com/mahmoodlab/viper](https://github.com/mahmoodlab/viper).

**License.**
The dataset is released under CC BY-NC-ND 4.0.

## Maintenance

**Who is supporting / maintaining the dataset?**
The Mahmood Lab at Harvard Medical School and Brigham and Women's Hospital.

**How can the curator be contacted?**
Issues and errata: [https://github.com/mahmoodlab/viper/issues](https://github.com/mahmoodlab/viper/issues). Direct
email: [faisalmahmood@bwh.harvard.edu](mailto:faisalmahmood@bwh.harvard.edu), [guillaume.jaume@unil.ch](mailto:guillaume.jaume@unil.ch).

**Will the dataset be updated?**
The benchmark version used for the paper is frozen. Future updates (new
organs, additional readers, expanded reader studies) will be released as
versioned snapshots on the Hub; the published code suite resolves a specific
revision via the `--dataset-revision` flag.