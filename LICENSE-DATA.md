# Data licensing

The `LICENSE` file at the root of this repository covers the code only. It does
not apply to the VIPER dataset, which is distributed separately at
[MahmoodLab/viper](https://huggingface.co/datasets/MahmoodLab/viper).

VIPER is a collection whose elements retain the terms they arrived under. It is
not a single relicensed derivative, and there is no blanket license over the
whole dataset. Each record carries `image_license` and `annotation_license`.

| Layer | Extent | License |
|---|---|---|
| ROI images derived from Open TG-GATEs | 63 images, 189 questions | CC BY-SA 2.1 JP |
| ROI images derived from MMO | 356 images, 1,062 questions | CC BY-NC 4.0 |
| VIPER annotations: questions, answer keys, distractors, scoring rubrics, and the organ, category and magnification metadata | all 1,251 records | CC BY-NC 4.0 |

Open TG-GATEs is credited to NIBIOHN (Igarashi et al., 2015). MMO is credited to
Citlalli et al. (2022). Reuse of a record must respect both the image license for
that record and the annotation license.

Note that the ShareAlike term on the TG-GATEs images permits commercial use of
those images, whereas the annotation layer does not. The two layers are licensed
separately for this reason.
