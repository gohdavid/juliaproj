Plot Style
==========

Use these conventions for maintained analysis figures:

- No plot titles; identify figures through filenames/captions.
- No x/y grid lines.
- Hide top and right axis spines.
- Histogram bars: Okabe-Ito blue `#0072B2`, semi-transparent when useful.
- Sample mean: Okabe-Ito vermillion `#D55E00`, dashed vertical line.
- Analytic reference PDF/mean: black solid PDF and black dashed mean.

`scripts/rouse_score_analytics.jl` implements these rules for Rouse video,
potential, and score plots.
