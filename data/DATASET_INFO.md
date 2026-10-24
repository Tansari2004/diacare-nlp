# DiaCare Support Message Dataset

630 synthetic messages; no patient records or personal health information. Five categories (126 rows each): appointment, billing, medical, other, tech_bug. Priority labels run from 1 to 5 and are not validated clinical annotations.

The original data has 613 distinct texts after lowercasing/trimming; related templates also recur. The original separate 20% row-level holdouts each contained seven texts also found in training and are superseded.

The corrected experiment uses all source rows, grouped into 84 manually reviewed families in `message_groups.csv`. One shared holdout contains 140 rows from 17 families; 490 rows from 67 families are used for training. Groups can span categories. No family or normalized text overlaps the split. See [grouping protocol](GROUPING_PROTOCOL.md) for the fixed seed, split rule, annotation limits, and evidence.

Family annotations are not independent clinical review, nor a guarantee that every paraphrase relationship is captured. Results remain a small synthetic experiment.
