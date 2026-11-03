# Next evaluation: frozen before further model selection

The 140-message grouped holdout is already observed. Do not rename it a final test, tune against it, or claim that the category-only scope increases all-message accuracy. The narrower scope deliberately stops automatic priority decisions.

## Independent annotation review (not completed)

Give a second reviewer `message_groups.csv` and the labeling guide, without model predictions. Ask them to record their name/role, review date, proposed family, category and rationale per row. Resolve disagreements before fitting a new experiment; retain the original annotations and a change log. An assistant-generated second pass is not an independent human/clinical review.

## Fresh final set (not yet collected)

Before any new tuning, collect permitted, de-identified support examples or independently authored scenarios representing actual intended traffic. Include unfamiliar wording, negation, multiple requests, ambiguous topics, regional phrasing, and rare requests. Do not reuse this repository's messages, families or templates. Have reviewers label without predictions. If priority is studied again, qualified reviewers must agree on what each priority means and how errors are handled.

Record an immutable ID, provenance/permission, text hash, family, reviewer labels, disagreements and split. Keep sensitive material in approved private storage. Freeze a version and SHA-256 before trying changes; compare the new set against all development messages for duplicate and paraphrase overlap. A hash alone cannot establish semantic independence.

Choose the evaluation sample size and required category accuracy/coverage from the intended use and uncertainty requirements before examining predictions. Keep development/calibration examples separate. Record model versions and every row-level prediction. Report all-message category accuracy/macro F1, category coverage, errors among returned suggestions, number requiring category review, and all manual priority reviews. Evaluate review time with people separately before claiming workload savings.

If changes are selected from final-set errors, that set becomes development data and another fresh final set is required. Until these steps are completed, evidence remains a synthetic retrospective prototype comparison.
