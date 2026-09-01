# Data privacy and publication policy

This repository is designed to contain code and synthetic examples only.

## Never commit

- Original, cleaned, sampled, or transformed patient-level data.
- Spreadsheets, CSV exports, notebooks with embedded outputs, or database extracts.
- Reports, slide decks, PDFs, screenshots, plots, or tables created from private data.
- Trained models, cached arrays, predictions, row-level error analyses, or feature exports.
- Names, dates, record numbers, free-text notes, study identifiers, local file paths, credentials, or internal company details.
- Aggregate statistics or model results from a private source unless the data owner has explicitly approved their publication.

## Safe demonstration material

`generate_synthetic_data.py` creates fictional records independently of the private source values. Generated files are ignored by Git and should be regenerated locally when needed.

## Before every push

1. Review `git status` and the exact staged diff.
2. Confirm that no ignored file has been force-added.
3. Search tracked files for credentials, personal paths, identifiers, and internal names.
4. Confirm that the Git history never contained private material; deleting a file in a later commit does not remove it from earlier commits.
5. Obtain any required employer, data-owner, ethics, or legal approval for publication of the code itself.

The ignore rules are a safety net, not a substitute for review.

