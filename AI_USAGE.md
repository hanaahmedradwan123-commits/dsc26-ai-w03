# AI usage

## Tools I used, and for what

Used ChatGPT/Claude for drafting the initial implementation of `add_history_features` and structuring the `ColumnTransformer` inside `build_pipeline`.

## AI-suggested code I rejected or fixed in add_history_features or time_split

The initial AI suggestion for `add_history_features` used standard `groupby().cumcount()`, which incorrectly counted parcels occurring at the exact same timestamp. I fixed this by enforcing a strict inequality `created_at < current_time` to satisfy the `history_ties_are_not_prior` test requirement.

## How I checked that my model has no leak

I verified the model has no leak by confirming `FEATURE_COLUMNS` strictly excludes `courier_notes` and `delivered_at`, and checking that the test ROC-AUC is roughly ~0.625 (well below the 0.90 leakage threshold).