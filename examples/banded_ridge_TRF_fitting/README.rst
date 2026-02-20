Fitting Banded Ridge TRF Models
-------------------------------

## Tips & Tricks: Feature Ordering in Banded Ridge

Because the **BandedTRF** uses an iterative "greedy" optimization, the order in which you fit your features matters. Here are the guiding principles for your research:

1. **Unique vs. Redundant Variance**: If Feature A and Feature B are highly correlated, the feature placed **first** will likely "claim" the shared variance, leaving only the unique residual variance for the second feature.
2. **Order by Hypothesis**: Place the feature you are most interested in (or the one known to have the strongest effect, like the Spectrogram) first. This ensures its  is optimized against a clean baseline.
3. **Low-D to High-D**: Generally, it is safer to fit lower-dimensional features (like a single broadband envelope) after higher-dimensional ones (like a spectrogram) if you want to see if the simpler feature adds any predictive power beyond the complex one (Delta R).
4. **Consistency**: When comparing participants, always use the same `feature_order` to ensure the resulting TRF shapes and  values are comparable across your cohort.

---