# Artifact Load Note

Old E3 artifact could not be loaded in the current Keras runtime.

Error type: `TypeError`

D1-D4 therefore rebuild/retrain a serializable reference-compatible teacher using `ChannelSlice` layers instead of Lambda slices.
