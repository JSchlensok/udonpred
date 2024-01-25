### Hydra & MLflow setup
`parameters/linreg.yaml` contains some default parameters (that are just a proof-of-concept). Running `train.py` without any additional parameters will perform training with those.
They can be overridden using [Hydra](https://hydra.cc/):
- simple: `python -m src.pipeline.train training.max_epochs=25`
- queueing multiple runs: `python -m src.pipeline.train --multirun data.embedding_type=esm2_3b,prott5 data.subset=strict,moderate,tolerant,unfiltered training.max_epochs
=25`

[MLflow](https://mlflow.org/docs/latest/index.html) tracks training and validation metrics averaged over all cross-validation folds automatically. To open its UI, run `mlflow ui` and open `localhost:5000`.

### Corrected pH values in unfiltered.json
```python
reference_df = read_reference_json("data/unfiltered.json")
with pl.Config(fmt_str_lengths=200):
    print(reference_df.filter(pl.col("pH") > 14).select(["pH", "citation_title"]))
```

```
shape: (2, 2)
┌───────┬────────────────────────────────────────────────────────────────────────────────────┐
│ pH    ┆ citation_title                                                                     │
│ ---   ┆ ---                                                                                │
│ f64   ┆ str                                                                                │
╞═══════╪════════════════════════════════════════════════════════════════════════════════════╡
│ 291.0 ┆ Complete 1H, 15N and 13C assignments of the MBD double mutant                      │
│       ┆                                                                                    │
│ 80.0  ┆ Backbone resonance assignment of an insect arylalkylamine N-acetyltransferase from │
│       ┆ Bombyx mori reveals conformational heterogeneity                                   │
│       ┆                                                                                    │
└───────┴────────────────────────────────────────────────────────────────────────────────────┘
```

**51547**
Citation title: Complete 1H, 15N and 13C assignments of the MBD double mutant 
Link: https://pubmed.ncbi.nlm.nih.gov/28236225/
Evidence:
> NMR experiments
> All NMR experiments were performed in a buffer consisting of 20 mM Tris pH=7.5, 100 mM KCl, 5 mM DTT, 0.5 mM EDTA and 7% D2O, at 30 °C, with sample concentrations in the range of 0.2–0.7 mM. 
pH 291 -> 7.5

**26962**
Citation title: Backbone resonance assignment of an insect arylalkylamine N-acetyltransferase from Bombyx mori reveals conformational heterogeneity
Link: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10325892/
Evidence:
> NMR sample preparation and assignments
> Purified uniformly-13C/15N wild-type (KVS) and its double (TVN) and triple (TAN) mutants were prepared in a mixed solvent of 90% H2O and 10% 2H2O (50 mM sodium phosphate, 50 mM NaCl, pH 6). All NMR experiments were carried out with protein concentrations of ∼0.5 mM on a Bruker Avance 800 MHz NMR spectrometer using a triple-resonance cryo probe.
pH 80 -> 6