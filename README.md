UdonPred is a light-weight disorder predictor.

## Abstract
Intrinsically disordered proteins (IDPs) are a unique class of proteins that lack a fixed or rigid three-dimensional structure under physiological conditions. They are pivotal in various biological processes, including signal transduction, transcriptional regulation, and cell-cycle control. The study of IDPs is crucial as their disorder is often linked to diseases such as cancer, cardiovascular diseases, and neurodegenerative disorders.
While the advancement in computational methods has significantly enhanced the study of IDPs, many predictors available are limited by a shortage of high-quality disorder annotations. To improve upon existing methods, UdonPred  uses a new dataset based on the [TriZOD](https://github.com/MarkusHaak/trizod) scoring scheme that assigns continuous disorder scores (G-scores) to residues based on nuclear magnetic resonance (NMR) spectroscopy chemical shifts in the entire Biological Magnetic Resonance Data Bank (BMRB), assessing how much experimentally determined chemical shifts deviate from random coil chemical shifts.
Based on this new dataset of over 15k peptides (10 times as large as the previously published CheZOD dataset), we trained slim neural networks to predict TriZOD G-scores from embeddings generated by the pre-trained protein language model (pLM) ProstT5. Needing only the information present in these embeddings, UdonPred itself is extremely fast and can predict per-residue disorder scores for a protein in a matter of milliseconds on a consumer-grade CPU.
UdonPred can successfully predict disorder from per-residue pLM embeddings: Evaluation on the commonly used OdinPred test set of 117 proteins exhibits promising results. With an area under the ROC curve (AUROC) of 0.907±0.01 between TriZOD G-scores predicted by UdonPred and manually computed binary CheZOD Z-scores, UdonPred is competitive with other state-of-the-art disorder predictors.

## Inference
For your convenience, use our Docker image:
```bash
docker run -it -v <data_directory>:/app/data -v <output_directory>:/app/out jschlensok/udonpred -i <path_to_input_fasta_file> [-e <path_to_embedding_h5_file>] [--write-to-one-file]  [--model-directory <model_directory>]
```
You can also build it from source using `docker build -t <image_name> .` and once built, generate predictions with the above command, substituting `<image_name>` for `jschlensok/udonpred`


### Parameters
The `<data_directory>` needs to contain the input FASTA file and corresponding embeddings in H5 format (if existent). If no HD5 file with ProstT5 per-residue embeddings is provided via the `-e` option, one will be generated (WARNING: THIS IS EXTREMELY SLOW ON CPU!)
The `<output_directory>` is used to persist predictions and timing files.
Importantly, `<data_directory>` and `<output_directory>` need to be specified with leading dots if they're relative paths, e.g. `./data:/app/data` instead of `data:/app/data`.
The input file paths need to be relative to the `/app` directory in the container, i.e. include the directory mounted to `/app/data`, e.g. `data/file.fasta`.
Output is formatted according to the [CAID 3 specifications](https://caid.idpcentral.org/challenge#participate). Predictions are stored in one file per protein unless the `--write-to-one-file` switch is toggled, in which case they're written to `<output_directory>/all.caid`.

### Flavours
UdonPred comes in three flavours:
- `UdonPred-TriZOD` that's trained on TriZOD scores. This is the default flavour.
- `UdonPred-DisProt` that's trained on DisProt annotations. Run this flavor by passing `--model-directory trained_models/disprot`.
- `UdonPred-combined` that's trained on TriZOD scores and fine-tuned on DisProt annotations. Run this flavor by passing `--model-directory trained_models/combined`.

## Training
### Hydra & setup
`config/config.yaml` contains default training parameters. Running `train.py` without any additional parameters will perform training with those.
They can be overridden using [Hydra](https://hydra.cc/):
- simple: `python -m src.pipeline.train training.max_epochs=250`
- queueing multiple runs: `python -m src.pipeline.train --multirun embedding_type=esm2_3b,prott5 dataset=strict,moderate,tolerant,unfiltered training.max_epochs=250`

A few default parameter sets for models and datasets are provided in the `config/models` and `config/datasets` folders which can be extended upon or overriden ad libitum using the command line. Additional parameters can be added with a prefixed `+`.

The training scripts automatically log loss (not hyperparameters because that just isn't working as expected using the API) using the PyTorch TensorBoard extension. Use it like `tensorboard --logdir <project_path>/runs`.

### Corrected pH values in unfiltered.json
Some samples had suspect pH values:
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
