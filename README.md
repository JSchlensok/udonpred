## Inference
A Dockerfile is provided to run inference easily. Build it using `docker build -t <image_name> .`, and once built, generate predictions as follows:
```bash
docker run -it -v <data_directory>:/app/data -v <output_directory>:/app/out <image_name> -i <path_to_input_fasta_file> [-e <path_to_embedding_h5_file>] [--write-to-one-file]  [--model-directory <model_directory>]
```
The `<data_directory>` needs to contain the input FASTA file and corresponding embeddings in H5 format (if existent). If no HD5 file with ProstT5 per-residue embeddings is provided via the `-e` option, one will be generated (WARNING: THIS IS EXTREMELY SLOW ON CPU!)
The `<output_directory>` is used to persist predictions and timing files.
Importantly, `<data_directory>` and `<output_directory>` need to be specified with leading dots if they're relative paths, e.g. `./data:/app/data` instead of `data:/app/data`.
The input file paths need to be relative to the `/app` directory in the container, i.e. include the directory mounted to `/app/data`, e.g. `data/file.fasta`.
Predictions are stored in one file per protein unless the `--write-to-one-file` switch is toggled, in which case they're written to `<output_directory>/all.caid`.

UdonPred comes in three flavours:
- `UdonPred-TriZOD` that's trained on TriZOD scores. This is the default flavour.
- `UdonPred-DisProt` that's trained on DisProt annotations. Run this flavor by passing `--model-directory trained_models/disprot`.
- `UdonPred-combined` that's trained on TriZOD scores and fine-tuned on DisProt annotations. Run this flavor by passing `--model-directory trained_models/combined`.

Output is formatted according to the [CAID 3 specifications](https://caid.idpcentral.org/challenge#participate).

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
