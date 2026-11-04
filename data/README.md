# Input data

The input data are not included in this repository.

The analysis uses the protein-level Proteome Discoverer 2.4 export of a TMTpro 16-plex
cochlear synaptosome experiment (sheet `Proteins`). The data were generated at Wayne State
University and are described in:

- Shahab M. Role of nitrative stress in cisplatin induced ototoxicity and synaptic dysfunction
  [doctoral dissertation]. Wayne State University, 2023.
  https://digitalcommons.wayne.edu/oa_dissertations/3846
- Shahab M, Rosati R, Stemmer PM, Dombkowski A, Jamesdaniel S. Quantitative profiling of
  cochlear synaptosomal proteins in cisplatin-induced synaptic dysfunction. Hear Res. 2024,
  447:109022. doi:10.1016/j.heares.2024.109022

Requests for access to the underlying data may be directed to the Office of the Vice President
for Research, Wayne State University, Detroit, Michigan, USA (https://research.wayne.edu).

To run the pipeline, place the export in this folder as `2363_MC-Loading_2.xlsx`, or set the
environment variable `P1_DATA_FILE` to its path.

Expected layout of the sheet: the reporter abundance columns are
`Abundances (Grouped): 126` to `Abundances (Grouped): 134N`. Channels are assigned to
groups in order: Control (126, 127N, 127C, 128N), Cisplatin (128C, 129N, 129C, 130N),
MnTBAP+Cisplatin (130C, 131N, 131C, 132N) and MnTBAP (132C, 133N, 133C, 134N).
Rows without a protein accession (spreadsheet working rows) are ignored.
