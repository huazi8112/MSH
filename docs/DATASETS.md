# Dataset sources and preprocessing

The revised manuscript uses nine empirical pairwise networks. All datasets are processed into simple, unweighted, undirected largest-connected-component graphs before scoring.

| Network | File/source used in repository | Source URL or collection | Original representation | Processing note |
|---|---|---|---|---|
| Lesmis | `lesmis.tar.gz` | SuiteSparse Matrix Collection / Newman | Character co-occurrence graph | Converted to simple undirected LCC. |
| Adjnoun | `adjnoun.tar.gz` | SuiteSparse Matrix Collection / Newman | Word adjacency graph | Converted to simple undirected LCC. |
| Jazz | `jazz.tar.gz` | SuiteSparse Matrix Collection / Arenas | Jazz musician collaboration graph | Converted to simple undirected LCC. |
| Usair | `usair.tar.gz` / `usair.zip` | Pajek / USAir97 | Air transportation graph | Converted to simple undirected LCC. |
| Infect | `ia-infect-dublin.mtx` / `infect.zip` | Network Repository | Face-to-face contact data | Temporal/repeated contacts are aggregated to a static unweighted graph. |
| Email | `email.tar.gz` | SuiteSparse Matrix Collection / Arenas | Email communication graph | Pairwise communications are aggregated to a static unweighted graph. |
| Polblogs | `polblogs.tar.gz`, `polblogs.gml`, etc. | Newman political blogs dataset | Directed hyperlink graph | Directed edges are symmetrized; LCC is extracted. |
| Hamster | `hamster.zip` / `soc-hamsterster.edges` | Network Repository / KONECT-style edge list | Online social network | Converted to simple undirected LCC. |
| Power | `power.tar.gz`, `power.gml`, etc. | Pajek / power grid dataset | Infrastructure graph | Converted to simple undirected LCC. |

## Licence and citation notes

Dataset licensing and citation terms are governed by the original repositories/providers. The repository keeps the local copies used in the experiments for reproducibility. When redistributing the repository publicly, please keep the original file names and cite the corresponding dataset providers as listed in the manuscript references and in the source collections above.

## Preprocessing summary

All datasets are processed using `network_loader.py`. The preprocessing rules are:

- directed edges -> undirected edges;
- weighted, repeated, or temporal contacts -> single unweighted static edge;
- self-loops removed;
- largest connected component extracted;
- nodes relabeled to consecutive integers from 0.

The exact before/after statistics can be regenerated with:

```bash
python -m tools.evaluate_preprocessing_original_vs_processed
```
