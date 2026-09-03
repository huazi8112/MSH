# Empirical datasets

The repository contains the nine empirical networks used in the manuscript. `network_loader.py` records the source location and expected local path for each dataset.

| Key | Manuscript name | Local path used by loader | Source type |
|---|---|---|---|
| lesmis | Lesmis | `networks_data/lesmis/lesmis.mtx` | SuiteSparse / Newman |
| adjnoun | Adjnoun | `networks_data/adjnoun/adjnoun.mtx` | SuiteSparse / Newman |
| jazz | Jazz | `networks_data/jazz/jazz.mtx` | SuiteSparse / Arenas |
| usair | Usair | `networks_data/USAir97/USAir97.mtx` | SuiteSparse / Pajek |
| infect | Infect | `networks_data/ia-infect-dublin.mtx` | local supplied Matrix Market file |
| email | Email | `networks_data/email/email.mtx` | SuiteSparse / Arenas |
| polblogs | Polblogs | `networks_data/polblogs/polblogs.mtx` | SuiteSparse / Newman |
| hamster | Hamster | `networks_data/soc-hamsterster.edges` | Network Repository |
| power | Power | `networks_data/power/power.mtx` | SuiteSparse / Pajek |

All methods use the processed simple, unweighted, undirected largest connected component. Node labels are converted to consecutive integers after preprocessing.

The original/source archives are retained when available so the loader can be audited against the extracted files.
