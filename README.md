# radarwow-ml

**radar-wow stage 9: Machine-learning QPE.** Learned estimators, with physics-based stages as inputs and baselines.

Part of [radar-wow](https://github.com/radar-wow), a set of repos that together build a radar QPE system that works for S, C and X band radars.
Every algorithm is one class implementing one paper, with the same interface: it takes an xradar sweep or volume with canonical moment names, declares the bands and moments it needs, and records provenance.

## Sub-components

| sub-component | algorithm | source | DOI | status |
|---|---|---|---|---|
| RainForest (random forest) | `RainForest` | Wolfensberger et al. 2021, AMT | [doi:10.5194/amt-14-3169-2021](https://doi.org/10.5194/amt-14-3169-2021) | planned |
| deep learning QPE | - | Chen et al. 2019, GRL | [doi:10.1029/2019GL084771](https://doi.org/10.1029/2019GL084771) | planned |
| satellite-to-radar | - | Hilburn et al. 2021, JAMC | [doi:10.1175/JAMC-D-20-0084.1](https://doi.org/10.1175/JAMC-D-20-0084.1) | planned |
| deep learning QPE | - | Li, Chen & Han 2023, IEEE TGRS | [doi:10.1109/TGRS.2023.3280799](https://doi.org/10.1109/TGRS.2023.3280799) | planned |
| deep learning QPE | - | Yo et al. 2021, Earth Space Sci. | [doi:10.1029/2020EA001340](https://doi.org/10.1029/2020EA001340) | planned |
| deep learning QPE | - | Osborne et al. 2023, AIES | [doi:10.1175/AIES-D-22-0053.1](https://doi.org/10.1175/AIES-D-22-0053.1) | planned |
| MCB-Net, cGAN blockage | - | talk |  | to review |

## Install

```bash
pip install git+https://github.com/radar-wow/radarwow-core git+https://github.com/radar-wow/radarwow-ml
```

## Usage

```python
from radarwow.core import load_sample
from radarwow.ml import *  # see table above

vol = load_sample('arm_csapr2_c')        # or open_volume('my_file')
# out = SomeAlgorithm()(vol)            # band read from the file, or band='S'|'C'|'X'
```

## Status legend

*implemented + notebook*: code, tests, executed notebook. *planned*: interface fixed, code to write. *stub*: class exists, raises `NotImplementedError`. *to review*: paper to read before deciding.

## License

MIT. Data used in the notebooks comes from public archives (NEXRAD on AWS, ARM, IMD via open-radar-data) under their own terms.
