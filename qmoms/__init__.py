# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2008-2024 Grigory Vilkov

Contact: vilkov@vilkov.net www.vilkov.net

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Original math is from
1. Bakshi, Kapadia, and Madan (RFS, 2003) for MFIV_BKM + Britten-Jones and Neuberger (JF, 2002) for MFIV_BJN
https://doi.org/10.1093/rfs/16.1.0101 and  https://doi.org/10.1111/0022-1082.00228
2. Martin (QJE, 2017) for SMFIV
https://doi.org/10.1093/qje/qjw034
3. Andersen, Bondarenko, and Gonzalez-Perez (RFS, 2015) for CVIX
https://doi.org/10.1093/rfs/hhv033
4. Vilkov, Xiao (2012) and Hamidieh (Journal of Risk, 2017) for TLM
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1940117 and
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2209654
5. Semivariances are based on CVIX derivation from #3/
following Feunou, Jahan-Parvar,and Okou, (Journal of Fin Econometrics, 2017)
https://doi.org/10.1093/jjfinec/nbx020
6. RIX following Gao, Gao and Song, (RFS, 2018)
https://doi.org/10.1093/rfs/hhy027
7. Slopeup and Slopedn measures used in Carbon Tail Risk, Ilhan, Sautner, and Vilkov, (RFS, 2021)  and
Pricing Climate Change Exposure, v.Lent, Sautner, Zhang, and Vilkov, (Management Science, 2023)
https://doi.org/10.1093/rfs/hhaa071 and https://doi.org/10.1287/mnsc.2023.4686
(mind the sign of slopeup -- we take '-Slope' here for better interpretability)
Originally Slopedn is used in https://doi.org/10.1111/jofi.12406 Kelly, Pastor, Veronesi (JF, 2016)
"""

__version__ = 0.1
name = 'qmoms'

from .qmoms_params import *
from .qmoms import qmoms_compute_bygroup, qmoms_compute
from .qmoms_utils import *

