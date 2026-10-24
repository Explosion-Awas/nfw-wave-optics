# Research paths and archived witnesses

**GPT-6.1sol Max & Hermes**

These are historical exploratory comparisons, not new runs and not additions to
the paper's frozen validation set. The [Writing account](https://explosion-awas.github.io/writing/nfw-wave-optics-lessons/)
explains how they changed the research question. Source-document hashes and
project-relative provenance are listed in `exploration_sources.json`.

## An order-one field and a tiny residual

An earlier high-precision NFW audit used two stabilized split-Hankel settings,
32 decimal digits with shape 1 and 48 decimal digits with shape 0.7. The settings
share an integral structure; their agreement is convergence evidence, not an
independent certificate of total error.

| kappa_s | y | w | abs(R), 48-digit audit | Relative difference between the two high-precision residuals |
| ---: | ---: | ---: | ---: | ---: |
| 0.03 | 5 | 0.1 | 1.9217189554e-3 | 2.87e-31 |
| 0.03 | 100 | 1000 | 5.8010940856e-13 | 7.64e-12 |
| 0.3 | 100 | 1000 | 5.7892801428e-12 | 7.53e-13 |

At the weakest point the converged residual was approximately
`8.6510223655e-16 + 5.8010876351e-13j`. These large-q audit points are separate
from the final fast-controller box. Completing the geometric square and using
image-centred coordinates removed most of the earlier double-precision failure
even when the algorithm still computed the full field before subtracting GO.
That result prevented an unstable ordinary baseline from becoming evidence for
a supposedly new residual-first advantage.

## A stronger Born baseline changed the question

Every error below uses the same exact image geometry, minimum-delay gauge and
complete complex field. `cut+image2` means an exact central endpoint combined
with a **truncated** image expansion, not the final complete-line evaluator.
These are the nine exploratory points from the archived checkpoint.

| w | y | kappa_s | M2+image2 absolute complex error | Full Born3 error | Exact cut+image2 error |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1.7 | 0.9 | 0.08 | 0.2857196739 | 2.870439006e-5 | 0.002729986115 |
| 2.2 | 1.3 | 0.01 | 0.002721113218 | 5.805593695e-9 | 3.247170899e-5 |
| 2.2 | 1.3 | 0.02 | 0.005402368743 | 9.406685806e-8 | 6.522895823e-5 |
| 2.2 | 1.3 | 0.04 | 0.01064606543 | 1.543415879e-6 | 0.0001307784266 |
| 0.05 | 20 | 0.5 | 0.0001114412425 | 1.461813005e-5 | 2.226627751e-6 |
| 0.025 | 40 | 0.5 | 7.550699874e-6 | 1.524614791e-6 | 9.293871954e-8 |
| 0.0125 | 80 | 0.5 | 4.781258806e-7 | 1.391954312e-7 | 3.297442394e-9 |
| 0.01 | 100 | 0.5 | 1.950805283e-7 | 6.266487624e-8 | 1.105661367e-9 |
| 0.00625 | 160 | 0.5 | 2.977516525e-8 | 1.095194667e-8 | 1.089468565e-10 |

Born3 has smaller error than M2+image2 at all five large-offset, fixed-q=1
points. Exact cut+image2 improves on Born3 at those five points but not the four
small-separation points. This supported retaining Born3 and moving from a higher
central expansion to a numerical representation with the **complete** potential
exponential and image contour. It does not establish a global ranking.

## Timing depends on what changes

The published `data/timing/` retains the archived analyses. The paper's cost
claim concerns a new dimensionless lens pair: preparation plus the first complete
frequency band. The four-band interleaved confirmation has 16 paired ratios and
32 executions, checked by `examples/replay_paper.py`. The band-0 cold outcome and
nonpassing cheaper band-5 setting are retained in the earlier cold analysis.

Prepared native FFT reevaluation is a different workload. On the original six
bands, that warm-optimized comparison passed the common timing accuracy target
and was about 493–794 times faster than the controlled evaluator, excluding its
one-time preparation. Improving the actual frequency interpolation density,
rather than just increasing the FFT point count, resolved an earlier FFT residual
failure. Both are reasons to state the workload before quoting a speed ratio.

The accuracy study uses a `1e-10` complete-complex target plus one-percent
integrated residual error with empirical reference margins. The separate timing
study uses `1e-8` plus one percent. An exact integral representation, a local
asymptotic statement and an empirical quadrature indicator answer different
questions; none should silently inherit the strongest label of the others.
