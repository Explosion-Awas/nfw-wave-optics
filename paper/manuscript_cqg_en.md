# Direct frequency-domain evaluation of single-image NFW wave-optics amplification with low per-lens preparation cost

**GPT-6.1sol Max & Hermes**

Code and numerical records: [https://github.com/Explosion-Awas/nfw-wave-optics](https://github.com/Explosion-Awas/nfw-wave-optics)

## Abstract

Lens-parameter scans require a new complex transfer function at each dimensionless lens configuration, making preparation and first-band evaluation a distinct computational workload. We construct a direct frequency-domain evaluator for the scalar thin-lens amplification of a standard untruncated Navarro–Frenk–White profile. An explicit modified-Bessel endpoint integral is combined with a complete complex straight-line integral that retains the lens-potential exponential. Starting from the original two-dimensional Gaussian-Abel integral, we prove the finite-contour decomposition, including its retained vertical tail, and then show how extending the line absorbs that tail exactly once. The identities hold for positive frequency and nonzero source momentum. The implementation is tested on 408 frequency samples from 24 lens configurations and six complete 801-frequency bands. All six bands satisfy a $10^{-10}$ full-complex error target and a one-percent integrated diffraction-residual target with the stated empirical reference margins. On these bands, the numerical representation meets a joint complex-field and diffraction-residual target beyond the accuracy of the tested image-plus-cusp and secant approximations. The timing comparison uses a common $10^{-8}$ full-complex and one-percent residual target. Paired comparisons on four of the six tested bands give median preparation-plus-first-band cost ratios of 3.66–6.34 in favor of the direct evaluator. The resulting evaluator targets first-band calculations for changing dimensionless lens models.

## 1. Introduction

Coherent gravitational lensing requires a complex transfer function, including its frequency-dependent phase. Weak-lensing wave-optics features have been studied as a source of information about lenses, with lens-model dependence and parameter degeneracies affecting the interpretation.[3]

Numerical cost depends on which quantities change between requests. A frequency-domain table for a fixed lens can make later evaluations inexpensive. A scan over lens strength and source offset instead asks for a new transfer function at each new dimensionless pair. This paper studies the preparation and first-band cost of that latter operation. Section 7 specifies which parameter changes require a new dimensionless response.

Complex deformations and nonperturbative resummations are established approaches to lensing diffraction integrals.[1]

Endpoint/saddle organization and efficient constructions of steepest-descent contours also precede this work.[2]

Single-image time-domain descriptions provide another nonperturbative treatment of wave-optics features.[3]

GLoW implements time-domain methods, singular/regular splitting, Fourier transforms and interpolation, including support for NFW lenses.[4][5][7]

We provide a specialized organization for the NFW potential: an explicit $K_0$ endpoint contribution, a straight line on the correct analytic sheet, direct treatment of the weak complex residual, and an empirical refinement/fallback rule. It avoids a finite expansion of the potential exponential and does not require a generic contour-flow search. We derive this representation directly from the original diffraction integral.

The analytical weak-lensing prescription of Wu et al. is a separate relevant comparison. Its central potential is replaced by a frequency-dependent secant construction, combined with a local image correction.[8] We reproduce the v2 formula with exact image geometry, distinguish it from an image2 replacement, and use the same frozen complex reference values for all comparisons. Accuracy of a finite asymptotic approximation and cost of a numerical method are assessed separately.

## 2. Model, phase convention and observables

Use the scale radius as lens-plane unit and write

\[
\psi(z)=2\kappa_s\left[\operatorname{Log}^2(z/2)+\arccos^2(1/z)\right],\qquad\psi(0)=0.
\tag{1}
\]

On real radii this is the conventional NFW normalization used in Wu et al.'s Eq. (33); on complex radii the sheet is continued from the positive real axis.[8] The apparent singularity at $z=1$ is removable. Small radii require a stable series, because subtracting the two large logarithmic squares loses precision. The left part of a complete contour must not be evaluated by replacing $\psi(z)$ with $\psi(-z)$ or $\psi(\sqrt{z^2})$.

Our convention for the amplification is

\[
F_+(w,y;\kappa_s)=\frac{w}{2\pi i}\int_{\mathbb R^2}
 e^{iw[|\boldsymbol x-\boldsymbol y|^2/2-\psi(x)-t_m]}\,d^2\boldsymbol x,
\quad w>0,\quad q=wy>0,
\tag{2}
\]

with a radial Gaussian-Abel definition and the exact minimum $t_m$ of the unshifted Fermat potential. We omit the subscript $+$ in numerical comparisons. The positive minimum image satisfies

\[
r_m-\psi'(r_m)=y,\qquad
\lambda_r=1-\psi''(r_m),\quad\lambda_t=y/r_m,\qquad
A=(\lambda_r\lambda_t)^{-1/2}.
\tag{3}
\]

The implementation checks the single-image geometry. All methods use the same minimum-delay convention and the same real GO amplitude $A$. We compare the full complex $F$. Define the common diffraction residual $R=F-A$ and the central delay $\tau_c=y^2/2-t_m$. The phase, amplitude and time-zero convention are fixed independently of the numerical references.

The central Fourier momentum $q$ and the image-separation parameter $s=wy^2/2=qy/2$ are different. A large value of $s$ does not imply a large $q$ or make a high-$q$ cusp expansion uniform.

## 3. Exact representation and numerical evaluation

### 3.1 From the original integral to a finite contour

For $t>0$, define

\[
h(t)=\log\frac{1+\sqrt{1+t^2}}{2},\qquad
 a(t)=2\kappa_s h(t)[2\log(t/2)-h(t)],\qquad
 b(t)=2\pi\kappa_s h(t).
\tag{4}
\]

Then $\psi(it)=a+ib$ and $\psi(-it)=a-ib$. A stable implementation evaluates $h$ with `log1p`. The standard Hankel–modified-Bessel connection supplies the kernel $K_0$.[6] Define

\[
\mathcal C=\frac{2w}{\pi}\int_0^\infty tK_0(qt)
 e^{-iw[t^2/2+a(t)]}\sinh[wb(t)]\,dt,
\tag{5}
\]
\[
T_r=\frac{w}{\pi}\int_r^\infty tK_0(qt)
 e^{-iw[t^2/2+a(t)]-wb(t)}\,dt.
\tag{6}
\]

For arbitrary $r>0$, let $\Gamma_r$ be the path $z=r+(1+i)u$, $-r\le u<\infty$, oriented from $-ir$ through $r$ into the upper-right sector. With

\[
I_{\Gamma_r}=-\frac{iw}{2}e^{iw\tau_c}\int_{\Gamma_r}
 zH_0^{(2)}(qz)e^{iw[z^2/2-\psi(z)]}\,dz,
\]

the exact finite-contour identity is

\[
F=I_{\Gamma_r}+e^{iw\tau_c}(\mathcal C+T_r).
\tag{7}
\]

Appendix A proves this statement directly from Eq. (2). First, for $q>0$, the two real-axis Hankel integrals have oscillatory tails $O(R^{-1/2})$; a real-variable Abel argument then identifies them with the regulated physical integral. After removing that regulator on the real axis, the first Hankel branch rotates to the positive imaginary axis. The second follows $0\to-ir\to\Gamma_r$, with only its far end rotated upwards. The proof explicitly controls the small origin arcs, the real-axis angular boundary layers and the imaginary-axis endpoint. The regulator is removed on the real axis before contour deformation.

### 3.2 Complete straight line and tail cancellation

Extend $\Gamma_r$ to $L_r:z=r+(1+i)u$, $-\infty<u<\infty$, on the lower-half-plane continuation of the same potential sheet. Its left extension satisfies

\[
I_{L_r}-I_{\Gamma_r}=e^{iw\tau_c}T_r,
\qquad
F=I_{L_r}+e^{iw\tau_c}\mathcal C.
\tag{8}
\]

The complete line absorbs the tail in Eq. (7) exactly once and retains the full potential exponential. The line is generally not a thimble of the full integrand.

Using $H_0^{(2),e}(u)=e^{iu}H_0^{(2)}(u)$, the numerically useful equivalent form is

\[
I_{L_r}=-\frac{iw}{2}\int_{L_r}zH_0^{(2),e}(qz)
 e^{iw[(z-y)^2/2-\psi(z)-t_m]}\,dz.
\tag{9}
\]

Completing this square before evaluation avoids unnecessarily subtracting large source-offset phases. The exact representation and its proof apply at positive $w$ and $q$.

### 3.3 Quadrature and empirical control

The numerical line passes through $r=r_m$. The local quadratic factor provides a Gaussian carrier, while the remaining integrand retains the full potential and scaled Hankel function. Gauss–Hermite quadrature evaluates the complete line. The residual is formed inside the integral using an analytically integrated free control, rather than relying only on a final subtraction of two nearly equal order-one numbers. Free-lens normalization is checked without adjustment.

The endpoint calculation uses stable exponential differences, subtraction of known small terms, and a downward-rotated contour with a near-endpoint panel and a shifted-Laguerre tail. Rotation analytically continues the functions in Eq. (4); at complex $t$, $a$ and $b$ no longer mean the real and imaginary parts of the potential.

The controller compares 48- and 96-node residuals, uses a doubled difference with a recorded phase-roundoff allowance, and can refine to 192 nodes or fall back to a stable split-Hankel reference. Its supported empirical box is $y\in[8,256]$, $\kappa_s\in[0.03,0.7]$, $q\in[0.3,12]$. The refinement indicator is calibrated empirically on this box. The implementation was fixed before reference validation.

## 4. Analytical comparisons

Let $p_n=\psi^{(n)}(r_m)$ and define

\[
I_1=A(1+ic_1/w),\qquad I_2=A(1+ic_1/w+c_2/w^2),
\]
\[
c_1=\frac{p_4}{8\lambda_r^2}+\frac{5p_3^2}{24\lambda_r^3}
 +\frac{p_3}{4r_m\lambda_r^2}
 +\frac{\lambda_r-\lambda_t}{8r_m^2\lambda_r\lambda_t}.
\tag{10}
\]

The first correction is the general bGO coefficient used by Wu et al.[8] The real coefficient $c_2$, with its sign included, is specified in Appendix B. Both image terms use the displayed polynomial truncation.

Define the displayed high-momentum central terms

\[
C_2=\frac{4\kappa_s}{w^2y^4},\qquad
C_3=\frac{32i\kappa_s}{w^3y^6}
 \left[-1+2\kappa_s\left(\log q+\gamma_E-\frac32\right)\right].
\tag{11}
\]

The leading NFW cusp term is an established result reproduced and attributed in Wu v2, Eq. (36).[8] Here $C_3$ specifies the next central term retained in the strengthened comparison. For the Wu secant prescription, set

\[
r_*=\frac{\sqrt2}{wy},\qquad
 f(q)=\frac{\psi(r_*)-\psi(0)}{r_*},\qquad
 C_W=\frac{f(q)}{wy^3}
     =\frac{\psi(\sqrt2/q)-\psi(0)}{\sqrt2\,y^2}.
\tag{12}
\]

The four full-field comparisons are

| Label | Full complex amplification | Status |
|---|---|---|
| Q1 | $I_1+e^{iw\tau_c}C_2$ | Leading image-plus-cusp comparison |
| Q2 | $I_2+e^{iw\tau_c}(C_2+C_3)$ | Strengthened image-plus-cusp comparison |
| W1 | $I_1+e^{iw\tau_c}C_W$ | Wu v2 published prescription, exact image geometry |
| W2 | $I_2+e^{iw\tau_c}C_W$ | Image2 substitution ablation |

W1 implements Wu v2 Eqs. (28), (30), (31)/(40), with the secant slope in Eq. (12) and exact image geometry.[8]

Q2 retains the image through $w^{-2}$ and the central sector through its next imaginary/logarithmic term $C_3$.

Q1 and Q2 use local high-frequency and central high-$q$ asymptotics. Wu v2 instead requires a finite central potential, a single image, $y\gg1$ and $wy^2/2\gg1$; its domain includes low-$w$/large-$y$ applications.[8] The common bands span finite $q$, including portions outside the strongly asymptotic regime. The comparison measures the full complex field and the integrated diffraction residual.

For a numerical perturbative comparison, B3 denotes the third-order full Fresnel–Born truncation with exact minimum-gauge phase transport. This phase-resummed truncation retains the free Fresnel phase.

## 5. Accuracy on the common parameter set

### 5.1 Data and reference uncertainty

The reference data contain 408 frequency samples from 24 lens configurations and six preselected 801-frequency bands. Together these are 5214 frequency cases grouped into 30 lens pairs. All methods are compared on this common reference set.

The references use stabilized split-Hankel contours with different shapes/tolerances and partial independent unsplit real-axis $J_0$ checks. There are 72 such checks in the point set and 54 in the six bands. The saved uncertainty $\sigma_j$ combines observed consistency/refinement information and reported numerical estimates. We use it as an empirical consistency margin.

For the complete bands use the original nodes with trapezoidal $d\log w$ weights. The residual and error tests are

\[
M_+=\max_j\bigl(|F_j-F_j^{\rm ref}|+\sigma_j\bigr),\qquad
\rho_+=\frac{\|F-F^{\rm ref}\|+\|\sigma\|}
                 {\|F^{\rm ref}-A\|-\|\sigma\|},
\quad\|g\|^2=\int|g|^2d\log w.
\tag{13}
\]

The denominator must be strictly positive. The primary target is $M_+\le10^{-8}$ and $\rho_+\le0.01$; the secondary target changes the absolute condition to $10^{-10}$. Point counts use the corresponding absolute-complex condition; the band criterion uses the integrated residual norm. All six denominators are positive, including the weakest band. All reported targets use the empirical reference margins defined above.

The band parameters are:

| Band | $\kappa_s$ | $y$ | $q$ range | $w y^2/2$ range |
|---:|---:|---:|---:|---:|
| 0 | 0.047 | 11.3 | 0.31–3.1 | 1.7515–17.515 |
| 1 | 0.183 | 27.7 | 0.47–5.3 | 6.5095–73.405 |
| 2 | 0.327 | 59.4 | 0.37–11.7 | 10.989–347.49 |
| 3 | 0.493 | 123.6 | 0.33–3.4 | 20.394–210.12 |
| 4 | 0.637 | 187.9 | 2.1–11.3 | 197.3–1061.64 |
| 5 | 0.061 | 249.1 | 8.2–11.8 | 1021.3–1469.69 |

### 5.2 Common analytical and numerical comparisons

The common-set comparison below reports the uncertainty-margin decisions. The complete complex values, raw errors, uncertainty margins and all 48 method–band combinations are retained in the supplementary records. Label C denotes the controlled numerical solver; the fixed-cut and stricter-controller variants also meet every listed point and band target.

| Method | Points at $10^{-8}$ / 408 | Points at $10^{-10}$ / 408 | Bands at primary target / 6 | Bands at secondary target / 6 |
|---|---:|---:|---:|---:|
| Q1 | 1 | 0 | 0 | 0 |
| Q2 | 1 | 0 | 0 | 0 |
| W1 | 8 | 0 | 0 | 0 |
| W2 | 8 | 0 | 0 | 0 |
| B3 | 196 | 90 | 1 | 1 |
| C | 408 | 408 | 6 | 6 |

The corresponding $\rho_+$ values are:

| Band | Q1 | Q2 | W1 | W2 | B3 | C |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 7.5364 | 19.28 | 0.49693 | 0.5336 | 2.9477e-05 | 1.8705e-10 |
| 1 | 3.7116 | 5.0259 | 0.26165 | 0.26369 | 0.0018217 | 2.9728e-10 |
| 2 | 4.9386 | 6.1266 | 0.21114 | 0.21122 | 0.0070876 | 1.0917e-09 |
| 3 | 5.6542 | 6.3698 | 0.18286 | 0.18283 | 0.0013454 | 1.6796e-09 |
| 4 | 0.49185 | 0.4921 | 0.23272 | 0.23272 | 0.058109 | 2.6188e-08 |
| 5 | 0.061701 | 0.061642 | 0.034003 | 0.034003 | 0.00031618 | 3.3799e-06 |

This table reports the residual condition; the joint decisions also include the absolute-complex maximum. B3 meets the complete joint target on band 5. The controlled solver meets both conditions on all six bands.

Band 5 illustrates why the two tests are kept separate. Every one of its 801 nodes passes the primary absolute condition for all four analytic methods, yet their residual errors are approximately 6.17%, 6.16%, 3.40% and 3.40%. The one-percent residual criterion therefore resolves an accuracy difference that the absolute-complex criterion alone would miss.

The controlled solver's largest uncertainty-adjusted complex discrepancy across the six bands is about $2.40\times10^{-12}$; its largest $\rho_+$ is $3.38\times10^{-6}$. The reported discrepancies include the empirical reference margins. Refining band 2 from 801 to 1601 nodes changes the reference residual RMS by a fractional $5.53\times10^{-5}$.

### 5.3 Contour identity and tail accounting

The finite-contour calculation was independently checked at four mathematical anchors, including a free lens and nonzero strengths. A direct $\Gamma_r$ integral, the explicit cut and the retained tail agree with the original unsplit-$J_0$ reference to a maximum observed absolute difference $1.314\times10^{-14}$. Two requested tolerances give a maximum observed component change $7.50\times10^{-13}$.

At the free anchor $(q,w,\kappa_s,r)=(0.3,0.1,0,0.5)$, $T_r=0.1325131391500408-0.1167886924374186i$. Omitting or double-counting its phased contribution causes an error about $0.1766333$, although the correctly assembled result is one to the observed quadrature precision. This check probes a substantial endpoint contribution. Appendix A establishes the identity, and these checks verify its implementation and tail accounting.

## 6. Preparation cost for changing lens parameters

### 6.1 Timing protocol

The optimized native comparison uses the recorded single-thread WSL environment, Python 3.11.17, NumPy 1.26.4, SciPy 1.15.3 and GSL 2.8, with GLoW fixed at commit `7cde9749283021814f5a905720884872d02420a7`. Its NFW normalization is `xs=1`, `psi0=4*kappa_s`. The exercised C/GSL quadrature is `qag31`. Build flags, effective settings and all accuracy results accompany the numerical records.

Cold cost is object preparation plus the first complete 801-frequency evaluation. It includes the controller and any lazy work actually invoked, but symmetrically excludes imports, process/WSL launch, serialization, disk I/O, provenance hashing and reference scoring. This definition isolates the transfer-function calculation. Timing ratios compare configurations that meet the stated accuracy target.

GLoW settings were optimized over a bounded set of publicly supported time windows, time-grid densities, FFT frequency coverage and interpolation-density settings on the same six bands. All tested configurations and their accuracy results are supplied with the numerical records. FFT calculations meet the primary joint target on all six bands.

### 6.2 Fixed-setting paired observations

For bands 1–4, the selected settings were measured in four interleaved, order-reversed pairs per band. The 32 complete-band evaluations all meet the primary target. The configurations and reference values were fixed throughout the paired measurements.

| Band | GLoW cold median / s | Cut cold median / s | Median paired GLoW / cut | Paired range |
|---:|---:|---:|---:|---:|
| 1 | 0.630964 | 0.167276 | 3.659 | 3.604–3.928 |
| 2 | 0.960528 | 0.150978 | 6.150 | 5.840–6.598 |
| 3 | 0.705753 | 0.166315 | 3.928 | 3.066–4.558 |
| 4 | 1.040994 | 0.160012 | 6.342 | 5.712–7.554 |

For these four bands, direct evaluation reduces preparation-plus-first-band cost by factors of 3.66–6.34 at the same primary accuracy target. The paired-ratio column is computed from the individual pairs; its ranges cover the four repetitions for each fixed configuration on this runtime.

Cold cost also depends on the lens configuration. A separate four-pair measurement on band 0 gives a median cut/GLoW ratio of 1.313. On band 5, the archived higher-grid native configuration meets the joint target, whereas a separate lower-grid trial gives $\rho_+=0.010044573839>0.01$. The paired cold-cost comparison above covers bands 1–4.

For repeated evaluation at a fixed prepared lens, native FFT interpolation provides the lower-cost reuse path: accepted complete-band reevaluation is about 493–794 times faster than the controlled solver in the measured warm configurations. The implementation evaluates a prepared Fourier grid by interpolation.[7] If lens and frequency vector are unchanged, both methods may instead return a complete cached answer. The preparation and reuse results therefore identify complementary operating regimes.

## 7. Application to changing lens parameters

### A transfer-function kernel inside a lens-model calculation

Weakly lensed gravitational waves have motivated lens-parameter inference, with model dependence and parameter degeneracies remaining distinct from the numerical calculation of the amplification factor.[3]

A concrete computational use case is a likelihood scan or a template bank over the dimensionless NFW strength and source offset. The following equations specify the interface of the measured transfer-function kernel to such a calculation. For angular-diameter distances $D_L,D_S,D_{LS}$, scale radius $r_s$, characteristic density $\rho_s$, and source angle $\beta$, write

\[
\Sigma_{\rm cr}=\frac{c^2D_S}{4\pi G D_LD_{LS}},\qquad
\kappa_s=\frac{\rho_s r_s}{\Sigma_{\rm cr}},\qquad
y=\frac{D_L\beta}{r_s},
\]
\[
t_s=\frac{(1+z_L)r_s^2D_S}{cD_LD_{LS}},\qquad w=2\pi f t_s.
\]

These expressions use $r_s$ as the lens-plane coordinate scale and $t_s$ as the physical time scale.[3]

Let the waveform Fourier transform be $\widetilde h(f)=\int h(t)e^{-2\pi i f t}dt$. The response needed by that convention is

\[
G(f;\theta_L)=F_+^*(2\pi f t_s,y;\kappa_s),\qquad
\widetilde h_L=G\,\widetilde h_0(f;\theta_S),
\]

where $F_+$ is the $\exp(+iwT)$ amplification calculated and compared throughout this paper. A delayed signal has the negative Fourier phase in this waveform convention. This Fourier convention requires conjugation, as also documented in the pinned GLoW source.[5]

With a minimum-delay transfer function, the waveform arrival-time parameter refers to the minimum-image arrival. Restoring an absolute lens delay instead requires its corresponding physical phase. The same arrival-time convention is used in every transfer-function comparison.

For example, a likelihood evaluation uses

\[
\log\mathcal L=-\tfrac12(d-Gh_0\mid d-Gh_0)+\mathrm{constant},\qquad
(a\mid b)=4\operatorname{Re}\int\frac{a(f)b^*(f)}{S_n(f)}\,df.
\]

A proposal changing $\kappa_s$ or $y$ changes $\psi(x)$ or the source geometry in the diffraction integral. It therefore changes the dimensionless transfer function and the equal-delay distribution. Computing the response at the new pair requires a fresh evaluation or a separately validated parameter-space approximation. The existing time-domain/FFT organization prepares and evaluates a frequency-domain object associated with a fixed time-domain lens object.[5][7]

The present solver addresses the cost component

\[
T_F^{\rm new\ pair}=T_{\rm prepare}+T_{\rm first\ band},
\]

without constructing a long equal-delay grid for each new pair. A template-bank construction that requests one band for each distinct pair is another instance of exactly this workload. The six-band measurements quantify this new-pair transfer-function cost.

### Cases where preparation can be reused

| Change requested | Reuse that remains available | Appropriate interpretation |
|---|---|---|
| Only source-waveform parameters change; lens and frequency vector stay fixed | Reuse the complete stored $F$ vector for either method | Reuse the cached transfer-function vector |
| Same $(\kappa_s,y)$, new frequencies inside a validated table range | Reuse the prepared frequency object; FFT interpolation can be very cheap | Use the measured warm-evaluation cost |
| Only $t_s$ changes while $(\kappa_s,y)$ remain fixed | Rescale $w=2\pi f t_s$ and reuse the same dimensionless object if coverage/resolution suffice | Reuse depends on dimensionless shape and frequency coverage |
| $\kappa_s$ or $y$ changes | Prepare a new exact dimensionless object, or introduce a separately validated parameter-space interpolant/surrogate | The one-new-pair workload measured here |

Here a source-only update leaves the response coordinates $(\kappa_s,y,t_s)$ fixed. Source-redshift or distance changes are classified through their effect on those coordinates.

At fixed distances, an explicit reusable family is $r_s\mapsto a r_s$, $\rho_s\mapsto\rho_s/a$, $\beta\mapsto a\beta$. It leaves $(\kappa_s,y)$ unchanged and sends $t_s\mapsto a^2t_s$. It changes only the frequency map. The coordinate relations and Fourier sign fix the waveform interface and its reuse conditions.

For $K$ different frequency requests at a fixed dimensionless pair, the relevant cost is instead

\[
T_F^{\rm fixed\ pair}=T_{\rm prepare}+\sum_{j=1}^K T_{{\rm evaluate},j}.
\]

Prepared FFT interpolation serves this repeated-frequency workload. Exact answer caching, adaptive reuse and parameter-space emulation can further reduce the number of new-pair evaluations.

### Transfer-function cost within the full calculation

If the transfer-function calculation occupies fraction $p$ of the original total iteration cost and its own speedup is $s$, leaving all other work unchanged gives

\[
S_{\rm total}=\frac{1}{1-p+p/s}.
\]

The six-band measurements determine the transfer-function cost component. Evaluating the total speedup additionally requires the application-specific fraction $p$ and reuse pattern, together with waveform generation, likelihood contractions and sampling overhead. The transfer-function timings include any controller fallback actually invoked.

The validation measure $d\log w$ and the one-percent relative diffraction-residual target quantify numerical accuracy of the transfer function. An application can combine this kernel with astrophysical priors, physical lens-model checks and detector-weighted waveform tolerances, using frequencies within the tested domain or a separately validated fallback.

## 8. Discussion and conclusion

The exact NFW endpoint/line representation connects the original regulated diffraction integral to a direct frequency-domain calculation. Appendix A establishes this connection, including the retained tail and its absorption into the complete line. The stabilized implementation meets the joint complex-field and diffraction-residual targets on all six tested bands with empirical reference margins.

The method specializes established contour and regular-part constructions to the NFW potential.[1][2][4] Its explicit endpoint term and complete sheet-correct line retain the lens-potential exponential while supporting direct evaluation of the weak diffraction residual.

For the new-dimensionless-pair workload in Section 7, the paired measurements on bands 1–4 give preparation-plus-first-band cost ratios of 3.66–6.34 at the same primary accuracy target. The common-set analytical comparisons identify the accuracy gained by retaining the full diffraction integral at finite $q$.

The explicit endpoint, complete sheet-correct line and direct residual quadrature supply a reproducible computational option for repeated changes of the dimensionless lens model.

## Data and code availability

The accompanying reproducibility archives contain the scientific source, reference and comparison values, all tested optimized-library settings, and replay instructions. The recorded native C/GSL environment supports timing reproduction, while saved-value replay supports independent checking of the numerical comparisons.

## Appendix A. Proof from the original Abel integral

### A.1 Statement and conventions

Let \(w>0\), \(q=wy>0\), and \(k=\kappa_s\geq0\). Fix the NFW potential gauge \(\psi(0)=0\), with
\[
 \psi(z)=2k\left\{\operatorname{Log}^2(z/2)
                    +\arccos^2(1/z)\right\}.                 \tag{A.1}
\]
In the right half-plane this means the analytic sheet continued from positive real radii, not an even reflection of a complex radius. The apparent singularity at \(z=1\) is removable. Boundary values on the imaginary axis are understood from that half-plane. For the whole-line corollary below, continue the same sheet into the open lower half-plane, without crossing the negative real axis.

For \(\boldsymbol y=(y,0)\), let \(t_{\min}\) denote the real minimum of \(|\boldsymbol x-\boldsymbol y|^2/2-\psi(|\boldsymbol x|)\), and set \(\tau_c=y^2/2-t_{\min}\). Define the original two-dimensional integral by radial Gaussian Abel regularization:
\[
 F=\lim_{\eta\downarrow0}F_\eta,\qquad
 F_\eta=\frac{w}{2\pi i}\int_{\mathbb R^2}
 e^{-\eta|\boldsymbol x|^2}
 e^{iw[|\boldsymbol x-\boldsymbol y|^2/2-\psi(|\boldsymbol x|)-t_{\min}]}
 \,d^2\boldsymbol x .                                      \tag{A.2}
\]
The proof also works for any fixed real choice of the time-zero constant. It does not require a unique image, a nondegenerate stationary point, or a stationary choice of contour intercept.

For an arbitrary fixed \(r>0\), orient
\[
 \Gamma_r:\quad z=r+(1+i)u,\quad -r\leq u<\infty,
\]
from \(-ir\), through \(r\), to the upper-right decaying sector, and define
\[
 I_{\Gamma_r}=-\frac{iw}{2}e^{iw\tau_c}
 \int_{\Gamma_r}zH_0^{(2)}(qz)
       e^{iw[z^2/2-\psi(z)]}\,dz.                            \tag{A.3}
\]
For \(t>0\), put
\[
 \begin{split}
 h(t)&=\log\frac{1+\sqrt{1+t^2}}2,\qquad L_t=\log(t/2),\\
 a(t)&=2kh(t)[2L_t-h(t)],\qquad b(t)=2\pi kh(t),\\
 d(t)&=t^2/2+a(t),
 \end{split}                                                \tag{A.4}
\]
and
\[
 \begin{split}
 C(w,q)&=\frac{2w}{\pi}\int_0^\infty
       tK_0(qt)e^{-iwd(t)}\sinh[wb(t)]\,dt,\\
 T_r(w,q)&=\frac{w}{\pi}\int_r^\infty
       tK_0(qt)e^{-iwd(t)-wb(t)}\,dt .
 \end{split}                                                \tag{A.5}
\]

**Theorem.** The Abel limit (A.2) exists and
\[
 \boxed{F=I_{\Gamma_r}+e^{iw\tau_c}\{C(w,q)+T_r(w,q)\}.}     \tag{A.6}
\]
All contour and \(K_0\) integrals on the right-hand side are absolutely convergent. The estimates used in the proof hold for fixed parameters and can be chosen locally uniformly when \(w,q,r\) range over positive compact sets and \(k\) over \([0,K]\). No uniform limit at \(q=0\) or \(w=0\), and no uniform numerical quadrature error estimate, is asserted.

### A.2 Analytic sheet, endpoint control, and elementary Bessel bounds

#### A.2.1 The potential on the required domains

The two limits of \(\arccos(1/z)\) across the interval \(0<z<1\) differ by sign, so their squares agree and extend analytically across the interval. At \(z=1\), the local inverse of \(\cos\sqrt v=1-v/2+O(v^2)\) shows that \(\arccos^2(1/z)\) is analytic. Together with the principal logarithm, these continuations specify a holomorphic function throughout \(\Re z>0\). It is real on the positive real axis, so Schwarz reflection gives the conjugate boundary values on the two imaginary half-axes. In the open lower half-plane \(1/z\) lies in the open upper half-plane, where the chosen inverse cosine has no cut. Thus (A.1) also defines the required lower-half-plane continuation; it is not \(\psi(-z)\).

For completeness, write \(\ell(z)=\operatorname{Log}(2/z)\) and
\[
 \delta(z)=\operatorname{Log}\frac{1+\sqrt{1-z^2}}2
          =-z^2/4+O(z^4)
\]
near zero, with the square root near \(+1\). Continuation of the positive-real identity gives
\[
 \psi(z)=2k\{\ell(z)^2-[\ell(z)+\delta(z)]^2\}
        =kz^2\ell(z)+O\bigl(k|z|^4[1+|\log|z||]\bigr).     \tag{A.7}
\]
This estimate is uniform in the closed right-half-plane sectors used below. In particular, \(\psi(z)\to0\) and \(e^{-iw\psi(z)}\) is bounded there at zero. Consequently the small circular arcs in either right quadrant contribute
\[
 O\bigl(\epsilon^2[1+|\log\epsilon|]\bigr)\longrightarrow0, \tag{A.8}
\]
because the zero-order Hankel functions have only a logarithmic singularity. The latter bound follows directly from the regular and logarithmic Frobenius solutions of the order-zero Bessel equation. The small arcs have length \(O(\epsilon)\), while the factor \(zH_0^{(j)}(qz)\) is \(O(\epsilon[1+|\log\epsilon|])\).

At large radius, the expansion of the analytic inverse cosine at zero gives, uniformly on the closed angular sectors in this proof,
\[
 \begin{split}
 \psi(z)&=2k\operatorname{Log}^2(z/2)+\frac{k\pi^2}{2}
                         +O(k/|z|),\\
 \Im\psi(Re^{i\theta})&=4k\theta\log(R/2)+O(k/R).
 \end{split}                                                \tag{A.9}
\]
In particular,
\[
 \psi'(x)=\frac{4k\log(x/2)}x+O(k/x^2),\qquad
 \psi''(x)=\frac{4k[1-\log(x/2)]}{x^2}+O(k/x^3).             \tag{A.10}
\]
Here differentiating the analytic large-radius expansion is legitimate on a slightly larger fixed sector. These estimates control the potential's contribution on the large arcs. On real radii, (A.7) makes the delay continuous at the origin, while (A.9) makes its quadratic term dominate at infinity. The real two-dimensional delay is therefore coercive and attains the minimum used in (A.2); no image classification is needed for its existence.

#### A.2.2 Bounds including the angular endpoints

The zero-order Hankel--modified-Bessel connection formulas used here are
\[
 H_0^{(1)}(it)=-\frac{2i}{\pi}K_0(t),\qquad
 H_0^{(2)}(-it)=\frac{2i}{\pi}K_0(t),\quad t>0.              \tag{A.11}
\]
These are the zero-order instances of the standard connection formula.[6]

The large-argument bounds below include the angular endpoints of the Gaussian decay sectors. Indeed, for \(|\arg\zeta|<\pi\),
\[
 K_0(\zeta)=e^{-\zeta}\sqrt{\frac{\pi}{2\zeta}}B(\zeta),
 \quad B(\zeta)=\frac1{\sqrt\pi}\int_0^\infty
 e^{-s}s^{-1/2}\left(1+\frac{s}{2\zeta}\right)^{-1/2}ds.    \tag{A.12}
\]
For positive \(\zeta\), this follows from \(K_0(\zeta)=\int_0^\infty e^{-\zeta\cosh v}dv\) by \(s=\zeta(\cosh v-1)\). Both sides continue analytically to the cut plane, which proves the stated version. All square roots in (A.12) use the corresponding principal branches. For \(|\arg\zeta|\leq3\pi/4\),
\(\left|1+s/(2\zeta)\right|\geq1/\sqrt2\) for \(s\geq0\). Thus \(B\) is uniformly bounded there. The integral, or its derivative, also gives
\[
 B(\zeta)=1+O(|\zeta|^{-1}),\qquad
 B'(\zeta)=O(|\zeta|^{-2})                                 \tag{A.13}
\]
as \(|\zeta|\to\infty\) in these sectors. For example, subtracting 1 in the integrand and integrating its derivative along the segment from 0 to \(s/(2\zeta)\) gives the first estimate; differentiation under the exponentially weighted integral gives the second.

Using (A.11) by analytic continuation, (A.12) yields
\[
 \begin{split}
 H_0^{(1)}(qz)&=\sqrt{\frac{2}{\pi qz}}\,
       e^{i(qz-\pi/4)}B(-iqz),\\
 H_0^{(2)}(qz)&=\sqrt{\frac{2}{\pi qz}}\,
       e^{-i(qz-\pi/4)}B(iqz).
 \end{split}                                                \tag{A.14}
\]
The first formula applies in the first quadrant. The second applies to the upper wedge \(0\leq\arg z\leq\pi/4\) and the lower sectors used below. In particular, it supplies bounds right up to the real and imaginary endpoints.

### A.3 From the two-dimensional Abel integral to ordinary Hankel integrals

For each \(\eta>0\), (A.2) is absolutely integrable, so polar coordinates and the exact angular integral give
\[
 F_\eta=-iw e^{iw\tau_c}\int_0^\infty
 xJ_0(qx)e^{-\eta x^2}e^{iw[x^2/2-\psi(x)]}\,dx.            \tag{A.15}
\]
Define, initially as ordinary improper oscillatory integrals,
\[
 A_\pm=\int_0^\infty f_\pm(x)\,dx,\qquad
 f_+(z)=zH_0^{(1)}(qz)e^{iw[z^2/2-\psi(z)]},\quad
 f_-(z)=zH_0^{(2)}(qz)e^{iw[z^2/2-\psi(z)]}.                \tag{A.16}
\]
They are integrable at zero by (A.7)--(A.8). To prove convergence at infinity, factor the real-axis integrands using (A.14):
\[
 f_\pm(x)=a_\pm(x)e^{ig_\pm(x)},\qquad
 g_\pm(x)=wx^2/2\pm qx-w\psi(x).
\]
Equations (A.13)--(A.14) give \(a_\pm=O(x^{1/2})\) and \(a_\pm'=O(x^{-1/2})\). Equations (A.10) give
\[
 g_\pm'=wx\pm q-w\psi'(x)\geq c x,\qquad g_\pm''=O(1)
\]
for sufficiently large \(x\), with \(c>0\). Integration by parts on \([R,S]\) gives
\[
 \int_R^S f_\pm(x)dx=
 \left[\frac{a_\pm(x)}{ig_\pm'(x)}e^{ig_\pm(x)}\right]_R^S
 -\int_R^S\left(\frac{a_\pm}{ig_\pm'}\right)'e^{ig_\pm}dx.
\]
The boundary amplitude is \(O(x^{-1/2})\), and the differentiated amplitude is \(O(x^{-3/2})\). Therefore the improper limits exist and
\[
 \left|\int_R^\infty f_\pm(x)dx\right|\leq C R^{-1/2}.      \tag{A.17}
\]
Finite-radius stationary points do not affect this exterior argument.

We identify this ordinary limit with (A.15) by a real-variable Abel argument. Write \(P_R(s)=\int_R^s f_\pm(x)dx\). Equation (A.17) implies \(\sup_{s\geq R}|P_R(s)|\leq2CR^{-1/2}\). For the decreasing real weight \(v_\eta(x)=e^{-\eta x^2}\), real-variable integration by parts gives
\[
 \int_R^\infty v_\eta(x)f_\pm(x)dx
       =-\int_R^\infty v_\eta'(x)P_R(x)dx,
 \quad
 \left|\int_R^\infty v_\eta f_\pm\right|\leq2CR^{-1/2}.     \tag{A.18}
\]
This holds uniformly in \(\eta>0\). On each finite interval, dominated convergence removes the regulator. Since \(J_0=(H_0^{(1)}+H_0^{(2)})/2\), (A.15)--(A.18) prove
\[
 F=-\frac{iw}{2}e^{iw\tau_c}(A_++A_-).                     \tag{A.19}
\]
The regulator has now been removed on the real axis, before any contour deformation. A fixed \(e^{-\eta z^2}\) would grow on the imaginary axis like \(e^{+\eta t^2}\).

### A.4 Rotation of the first Hankel branch

Rotate \(A_+\) through the first quadrant. On \(z=Re^{i\theta}\), \(0\leq\theta\leq\pi/2\), (A.9) and (A.14) bound the modulus of the integrand by
\[
 C R^{1/2}\exp\left[-\frac{wR^2}{2}\sin2\theta
                    -qR\sin\theta
                    +4wk\theta\log(R/2)+O(R^{-1})\right].  \tag{A.20}
\]
On \(0\leq\theta\leq\pi/4\), \(\sin2\theta\geq4\theta/\pi\); for sufficiently large \(R\) the exponent is at most \(-cR^2\theta\), apart from a fixed constant. Including the arc length \(R\,d\theta\), this part is \(O(R^{-1/2})\). It resolves the real-axis angular boundary layer explicitly.

On \(\pi/4\leq\theta\leq\pi/2\), the quadratic factor has modulus at most 1 and the Hankel exponential supplies \(e^{-qR/\sqrt2}\). The potential contributes at worst \(C R^{2\pi wk}\). The remaining arc is thus bounded by
\[
 C R^{3/2+2\pi wk}e^{-qR/\sqrt2}\longrightarrow0.           \tag{A.21}
\]
In particular the endpoint near the positive imaginary axis, where the quadratic damping vanishes, is controlled by \(q>0\), not by that damping. The small arc vanishes by (A.8). Cauchy's theorem therefore gives the oriented identity
\[
 A_+=\int_{0}^{i\infty}f_+(z)dz.                            \tag{A.22}
\]

To evaluate it, use
\[
 \arccos(-i/t)=\pi/2+i\operatorname{asinh}(1/t),\quad
 L_t+\operatorname{asinh}(1/t)=h(t).
\]
Substitution into (A.1), with \(\operatorname{Log}(it/2)=L_t+i\pi/2\), gives the exact boundary values
\[
 \psi(it)=a(t)+ib(t),\qquad \psi(-it)=a(t)-ib(t).             \tag{A.23}
\]
Here \(h\geq0\), so \(b\geq0\). Define
\[
 J_+^{\alpha,\beta}=\int_\alpha^\beta
       tK_0(qt)e^{-iwd(t)+wb(t)}dt,\qquad
 J_-^{\alpha,\beta}=\int_\alpha^\beta
       tK_0(qt)e^{-iwd(t)-wb(t)}dt.                          \tag{A.24}
\]
The superscript endpoints may include 0 and \(\infty\). Because \(z\,dz=-t\,dt\) on \(z=it\), (A.11), (A.22), and (A.23) imply
\[
 \boxed{A_+=\frac{2i}{\pi}J_+^{0,\infty}.}                 \tag{A.25}
\]

These axis integrals are absolutely convergent. Near zero, \(h=O(t^2)\), \(a=O(t^2[1+|\log t|])\), \(b=O(t^2)\), and \(tK_0(qt)=O(t[1+|\log t|])\). At infinity, \(h=\log(t/2)+O(t^{-1})\), and (A.12) bounds the integrands by an exponential \(e^{-qt}\) times a fixed power of \(t\). Both axis integrals in (A.24) are therefore absolutely convergent.

### A.5 Finite deformation of the second Hankel branch

For \(A_-\), use the path \(0\to-ir\to\Gamma_r\to\infty\). Only its far end is rotated upwards, into an angle strictly smaller than \(\pi/4\). In particular, the entire second Hankel branch is **not** rotated into the lower half-plane.

The finite deformation can be made with two ordinary Cauchy domains: the lower-right triangle with vertices \(0,-ir,r\), with a small origin indentation, and the upper-right domain between the real ray starting at \(r\) and the ray of \(\Gamma_r\) starting at \(r\). This avoids treating the curve that crosses the real axis at \(r\) as a single simple closed boundary. Both domains use only the analytic right-half-plane sheet established above.

For \(R>r\), let \(P_R\) be the intersection of the outgoing part of \(\Gamma_r\) with \(|z|=R\). Its argument satisfies \(0<\theta_R<\pi/4\). The large connecting arc is bounded, by (A.9) and (A.14), by
\[
 C R^{3/2}\int_0^{\theta_R}
 \exp\left[-\frac{wR^2}{2}\sin2\theta
             +qR\sin\theta+4wk\theta\log(R/2)\right]d\theta.
\]
Since \(\sin2\theta\geq4\theta/\pi\) and \(\sin\theta\leq\theta\), its exponent is at most
\[
 \left[-\frac{2w}{\pi}R^2+qR+4wk\log(R/2)\right]\theta
 \leq-cR^2\theta
\]
for large \(R\). The arc is again \(O(R^{-1/2})\). In particular, the \(H_0^{(2)}\) growth \(e^{qR\sin\theta}\) is dominated by the quadratic decay, including its real-axis boundary layer.

On the outgoing contour itself, \(z=r+(1+i)u\), the combined free and Hankel exponential has modulus
\[
 \left|e^{iwz^2/2-iqz}\right|
       =\exp[-wu^2+(q-wr)u].                               \tag{A.26}
\]
The potential adds at most a fixed power of \(1+u\) by (A.9); hence the outgoing integral is absolutely convergent. The finite segment has no singularity: \(|z|\geq r/\sqrt2\), and \(z=1\), if encountered, is removable.

Let \(A_{\Gamma_r}=\int_{\Gamma_r}f_-(z)dz\). Taking the large and small radius limits of the two Cauchy identities now proves
\[
 A_-=\int_0^{-ir}f_-(z)dz+A_{\Gamma_r}.
\]
For the first term, \(z=-it\) has \(t:0\to r\), \(z\,dz=-t\,dt\), and \(H_0^{(2)}(-iqt)=2iK_0(qt)/\pi\). Therefore
\[
 \boxed{A_-=A_{\Gamma_r}-\frac{2i}{\pi}J_-^{0,r}.}         \tag{A.27}
\]
The minus sign here comes from \(z\,dz\) and gives the positive retained-tail contribution in the final answer.

### A.6 Assembly, the positive vertical tail, and the whole-line corollary

Combining (A.19), (A.25), and (A.27) gives
\[
 \begin{split}
 F&=I_{\Gamma_r}+\frac{w}{\pi}e^{iw\tau_c}
                  \left[J_+^{0,\infty}-J_-^{0,r}\right]\\
  &=I_{\Gamma_r}+\frac{w}{\pi}e^{iw\tau_c}
       \left[J_+^{0,\infty}-J_-^{0,\infty}+J_-^{r,\infty}\right].
 \end{split}                                                \tag{A.28}
\]
By (A.24), \((w/\pi)(J_+^{0,\infty}-J_-^{0,\infty})=C\), whereas \((w/\pi)J_-^{r,\infty}=T_r\). This proves (A.6), including the normalization and the positive sign of \(T_r\). Since \(b\geq0\), one also has the explicit analytic tail bound
\[
 |T_r|\leq\frac{w}{\pi}\int_r^\infty tK_0(qt)dt
          =\frac{wr}{\pi q}K_1(qr).                        \tag{A.29}
\]
The last equality follows by differentiating \(tK_1(qt)\). Equation (A.29) bounds the retained contour tail.

To connect the result to a whole Gaussian line, define
\[
 L_r:\ z=r+(1+i)u,\quad-\infty<u<\infty,\qquad
 I_{L_r}=-\frac{iw}{2}e^{iw\tau_c}\int_{L_r}f_-(z)dz.        \tag{A.30}
\]
Its left extension is oriented from lower-left infinity to \(-ir\). The sweep from that extension to the negative imaginary tail lies in
\(-3\pi/4\leq\arg z\leq-\pi/2\), entirely in the open lower-half-plane sheet. On a joining large circle, \(\sin2\theta\geq0\), \(\sin\theta\leq-1/\sqrt2\), and the leading potential exponent in (A.9) is nonpositive for \(R\geq2\). Equations (A.9) and (A.14) bound its integral by
\[
 C R^{3/2}e^{-qR/\sqrt2}\longrightarrow0.                  \tag{A.31}
\]
Thus the left extension equals \(\int_{-i\infty}^{-ir}f_-(z)dz\). On this oriented axis, \(t\) decreases from \(\infty\) to \(r\), so
\[
 \int_{-i\infty}^{-ir}f_-(z)dz=\frac{2i}{\pi}J_-^{r,\infty},
 \qquad
 \boxed{I_{L_r}-I_{\Gamma_r}=e^{iw\tau_c}T_r.}             \tag{A.32}
\]
Both ends of \(L_r\) are absolutely integrable, by the Gaussian term in (A.26) and the corresponding fixed-power potential bound. Substitution of (A.32) into (A.6) gives the equivalent representation
\[
 \boxed{F=I_{L_r}+e^{iw\tau_c}C(w,q).}                     \tag{A.33}
\]
The finite-contour formula (A.6) retains \(T_r\); the whole-line formula (A.33) has already absorbed it. Adding \(T_r\) to (A.33) would double count the left extension. The potential exponential is retained exactly. The straight line is a convenient exact contour, not in general a steepest-descent contour of the full NFW phase. This completes the proof.

### A.7 Free normalization

For \(k=0\), take \(t_{\min}=0\). Direct Gaussian integration in (A.15) gives
\[
 F_\eta=\frac{-iw}{2(\eta-iw/2)}
       \exp\left[\frac{iwy^2}{2}
                  -\frac{q^2}{4(\eta-iw/2)}\right]
       \longrightarrow1,\qquad q=wy.                      \tag{A.34}
\]
Also \(C=0\), but \(T_r\) generally does not vanish. Thus \(I_{\Gamma_r}+e^{iw\tau_c}T_r=1\) and \(I_{L_r}=1\), providing a nontrivial check of both the contour endpoint and the Fresnel prefactor.

The proof controls exact contour identities and their defining limits for \(w,q,r>0\). The use of positive compact parameter sets in the estimates is essential to the stated uniformity. The name “cut” denotes the combination of the two oriented imaginary-axis endpoint contributions.

## Appendix B. Second image coefficient for the analytical comparisons

For $r=r_m$, $p_n=\psi^{(n)}(r_m)$ and the eigenvalues in Eq. (3), the real coefficient in $I_2=A(1+ic_1/w+c_2/w^2)$ is

\[
\begin{aligned}
c_2={}&-\frac{3(\lambda_r-\lambda_t)(3\lambda_r+5\lambda_t)}{128r^4\lambda_r^2\lambda_t^2}
 +\frac{p_3(\lambda_r-5\lambda_t)}{32r^3\lambda_r^3\lambda_t}
 +\frac{p_4(5\lambda_t-\lambda_r)}{64r^2\lambda_r^3\lambda_t}\\
&-\frac{p_5}{16r\lambda_r^3}-\frac{p_6}{48\lambda_r^3}
 +\frac{5p_3^2(7\lambda_t-\lambda_r)}{192r^2\lambda_r^4\lambda_t}
 -\frac{35p_3p_4}{96r\lambda_r^4}\\
&-\frac{7p_3p_5}{48\lambda_r^4}-\frac{35p_4^2}{384\lambda_r^4}
 -\frac{35p_3^3}{96r\lambda_r^5}
 -\frac{35p_3^2p_4}{64\lambda_r^5}
 -\frac{385p_3^4}{1152\lambda_r^6}.
\end{aligned}
\]

This coefficient specifies the strengthened comparison and the image2 ablation. The image implementation prepares $p_5,p_6$ using a cloned 60-digit differentiation context and evaluates frequency arrays in ordinary arithmetic. General image stationary-phase expansions provide the basis of these analytical comparisons.[3]

## Sources

[1] https://arxiv.org/html/2602.21493v1 — Feldbrugge J, Crew S and Pen U-L. Resurgence and Hyperasymptotics in Wave Optics Astronomy. arXiv:2602.21493v1.
[2] https://arxiv.org/html/2409.12991v1 — Shi X. Acquiring the Lefschetz thimbles: efficient evaluation of the diffraction integral for lensing in wave optics. arXiv:2409.12991v1.
[3] https://arxiv.org/html/2306.05282v2 — Savastano S, Tambalo G, Villarrubia-Rojo H and Zumalacarregui M. Weakly Lensed Gravitational Waves: Probing Cosmic Structures with Wave-Optics Features. arXiv:2306.05282v2.
[4] https://arxiv.org/html/2409.04606v1 — Villarrubia-Rojo H et al. GLoW: novel methods for wave-optics phenomena in gravitational lensing. arXiv:2409.04606v1.
[5] https://github.com/glow-astro/GLoW/blob/7cde9749283021814f5a905720884872d02420a7/freq_domain.py — GLoW developers. freq_domain.py, fixed commit 7cde9749283021814f5a905720884872d02420a7.
[6] https://dlmf.nist.gov/10.27 — NIST Digital Library of Mathematical Functions. Section 10.27, connection formulas, Eq. 10.27.8.
[7] https://github.com/glow-astro/GLoW/blob/7cde9749283021814f5a905720884872d02420a7/freq_domain_c.py — GLoW developers. freq_domain_c.py, fixed commit 7cde9749283021814f5a905720884872d02420a7.
[8] https://arxiv.org/html/2508.17486v2 — Zhao-Feng Wu et al., Weak Lensing Approximation of Wave-optics Effects from General Symmetric Lens Profiles, v2
