# Mathematical Formulations in the Fovea ISP Pipeline

This document details the mathematical models and signal processing formulations implemented in the Fovea Image Signal Processor (ISP) and compiler pipeline.

---

## 1. Sensor Black Level Pedestal Subtraction

Modern CMOS image sensors introduce a digital pedestal (typically 64 or 256 for 10/12-bit ADCs) to prevent dark current noise fluctuations from clipping below zero:

$$
I_{\text{calibrated}}(x, y) = \frac{\max(0, I_{\text{raw}}(x, y) - I_{\text{pedestal}})}{I_{\text{saturation}} - I_{\text{pedestal}}}
$$

where $I_{\text{saturation}} = 2^B - 1$ for a sensor with bit depth $B$.

---

## 2. Radial Lens Shading (Vignetting) Correction

Due to the cosine-fourth law of illumination ($\cos^4 \theta$) and physical barrel vignetting, peripheral pixels receive less irradiance than on-axis pixels. We model lens attenuation as an even-order radial polynomial centered at optical principal point $(c_x, c_y)$:

$$
r^2 = \left(\frac{x - c_x}{R_{\max}}\right)^2 + \left(\frac{y - c_y}{R_{\max}}\right)^2
$$

$$
G(r) = 1.0 + k_0 r^2 + k_1 r^4 + k_2 r^6 + k_3 r^8
$$

$$
I_{\text{corrected}}(x, y) = \min\left(1.0, I(x, y) \cdot G(r)\right)
$$

---

## 3. Dynamic Defective Pixel Correction (DPC)

Cosmic radiation and semiconductor impurities produce defective pixels (dead pixels with zero output or hot pixels with maximal leakage). A pixel $p$ at coordinate $(x, y)$ is classified as an outlier if:

$$
|p - \text{median}(\mathcal{N}_8(p))| > \tau \cdot \sigma_{\text{local}}
$$

where $\mathcal{N}_8(p)$ denotes the 8-connected neighborhood of same-color Bayer samples. Outliers are replaced by the local neighborhood median:

$$
p_{\text{corrected}} = \text{median}(\mathcal{N}_8(p))
$$

---

## 4. Bayer Demosaicing Formulations

### Malvar-He-Cutler (MHC) 5x5 Optimal Linear FIR Filter
Malvar, He, and Cutler derived the optimal minimum mean square error linear filter that exploits inter-channel high-frequency correlation.

#### Green at Red / Blue Positions:
$$
G(i, j) = \frac{1}{8} \left[ 2(C_{0, -1} + C_{0, 1} + C_{-1, 0} + C_{1, 0}) + 2 C_{0, 0} - (C_{0, -2} + C_{0, 2} + C_{-2, 0} + C_{2, 0}) \right]
$$

#### Red at Green Positions (Horizontal Red Neighbors):
$$
R(i, j) = \frac{1}{16} \left[ 8(C_{0, -1} + C_{0, 1}) + 10 C_{0, 0} - 2(C_{0, -2} + C_{0, 2} + C_{-1, -1} + C_{-1, 1} + C_{1, -1} + C_{1, 1}) - (C_{-2, 0} + C_{2, 0}) \right]
$$

#### Red at Blue Positions (Diagonal Cross):
$$
R(i, j) = \frac{1}{16} \left[ 4(C_{-1, -1} + C_{-1, 1} + C_{1, -1} + C_{1, 1}) + 12 C_{0, 0} - 3(C_{0, -2} + C_{0, 2} + C_{-2, 0} + C_{2, 0}) \right]
$$

---

## 5. Color Calibration and Spectral Transformation

### von Kries Illuminant Adaptation (White Balance)
Illuminant chromaticity is compensated using diagonal gain scaling in sensor cone space:

$$
\begin{bmatrix} R_{\text{wb}} \\ G_{\text{wb}} \\ B_{\text{wb}} \end{bmatrix} = \begin{bmatrix} g_r & 0 & 0 \\ 0 & g_g & 0 \\ 0 & 0 & g_b \end{bmatrix} \begin{bmatrix} R \\ G \\ B \end{bmatrix}
$$

Under the Gray World assumption, gains satisfy:

$$
g_r = \frac{\mu_G}{\mu_R}, \quad g_g = 1.0, \quad g_b = \frac{\mu_G}{\mu_B}
$$

### 3x3 Sensor Color Correction Matrix (CCM)
Camera color filter arrays do not satisfy the Luther-Condition (their spectral responses are not linear combinations of CIE color matching functions). A calibrated 3x3 matrix maps sensor RGB into standard sRGB space:

$$
\begin{bmatrix} R_{\text{sRGB}} \\ G_{\text{sRGB}} \\ B_{\text{sRGB}} \end{bmatrix} = \begin{bmatrix} m_{00} & m_{01} & m_{02} \\ m_{10} & m_{11} & m_{12} \\ m_{20} & m_{21} & m_{22} \end{bmatrix} \begin{bmatrix} R_{\text{wb}} \\ G_{\text{wb}} \\ B_{\text{wb}} \end{bmatrix}
$$

with the energy conservation constraint $\sum_{j} m_{ij} = 1$ for neutral gray preservation.

---

## 6. Edge-Preserving Denoising

### Joint Bilateral Filter
The joint bilateral filter uses a high-SNR guidance image $G$ (typically the green channel or luminance) to calculate edge weights for filtering noisy channel $I$:

$$
I_{\text{denoised}}(p) = \frac{1}{W_p} \sum_{q \in \Omega} I(q) \exp\left( -\frac{\|p - q\|^2}{2\sigma_s^2} \right) \exp\left( -\frac{\|G(p) - G(q)\|^2}{2\sigma_r^2} \right)
$$

where normalization factor $W_p$ is:

$$
W_p = \sum_{q \in \Omega} \exp\left( -\frac{\|p - q\|^2}{2\sigma_s^2} \right) \exp\left( -\frac{\|G(p) - G(q)\|^2}{2\sigma_r^2} \right)
$$

### Guided Image Filter
He, Sun, and Tang established an edge-preserving filter assuming a local linear model between guidance $I$ and output $q$:

$$
q_i = a_k I_i + b_k, \quad \forall i \in \omega_k
$$

Minimizing the squared difference with regularization $\epsilon$:

$$
a_k = \frac{\frac{1}{|\omega|} \sum_{i \in \omega_k} I_i p_i - \mu_k \bar{p}_k}{\sigma_k^2 + \epsilon}
$$

$$
b_k = \bar{p}_k - a_k \mu_k
$$

The output pixel is the average of all linear models covering pixel $i$:

$$
q_i = \bar{a}_i I_i + \bar{b}_i
$$

---

## 7. Multi-Exposure Fusion (Mertens et al.)

Given $N$ bracketed exposure frames $I_k$, each pixel is assigned quality weights based on three visual metrics:

1. **Contrast Metric** ($C$): Magnitude of the discrete Laplacian filter:
   $$
   C_k(x, y) = |\nabla^2 I_k(x, y)|
   $$
2. **Saturation Metric** ($S$): Standard deviation across RGB color channels:
   $$
   S_k(x, y) = \sqrt{\frac{1}{3} \sum_{c \in \{R, G, B\}} (I_{k, c}(x, y) - \mu_k(x, y))^2}
   $$
3. **Well-Exposedness Metric** ($E$): Gaussian distance from ideal midtone $0.5$ with standard deviation $\sigma = 0.2$:
   $$
   E_k(x, y) = \prod_{c \in \{R, G, B\}} \exp\left( -\frac{(I_{k, c}(x, y) - 0.5)^2}{2(0.2)^2} \right)
   $$

The aggregate weight is the product of metrics:

$$
W_k(x, y) = [C_k(x, y)]^{\omega_C} \times [S_k(x, y)]^{\omega_S} \times [E_k(x, y)]^{\omega_E}
$$

Normalized weights:

$$
\hat{W}_k(x, y) = \frac{W_k(x, y)}{\sum_{j=1}^N W_j(x, y) + \epsilon}
$$

Multi-scale blending is performed across Laplacian decomposition bands $\mathcal{L}\{I_k\}$ and Gaussian pyramid weights $\mathcal{G}\{\hat{W}_k\}$:

$$
\mathcal{L}_{\text{fused}}^l(x, y) = \sum_{k=1}^N \mathcal{G}^l\{\hat{W}_k\}(x, y) \cdot \mathcal{L}^l\{I_k\}(x, y)
$$

---

## 8. Perceptual Tone Reproduction

### Reinhard Global Photographic Operator
$$
L_{\text{white}} = 1.5 \cdot \max(L)
$$

$$
L_{\text{scaled}} = \frac{\alpha}{\bar{L}_w} L(x, y)
$$

$$
L_{\text{display}}(x, y) = \frac{L_{\text{scaled}} \left( 1 + \frac{L_{\text{scaled}}}{L_{\text{white}}^2} \right)}{1 + L_{\text{scaled}}}
$$

### ACES Filmic Tone Curve
Fitted by Krzysztof Narkowicz for real-time graphics and cinematic production pipelines:

$$
f(x) = \frac{x(2.51x + 0.03)}{x(2.43x + 0.59) + 0.14}
$$

---

## 9. Electro-Optical Transfer Function (sRGB EOTF)

Standard IEC 61966-2-1 defines the transfer function converting linear scene radiance into non-linear encoded code values:

$$
C_{\text{sRGB}} = \begin{cases}
12.92 \cdot C_{\text{linear}}, & C_{\text{linear}} \le 0.0031308 \\
1.055 \cdot C_{\text{linear}}^{1/2.4} - 0.055, & C_{\text{linear}} > 0.0031308
\end{cases}
$$
