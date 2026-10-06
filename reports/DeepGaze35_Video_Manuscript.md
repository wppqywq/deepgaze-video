# From Static Scanpaths to Video Gaze: Controlled Temporal Adaptation of DeepGaze3.5-VL

**Youyou Yang**  
**Research manuscript - 6 October 2026**

## Abstract

Pretrained gaze models offer a way to predict attention in videos, but improvements after adaptation can conflate learning a new gaze task with using recent visual history. We study these effects by adapting DeepGaze3.5-VL to predict the first gaze position of the next foveation at an annotated saccade onset. All models receive causal inputs and are benchmarked by information gain over one training-fitted center-bias distribution on a shared 100 x 100 spatial grid. A repeated-current-frame control matches the four-frame model's input structure while removing distinct historical images. On a Hollywood development subset with 7,144 training and 2,571 validation transitions, adapted single-frame DeepGaze achieves 2.342 bits/transition above center bias, compared with 1.466 for its unadapted initialization and 0.473 for a frozen-vision GRU-Gaussian-mixture autoregressive baseline. Four historical frames achieve 2.403 bits/transition, adding 0.051 over repeated-frame input and 0.061 over a single current frame. These DeepGaze contrasts average three training seeds and favor historical frames in every seed, although most of the historical-content gain comes from two of four validation films. At a common final training step, the gain over repeated frames is 0.087 bits. The results support substantial task adaptation and a smaller benefit of historical image content within this cohort. Validation-based selection, limited stimulus coverage, unequal baseline capacities, and incomplete acquisition calibration constrain generalization and mechanistic interpretations.

## 1. Introduction

A person watching a film does not choose each gaze location independently. The next look follows a sequence of earlier looks, while the scene itself changes. A character may turn toward another person, an object may begin to move, or an edit may replace the scene. A useful gaze model should assign probability to the next location in light of both what the viewer has already inspected and what has happened in the video.

This problem has two distinct temporal dimensions: the history of the observer's gaze and the history of the visual stimulus. A model that uses earlier gaze positions is sequential even when it sees only one image. Conversely, a model that receives multiple video frames need not use their temporal differences effectively. Keeping these dimensions separate is essential when interpreting improvements in video gaze prediction.

We investigate a pretrained vision-language gaze model, DeepGaze3.5-VL, as a common starting point for this analysis. Its coordinate-token interface provides a practical route from a multimodal input to a normalized spatial prediction. Our question is specific: **does recent visual history improve the prediction of the next gaze landing beyond the current frame and the same observed gaze history?**

A comparison between one and four images alone cannot answer this question cleanly. Four images introduce additional visual tokens and change the structure of the model's input. We therefore include a repeated-frame control: the four temporal slots remain, but every slot contains the current image. The difference between real and repeated frames tests the value of distinct historical image content within a matched input structure. It does not, by itself, identify a motion computation or a human perceptual mechanism.

We also define the behavioral target at a gaze transition rather than at every video frame. Dense, fixed-rate gaze prediction includes many samples from periods in which gaze remains near its previous location. Here, a prediction is requested when an annotated saccade begins, and the target is the onset position of the following foveation. This focuses evaluation on where a gaze shift lands. The event time is supplied by the dataset; the model does not predict when a saccade will occur.

The study contributes an event-conditioned formulation of video gaze with pursuit-aware history; a controlled adaptation of DeepGaze3.5-VL that distinguishes current appearance, repeated-image structure, and historical content; and a common likelihood benchmark for coordinate-token and Gaussian-mixture predictors. Every model is evaluated by information gain over the same training-fitted center bias. This reference asks how much a predictor explains beyond the population's marginal spatial preference, while the paired DeepGaze controls ask where its additional performance comes from.

Across three training seeds, adaptation produces a substantial improvement, while historical visual content contributes a smaller gain whose magnitude and distribution across films depend on checkpoint selection. The consistent direction across seeds strengthens the evidence for a benefit on this development cohort. Its concentration in two films motivates a more specific next question: which transitions benefit from historical images, and does that benefit transfer to new films? The resulting framework makes those questions measurable without changing the prediction target or probability metric.

## 2. Background and Related Work

### 2.1 DeepGaze3.5-VL: the starting point

Agrawal, Bethge, and Kummerer introduce DeepGaze3.5-VL as a formulation of scanpath prediction through autoregressive coordinate-token generation. A vision-language model receives an image and gaze context, then predicts coordinates written as zero-padded digits. The paper uses digit-restricted probabilities to obtain exact fixation likelihoods and explores conditioning on viewing task, observer identity, and fixation duration. It reports 2.18 bits per fixation of information gain on MIT1003. This is a result under the original image-based benchmark and its baseline; it is not directly comparable to the video results reported here. [1](https://arxiv.org/abs/2607.02083)

Our implementation starts from the released combined free-viewing adapter on InternVL3.5-8B. The release describes joint training on five image-based gaze datasets and provides the adapter and inference code. We retain this pretrained initialization rather than training a general-purpose vision-language model to produce gaze coordinates from scratch. [2](https://github.com/Susmit-A/DeepGaze3.5-VL)

The distinction between inheritance and extension is central. Coordinate tokenization, the pretrained gaze adapter, and its underlying multimodal architecture are inherited. Our extension concerns the dynamic behavioral target, the causal video-and-gaze input, and the controlled evaluation of historical visual content. A sequence decoder's ability to generate successive digits does not supply visual evidence from frames that were never included in its input.

### 2.2 From saliency to conditional gaze prediction

Saliency prediction and next-gaze prediction answer different questions. A saliency map describes a distribution of viewing locations, whereas a conditional gaze model predicts the next location given a particular preceding history. Kummerer and Bethge advocate evaluating scanpath models by their predictions for each observed fixation conditioned on earlier fixations, rather than relying exclusively on similarity between sampled and observed trajectories. We use the analogous conditional evaluation for video foveation transitions. [3](https://arxiv.org/abs/2102.12239)

Static appearance is a strong competitor in video prediction. Tangemann and colleagues showed that their static DeepGaze MR model explained at least 75% of the explainable information on LEDOV under their evaluation, despite excluding temporal patterns. Their findings motivate a strong current-frame reference and an explicit test of temporal information; they do not establish the expected effect size on our different task and dataset. [4](https://bethgelab.org/publication/2020_01_tangemann/)

Dynamic gaze also includes pursuit. Roth and colleagues model gaze in real-world dynamic scenes using object-based exploration, illustrating an approach that explicitly represents scene structure and gaze dynamics. We instead ask how far a pretrained gaze model can transfer with an event-based input and without adding object identities or an explicit motion pathway. These are different modeling choices, rather than a controlled comparison with their method. [5](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011512)

### 2.3 Relation to video gaze generation

Video gaze modeling is an established research area. Ozdel and colleagues use a transformer-based reinforcement-learning approach to simulate gaze in videos from VirtualHome, with activity recognition as the viewing context. [6](https://arxiv.org/abs/2404.07351) Kang and colleagues propose autoregressive diffusion for continuous, timestamped gaze generation over videos of arbitrary length. [7](https://arxiv.org/abs/2603.24938)

Our evaluation addresses a narrower question: conditional spatial prediction at an observed event boundary, with a fixed recent context and a tractable probability for the actual landing. We neither evaluate unrestricted trajectory generation nor claim the first application of sequence modeling to video gaze. The contribution is a controlled temporal adaptation study of a pretrained coordinate-token gaze model.

## 3. Task: Predicting the Next Foveation Landing

### 3.1 Behavioral unit and prediction time

We use *foveation* to denote a continuous episode of visual intake that may contain fixation or smooth pursuit. Event labels originate from REMoDNaV, which supports fixations, pursuits, saccades, and post-saccadic oscillations in dynamic recordings. [8](https://doi.org/10.3758/s13428-020-01428-x)

Within a valid continuous segment, adjacent fixation and pursuit labels are merged. A switch between these two labels does not itself create a new prediction target. Saccades separate successive foveations; associated post-saccadic oscillations belong to the transition. Missing observations, unexplained event gaps, and invalid segments prevent a transition from being treated as a valid adjacent pair. We do not skip an invalid intervening event to construct an easier target.

Let \(c_t\) be the annotated onset time of the saccade following foveation \(t\). The target \(Y_{t+1}\) is the spatial grid cell containing the first raw gaze sample of the next foveation. A target is excluded if that initial sample is invalid or outside the assumed video display rectangle. A later valid sample is not substituted for it.

The model estimates

\[
p_\theta\!\left(Y_{t+1}\mid H_t,V_{\leq c_t},\text{free viewing}\right),
\]

where \(H_t\) summarizes completed foveations and \(V_{\leq c_t}\) contains the selected causal video frames. Gaze samples used in the history are strictly earlier than \(c_t\); video frames may have presentation timestamps equal to \(c_t\). The target's onset time and subsequent trajectory are excluded from the input.

This is an **event-conditioned spatial task**. Event boundaries are obtained offline, and the model is given the opportunity to predict at an annotated transition. Causal model inputs do not imply an online event detector, and these experiments do not establish end-to-end real-time anticipation of saccades.

### 3.2 Gaze history

The input contains up to four completed foveations, ordered from oldest to newest. Each event contributes its start position, end position, start time, and end-boundary time, expressed relative to the prediction cutoff. Preserving both endpoints allows pursuit to contribute a displacement rather than being reduced to a stationary point. The last raw sample of every event must precede the cutoff; an event boundary may coincide with it.

The number of events and the video time span are separate restrictions. The model receives the last four eligible events, which can extend further back than the one-second video window. It does not receive the entire earlier scanpath or a recurrent state carried over from previous predictions. Observer identities, film names, and fixation/pursuit class labels are withheld from the prompt.

### 3.3 Spatial prediction space

Gaze coordinates are mapped to the video's assumed display rectangle and normalized to \([0,1)^2\). For normalized position \((u,v)\), the target cell is

\[
Y=(\lfloor100u\rfloor,\lfloor100v\rfloor).
\]

Thus, all conditions predict the same 10,000 possible cells. Targets outside this rectangle are excluded rather than clipped to the boundary. The distinction between cell probability and continuous density is maintained throughout evaluation: a Gaussian baseline is integrated over a cell, while the token model directly assigns probability to the corresponding pair of integers.

This definition conditions the analysis on eligible, in-rectangle transitions. It does not measure performance on blinks, missing segments, or all gaze samples in the original recordings.

## 4. Model and Temporal Controls

### 4.1 A common pretrained gaze model

All principal conditions use the released combined DeepGaze3.5-VL adapter on InternVL3.5-8B. In the pinned model configuration, a 24-layer vision transformer encodes each image at 448 x 448 resolution with 14 x 14 patches and hidden width 1,024. Spatial token reduction and a multimodal projector produce 256 image tokens of width 4,096 per image. These tokens are inserted into the same sequence as the gaze-history text, frame timestamps, and task instruction. A 36-layer Qwen3 decoder with hidden width 4,096 then predicts coordinate digits through causal self-attention. The model therefore combines image and gaze context in the decoder rather than through a separate recurrent gaze module. The official model card describes the InternVL backbone. [11](https://huggingface.co/OpenGVLab/InternVL3_5-8B-HF)

The visual encoder, multimodal projector, and base decoder weights are frozen. We update one existing rank-32 LoRA adapter in every decoder layer's q/k/v/o attention projections and gate/up/down feed-forward projections, with scaling parameter 64 and dropout 0.05. At inference, an adapted projection is \(W'=W+(64/32)BA\), where only the low-rank factors \(A\) and \(B\) are trainable. The factors retain the released gaze initialization. [9](https://arxiv.org/abs/2106.09685)

There are 87,293,952 trainable adapter parameters. The base model uses BF16 and the adapter parameters use FP32. Each training condition starts independently from the same adapter weights. No additional temporal adapter or external motion encoder is introduced.

Inputs combine video images, their relative timestamps, the common gaze history, and a free-viewing task instruction. Each image contributes one 448 x 448 patch and 256 visual tokens under the fixed processor configuration. We disable dynamic image tiling and input truncation. The one-frame condition uses 256 visual tokens; both four-frame conditions use 1,024. This ensures that the real-frame and repeated-frame models have the same visual-token budget.

### 4.2 Separating adaptation, input structure, and historical content

For each target, we construct the following conditions.

| Condition | Visual input | Video-task training | Purpose |
|---|---|---|---|
| StaticD1 | Current frame | None | Evaluate the released gaze adapter under the new task interface |
| D1 | Current frame | Yes | Measure adaptation with no earlier video images |
| R4 | Current frame repeated in four slots | Yes | Control for four-image input structure |
| D4 | Four causal frames spanning approximately one second | Yes | Add distinct historical visual content |

**Table 1. Principal experimental conditions.** All four conditions receive the same completed gaze history and are evaluated on identical target IDs. "Static" describes the absence of video-task adaptation, not the absence of gaze-history conditioning.

D4 requests frames at \(c_t-1\), \(c_t-2/3\), \(c_t-1/3\), and \(c_t\) seconds. Each request is resolved to the latest decoded frame whose presentation timestamp does not exceed the requested time. Actual timestamps therefore differ slightly from the nominal schedule, and the earliest selected frame can be just older than one second. D1 uses the final selected frame.

R4 retains D4's four slot timestamps and text, but replaces all image content with D1's current image. These timestamps identify the control's slots; they should not be interpreted as the acquisition times of four distinct images. The repeated images remain available by the prediction cutoff. R4 is deliberately synthetic, and it is trained in that form rather than introduced only as a test-time corruption.

The resulting contrasts have different interpretations:

\[
\Delta_{\mathrm{adapt}}=L(\mathrm{D1})-L(\mathrm{StaticD1}),
\]

\[
\Delta_{\mathrm{content}}=L(\mathrm{D4})-L(\mathrm{R4}),
\qquad
\Delta_{\mathrm{total}}=L(\mathrm{D4})-L(\mathrm{D1}).
\]

The first measures video-task adaptation under the same single-frame interface. The second is the primary comparison for distinct historical image content at matched input length. The third measures the overall change from one current image to four historical images. R4-D1 additionally shows how much a repeated-image input structure changes performance.

D4-R4 does not isolate motion alone. Earlier frames may provide another view of a face, reveal content absent in the current frame, or indicate a scene change. A positive difference supports the utility of historical content under this protocol. Establishing sensitivity to temporal order would require additional controls, such as shuffling the historical frames.

### 4.3 Exact probability and training objective

The response has fixed syntax, `(XX, YY)`, with two digits for each coordinate. For example, `(47, 06)` is represented by the digits 4, 7, 0, and 6. Let \(d_1,d_2,d_3,d_4\) denote those digits and \(C_t\) the multimodal context. At digit position \(j\), we normalize the model's logits over all ten digit tokens:

\[
q_\theta(d_j\mid C_t,d_{<j})
=\frac{\exp z_{j,d_j}}{\sum_{a=0}^{9}\exp z_{j,a}}.
\]

Fixed punctuation is part of the prefix but does not receive a spatial-probability factor. The probability of a cell is

\[
p_\theta(Y\mid C_t)=\prod_{j=1}^{4}q_\theta(d_j\mid C_t,d_{<j}).
\]

Every branch has ten possible digits, so this factorization defines a normalized distribution over the 10,000 cells. Training minimizes the negative sum of the four target-digit log probabilities. Prompt tokens and punctuation are not additional supervised targets.

For evaluation, a causal teacher-forced forward pass supplies the logits for the four observed digits. Earlier target digits condition later ones, as required by the factorization; no later digit is available to an earlier prediction. All ten logits are retained at every position. This yields the exact observed-cell likelihood without sampling candidate coordinates or constructing the full spatial map.

A full probability map is defined by the same factorization, but the present VLM evaluation computes scalar target likelihoods. Qualitative maps from a separate Gaussian-mixture baseline must not be presented as DeepGaze maps.

### 4.4 Population autoregressive reference

We compare the coordinate-token architecture with a compact autoregressive (AR) model that predicts a distribution for the next landing from the same four gaze events and D4 frame plan. Here, *autoregressive* describes conditioning the next gaze location on preceding gaze; it is distinct from DeepGaze's autoregression over coordinate digits. The population AR has no observer-specific parameters.

Each video frame is aspect-fitted and black-padded to 224 x 224, normalized with ImageNet statistics, and encoded by frozen ImageNet-pretrained ResNet18 through its final convolutional block. [12](https://arxiv.org/abs/1512.03385) Adaptive 2 x 2 pooling yields a 2,048-dimensional feature vector. A shared linear layer with tanh projects each frame to 32 dimensions. The four projected vectors, frame-presence masks, and relative timestamps are concatenated in chronological order and projected from 136 to 32 dimensions to form visual context \(v_t\). This is a learned summary of four ordered frames, not recurrence along the video stream.

Let \(h_j\in\mathbb R^6\) contain an event's end XY, start XY, and relative start/end times. Coordinates use normalized grid-cell centers and times use seconds at the prompt's recorded precision. Valid history events are processed oldest first by a 32-unit GRU. [13](https://arxiv.org/abs/1406.1078)

\[
s_j=\operatorname{GRUCell}([h_j;v_t],s_{j-1}),\qquad s_0=0.
\]

Missing history slots leave the state unchanged. The state is reset for every prediction, so this model has bounded history rather than a state carried through an entire trial. Input feature means and standard deviations are fitted on training data only. A 25-output linear head on the final state supplies five weights, ten means, and ten diagonal scales:

\[
\pi_k=\operatorname{softmax}(a)_k,\quad
\mu_k=\operatorname{sigmoid}(b_k),\quad
\sigma_k=0.02+0.48\operatorname{sigmoid}(r_k),\qquad k=1,\ldots,5.
\]

The resulting Gaussian mixture is integrated over each grid cell and conditioned on the video rectangle \(R=[0,1]^2\):

\[
p_{\mathrm{AR}}(Y)=
\frac{\sum_{k=1}^{5}\pi_k\int_{B_Y}\mathcal N(z;\mu_k,\operatorname{diag}(\sigma_k^2))\,dz}
{\sum_{k=1}^{5}\pi_k\int_R\mathcal N(z;\mu_k,\operatorname{diag}(\sigma_k^2))\,dz}.
\]

The denominator conditions the full mixture, rather than normalizing each component separately. Gaussian CDF differences provide exact cell masses; log-domain evaluation avoids numerical zero in far tails. Training minimizes the observed-cell negative log likelihood. The mixture count is fixed at five, while weights, means, and scales vary with the input. The 0.02 scale floor corresponds to two grid-cell widths and constrains spatial sharpness; this is a modeling choice, not a calibrated measurement-noise estimate. The complete trainable model has 77,689 parameters.

Both architectures output normalized probabilities for the same target cells. They differ substantially in visual pretraining, spatial resolution, capacity, and optimization. Their comparison benchmarks the implemented predictors; it is not an ablation that isolates the causal value of images or recurrence.

## 5. Data and Experimental Design

### 5.1 Development cohort

We use recordings from the public Hollywood video eye-movement database of Costela and Woods. The resource contains naturalistic movie stimuli and timestamped gaze recordings. Its acquisition description includes more than one display configuration, making the mapping from each recording to its viewing geometry relevant to reconstruction. [10](https://pubmed.ncbi.nlm.nih.gov/31428665/)

The experiments reported here use an existing fixed development subset, not the complete database. The audited subset contains 151 recordings over 16 video clips. Training and validation are separated by source film, so clips from a validation film are not used in training.

| Partition | Source films | Audited recordings | Eligible transitions |
|---|---:|---:|---:|
| Training | 10 | 109 | 7,144 |
| Validation | 4 | 42 | 2,571 |
| Total | 14 | 151 | 9,715 |

**Table 2. Development data used by all principal conditions.** Recordings are observer-clip trials; transition counts follow the common event and validity rules. The split evaluates transfer across source films, not a separately designed held-out-observer task.

The four validation films are *Batman Forever*, *Deep Blue*, *Quiz Show*, and *March of the Penguins*. Every condition uses the same eligibility mask, including the requirement that a complete one-second frame-request window be available. D1 therefore does not obtain additional early targets unavailable to D4.

The present results contain no untouched final-test evaluation. A historically named test partition was inspected during earlier development and is not treated as an independent test resource in this manuscript.

### 5.2 Reconstruction assumptions

The current reconstruction uses elapsed time from the first untrimmed raw timestamp as its video-time origin. Video frames are indexed by decoded presentation timestamps. Gaze positions are mapped through a centered, aspect-preserving video rectangle on an assumed 2560 x 1440 reference screen.

Recovered event preprocessing interpolates eligible raw observations to 250 Hz for REMoDNaV, with a maximum interpolation gap of 10 ms. Raw missingness flags are retained, and interpolated samples spanning invalid observations are not treated as valid. Event boundaries are mapped back to raw samples; the prediction target remains an original sample rather than an interpolated detector coordinate.

Independent playback-onset markers, trial-specific display geometry, and complete acquisition-to-MAT export records are unavailable for this cohort. Observed median raw timestamp intervals vary between recordings: 90 trials have a median interval of 1 ms, 17 have 2 ms, and 44 have 6 ms. These observations do not establish the original acquisition or export procedure. They motivate retaining the raw timestamps and reporting the reconstruction assumptions explicitly.

Consequently, all reported comparisons are development evidence conditional on a shared reconstruction. Identical preprocessing makes the experimental conditions comparable within that reconstruction; it does not establish physiological timing accuracy or eliminate the possibility that misalignment attenuates a temporal effect.

### 5.3 Training and model selection

D1, R4, and D4 are each trained for 1,000 optimizer updates with seeds 0, 1, and 2, giving nine runs. Within each seed, conditions share target order, initialization, optimizer schedule, and validation schedule. Across seeds, the pretrained initialization remains fixed while training randomness changes. AdamW uses a peak learning rate of \(2\times10^{-5}\), weight decay 0.01, gradient clipping at 1.0, and an effective batch size of 16. A 50-update warmup is followed by cosine decay. Each run processes 15,984 target presentations, including partial batches at training-set boundaries.

Validation is performed every 100 updates. The primary selection rule chooses the checkpoint with the highest validation source-film mean, breaking ties in favor of the earlier checkpoint. Each condition therefore has ten selection opportunities. We also report all conditions at the common final update, 1,000, as a sensitivity analysis.

Equal updates and examples do not imply equal computation: four-frame inputs are longer. The matched budget is an optimization-and-data budget, not a wall-clock budget. Seeds 1 and 2 replicate the seed-0 protocol without changing the data, architecture, or training budget. These replications assess training variability within the existing development experiment; they do not constitute a new held-out evaluation.

### 5.4 Reference baselines

**Common center bias.** The benchmark reference is a Gaussian kernel density estimate fitted to the 7,144 training landing locations, with no gaze history, image, timestamp, film identity, or observer identity. For training cell \((x_n,y_n)\), the kernel center is \(z_n=((x_n+0.5)/100,(y_n+0.5)/100)\). The density is an equal-weight mixture,

\[
\widetilde p_{\mathrm{CB}}(z)=\frac1N\sum_{n=1}^{N}
\mathcal N(z;z_n,\operatorname{diag}(b_x^2,b_y^2)),\qquad
b_a=\widehat\sigma_aN^{-1/6}.
\]

Scott's two-dimensional rule gives bandwidths \((b_x,b_y)=(0.036023,0.039213)\). We integrate the mixture over each cell and normalize by its total mass in \([0,1]^2\), exactly as for the AR mixture. This produces one positive 100 x 100 probability map held fixed for all validation targets and all models. It captures the marginal spatial preference of eligible training landings; the term *center bias* does not imply a centered or symmetric Gaussian. Its validation film-macro log2 likelihood is -12.318787 bits/transition.

An axis-aligned training Gaussian and a training histogram with one pseudocount per cell are additional spatial references. The same KDE remains the denominator throughout the benchmark; no model receives its own reference map. These spatial controls were added after inspecting initial development results, so the benchmark is retrospective rather than preregistered. Kernel bandwidths and model fitting nevertheless use training locations only, with no validation bandwidth search.

**Linear history references.** Persistence centers a diagonal Gaussian on the latest completed foveation endpoint. AR(1) and AR(4) predict the next XY from one or four completed endpoints using ridge regression with an unpenalized intercept. Full-field linear AR additionally receives all start/end coordinates and relative times from the gaze prompt. Missing lags have presence masks, center-filled coordinates, and zero times. Feature normalization is fitted within each training fold. Residual diagonal scales are multiplied by a selected factor and have a half-cell numerical floor; cell masses are integrated and conditioned on the same rectangle.

Ridge strengths \(\{0.1,10,1000\}\) and scale multipliers \(\{0.5,1,2\}\) are selected by equal-film mean likelihood in leave-one-training-film-out cross-validation, then refitted on all training data. Persistence selects only its scale multiplier, using residual RMS about its fixed last-endpoint mean. Endpoint AR(1)/AR(4) select ridge 0.1; full-field linear AR selects ridge 10. All select scale multiplier 1. This avoids tuning these fits on the four external validation films. Their ridge least-squares mean fits and continuous residual estimates are approximations, rather than joint maximum-likelihood fits of the truncated discrete distribution.

These models are deliberately simple probability references. They differ from DeepGaze in representation, pretraining, capacity, and fitting objective. Their score differences cannot identify the contribution of visual information alone.

The population visual AR in Section 4.4 is optimized with Adam at learning rate 0.001, batch size 128, and gradient clipping at 5 for 20 epochs, using seed 0. Its checkpoint maximizes validation film-macro likelihood, with earliest ties retained. This gives 20 selection opportunities versus ten for each DeepGaze run. It receives no identity, optical flow, event-type input, or duration-prediction head. Its training budget and capacity are reported explicitly rather than treated as matched to DeepGaze.

### 5.5 Evaluation and uncertainty

The primary benchmark is **information gain over the common center bias**, following the likelihood-based evaluation used by DeepGaze3.5-VL. [1](https://arxiv.org/html/2607.02083v1#S2.SS1) For target \(i\),

\[
g_i(M)=\log_2\frac{p_M(Y_i\mid C_i)}{p_{\mathrm{CB}}(Y_i)}.
\]

Positive values mean that the model assigns greater probability to observed landings on average than the population spatial reference. A gain of one bit corresponds to a factor of two in the geometric mean likelihood ratio. The score applies identically to DeepGaze's coordinate-token probabilities, AR's integrated Gaussian-mixture probabilities, and the simpler references. It measures prediction beyond marginal location bias; it does not isolate image information from gaze-history information.

We give each validation film equal weight:

\[
\operatorname{IG}(M)=\frac{1}{F}\sum_{f=1}^{F}
\frac{1}{N_f}\sum_{i\in f}g_i(M)
=L(M)-L(\mathrm{CB}),
\]

where \(L\) denotes the same film-macro log2 likelihood. Larger films therefore do not dominate the headline result. For the identical target population, subtracting the common center bias preserves every model ranking and paired contrast: \(\operatorname{IG}(A)-\operatorname{IG}(B)=L(A)-L(B)\). It also leaves validation checkpoint selection unchanged because the reference term is constant across checkpoints. Absolute likelihoods are included as supporting values; all headline model scores use the common-reference IG. This reference is an evaluation denominator, not an extra prior multiplied into the learned predictions.

For the three-seed results, we first average paired model differences across seeds within each film, then average equally across the four films. Descriptive intervals use 5,000 bootstrap resamples of those four seed-averaged film differences. The independent stimulus units remain four films, not twelve film-seed combinations. We separately report the sample standard deviation of the three seed-specific film-macro differences to describe training variability. No seed ensemble is formed: the reported mean averages log scores of separately trained predictors, rather than scoring a mixture of their probability distributions.

The film-bootstrap intervals do not correct for checkpoint selection on the same validation data, integrate seed uncertainty, or provide independent-test confirmation. With only four films, they offer a limited description of stimulus variability. Target-level resampling would overstate the number of independent stimulus units and is not used.

## 6. Results

### 6.1 A common center-bias benchmark

| Predictor | Visual context | Fit or checkpoint | IG above center bias (higher is better) | Log2 likelihood |
|---|---|---|---:|---:|
| Center bias: training KDE | None | Training-only Scott rule | 0.000 | -12.319 |
| Training Gaussian | None | Training-only fit | -0.001 | -12.320 |
| Training histogram | None | Training-only fit | -0.441 | -12.760 |
| Last-endpoint persistence | None | Training-film CV | 0.411 | -11.908 |
| Endpoint AR(1) | None | Training-film CV | 0.598 | -11.721 |
| Endpoint AR(4) | None | Training-film CV | 0.612 | -11.707 |
| Full-field linear AR | None | Training-film CV | 0.668 | -11.651 |
| Population visual AR: ResNet18-GRU-mixture | Four historical frames | Epoch 4 of 20; one seed | 0.473 | -11.846 |
| StaticD1: released DeepGaze3.5-VL | Current frame | No video adaptation | 1.466 | -10.852 |
| D1: adapted DeepGaze3.5-VL | Current frame | Selected; three seeds | 2.342 | -9.977 |
| R4: adapted DeepGaze3.5-VL | Current frame repeated four times | Selected; three seeds | 2.351 | -9.967 |
| D4: adapted DeepGaze3.5-VL | Four historical frames | Selected; three seeds | **2.403** | **-9.916** |

**Table 3. Shared-reference development benchmark, in bits per transition.** Every IG subtracts the same training KDE score on the same 2,571 validation targets, with equal film weights. DeepGaze D1/R4/D4 scores average three separately trained models' log scores; visual AR uses one seed. StaticD1 and the spatial/linear fits are single fixed references. Checkpoint budgets differ across architecture families. Negative IG denotes performance below the KDE, not an invalid probability distribution.

Adapted D1 explains 2.342 bits per transition beyond center bias, compared with 1.466 for the released gaze initialization. Thus, adaptation adds 0.875 bits under the same single-frame interface. Seed-specific adaptation gains are 0.861, 0.867, and 0.898 bits. Averaging seeds within each film gives positive gains on all four films, ranging from 0.531 to 1.128 bits per transition. This supports adapting the pretrained gaze representation to the event-conditioned video task even without providing earlier images.

The contrast includes learning the new prompt, target definition, and data distribution. It should therefore be interpreted as task adaptation, not as evidence that temporal visual representations have been learned. StaticD1 itself is the released gaze adapter under the new interface, rather than a separately optimized static benchmark model for this cohort.

### 6.2 Historical image content gives a smaller gain

| Paired IG contrast | Mean difference | Across-seed SD | Descriptive 95% film-bootstrap interval |
|---|---:|---:|---:|
| D1 - StaticD1 | +0.8753 | 0.0202 | [0.6376, 1.0671] |
| D4 - R4 | +0.0513 | 0.0150 | [0.0027, 0.0999] |
| D4 - D1 | +0.0610 | 0.0204 | [0.0150, 0.1138] |
| R4 - D1 | +0.0097 | 0.0085 | [-0.0170, 0.0338] |

**Table 4. Distinct effects within the matched comparison, in bits per transition.** Means and film-bootstrap intervals use seed-averaged film differences. Across-seed SD describes the three seed-specific film-macro differences and is not a standard error. Intervals remain descriptive because the four films also determine checkpoint selection.

At the selected checkpoints, D4 exceeds R4 by 0.051 bits per transition. The three seed-specific differences are +0.037809, +0.067422, and +0.048741 bits. D4 also exceeds D1 in every seed, by +0.049179, +0.084589, and +0.049209 bits. Thus, the direction is consistent across training seeds for both contrasts. The historical-content effect remains much smaller than the adaptation effect, and the positive film-bootstrap interval should be read alongside the concentration of gains in particular films.

| Validation film | Transitions | Visual AR IG | D1 IG | D4 IG | D4 - R4 |
|---|---:|---:|---:|---:|---:|
| *Batman Forever* | 375 | 0.740 | 2.536 | 2.681 | +0.1036 |
| *Deep Blue* | 512 | 0.212 | 1.629 | 1.697 | +0.0962 |
| *Quiz Show* | 999 | 0.436 | 3.014 | 3.034 | +0.0094 |
| *March of the Penguins* | 685 | 0.505 | 2.188 | 2.199 | -0.0039 |

**Table 5. Film-level information gain over the same center-bias map.** Within each film, each model subtracts that film's mean KDE log likelihood. D1/D4 and the paired contrast average three seeds; visual AR uses its selected seed-0 checkpoint. The common map is not refitted by film. Transition counts show why micro-averaging would weight these stimuli unequally.

Batman Forever and Deep Blue account for most of the historical-content advantage. Quiz Show improves by only 0.009 bits, while March of the Penguins is a small counterexample at -0.004 bits. Repeating training therefore strengthens the evidence for a modest benefit on this cohort without establishing broad stimulus generalization. The average does not reveal whether gains occur around motion, edits, pursuit, or particular semantic content; those event-level explanations require separate analyses.

### 6.3 Checkpoint choice changes the temporal contrast

| Condition | IG above center bias at update 1,000 | Log2 likelihood |
|---|---:|---:|
| D1 | 2.240 | -10.079 |
| R4 | 2.258 | -10.061 |
| D4 | **2.345** | **-9.973** |

**Table 6. Common-update sensitivity analysis, averaged across three seeds.** These checkpoints are fixed by training budget, rather than selected independently for each condition.

At update 1,000, D4 exceeds R4 by 0.087359 bits, with a descriptive interval of [0.051176, 0.123422] and positive seed-averaged differences on all four films. Seed-specific D4-R4 differences are +0.090107, +0.059326, and +0.112644 bits, with SD 0.026765. D4 also exceeds D1 by 0.105438 bits on average. D1 and R4 lose more validation performance after their best checkpoints than D4 does.

This analysis shows that the magnitude and film-wise consistency of the temporal contrast depend on the checkpoint rule. It does not replace the primary selected-checkpoint result. Both rules favor D4 in all three seeds, but better late-training retention could reflect differences in optimization or regularization as well as useful historical information. Seed replication alone does not distinguish those explanations.

### 6.4 What the AR references establish

The full-field linear AR captures 0.668 bits beyond center bias, whereas endpoint-only AR(4) captures 0.612 bits. The population visual AR captures 0.473 bits, below the full-field linear predictor by 0.195 bits. It improves on that predictor in two films and loses in two. Its final-epoch score is -0.690 bits relative to center bias, showing substantial validation deterioration after the selected fourth epoch. This behavior limits the strength of the neural baseline: adding visual features and a GRU under this recipe does not establish an improvement over a simpler history model.

Adapted D1 exceeds both AR references on every validation film (Table 5 for visual AR). However, the models differ in capacity, pretraining, image resolution, and training budget. This is evidence about the predictive systems as implemented, not proof that the transformer architecture or images alone explain the gap. In particular, the modest five-component mixture and its scale constraints are not an exhaustive probabilistic AR benchmark. A stronger tuned AR, replicated across seeds, would test whether the gap survives a better reference; a matched DeepGaze history-only control would address the separate question of what its images contribute.

The Gaussian and KDE center-bias models perform almost identically, while the histogram is 0.441 bits worse. Using the histogram as the headline reference would therefore inflate every model's reported IG by 0.441 bits. Holding the KDE fixed prevents the DeepGaze and AR results from inheriting different apparent advantages from different spatial references.

## 7. Discussion

### 7.1 What transfers from a pretrained gaze model?

The strongest result is the adapted current-frame model's 2.342 bits of information gain over center bias, including 0.875 bits gained through video-task adaptation. The unadapted model already captures 1.466 bits beyond the same reference, so the experiment demonstrates both transfer from image-based gaze pretraining and further improvement under the new target definition. It also identifies the appropriate competitor for a temporal extension: an adapted current-frame model receiving the same gaze history.

The task provides a common probability interface for this transfer. Gaze events, video timestamps, and current appearance can be expressed as context while the output remains a distribution over spatial cells. That interface allows the research question to change without also changing the meaning of the score. It is particularly useful when comparing a coordinate-token model with density-based baselines.

### 7.2 Why a small temporal effect can still guide the project

Temporal information may be important only for a subset of gaze transitions. A scene can contain enough information in its current frame to make recent images largely redundant, while a different transition may depend on an action that has just occurred. A small average gain is compatible with this possibility, but does not establish it.

The repeated-frame control makes the role of historical content testable, and the three-seed comparison supplies a consistent aggregate direction. The film-level differences now motivate a finer analysis using independently defined scene changes, motion, pursuit history, or target displacement. Those categories should be specified before comparing subgroup scores and evaluated on new data where possible. The present film averages cannot explain why historical frames help some transitions more than others.

The next evidential priority is stimulus generalization and verified alignment. More training seeds on the same films would refine optimization variability without resolving either issue. Order controls could distinguish a benefit from additional views from sensitivity to their temporal arrangement. Together, these tests would help determine whether more frames, a different time span, or persistent visual memory are justified. The present experiment does not establish a need for any particular larger architecture, and no practical effect-size threshold was preregistered for this development comparison.

## 8. Limitations

**Development evidence.** The primary results average three training seeds on four validation films, which also select checkpoints. Replication shows a consistent direction across these seeds, but three seeds provide only a limited estimate of training variability. They do not increase the number of independent films or remove validation-selection bias. Film-bootstrap intervals remain descriptive, and broad generalization requires additional held-out films or another dataset. A previously inspected six-film confirmation set is already used development evidence and cannot serve as an untouched final test. The population visual baseline has only one training seed and a different optimization budget.

**Acquisition uncertainty.** Playback origin, per-trial viewing geometry, and raw export procedures remain incompletely verified. A shared preprocessing assumption can preserve internal experimental consistency while still biasing the absolute target locations or weakening apparent temporal dependence. Dataset alignment therefore remains a scientific limitation, not merely a reproducibility detail.

**Restricted population and target.** The development subset is not the full Hollywood resource. Eligibility excludes invalid histories, transitions across gaps, out-of-rectangle landings, and early windows without sufficient video context. The estimates apply to retained transitions. They do not describe all eye movements or all observers in the source database.

**Limited temporal representation.** Four sampled frames and four gaze events provide bounded context. The vision encoder remains frozen, and no explicit optical-flow, scene-cut, or persistent-memory module is trained. The comparison tests historical images within this representation, not the maximum information available in the full video.

**Interpretation of controls.** R4 matches input structure but is a synthetic viewing condition. D4-R4 can reflect distinct appearance as well as motion, and does not prove sensitivity to frame order. Comparisons with linear or small recurrent baselines additionally differ in capacity and pretraining. None of these contrasts constitutes a causal account of human attention.

**Benchmark scope.** The KDE measures population spatial preference under the retained-target definition, rather than an image-conditioned or observer-specific baseline. IG above it combines gains from gaze history, appearance, pretraining, and adaptation. The benchmark does not estimate the fraction of explainable gaze information, establish state of the art, or make its bit values directly comparable to the original MIT1003 benchmark. Spatial-reference and AR choices were made during development, so stronger confirmation requires freezing these choices before new-data evaluation.

**Conditional rather than free-running prediction.** Observed event boundaries and gaze history are supplied. Error accumulation under generated histories, duration calibration, pursuit trajectories, and saccade-onset prediction are outside the reported evaluation.

## 9. Conclusion

We formulate video gaze adaptation around a measurable distinction: learning the distribution of video landings and learning to use recent visual history are separate achievements. DeepGaze3.5-VL supplies a pretrained probabilistic foundation; event-conditioned targets and matched repeated-frame controls make these achievements distinguishable.

Under a common center-bias benchmark, adapted current-frame DeepGaze achieves 2.342 bits per transition, historical-frame DeepGaze achieves 2.403 bits, and the implemented population visual AR achieves 0.473 bits. The matched DeepGaze comparisons separate a 0.875-bit task-adaptation gain from a 0.051-bit historical-content gain beyond repeated frames. The latter is positive in all three seeds but concentrated in two of four validation films. This study demonstrates a transferable probability interface for video landings and a controlled way to measure the value of recent visual content. New films, verified acquisition alignment, stronger AR references, and frame-order controls are needed to establish its broader scope.

## References

1. Susmit Agrawal, Matthias Bethge, and Matthias Kummerer. **DeepGaze3.5-VL: Modeling Scanpaths via Autoregressive Token Prediction.** arXiv:2607.02083, 2026. [Paper](https://arxiv.org/abs/2607.02083).
2. Susmit Agrawal and contributors. **DeepGaze3.5-VL: official model and code release.** Revision `5cd84d2225beb92e77617b1ee5c5480cae948ab7`. [Repository](https://github.com/Susmit-A/DeepGaze3.5-VL).
3. Matthias Kummerer and Matthias Bethge. **State-of-the-Art in Human Scanpath Prediction.** arXiv:2102.12239, 2021. [Paper](https://arxiv.org/abs/2102.12239).
4. Matthias Tangemann, Matthias Kummerer, Thomas S. A. Wallis, and Matthias Bethge. **Measuring the Importance of Temporal Features in Video Saliency.** ECCV, 2020. [Author publication page](https://bethgelab.org/publication/2020_01_tangemann/).
5. Nicolas Roth, Martin Rolfs, Olaf Hellwich, and Klaus Obermayer. **Objects guide human gaze behavior in dynamic real-world scenes.** PLOS Computational Biology 19(10):e1011512, 2023. [Paper](https://doi.org/10.1371/journal.pcbi.1011512).
6. Suleyman Ozdel, Yao Rong, Berat Mert Albaba, Yen-Ling Kuo, Xi Wang, and Enkelejda Kasneci. **A Transformer-Based Model for the Prediction of Human Gaze Behavior on Videos.** ETRA, 2024. [Paper](https://arxiv.org/abs/2404.07351).
7. Jenna Kang, Colin Groth, Tong Wu, Finley Torrens, Patsorn Sangkloy, Gordon Wetzstein, and Qi Sun. **Infinite Gaze Generation for Videos with Autoregressive Diffusion.** arXiv:2603.24938, 2026. [Paper](https://arxiv.org/abs/2603.24938).
8. Asim H. Dar, Adina S. Wagner, and Michael Hanke. **REMoDNaV: robust eye-movement classification for dynamic stimulation.** Behavior Research Methods 53:399-414, 2021. [Paper](https://doi.org/10.3758/s13428-020-01428-x).
9. Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. **LoRA: Low-Rank Adaptation of Large Language Models.** arXiv:2106.09685, 2021. [Paper](https://arxiv.org/abs/2106.09685).
10. Francisco M. Costela and Russell L. Woods. **A free database of eye movements watching "Hollywood" videoclips.** Data in Brief 25:103991, 2019. [Paper](https://doi.org/10.1016/j.dib.2019.103991).
11. OpenGVLab. **InternVL3.5-8B-HF: official model release and configuration.** Revision `741a7d03020411e666c6109218ab71e08151ef86`. [Model card](https://huggingface.co/OpenGVLab/InternVL3_5-8B-HF).
12. Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. **Deep Residual Learning for Image Recognition.** CVPR, 2016. [Paper](https://arxiv.org/abs/1512.03385).
13. Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. **Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation.** EMNLP, 2014. [Paper](https://arxiv.org/abs/1406.1078).

## Appendix A. Reproduction Details

### A.1 Fixed settings

| Component | Setting |
|---|---|
| Base model | `OpenGVLab/InternVL3_5-8B-HF` |
| Base revision | `741a7d03020411e666c6109218ab71e08151ef86` |
| Gaze initialization | Official combined free-viewing adapter, preserved without video-task training |
| LoRA | Rank 32; alpha 64; dropout 0.05; one adapter |
| Trainable modules | Language-model attention q/k/v/o projections and gate/up/down feed-forward projections |
| Trainable parameters | 87,293,952 |
| Precision | BF16 base; FP32 LoRA |
| Gaze context | Up to four completed foveations; start/end XY and relative times |
| Video context | One frame or four slots; nominal one-second span |
| Image processor | One 448 x 448 patch per image; no dynamic tiling or truncation |
| Maximum token length | 4,096 |
| Observed D1 token range | 443-532 |
| Observed D4/R4 token range | 1,266-1,355 |
| Optimizer | AdamW; learning rate 2e-5; weight decay 0.01 |
| Batch and clipping | Microbatch 1; effective batch 16; gradient norm 1.0 |
| Schedule | 1,000 updates; 50-update warmup; cosine decay |
| Checkpoint comparison | Every 100 updates; highest validation film-macro score; earliest tie |
| Principal DeepGaze training seeds | 0, 1, 2; matched across conditions |
| Reference-baseline comparison | Seed-0 D1 and neural AR; deterministic spatial/linear fits |
| Shared IG reference | Training-only Scott KDE; one fixed 100 x 100 probability map for every model |
| Visual AR encoder | Frozen torchvision ResNet18; `ResNet18_Weights.IMAGENET1K_V1`; 2 x 2 pooling |
| Bootstrap | 5,000 film resamples; seed 20260929 |
| Event processing | REMoDNaV 1.1.2; 250 Hz interpolation; maximum gap 10 ms |
| Reference angular conversion | 0.01304628456 degrees/pixel; not independently verified per trial |
| Runtime | PyTorch 2.9.0+cu128; Transformers 4.57.6; PEFT 0.18.1 |

The selected D1 updates for seeds 0/1/2 are 400/400/300; R4 and D4 select 500/400/300. The seed-0 D1, R4, and D4 runs used one H100 GPU each and took approximately 4.19, 4.77, and 4.71 hours respectively, including their scheduled validations. These are whole-run elapsed times for those three runs, not the total cost of the nine-run study or per-target inference benchmarks. Every reported principal run completed 1,000 updates and ten scheduled validations. Frozen base-weight checksums remain unchanged after adaptation.

### A.2 Traceability of the reported evidence

The data and result artifacts identify the exact experimental population and checkpoint selection. Their identifiers are included here for reproducibility; they are not additional experiments.

| Evidence | Artifact within the project storage root |
|---|---|
| Event validity and reconstruction assumptions | `runs/audit-5819954/audit/summary.json` |
| Target definition, counts, and exclusions | `runs/build-5819982/targets/build-summary.json` |
| Model-input scan and data hashes | `runs/export-5820032/inputs/scan.json` |
| Seed-0 D1 / R4 / D4 training | `runs/train-5820037/training/`, `runs/train-5820038/training/`, `runs/train-5820039/training/` |
| Seed-1 D1 / R4 / D4 training | `runs/replicate-5823926/training/`, `runs/replicate-5823927/training/`, `runs/replicate-5823928/training/` |
| Seed-2 D1 / R4 / D4 training | `runs/replicate-5823929/training/`, `runs/replicate-5823930/training/`, `runs/replicate-5823931/training/` |
| Seed-0 paired comparisons | `runs/compare-5823916/{best,final}/comparison.json` |
| Seed-1 paired comparisons | `runs/compare-5824644/{best,final}/comparison.json` |
| Seed-2 paired comparisons | `runs/compare-5824754/{best,final}/comparison.json` |
| Tables 3-5: three-seed selected checkpoints | `runs/aggregate-5824755/best/comparison.json` |
| Table 6: three-seed common final update | `runs/aggregate-5824755/final/comparison.json` |
| Tables 3 and 5: common KDE, spatial/history references and visual AR | `runs/baseline-5823952/comparison/comparison.json` |
| Corrected spatial/history baseline audit | `runs/baseline-5823937/comparison/comparison.json` |
| Neural AR checkpoints, scores and heatmaps | `runs/ar-5823951/training/` |

The input scan SHA-256 is `c99ffc5c65b8c5ee84bffe1edd6606b3ba06547de8d411b5fc5fe9c735501dfb`; the target manifest SHA-256 is `3d5f6de0e8f67bb8ec95174a750d83c11eb9355cdd2aa06ed826f3adb8e626e0`. Per-condition protocols record training-source hashes, initialization identity, and runtime versions. Selected comparisons verify input identity and saved prediction/checkpoint hashes. The shared-reference IG values are derived from these paired likelihood exports, not from the `ig_bits` fields in older training logs: those fields use the historical histogram prior. The new presentation subtracts the same KDE log likelihood from every model. These records support reconstruction of the stated development comparison but do not resolve the acquisition assumptions discussed in Section 5.2.

### A.3 Availability

The source dataset and original pretrained model are publicly documented in references 10, 2, and 11. This repository includes active source code, tests, the aggregate benchmark in `reports/results.json`, and a script to regenerate its figure. Project-specific targets, historical code snapshots, paired predictions, and trained adapters remain in project storage; no external download endpoint is provided for those artifacts. Full independent reproduction would additionally require the fixed split, recovered event annotations, exact preprocessing records, and model artifacts. The released aggregate summary supports checking the stated comparisons and regenerating the figure, but does not replace those inputs.

The experiments use existing public data and pretrained model releases; no new participant data were collected for this study. Access and redistribution remain subject to the original data and model terms. The present manuscript does not claim a newly released public benchmark or an independently validated clinical or cognitive assessment tool.
