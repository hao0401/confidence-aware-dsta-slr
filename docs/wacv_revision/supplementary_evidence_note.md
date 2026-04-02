# Supplementary Evidence Note

## 1. Confidence Signal Validity

Under the same code path on WLASL100, the four-stream Top-1 results are:

- original confidence: `52.35%`
- constant confidence: `46.14%`
- shuffled confidence: `47.32%`
- removed confidence signal: `41.78%`

Interpretation:

- The gain cannot be explained by merely appending one extra channel.
- Preserving the original correspondence between confidence and observation quality is important.
- This result supports the claim that confidence acts as a useful reliability prior rather than a generic auxiliary feature.

Suggested wording:

To further verify that the improvement does not simply arise from adding one extra input channel, we compare original confidence, constant confidence, shuffled confidence, and removed confidence under the same code path. The four-stream Top-1 results are 52.35%, 46.14%, 47.32%, and 41.78%, respectively, indicating that the original confidence-quality correspondence is materially informative for recognition.

## 2. Public Strong Baseline Comparison

Under the same perturbation protocol, the joint-stream comparison against the public DSTA-SLR strong baseline is:

- clean: `66.44% vs. 65.60%` (`-0.84`)
- missing 10%: `54.53% vs. 63.42%` (`+8.89`)
- missing 20%: `30.37% vs. 58.05%` (`+27.68`)
- missing 30%: `14.77% vs. 48.83%` (`+34.06`)
- noise 5: `64.77% vs. 63.76%` (`-1.01`)
- noise 10: `62.42% vs. 59.06%` (`-3.36`)
- noise 20: `53.69% vs. 43.46%` (`-10.23`)

Interpretation:

- The proposed structure enhancement does not outperform the public strong baseline under every condition.
- Its main advantage lies in missing-joint scenarios, especially under severe keypoint dropout.
- This evidence supports the paper’s intended positioning: a robustness module specialized for degraded observations, not a universally stronger clean-set recognizer.

Suggested wording:

Compared with the public DSTA-SLR strong baseline under the same perturbation protocol, the proposed structure-enhanced version does not dominate in clean or heavy-noise settings, but shows substantial gains under missing-joint perturbations, especially at 20% and 30% dropout. This observation is consistent with our main conclusion that structural reliability modeling primarily targets missing-type degradation, while strong-noise robustness requires additional training-stage consistency constraints.

## 3. Recommended Placement

- Main paper:
  - Mention the confidence-signal validity result briefly in Discussion or Conclusion.
  - Mention the public-strong-baseline comparison briefly in Discussion.
- Supplementary:
  - Add one subsection titled `Confidence Signal Validity`.
  - Add one subsection titled `Comparison with a Public Strong Baseline Under Unified Perturbations`.

## 4. Core Message

These additional results strengthen the paper in two ways:

1. They make the novelty claim more defensible by showing that the gain is not caused by a trivial auxiliary channel.
2. They make the robustness claim more credible by showing where the method helps, and where it does not.
