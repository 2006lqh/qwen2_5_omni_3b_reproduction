# Third-Party Source

`third_party/EfficientAI` points to the upstream EfficientAI repository:

```text
https://github.com/alibaba/EfficientAI
```

The reproduction uses upstream commit:

```text
3d32ae427eec57166ea67f3018cd4568be84496f
```

After initializing the submodule, apply:

```bash
cd third_party/EfficientAI
git apply ../../patches/efficientai_masquant_qwen2_5_omni_w4a8.patch
```

The patch records runtime compatibility changes used for this reproduction, including SDPA attention fallback, local lmms-eval task loading, result JSON emission, and Qwen2.5-Omni audio/input-device handling.
