---
title:  "VLM"

categories:
  - Blog
tags:
  - [VLM]

toc: true
toc_sticky: true
 
date: 2024-10-28
last_modified_at: 2024-10-28
---


<details>
<summary>🔗 참고자료</summary>
<div markdown="1">       
https://huggingface.co/blog/vision_language_pretraining
</div>
</details>

# VLM?
: 이미지, 텍스트를 contrastive하게 정렬하는 방식

- 이미지, 자연어(텍스트) 함께 처리하여 시각적, 언어적 특성 통합하는 모델
    - input, output, task에 따라 프로세스 달라짐
- 3 key elements로 구성
: text encoder + image encoder + 2개의 Encoder로부터 받은 정보 어떻게 처리할지?
- 최근에는 대부분 Transformer 아키텍처 사용
- *여러 downstream task 수행 위한 pre-training
: Contrastive learning, PrefixLM, Cross attention …*

# 1. Contrastive learning
![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7f21b3a3-26e9-46e9-b75b-9ad564d671e6/9cd86e8d-2054-4498-a704-2320a5e0baff/image.png)

> *Contrastive learning*?
> 
> - 입력 이미지와 텍스트를 동일한 feature space에 매핑하여 
> 이미지-텍스트 쌍의 임베딩 사이의 거리가 일치할 때 최소화, 일치하지 않는 경우 최대화

- {image, caption}으로 구성된 대규모 데이터셋
: 텍스트 인코더와 이미지 인코더를 contrastive loss 사용하여 함께 학습
→ 시각적 특성과 언어적 특성 연결
- `CLIP`: 텍스트, 이미지 임베딩 사이의 코사인 거리
- `ALIGN`, `DeCLIP`: noisy한 데이터 고려하여 자체 distance metric 사용

# 2. PrefixLM
: 이미지를 텍스트의 prefix로 사용 → 이미지, 텍스트 임베딩 함께 학습하는 방식

1. **PrefixLM
:** 입력 텍스트가 주어지면 다음 토큰 예측하는 방식
*"A man is standing at the” → "corner”*

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7f21b3a3-26e9-46e9-b75b-9ad564d671e6/786c749a-4dcd-4065-b3d2-f881b38bb95d/image.png)

- `SimVLM`
: ViT(이미지를 패치 단위로 나눠서 모델에 입력, 예측하는 방식) 아이디어 적용한 아키텍처
    - Encoder: [연결된 이미지 패치 시퀀스 + Prefix 텍스트 시퀀스] 입력
    → Decoder: 다음에 올 텍스트 시퀀스 예측

: prefixLM 만 사용한 모델은 image captioning, Visual QA …에만 제한적으로 응용 가능

⇒  multi-modal representation 학습 or 하이브리드 접근방식 사용 모델
: object detection, image segmentation 같은 다양한 task에 맞게 조정 가능

2. **Frozen PrefixLM**
    - 시각 정보 LM에 융합 → 성능 ↑
    - LM fine-tuning 없이 사용 가능
        - Vision encoder가 입력 이미지로부터 임베딩 생성 → LM이 텍스트로 변환
        : LM은 frozen, vision encoder만 학습됨
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7f21b3a3-26e9-46e9-b75b-9ad564d671e6/ac29b0f0-3f14-48ca-bd40-b849460a6cee/image.png)
    
    - `Frozen`, `ClipCap`
        - [이미지 임베딩 + prefix text] 주어지면 캡션에서 다음 토큰 생성하도록 정렬된 image-text 데이터셋으로 학습
    - `Flamingo`
        - Vision encoder, Frozen LM 사이에 cross-attention layer 추가
        → few shot 성능 ↑


# 3. Multi-modal Fusing with Cross Attention
![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7f21b3a3-26e9-46e9-b75b-9ad564d671e6/e2a4961e-7e17-4f88-b43c-b7727f25f5df/image.png)

> *Cross attention*
> 
> - 하나의 모달리티(이미지)에서 추출된 정보를 다른 모달리티(텍스트)와 결합해서 서로 참조하는 방식 → 각 모달리티가 서로의 정보 보고 활용할 수 있도록 함
> - 텍스트 생성 능력 — 시각적 정보 조합 효율적으로 균형 있게 유지
- `VisualGPT`, `VC-GPT`, `Flamingo`→ image captioning, visual QA
- 효율적 데이터 활용 가능 ⇒ 큰 multimodal 데이터셋 없을 때 중요


# 4. MLM (Masked-Language Modeling) + ITM (Image-Text Matching)
: 이미지 세부 정보를 텍스트로 정렬하고 downstream task 수행

> **MLM (Masked-Language Modeling)**
> 
> - 일부 단어 마스킹한 뒤 모델이 예측하도록 학습
> ⇒ 이미지 정보, 텍스트 정보 결합해서 마스킹된 단어 예측하도록 사용
> - 주변 단어, 이미지에 포함된 시각적 정보 사용하여 마스킹 된 단어 예측하도록 학습
> → 이미지 특정 부분과 특정 단어가 align 되는 관계 학습하게 됨
> 
> **ITM (Image-Text Matching)**
> 
> - 이미지-텍스트 쌍이 실제로 일치하는지 여부 예측
> : 이미지, 텍스트 간의 연관성 이해, 서로 잘 맞는지 판별할 수 있도록 함

![Aligning parts of images with text ([**image source**](https://arxiv.org/abs/1908.02265))](https://prod-files-secure.s3.us-west-2.amazonaws.com/7f21b3a3-26e9-46e9-b75b-9ad564d671e6/05d3c428-f8a4-4543-959c-3368aa83c186/image.png)

Aligning parts of images with text ([**image source**](https://arxiv.org/abs/1908.02265))

- **MLM + ITM → 이미지-텍스트 align, 이해 능력 ↑**
    - MLM: 텍스트, 이미지 내 개별 객체, 부분 간의 연관성 학습
    ITM: 이미지, 텍스트의 전반적 일치 여부 학습
        
        → 단순한 일치 여부 판별 넘어서 텍스트의 특정 단어가 이미지 내 특정 영역과 어떻게 관련되는지 자세하게 이해 가능
        
- `VisualBERT`, `LXMERT`
: Faster R-CNN 통해 인식된 이미지 객체와 텍스트 정렬, MLM+ITM 결합하여 학습
- `FLAVA`
: MLM, ITM 이외에도 Masked Image Modeling(MIM), contrastive learning 사용하여 학습
→ 범용성 ↑


# 5. No training
: 추가적 학습, fine-tuning 없이 기존에 pre-train된 이미지, 텍스트 모델 활용

- 기존 pretrained 모델 사용하여 이미지와 텍스트 간의 representation 연결하거나 새로운 멀티모달 태스크에 적용하는 방식
    
    : 데이터셋 부족하거나 학습 리소스 제한적인 상황에서 효율적으로 태스크 수행 가능
    

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7f21b3a3-26e9-46e9-b75b-9ad564d671e6/c2dea58f-6d34-4247-82ce-e01e09b56178/image.png)

- `MaGiC`: CLIP + pre-trained autoregressive LM
    - 반복적 최적화 통해 이미지 캡션 생성
    - CLIP 기반의 *magic score* 사용하여 캡션 품질 평가
- `ASIF`
    - 핵심 아이디어: *“유사한 이미지는 캡션도 유사할 것!”*
    - 소규모 멀티모달 데이터셋에서 이미지-텍스트 쌍 기준으로 similarityu-based search 수행
    : 새로 입력된 이미지와 유사한 이미지 찾고 해당 유사 이미지의 캡션 재사용하는 방식으로 캡션 생성
