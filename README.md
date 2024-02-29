
# UniEdit: A Unified Tuning-Free Framework for Video Motion and Appearance Editing
## [<a href="https://jianhongbai.github.io/UniEdit/" target="_blank">Project Page</a>] [<a href="https://arxiv.org/abs/2402.13185" target="_blank">arXiv</a>]


![fig1_demo_video](https://github.com/JianhongBai/UniEdit/assets/74419686/c18d11a1-fe02-473e-9133-65758b787bed)


## Support Various Editing Scenarios

**UniEdit** supports both video ***motion*** editing in the time axis (i.e., from playing guitar to eating or waving) and various video ***appearance*** editing scenarios (i.e., stylization, rigid/non-rigid object replacement, background modification).


<div align="center">
  
  **1. Motion editing.**
  <img src="https://github.com/JianhongBai/UniEdit/assets/74419686/4b66c473-21f0-41ee-9371-d1c1441cef86" style="width: 50%; margin-top: 0px;"/>
</div>
<div align="center">

  **2. Stylization.**
  <img src="https://github.com/JianhongBai/UniEdit/assets/74419686/c3caf23b-933e-400d-aaa3-343c9a440a4a" style="width: 50%; margin-top: 0px;"/>
</div>
<div align="center">
  
  **3. Rigid object replacement.**
  <img src="https://github.com/JianhongBai/UniEdit/assets/74419686/407747c7-45c9-44f4-8c28-c698bbdd2757" style="width: 50%; margin-top: 0px;"/>
</div>
<div align="center">
  
  **4. Non-rigid object replacement.**
  <img src="https://github.com/JianhongBai/UniEdit/assets/74419686/2e7117cd-0641-48f6-ac39-8da3cfbb7f70" style="width: 50%; margin-top: 0px;"/>
</div>
<div align="center">
  
  **5. Background modification.**
  <img src="https://github.com/JianhongBai/UniEdit/assets/74419686/90f6eff3-9a15-4dee-8d11-5ee2a7223514" style="width: 50%; margin-top: 0px;"/>
</div>

## Introduction
>**Abstract:** Recent advances in text-guided video editing have showcased promising results in appearance editing (e.g., stylization). However, video motion editing in the temporal dimension (e.g., from eating to waving), which distinguishes video editing from image editing, is underexplored. In this work, we present UniEdit, a tuning-free framework that supports both video motion and appearance editing by harnessing the power of a pre-trained text-to-video generator within an inversion-then-generation framework. To realize motion editing while preserving source video content, based on the insights that temporal and spatial self-attention layers encode inter-frame and intra-frame dependency respectively, we introduce auxiliary motion-reference and reconstruction branches to produce text-guided motion and source features respectively. The obtained features are then injected into the main editing path via temporal and spatial self-attention layers. Extensive experiments demonstrate that UniEdit covers video motion editing and various appearance editing scenarios, and surpasses the state-of-the-art methods.

**Features**:<br>

- **Versatile**: supports both video motion editing and various video appearance editing scenarios.
- **Tuning-free**: no training or optimization required.
- **Flexibility**: compatible with off-the-shelf T2V models.

## Demo
<!-- https://github.com/JianhongBai/UniEdit/assets/74419686/0f1bf5a8-600b-4834-a734-74ea8104971f -->


Please visit the [project webpage](https://jianhongbai.github.io/UniEdit/) to see more results and information.

## Updates

- [ ] :computer: Code (The code will be released when the paper is accepted).
- [x] :page_facing_up: Paper released on [arXiv](https://arxiv.org/abs/2402.13185).
