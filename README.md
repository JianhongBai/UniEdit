
# UniEdit: A Unified Tuning-Free Framework for Video Motion and Appearance Editing
## [<a href="https://jianhongbai.github.io/UniEdit/" target="_blank">Project Page</a>] [<a href="https://arxiv.org/abs/2402.13185" target="_blank">arXiv</a>]


![fig1_demo_video](https://github.com/JianhongBai/UniEdit/assets/74419686/c18d11a1-fe02-473e-9133-65758b787bed)

**UniEdit** supports both video ***motion*** editing in the time axis (i.e., from playing guitar to eating or waving) and various video ***appearance*** editing scenarios (i.e., stylization, rigid/non-rigid object replacement, background modification).

>**Abstract:** Recent advances in text-guided video editing have showcased promising results in appearance editing (e.g., stylization). However, video motion editing in the temporal dimension (e.g., from eating to waving), which distinguishes video editing from image editing, is underexplored. In this work, we present UniEdit, a tuning-free framework that supports both video motion and appearance editing by harnessing the power of a pre-trained text-to-video generator within an inversion-then-generation framework. To realize motion editing while preserving source video content, based on the insights that temporal and spatial self-attention layers encode inter-frame and intra-frame dependency respectively, we introduce auxiliary motion-reference and reconstruction branches to produce text-guided motion and source features respectively. The obtained features are then injected into the main editing path via temporal and spatial self-attention layers. Extensive experiments demonstrate that UniEdit covers video motion editing and various appearance editing scenarios, and surpasses the state-of-the-art methods.

**Features**:<br>

- **Versatile**: supports both video motion editing and various video appearance editing scenarios.
- **Tuning-free**: no training or optimization required.
- **Flexibility**: compatible with off-the-shelf T2V models.

https://github.com/JianhongBai/UniEdit/assets/74419686/1d740f83-705d-4f59-89a6-21a61d1ebf14


Please visit the [project webpage](https://jianhongbai.github.io/UniEdit/) to see more results and information.

## Updates

- [ ] :computer: Code (coming soon).
- [x] :page_facing_up: Paper released on [arXiv](https://arxiv.org/abs/2402.13185).
