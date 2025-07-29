# ICCV2025: SI-Attack: Jailbreaking Multimodal Large Language Models via Shuffle Inconsistency

The core code of Jailbreaking Multimodal Large Language Models via Shuffle Inconsistency. 

Our paper can be viewed in [Here](https://arxiv.org/abs/2501.04931)

The evaluation result on GPT-4o-05-13 can be found in [Here](https://drive.google.com/drive/folders/1F2VdH_mPblwe2_PZCfbsfgqAsjy5OMR4?usp=drive_link)

The core code includes the SI-Attack on MMsafetybench, you can also apply the HADES and Figstep following the original instructions. And the target MLLMs are based on Llava-Next, you can easily change into other MLLMs based on this core code.

It should be mentioned that the toxic score is based on the judge prompt. The detailed implementation of using ChatGPT-3.5(Azure) is not provided.  You need to do this by yourself in function judge(prompt=""). After obtaining the judge response from GPT, you can get the final score and reason based on the extract_content function.

### Citation

```bash
@inproceedings{Zhao2025Jailbreaking,
title={Jailbreaking Multimodal Large Language Models via Shuffle Inconsistency},
author={Shiji Zhao and Ranjie Duan and Fengxiang Wang and Chi Chen and Caixin Kang and Shouwei Ruan and Jialing Tao and YueFeng Chen and Hui Xue and Xingxing Wei},
booktitle={International Conference on Computer Vision},
year={2025},
}
```

