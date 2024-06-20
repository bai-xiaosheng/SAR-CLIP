# CLIP Multimodal Model for SAR Image Few-Shot Recognition

## Project Overview

This project leverages CLIP and other multimodal models to achieve groundbreaking few-shot and zero-shot recognition for SAR (Synthetic Aperture Radar) images. Our goal is to enable accurate target identification and classification in real-world environments without requiring a large number of enemy target samples. This work holds significant potential not only in the military domain but also in fields such as remote sensing and medical imaging.

## Objectives

- **Multimodal Models**: Utilize advanced models like CLIP to achieve few-shot and zero-shot recognition of SAR images.
- **Real-World Application**: Address the challenge of identifying targets in real-world scenarios with limited target samples.

## Methodology

### Approach Selection

- **Multimodal Technology**: Given that current technology requires textual or other auxiliary information to achieve zero-shot recognition, multimodal models are the optimal choice.
- **Model Research**: We conducted thorough research and evaluation of relevant SAR image models, remote sensing models, medical imaging models, and general models.

### Dataset Construction

- **Custom Dataset**: There is currently no existing multimodal SAR image dataset. We constructed a text-image dataset based on the MSTAR dataset to ensure data diversity and relevance.

### Model Training

- **Fine-Tuning Large Models**: Considering the high computational power and large datasets required for comprehensive fine-tuning, we adopted a fine-tuning approach.
  - **Input-Side Fine-Tuning**: Refine the input aspects of the model to enhance sensitivity and adaptability to different input information.
  - **Output-Side Fine-Tuning**: Optimize the output aspects of the model to improve recognition accuracy and robustness.

### Dataset Construction

Based on the features of each sample image in the MSTAR dataset, we constructed text descriptions. Here are some examples:  
- 'a photo of an infantry fighting vehicle'  
- 'a photo of an Armoured Transporter'  
- 'a photo of a T-72 Main Battle Tank'  
- 'a photo of a Wheeled armored transport vehicle'  
- 'a photo of a Self-propelled howitzer'  
- 'a photo of an Armored reconnaissance vehicle'  
- 'a photo of a bulldozer'  
- 'a photo of a T-62 Main Battle Tank'  
- 'a photo of a Cargo truck'  
- 'a photo of anti-aircraft guns'  

### Model Optimization

- **Continuous Optimization**: Ongoing model optimization to enhance recognition capabilities and accuracy across various environments and scenarios.

## Experimental Results

We will showcase the experimental results of the project through a dedicated webpage. Stay tuned!

## Contact Information

If you are interested in our project or have any questions, please contact us at:  
**Email**: 22021211621@stu.xidian.edu.cn

---

This project combines cutting-edge multimodal model technology with practical application needs, showcasing our team's innovation and efforts in the field of SAR image recognition. We look forward to your attention and feedback as we advance this field together!
