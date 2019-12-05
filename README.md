# Food Generation From Ingredients and Cooking Method
## Team
**Team name**: 0 error, 0 warning

**Team members**: Hanyuan Xiao, Kaijie Cai, Buke Ao, Heng Zhang

<p align="center">
  <img src="img_1.PNG" alt="Realistic Generated Food Images"/>
</p>

## Motivation
Food image generation as one of image generation tasks is useful in visualization for almost everyone. Given food ingredients and cooking methods (e.g. bake, grill), people may wonder name and image of the dish that can be cooked. For example, chefs may want to try so many new ingredients and cooking method to invent new menu. Parents may be worried about whether dinner will be attractive to their children and consider nutrients at the same time. Based on the same ingredients, can we make something new and interesting? Even students who have a deadline may want to spend the minimum time to cook their lunch or dinner with whatever in the fridge. Therefore, such an image generator can provide a high-level idea about what they can cook. 

Besides sparks and interest that can be brought to the public in this project, outputs of our model can also be used to evaluate and quantify vital criteria of food with attention drawn by Computational food analysis (CFA) [0] such as meal preference forecasting, and computational meal preparation. Therefore, the model defines its importance and usage in real life and is crucial to human life. Existing approaches such as The Art of Food do not take cooking method as input. However, the importance has been overshadowed since the same ingredients can be made into different dishes. For instance, chicken and noodles can be made in ramen or fried noodles by boiling and stir-fry, respectively. Therefore, this project aims at developing a reliable method to generate food image that fits in any specific class.

## Problem Statement
For this project, we trained a deep learning network to learn and generate food images with ingredients and cooking methods as input. Typically, we leveraged Generative Adversarial Network (GAN) for our task.

## Related Works & Method
### Related Works
Objects in Images have many attributes that could represent their visual information. On the other hand, the attributes could be described by texts either. Hence, if the connection between images and texts is learned, then we could generate images by texts. Furthermore, the problem could be solved by two steps, first is to learn the text feature representations that related to the key visual details and the second is to generate images from the text feature representations where the visual attributes are the same to word descriptions.

The connection between the image pixel and the text description is highly multimodal, there are many possible mapping relationships between them. This multimodal learning is hard and finding the shared representation across different modalities is essential, besides, the generalization to unseen data is also a basic problem.

One way to generate images from texts is implemented by coding the texts into class labels, which may lose information and inaccurate because the class labels are not good representations of original texts. Instead of directly using class labels, [1] proposes an end-to-end architecture to generate images from text encodings by RNN and GAN, but the associations between texts and images are not well established. In this report, we use two stages of an association model and a generative model to address this problem.

### Method
To address this problem, we use a recipe association model which could find the common representations (embeddings) between images and text ingredients, and then use a GAN to generate images from the embeddings.
<p align="center">
  <img src="img_2.jpg" alt="Association Model Between Ingredients + Methods and Images"/>
  <em>Figure 2. Association Model Between Ingredients + Methods and Images</em>
</p>

The loss function of association model is:

![equation](https://latex.codecogs.com/svg.latex?\inline&space;V(F_p,F_q)=\mathop{\mathbb{E}}_{\hat{p}(r^+,v^+),\hat{p}(v^-)}min([cos[\textbf{p}^+,\textbf{q}^+]-cos[\textbf{p}^+,\textbf{q}^-]-\epsilon],0)\newline\hphantom{asdfasdfsadf}+\mathop{\mathbb{E}}_{\hat{p}(r^+,v^+),\hat{p}(r^-)}min([cos[\textbf{p}^+,\textbf{q}^+]-cos[\textbf{p}^-,\textbf{q}^+]-\epsilon],0)),

where ![equation](https://latex.codecogs.com/svg.latex?\inline&space;(\textbf{p}^+,\textbf{q}^+)) is positive pair between ingredient embeddings and image embeddings. ![equation](https://latex.codecogs.com/svg.latex?\inline&space;(\textbf{p}^+,\textbf{q}^-)), ![equation](https://latex.codecogs.com/svg.latex?\inline&space;(\textbf{p}^-,\textbf{q}^+)) are negative paris. ![equation](https://latex.codecogs.com/svg.latex?\inline&space;\epsilon) is the bias to train the model on pairs not correctly associated.

This model takes ingredients and cooking methods as text inputs, and uses images as inputs of another side in Figure 2. The ingredients and cooking methods are encoded by LSTM and concatenated together to get the representation embedding of texts. The embeddings of images are generated by ResNet and then fine-tuned by our dataset. Finally a cosine similarity is used as the loss function. The LSTM has XXX hidden units and XXX layers. We trained the model XXX epoches to get a final loss of XXX. 

<p align="center">
  <img src="img_3.jpg" alt="Image Generation by StackGAN"/>
  <em>Figure 3. Image Generation by StackGAN</em>
</p>
We first get the text embeddings from the ingredients by the trained LSTM networks in the association model. The text embeddings are then used as the condition codes in our stackGAN. In order to ascertain the food image has the expected ingredients that it depends on, we add a cycle-consisitency term to make the similarity between the fake images and text embeddings of ingredients strong.

The overall loss function of image generation by GAN is:

![equation](https://latex.codecogs.com/svg.latex?\inline&space;L_G=\sum_{i=0}^2(L_{G_i}^{cond}+\lambda_{uncond}L_{G_i}^{uncond}-\lambda_{cycle}L_{C_i})+\lambda_{ca}L_{ca})

In the equation, the losses of generator from both conditioned and unconditioned are considered. The loss of cycle-consistency constraint is also incorporated and the last part is the regularization factor, so as to make the distribution of conditions given image embeddings approximates the standard Gaussian distribution as closely as possible.

## Experiment
### Dataset
We conduct our experiments using data from recipe1M [4]. The recipe1M dataset consists of more than 1 million food images with corresponding ingredients and instructions. We manually chose 12 different types of cooking method that we think are meaningful and distinguishable, and generated cooking method for each image by searching for keywords in the instruction text. We also reduced the number of different ingredients from around 18,000 to around 2,000 by removing ingredients with low frequency ( < 500 occurrence in the dataset)  and combining ingredients that are the same or very close.  Because of the limit of time and computing resources we used only 10,000 data from the dataset.

### Input
We feed association model with paired and unpaired 128 &#215; 128 image and words. For the stackgan model, we use both 64 &#215; 64 and 128 &#215; 128 images because there are two discriminators differential two resolution images from generators.

## Evaluation
|     |              |             | im2rcp     |            |             |             | rcp2im     |            |             |
|-----|--------------|-------------|------------|------------|-------------|-------------|------------|------------|-------------|
|     |              | MedR&#8595; | R@1&#8593; | R@5&#8593; | R@10&#8593; | MedR&#8595; | R@1&#8593; | R@5&#8593; | R@10&#8593; |
| 1K  | Model in [1] | 5.500       | 0.234      | 0.503      | 0.618       | 5.750       | 0.230      | 0.491      | 0.615       |
|     | Ours         | **4.400**   | **0.261**  | **0.549**  | **0.679**   | **4.200**   | **0.270**  | **0.556**  | **0.682**   |
| 5K  | Model in [1] | 24.000      | 0.099      | 0.265      | 0.364       | 25.100      | 0.097      | 0.259      | 0.357       |
|     | Ours         | **17.900**  | **0.116**  | **0.299**  | **0.406**   | **16.700**  | **0.129**  | **0.315**  | **0.421**   |
| 10K | Model in [1] | 47.000      | 0.065      | 0.185      | 0.267       | 48.300      | 0.061      | 0.178      | 0.261       |
|     | Ours         | **34.900**  | **0.077**  | **0.212**  | **0.301**   | **32.700**  | **0.088**  | **0.229**  | **0.319**   |


## Future Improvements
From the experiments, we find that there are some improvements can be made in the future. Firstly, reducing the number of ingredients further. For example, we may combine different kinds of cheeses as they have similar appearance and contribution to the generated images. Such change will reduce the redundancy in the dataset and make it easier to learn. Secondly, balancing the number of images with different color to prevent the model from the inclination to generate reddish and yellowish images. Finally, we may further investigate the way to better control the contribution of conditional inputs as we found that it sometimes generated irrelevant image. Attention mechanism and regularization loss can be the options.

## Contributions
We acknowledge the assistance and advice from professor [Joseph Lim](https://viterbi-web.usc.edu/~limjj/) and TAs of course CS-566 (Deep Learning and its Applications). With their guidance, we developed the project and made the following contributions.
* A conditional GAN model for food image generation task with ingredients and cooking methods as input
* A refined version of dataset Recipe1M which further contains cooking methods extracted from instructions
* Quantitative data that proves cooking method as a useful and valuable input to food image generation tasks

## References
[1] Reed, Scott, et al. "Generative adversarial text to image synthesis." arXiv preprint arXiv:1605.05396 (2016).
