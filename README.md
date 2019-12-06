# Food Generation From Ingredients and Cooking Method
## Team
**Team name**: 0 error, 0 warning

**Team members**: Hanyuan Xiao, Kaijie Cai, Buke Ao, Heng Zhang

<p align="center">
  <img src="img_1.PNG" alt="Realistic Generated Food Images"/>
  <br><em>Figure 1. Realistic Generated Food Images</em></br>
</p>

## Motivation
Food image generation as one of image generation tasks is useful in visualization for almost everyone. Given food ingredients and cooking methods (e.g. bake, grill), people may wonder name and image of the dish that can be cooked. For example, chefs may want to try so many new ingredients and cooking method to invent new menu. Parents may be worried about whether dinner will be attractive to their children and consider nutrients at the same time. Based on the same ingredients, can we make something new and interesting? Even students who have a deadline may want to spend the minimum time to cook their lunch or dinner with whatever in the fridge. Therefore, such an image generator can provide a high-level idea about what they can cook. 

Besides sparks and interest that can be brought to the public in this project, outputs of our model can also be used to evaluate and quantify vital criteria of food with attention drawn by Computational food analysis (CFA) [0] such as meal preference forecasting, and computational meal preparation. Therefore, the model defines its importance and usage in real life and is crucial to human life. Existing approaches such as The Art of Food do not take cooking method as input. However, the importance has been overshadowed since the same ingredients can be made into different dishes. For instance, chicken and noodles can be made in ramen or fried noodles by boiling and stir-fry, respectively. Therefore, this project aims at developing a reliable method to generate food image that fits in any specific class.

## Problem Statement
For this project, we trained a deep learning network to learn and generate food images with ingredients and cooking methods as input. Typically, we leveraged Generative Adversarial Network (GAN) for our task.

## Related Works & Method
### Related Works
Objects in images have many attributes that represent their visual information. On the other hand, the attributes could be described by texts either. Hence, if the connection between images and texts is learned, then we are able to generate images with text as input. Furthermore, the problem could be solved by two steps. 

* The first is to learn the text feature representations that are related to the key visual details.
* The second is to generate images from the text feature representations where the visual attributes are the same to word descriptions.

The connection between the image pixel and the text description is highly multimodal, there are many possible mapping relationships between them. This multimodal learning is hard but finding the shared representation across different modalities is essential, besides, the generalization to unseen data is also a basic problem.

One way to generate images from texts is implemented by encoding the texts into class labels, which may cause loss of information and inaccuracy because the class labels are not good representations of original texts and there can be a large number of classes due to diverse combination of texts that the model cannot handle. Instead of directly using class labels, [1] proposed an end-to-end architecture to generate images from text encodings by RNN and GAN, but the associations between texts and images as well as loss functions are not well established. In this project, we use two stages -- an association model and a generative model -- to address this problem.

### Method
To address this problem, we use a recipe association model which is able to find the common representations (i.e. text embeddings) between images and text input, and then a GAN to generate images from the embeddings.
#### Cross-modal Association Model ####
<p align="center">
  <img src="img_2.jpg" alt="Association Model From Ingredients + Methods and Images"/>
  <em>Figure 2. Association Model From Ingredients + Methods and Images</em>
</p>

The loss function of association model is:

![equation](https://latex.codecogs.com/svg.latex?\inline&space;V(F_p,F_q)=\mathop{\mathbb{E}}_{\hat{p}(r^+,v^+),\hat{p}(v^-)}min([cos[\textbf{p}^+,\textbf{q}^+]-cos[\textbf{p}^+,\textbf{q}^-]-\epsilon],0)\newline\hphantom{asdfasdfsadf}+\mathop{\mathbb{E}}_{\hat{p}(r^+,v^+),\hat{p}(r^-)}min([cos[\textbf{p}^+,\textbf{q}^+]-cos[\textbf{p}^-,\textbf{q}^+]-\epsilon],0)),

where ![equation](https://latex.codecogs.com/svg.latex?\inline&space;(\textbf{p}^+,\textbf{q}^+)) is positive pair between text embeddings and extracted image features. ![equation](https://latex.codecogs.com/svg.latex?\inline&space;(\textbf{p}^+,\textbf{q}^-)), ![equation](https://latex.codecogs.com/svg.latex?\inline&space;(\textbf{p}^-,\textbf{q}^+)) are negative paris. ![equation](https://latex.codecogs.com/svg.latex?\inline&space;\epsilon) is the bias to train the model on pairs that are not correctly associated.

This network takes ingredients and cooking methods as input from one side, and uses images as input from another side as shown in Figure 2. The ingredients and cooking methods are encoded by LSTM and concatenated together to get the representative text embedding. The feature extraction from images is achieved by [ResNet](https://arxiv.org/abs/1512.03385) and then tuned based on our dataset and task. Finally, a cosine similarity is used to compute Euclidean distance between image features and text embedding. Ideally, for positive pairs of image and corresponding text embedding, the similarity is as large as 1; for negative pairs, the similarity is a smaller than a marginal value based on task and dataset.

#### Conditional StackGAN ####
<p align="center">
  <img src="img_3.jpg" alt="StackGAN for Image Generation"/>
  <em>Figure 3. StackGAN for Image Generation</em>
</p>
After we extracted meaningful and respresentative text embedding from ingredients and cooking methods by trained network in the association model. The text embedding for each training case is then used as the conditional code in StackGAN. In order to ascertain the food image has the expected ingredients and methods that it depends on, we added cycle-consistency constraint [] to guarantee the similarity between generated fake images and text embedding strong.

The loss function [] for image generation used in conditional GAN is:

![equation](https://latex.codecogs.com/svg.latex?\inline&space;L_G=\sum_{i=0}^2(L_{G_i}^{cond}+\lambda_{uncond}L_{G_i}^{uncond}-\lambda_{cycle}L_{C_i})+\lambda_{ca}L_{ca})

In the equation, we exploited both conditioned and unconditioned loss for discriminator. The loss of cycle-consistency constraint is  incorporated as the <img src="L_c_i.PNG"> term. The last part is the regularization factor, which aims at ensuring the distribution of conditions given extracted image features to approximate the standard Gaussian distribution as closed as possible.

## Experiment
### Dataset
We conduct our experiments using data from Recipe1M [4]. Recipe1M dataset consists of more than 1 million food images with corresponding ingredients and instructions. We manually extracted and chose 12 different types of cooking methods that are believed to be meaningful and distinguishable statistically, and then generated cooking methods for each training data by searching for keywords in the instruction text. We also reduced the number of different ingredients from around 18,000 to around 2,000 by removing ingredients with low frequency ( < 500 occurrence in the dataset) and then combined ingredients that belong to the same kind contextually (e.g. different kinds of oil which have the same features in images) or trivially (e.g. 1% milk and 2% milk). Because of the limit of time and computing resources we used only 10,000 data from the dataset to train.

### Input
We fed association model with paired and unpaired 128 &#215; 128 image and text input. For the StackGAN model, we used both 64 &#215; 64 and 128 &#215; 128 images because there are two discriminators for two resolution images from generators, respectively.

## Evaluation
We evaluated our task and approach via qualitative and quantitative results. In qualitative part, we demonstrate that our results are valid and meaningful under different conditions. In quantitaive part, we show two tables to compare the performance of our model with prior work.
### Qualitative
Besides Figure 1 where we show several realistic generated images from our model, here we compare the influence of two inputs -- ingredient and cooking method -- on image generation.

In Figure 4, ingredients are fixed as pork chops, green pepper and butter, but cooking method is changed from stir+fry to boil.
<p align="center">
  <img src="img_4.PNG" alt="fixed ingredients, change cooking method (1)"/>
  <br><em>Figure 4. Fixed ingredients (pork chops, green pepper and butter) and change cooking method</em></br>
</p>

In Figure 5, ingredients are fixed as cheese, egg and pizza sauce, but cooking method is changed from boil+heat to bake+stir.
<p align="center">
  <img src="img_5.PNG" alt="fixed ingredients, change cooking method (2)"/>
  <br><em>Figure 5. Fixed ingredients (cheese, egg and pizza sauce) and change cooking method</em></br>
</p>

In Figure 6, cooking method are fixed as bake as for muffin, but blueberry is added as extra ingredient. Blueberry is added to the top and inside muffin and we can see such dip in muffin with blueberries.
<p align="center">
  <img src="img_6.PNG" alt="fixed cooking method, change ingredients (1)"/>
  <br><em>Figure 6. Fixed cooking method and add blueberry</em></br>
</p>

In Figure 7, cooking method are fixed as bake as for muffin, but chocolate is added as extra ingredient. Chocolate is mixed with flour to prepare base for muffin and we can see muffin with chocolate in a darker color which represents chocolate.
<p align="center">
  <img src="img_7.PNG" alt="fixed cooking method, change ingredients (2)"/>
  <br><em>Figure 7. Fixed cooking method and add chocolate</em></br>
</p>

### Quantitative
We used MedR and R@K to evaluate our cross-modal association model the same as in [1]. In Table 1, we show the quantities from different sides of model under different conditions.

<p align="center">
  <img src="table_1.PNG" alt="Quantitative Evaluation for Cross-modal Association Model"/>
  <em>Table 1. Quantitative Evaluation for Cross-modal Association Model</em>
</p>

We used inception score (IS) and Fréchet Inception Distance (FID) to evaluate results of GAN. The higher IS and lower FID are, the better quality and diversity are for our generated images. In Table 2, the comparison is based on same model structure, parameters, training and test cases and approximately the same IS for real image sets. The only difference is the input type. The image-input model has only noise as input for generator. The ingredient-input model has noise and ingredient text embedding as input for generator. The ingredient+method model has noise, ingredient text embedding and cooking method text embedding as input.

<p align="center">
  <img src="table_2.PNG" alt="Quantitative Evaluation for GAN"/>
  <em>Table 2. Quantitative Evaluation for GAN</em>
</p>



|                         | Inception Score (IS)&#8593; | Fréchet Inception Distance (FID)&#8595; |
|-------------------------|-----------------------------|-----------------------------------------|
| Image Input Model       | 3.43041                     | 34.31625                                |
| Ingredient Input Model  | 3.51826                     | 32.65582                                |
| Ingredient+Method Model | **3.53567**                 | **25.90622**                            |

Based on Table 2, we successfully proved that cooking method, as an extra input, is a useful and valuable input for food image generation task.

## Future Improvements
From the experiments, we find that there are some improvements can be made in the future. Firstly, reducing the number of ingredients further. For example, we may combine different kinds of cheeses as they have similar appearance and contribution to the generated images. Such change will reduce the redundancy in the dataset and make it easier to learn. Secondly, balancing the number of images with different color to prevent the model from the inclination to generate reddish and yellowish images. Finally, we may further investigate the way to better control the contribution of conditional inputs as we found that it sometimes generated irrelevant image. Attention mechanism and regularization loss can be the options.

## Contributions
We acknowledge the assistance and advice from professor [Joseph Lim](https://viterbi-web.usc.edu/~limjj/) and TAs of course CS-566 (Deep Learning and its Applications). With their guidance, we developed the project and made the following contributions.
* A conditional GAN model for food image generation task with ingredients and cooking methods as input
* A refined version of dataset Recipe1M which further contains cooking methods extracted from instructions
* Quantitative data that proves cooking method as a useful and valuable input to food image generation tasks

## References
[1] Reed, Scott, et al. "Generative adversarial text to image synthesis." arXiv preprint arXiv:1605.05396 (2016).
