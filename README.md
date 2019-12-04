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

## Method & Approach

## Experiment
### Dataset
We conduct our experiments using data from recipe1M [4]. The recipe1M dataset consists of more than 1 million food images with corresponding ingredients and instructions. We manually chose 12 different types of cooking method that we think are meaningful and distinguishable, and generated cooking method for each image by searching for keywords in the instruction text. We also reduced the number of different ingredients from around 18,000 to around 2,000 by removing ingredients with low frequency ( < 500 occurrence in the dataset)  and combining ingredients that are the same or very close.  Because of the limit of time and computing resources we used only 10,000 data from the dataset.

### Input
We feed association model with paired and unpaired 128 &#215; 128 image and words. For the stackgan model, we use both 64 &#215; 64 and 128 &#215; 128 images because there are two discriminators differential two resolution images from generators.

## Evaluation
<table>
   <tr>
      <td></td>
      <td></td>
      <td></td>
      <td>im2rcp</td>
      <td></td>
      <td></td>
      <td></td>
      <td>rcp2im</td>
      <td></td><td></td>
   </tr>
   <tr>
      <td></td>
      <td></td>
      <td>MedR&#8595;</td>
      <td>R@1&#8593;</td>
      <td>R@5&#8593;</td>
      <td>R@10&#8593;</td>
      <td>MedR&#8595;</td>
      <td>R@1&#8593;</td>
      <td>R@5&#8593;</td>
      <td>R@10&#8593;</td>
   </tr>
   <tr>
      <td>1K</td>
      <td>model in [1]</td>
      <td>5.500</td>
      <td>0.234</td>
      <td>0.503</td>
      <td>0.618</td>
      <td>5.750</td>
      <td>0.230</td>
      <td>0.491</td>
      <td>0.615</td>
   </tr>
   <tr>
      <td></td>
      <td>our model</td>
      <td><b>4.400</td>
      <td><b>0.261</td>
      <td><b>0.549</td>
      <td><b>0.679</td>
      <td><b>4.200</td>
      <td><b>0.270</td>
      <td><b>0.556</td>
      <td><b>0.682</td>
   </tr>
   <tr>
      <td>5K</td>
      <td>model in [1]</td>
      <td>24.000</td>
      <td>0.099</td>
      <td>0.265</td>
      <td>0.364</td>
      <td>25.100</td>
      <td>0.097</td>
      <td>0.259</td>
      <td>0.357</td>
   </tr>
   <tr>
      <td></td>
      <td>our model</td>
      <td><b>17.900</td>
      <td><b>0.116</td>
      <td><b>0.299</td>
      <td><b>0.406</td>
      <td><b>16.700</td>
      <td><b>0.129</td>
      <td><b>0.315</td>
      <td><b>0.421</td>
   </tr>
   <tr>
      <td>10K</td>
      <td>model in [1]</td>
      <td>47.000</td>
      <td>0.065</td>
      <td>0.185</td>
      <td>0.267</td>
      <td>48.300</td>
      <td>0.061</td>
      <td>0.178</td>
      <td>0.261</td>
   </tr>
   <tr>
      <td></td>
      <td>our model</td>
      <td><b>34.900</td>
      <td><b>0.077</td>
      <td><b>0.212</td>
      <td><b>0.301</td>
      <td><b>32.700</td>
      <td><b>0.088</td>
      <td><b>0.229</td>
      <td><b>0.319</td>
   </tr>
</table>


## Future Improvements
From the experiments, we find that there are some improvements can be made in the future. Firstly, reducing the number of ingredients further. For example, we may combine different kinds of cheeses as they have similar appearance and contribution to the generated images. Such change will reduce the redundancy in the dataset and make it easier to learn. Secondly, balancing the number of images with different color to prevent the model from the inclination to generate reddish and yellowish images. Finally, we may further investigate the way to better control the contribution of conditional inputs as we found that it sometimes generated irrelevant image. Attention mechanism and regularization loss can be the options.

## Contributions
We acknowledge the assistance and advice from professor [Joseph Lim](https://viterbi-web.usc.edu/~limjj/) and TAs of course CS-566 (Deep Learning and its Applications). With their guidance, we developed the project and made the following contributions.
* A conditional GAN model for food image generation task with ingredients and cooking methods as input
* A refined version of dataset Recipe1M which further contains cooking methods extracted from instructions
* Quantitative data that proves cooking method as a useful and valuable input to food image generation tasks

## References
