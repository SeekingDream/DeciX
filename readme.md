
# DeciX


*DeciX* is a **DE**pendency-aware **C**ausal **I**nference framework for e**X**plaining the decision-making in deep learning-based code generation applications.
*DeciX* can (i) model the output-output dependency in code generation applications; (ii) handle the non-numberic data format; (iii) supporting black-box settings.
In detail, *DeciX* provides token-level explanations by constructing a causal relation graph and decomposing the edge weights in the graph. 


## A Demo Example to Explain DeciX

<div  align="center">    
 <img src="https://github.com/anonymousGithub2022/CodeGenExp/blob/main/fig/detail.png" width="720" height="800" alt="Design Overview"/><br/>
</div>    

In our step 1, we randomly select 20% of the input token and replace the selected tokens with random tokens. We then compare the mutant with the original input and get the binarized causal input/output.

In our step 2, we construct the graph with two tyeps of edges based on the input token order and output token order.

In step 3, we treat each output token as a target and train a regression model to fit the causal contribution scores. The above figure shows an example to compute the causal contribution scores for the third output token **now**. Although Decix and LIME both use a regression model in this step, Decix differentiates from LIME from the following two perspectives: (i) Decix fits a regression model on the binarized casual input/output, while LIME fits a regression model directly on the input/output. (ii) Decix considers not only the input token but also previous output tokens to fit the regression model (the green box in step 3), which is the unique part in Decix and does not exist in existing work.

In our step 4, we decompose the output-output dependency in our previous step.



## File Structure
* **src** -main source codes.
  * **./src/CodeBert** - the application of CodeBert.
  * **./src/GPT2** - the application of PyGPT2.
  * **./src/DeepAPI** -the application of DeepAPI.
  * **./src/wrapper_model.py** -the wrapper model of the mentioned applications.
  * **./src/xai** -the lib that includes the implementation of each explanation methods.
    * **./src/xai/codeexpgen** - the implementaion of our approach.
 
* **utils.py** -the basical functions to load DNNs.
* **generate_explanation.py** -the script is used for explanation each DL-based code generation applications.
* **evaluate_explanation.py** -this script is used to evaluate the accuracy of the explanations.
* **post_acc.py**.   -get the accuracy results.
* **bashXX.sh** -bash script to run experiments (**XX** are integer numbers that represent the code generation model ID).
* **requirement.txt** -the dependent libraries.

## Setup
We strongly recommend the user use the *conda* to manage the virtual environment.

First create an environment with *conda*.
`conda create -n your_env_name python=3.7`

Second, activate the virtual environment.
`conda activate your_env_name`.

Next, install the basic library dependency.
`pip install -r requirement.txt`.

Finally, download the pre-trained model weights from [model_weight](https://drive.google.com/drive/folders/1KJBahf25i9ttQr8VWF8tA9e87IBw5Q0E?usp=sharing)
and put the model weights in the directory `model_weight`.
The model weights will be `model_weight/deepAPI` and `model_weight/pytorch_model.bin`.


## Quick Start

We have run the explanation scripts offline and stored the explanation results in the directory `exp_res`. 

To quickly evaluate the explanation quality, run `bash demo_bash1.sh`, `bash demo_bash2.sh` and `bash demo_bash3.sh`. 

After that, run `python post_acc.py` to plot the figures.

All explanation results are stored in the directory `final_res`.


## How to run

We provide the bash script that generate adversarial examples and measure the efficiency in **bash1.sh**. **bash2.sh**, **bash3.sh**, are implementing the similar functionality but for different gpus. 

So just run `bash bash1.sh`, `bash bash2.sh`, and `bash bash3.sh`.
 
After get the results, run `python post_acc.py` to plot the results.












