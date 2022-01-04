
# Inverse deep learning methods and benchmarks for artificial electromagnetic material design

This repository is the implementation of [Inverse deep learning methods and benchmarks for artificial electromagnetic material design](https://arxiv.org/pdf/2112.10254.pdf). Currently under review, by [Simiao Ren*](https://bensonren.github.io/), Ashwin Mahendra*, [Omar Khatib](https://scholar.google.com/citations?user=xPo2CTkAAAAJ&hl=en), [Yang Deng](https://scholar.google.com/citations?hl=en&user=dnh1U0sAAAAJ), [Willie J. Padilla](https://ece.duke.edu/faculty/willie-padilla) and [Jordan Malof](https://ece.duke.edu/faculty/jordan-malof).


## Environment
The detailed conda environment is packaged in [.yml file](./demo/environment.yml).

### Flow chart process
Individual steps are illustrated below. This is laying the roadmap of how you can recreate the result in our paper. The blue fineprint above each step is the functions you should run and please check the detailed illustration below before you run it. 
![Flow_chart](./demo/flow_chart.png) 

### Folder names
| Folder name | Usage |
|:---------------------------------------------:|:------------------------------------------------------------------:|
| Data |  The folder holding the datasets and the generating code for the benchmarked dataset| 
| NA|  Neural adjoint method (Also contains the code to run neural simulator of ADM dataset)| 
| cINN|  Conditional invertible neural network algorithm| 
| INN_FrEIA|  Invertible neural network| 
| VAE|  Conditional variational autoencoder| 
| MDN| Mixture density network| 
| GA|  Genetic algorithm| 
| Tandem|  Tandem model| 
| inverse|  The direct naive network method (NN)| 
| utils|  helper functions to be called| 

### Generating benchmarking datasets

The code to generate the benchmarking dataset can be found in [Simulated_Datasets](./Data)
 
#### ADM dataset (a.k.a. Yang_sim / Yang / Deng dataset)
Following the generation procedure that published the ADM simulator [NeurIPS2020 paper Github](https://github.com/BensonRen/BDIMNNA), first go to Data/Yang_sim and run the random generation of the geometry variable

```Generate_geometry_ADM
generate_mm_x.py
```

Then, to get the spectra response of the generated geometries, go to NA/ folder 

```create_mm_dataset
python predict.py  (Make sure you have the one line says creat_mm_dataset activated in this file)
```

The generated file called "MM_data_full.csv" is the Meta-material data that you should read for further training and\
 is regarded as the ground truth for next stages. Move the file to dataIn/ folder (as the only file in the folder).

## Training

To train the models in the paper, go to the each folders for algorithms,
 run this command:

```train
python train.py 
```

> This would read the flags.obj file in the 'models' folder to retrieve the hyper-parameters for each of the
> datasets, afterwhich a training process would be initiated to train a network described in Section 4 of main paper.
> 
> The trained models would be placed in the same 'models' folder with training time and checkpoint file produced.
> Currently the 'models' folders contains the trained hyper-parameters 
> 
> To counter the stochastic nature of the neural network training process, the training is done 10 times (set up already) and make sure you use the evaluation code below to evaluate and get a good model instead of one stuck in local minimum that gives bad performance

## Evaluation

To evaluate the models, for each of the algorithms, run this command:
```eval
python evaluate.py
```

> This would read the flags.obj and best_model.pt file from the 'models' folder and write the evaluation data.
> Note that since we are benchmarking and comparing the time-performance trade-off of various networks structures,
> for a single query data point, multiple trails (200) would be run and each trail would get a different inverse 
> solution (except for Tandem model which is deterministic) as depicted in Fig 3 in main paper.
>
> The function evaluate.py has multiple options (all documented) and one can use different combination of them to test 
> experiment relevant to the applicaiton. Currently the code is prepared to evaluate all the dataset with all 10 training instance for single point evaluation (single point means each algorithm gets 1 chance to provide an inverse answer to a inquiry y point where multi_eval means each algorithm gets multiple chances for each inquiry y and best accuracy is selected. 
> 
> The evaluation results would be stored at the "data/" folder in current directory with plot of the histogram of the MSE. You can change the number of inference point (test set ratio) in code "utils/evaluation_helper/get_test_ratio_helper".
> There are generally 2 modes of evaluation, the single evaluation to determine whether you have a good model trained and another multi_eval mode where the various T values are used and compare to see how efficiently these methods explore the solution space. You would find revelant documentation there.
> One special thing to mention is that for meta-material dataset, since there is no closed form "Simulator", we would run the neural simulator again just like in meta-material dataset creation step. After you run evaluation, simply go to NA/ folder and run the predict.py with relevant code at __main__ function.
> After comparing the 10 instance of training and selecting the best performing one, modify the code in predict.py function "evaluate_different_dataset" line 3: for j in range(10) to become only evaluating the best one you choosed from for next step.
> If the previous step of using validation performance as test set performance bothers you, feel free to run the data generation code again to get a new set of "unseen" data. Our experience is that only selecting from 10 training instance with the same hyper-parameter does not cause significant validation set overfitting and therefore we did not make this default choice here.

## Comparing the efficiency of the algorithms

To compare the efficiency of the algorithms on exploring the solution space, we proposed R_t metric which resembles the top_k_accuracy in computer vision task in inverse problem. To get the results of our paper figure 3, run the multi_evaluation in the above section for each of the algorithms. By then you would find multi_eval folder full of evalutation files. To do some housekeeping for our code, we need to delete and rename some files, go to multi_eval/ folder and run

```delete
python delete.py
```

Then, if you are using the meta-material dataset, run the predict.py for the meta-material dataset in the NA/ folder to get the Ypred files for meta-material.

You are almost there! Now you have all the evaluation results, to plot them go to utils/ folder and run

```plot
python plotsAnalysis.py
```

to get the plot, which would be located in the multi_eval/ folder. We also provided the 25% 75% as a basic error bar for a visualization of the uncertainty of the methods.  

> One thing to mention is that due to random initialization of weights, local minimums caused by the rough loss surface and stochastic nature of these algorithms, the plot generated is not going to be exactly like the one on the paper. However, with multiple training (mentioned above), the trend is very stable.
> Also, after further experiment, we did managed to find better hyper-params for cINN in ballistics dataset and hence included the updated hyper-params in this repo.


## Results

Our model achieves the following performance on :

![Inverse model performance as a function of time](./demo/3.png) 


## Contributing


## Credit
We would like to express special thanks to [Visual Learning Lab Heidelberg](https://github.com/VLL-HD) who provided open sourced implementation of their Invertible network structure called Framework for Easily Invertible Architectures (FrEIA) above. Their code base is undergoing non-backward-compatible updates. Therefore to simplify the process we have enclosed their v0.2 code that we used along with our code in this repo. We were inspired a lot from their work and grateful for the implementation of INNs. 

We would also like to express thanks to [Manu Joseph](https://github.com/manujosephv) who published the mixture density network (MDN) implementation that we were greatly inspired from. The implementation of our MDN is adapted from their [Pytorch tabular repo](https://github.com/manujosephv/pytorch_tabular).

