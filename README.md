##**Practical Machine Learning Project on Coursera**

###**To view html report online, please click** [here](https://tihanam.github.io/practical_machine_learning/)

- [Practical_Machine_Learning_project.md] (https://github.com/TihanaM/practical_machine_learning/blob/gh-pages/Practical_Machine_Learning_project.md): Markdown document for project report.
- [Practical_Machine_Learning_project.Rmd] (https://github.com/TihanaM/practical_machine_learning/blob/gh-pages/Practical_Machine_Learning_project.Rmd): R markdown document for project report.
- [Practical_Machine_Learning_project.html] (https://github.com/TihanaM/practical_machine_learning/blob/gh-pages/Practical_Machine_Learning_project.html): Compiled html file for project report.

- pml_data_training.csv and pml_data_testing.csv are the provided training and testing data sets for the project.



##**1 Introduction (Synopsis)**

Owing to the advancements in wearable technology for the tracking of personal activity, such as Jawbone Up, Nike FuelBand, and Fitbit, a large amount of data on physical performance is easily collected. The primary goal of these devices is to allow customers to quantify self movement, but often, the quality of the exercise performance gets neglected. In this project, data collected from 6 individuals while performing barbell lifts from accelerometers on the belt, forearm, arm, as well as the dumbbell itself, is used in order to build a predictive model for the classification of how well the weightlifting exercise was performed. 

The quality of the execution of the weightlifting exercise was defined in five different categories:  

*  **CLASS A**: exactly according to the specification  
*  **CLASS B**: throwing the elbows to the front  
*  **CLASS C**: lifting the dumbbell only halfway
*  **CLASS D**: lowering the dumbbell only halfway
*  **CLASS E**: throwing the hips to the front

The machine learning algorithm developed in this project is outlined below. The classification tree approach, the generalized boosted model, the random forest classification were compared and assessed for accuracy. Consequently the most accurate model, in this case the random forest classification, was applied to the test data set in order to obtain predictions in which manner the exercises were performed in (class A to E).
