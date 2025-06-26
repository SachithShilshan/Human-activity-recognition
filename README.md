# Human-activity-recognition

## 1. Introduction
This final assessment applies a Decision Tree Classifier to the HARTH dataset, a time-series human activity recognition dataset collect using wearable accelerometers. I used statistical feature engineering, data visualization, and scikit-learn modeling to classify physical activities like walking, sitting, cycling, etc. So, The goal is to classify these activities using a Decision Tree Classifier.

### Dataset Overview

The Human Activity Recognition Trondheim (HARTH) dataset is a professionally-annotated dataset containing 22 subjects wearing two 3-axial accelerometers for around 2 hours in a free-living setting. The sensors were attached to the right thigh and lower back. The professional recordings and annotations provide a promising benchmark dataset for researchers to develop innovative machine learning approaches for precise HAR in free living.
The provided sampling rate is 50Hz. Video recordings of a chest-mounted camera were used to annotate the performed activities frame-by-frame.
Link :- https://archive.ics.uci.edu/dataset/779/harth

![image](https://github.com/user-attachments/assets/8a8119c3-03ee-4643-af62-ebe82f289037)
 
## 2. Data Handling and Label Mapping
	Using pandas to load the multiple CSV files provided and numpy for numerical operations. 
	The dataset consists of continuous time-series sensor data stored in multiple CSV files. 
	Using used the glob module to automatically gather all CSV files from the dataset directory. This is especially useful because this dataset is stored across multiple CSV files, one for each subject. The glob() function searching for all files matching a pattern.
 ![image](https://github.com/user-attachments/assets/28fef329-844a-484c-a8bd-9fa8d20f6639)

•	Information about the DataFrame

![image](https://github.com/user-attachments/assets/b1b3a324-d4ca-43e7-b950-da681ce1ccc1)

•	Activity labels in the dataset are encoded as numbers.So,to make the outputs human readable, defining a label-to name mapping.
 ![image](https://github.com/user-attachments/assets/ebe742d8-ca5c-42ec-bcb2-289b676c32fc)

## 3. Feature Engineering
To prepare the dataset for modeling
•	Divid data into non-overlapping 2-second windows (100 samples at 50Hz)
•	Compute statistical summaries: mean and standard deviation of back_x/y/z and thigh_x/y/z
Each window label by the most common activity during that segment.
![image](https://github.com/user-attachments/assets/13ecd136-88da-49d8-b785-d9f5d9a6ee5a)

•	So ,this features provide a compact, descriptive summary of movement patterns .
•	The resulting dataset (features_df) contain 12 features and one label per window & applied mapping.
![image](https://github.com/user-attachments/assets/cf8b4da3-d550-4a0b-98ef-36a1678ded59)

## 4. Visual Explorations
Using Matplotlib and Seaborn for visualizations. So, Matplotlib is Use for plot the decision tree and confusion matrix . Seaborn is use for quick, clean statistical visualizations like boxplots, heatmaps, and bar charts.

### Correlation Analysis
To explore relationships between features for using Seaborn and Matplotlib to create a correlation heatmap.
  ![image](https://github.com/user-attachments/assets/3f0fba88-88f3-4c93-b1a2-16f4f21d2579)

From the heatmap, observed that
	Strongly Correlated Features (r ≥ 0.9): std_back_x & std_ thigh _z  , std_thigh_y & std_thigh_z , std_back_y & std_back_z , mean_thigh_x & mean_back_x
	Moderately Correlated Features (r ≈ 0.7): std_back_y & mean_thigh _x
	Weakly or Independently Correlated Features (|r| < 0.3): mean_back_x & std_ thigh_z

Using seaborn and matplotlib create boxplot that clearly show that std_thigh_y helps distinguish dynamic from static activities.
![image](https://github.com/user-attachments/assets/034c30f4-0464-4d42-9dbf-510b8e484aa8)

Key insights are
•	Running and walking show high variation (high std_thigh_y).
•	Sitting, lying, and standing have very low movement (low std_thigh_y).
•	Stairs and shuffling show moderate values.
This shows that std_thigh_y is useful for separating active vs. inactive behaviors.

## 5. Decision Tree Classification
Model Optimization Chart
To determine the best max_depth for our Decision Tree, evaluating accuracy across depths from 1 to 15.So, the following graph shows the results.
 ![image](https://github.com/user-attachments/assets/8f2a6751-3343-49fb-9e66-3494b5bfb03a)

As shown, accuracy increases rapidly until about depth 6, then levels off. So ,suggesting max_depth is 5 or 6 is a strong choice for balancing accuracy and simplicity while avoid overfitting.

### Model Training
training a Decision Tree Classifier using scikit-learn.
![image](https://github.com/user-attachments/assets/7f1d14bf-d9cd-4ee2-837a-cd39b51d045a)

Evaluation
![image](https://github.com/user-attachments/assets/c55159ac-88ee-4e10-9a24-d47b01d90c55)

The model achieved ~90% accuracy. 

## 6. Feature Importance and Tree Visualization
Feature Importance
 Visualizing the most important features.

 ![image](https://github.com/user-attachments/assets/2c3e13eb-ac5d-4691-a286-a3f51adeae67)

This bar chart shows how much each feature contributing to the Decision Tree prediction. So, Key points are
•	mean_thigh_z was the most important feature that help to separate activities with strong vertical leg motion.
•	mean_back_x and std_thigh_x were also important for that identify posture and dynamic movement.

### Tree Visualization
![image](https://github.com/user-attachments/assets/aebffa05-eda1-44ec-92e1-cd5ef6fac864)
![image](https://github.com/user-attachments/assets/c03c5653-cf19-4c3f-a6ee-f0a354c48d46)

This visualization shows the full structure of the trained Decision Tree Classifier. Key insights are
•	The first split is based on mean_thigh_z  that key feature separates high leg motion from low-motion activities.

 ![image](https://github.com/user-attachments/assets/8a8756e9-8f68-46e0-8b29-68bc2e6f37b0)

•	Other important feature( second splits) are std_thigh_x and mean_back_x.

  ![image](https://github.com/user-attachments/assets/bd61ec87-a1ff-4f99-a8ee-2d5f537b8cd1)
  
  ![image](https://github.com/user-attachments/assets/810e4fbb-55ea-4181-8de8-efee230ea750)

•	The tree structure clearly shows that how combination of features are used to classify each activity class and this helps us understand which features matter most.

•	The high-resolution decision tree visualization Link – https://drive.google.com/file/d/174EvyXTN6NZynIurpMK4sGbiLC8rtWjT/view?usp=sharing

## 7. Conclusion
The HARTH dataset contains detailed motion signals and that transforming into 12 statistical features per time window. Through exploratory data analysis and decision tree modeling,  achieved ~90% classification accuracy. Label mapping enabled us to generate clear, readable visualizations and outputs. Heatmap helps to explore relationships between features and boxplot shows that std_thigh_y is useful for separating active vs. inactive behaviors. Features like mean_thigh_z , std_thigh_x and mean_back_x were most important in separate active from passive movements.


