# Report on Alphabet Soup Charity Funding Success Prediction

## Executive Summary
As a Data Analyst at Alphabet Soup Charity, my objective was to create a binary classifier to predict the success of funding applications. By utilizing machine learning and neural networks on a dataset with over 34,000 past applications, I developed a model aimed at distinguishing applicants most likely to succeed upon receiving funding. This report documents the methodologies, processes, and findings of the analytical exercise.

## Data Preprocessing
The foundational stage of data analysis involved loading the dataset for initial exploration. The preprocessing steps encompassed:

- **Target Variable Identification**: The 'IS_SUCCESSFUL' column was identified as the target variable, indicating the effectiveness of the funding usage.
- **Feature Selection**: Identifiers such as 'EIN' and 'NAME' were dropped to focus on predictive attributes.
- **Unique Value Analysis**: Evaluated the dataset for unique values, aiding in understanding the data's complexity.
- **Categorical Variable Binning**: 'APPLICATION_TYPE' and 'CLASSIFICATION' columns underwent binning to condense sparse categories and reduce overfitting risks.
- **One-Hot Encoding**: Categorical variables were transformed into numerical format via `pd.get_dummies()`.
- **Data Splitting**: Segregated data into feature (X) and target (y) sets, followed by a train-test split.
- **Feature Scaling**: Normalized feature scales using `StandardScaler` to ensure optimal neural network model input.

## Model Development
The neural network model was architected with:

- **Input Layer**: Defined by the post-encoding feature count.
- **Hidden Layers**: Incorporated three hidden layers with a mix of 'relu' and 'leaky_relu' activations for non-linearity.
- **Dropout Regularization**: Integrated at a rate of 0.1 for each hidden layer to mitigate overfitting.
- **Output Layer**: Single neuron with 'sigmoid' activation tailored for binary outcome prediction.
- **Compilation**: Employed Nadam optimizer for convergence efficacy and binary cross-entropy loss suitable for binary classification.
- **Callbacks**: `EarlyStopping` was utilized to cease training when no validation loss improvement was detected, avoiding overfitting.

## Training and Evaluation
Model training was executed over multiple epochs with batch size 32, while training and validation accuracies were monitored for learning progression.

## Results
The model attained an accuracy close to 73% on the test set. Though this falls short of the 75% project goal, it establishes a baseline for future enhancement.

## Key Steps for Improvement
- **Hyperparameter Tuning**: Plans include methodical tuning of hyperparameters like neurons, layers, and learning rates.
- **Advanced Feature Engineering**: Further sophisticated feature engineering is expected to improve model performance.
- **Ensemble Methods**: Application of model ensembles to refine predictions.

## Conclusion and Recommendations
The constructed neural network model lays the groundwork for predictive analytics in funding application success. Despite not achieving the 75% accuracy benchmark, the model's potential is evident. I recommend a progressive model refinement strategy and consideration of alternative machine learning techniques for enhanced accuracy.

## Future Directions
To advance this project, the following is proposed:

- **Implement Cross-Validation**: To assure model consistency across different data subsets.
- **Explore Alternative Models**: To benchmark against other models and find the most accurate approach.
- **Ongoing Data Training**: To maintain model relevancy with emerging application trends and patterns.

###### Data provided by IRS. Tax Exempt Organization Search Bulk Data Downloads. https://www.irs.gov/Links to an external site.
