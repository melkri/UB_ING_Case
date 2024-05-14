import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.impute import KNNImputer
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import shap

def filter_df(df):
    # Group by ID and sum the default column
    default_sums = df.groupby('ID')['default'].sum()

    # Filter in only values where the sum is less than 1 (meaning no occurrence of default = 1)
    no_default_indices = df[df['ID'].isin(default_sums[default_sums < 1].index)]
    no_default_indices['filter_condition'] = True
    # Sort the DataFrame by 'ID' and 'obs_date' to ensure observations are ordered correctly
    df.sort_values(by=['ID', 'obs_date'], inplace=True)

    first_default_indices = df[df['default'] == 1].groupby('ID').head(1)
    default_dates = first_default_indices[['ID', 'obs_date']].rename(columns={'obs_date': 'cutoff_date'})

    df = df.merge(default_dates, on='ID', how='left')
    df['cutoff_date'] = df.groupby('ID')['cutoff_date'].ffill()

    df['filter_condition'] = df['obs_date'] <= df['cutoff_date']
    df_filtered = df[df['filter_condition']]


    # Concatenate df_filtered with no_default_indices
    df_filtered = pd.concat([df_filtered, no_default_indices])
    df_filtered.drop(columns=['cutoff_date', 'filter_condition'], inplace=True)
    return df_filtered


def preprocess_data(df):
    # List of columns to be processed
    var_columns = [f'Var_{i:02d}' for i in range(1, 40) if i not in [5, 6, 20, 22, 23, 24, 32, 33, 36, 37, 39]]

    # Replace negative values with 0 in the specified columns
    df[var_columns] = df[var_columns].clip(lower=0)

    return df


def binning_with_decision_tree(df, target='default'):
    df_binned = pd.DataFrame(index=df.index)

    for col in df.columns:
        if col.startswith('Var_'):
            # Split data into features (X) and target (y)
            X = df[[col]]
            y = df[target]

            # Fit decision tree classifier
            dt = DecisionTreeClassifier(max_leaf_nodes=10, random_state=42)
            dt.fit(X, y)

            # Extract decision tree thresholds
            splits = dt.tree_.threshold
            splits = splits[splits != -2]
            splits = splits.tolist()
            splits.append(float('inf'))
            splits.append(float('-inf'))
            splits.sort()

            # Create bins based on decision tree splits
            bin_labels = pd.cut(df[col], bins=splits, labels=False)

            # Assign bin labels to observations
            df_binned[col + '_bin'] = bin_labels

    # Add the target variable "default" to the binned DataFrame
    df_binned[target] = df[target]

    return df_binned


def calculate_woe(df_binned, target='default'):
    df_woe = pd.DataFrame(index=df_binned.index)

    for col in df_binned.columns:
        if col.endswith('_bin'):
            # Calculate the number of observations in each category (bin)
            grouped = df_binned.groupby(col)[target].agg(['count', 'sum'])

            # Calculate total positive and negative outcomes
            total_positive = grouped['sum'].sum()
            total_negative = grouped['count'].sum() - total_positive

            # Calculate WoE for each category
            grouped['woe'] = np.log((grouped['sum'] + 1) / total_positive) - np.log((grouped['count'] - grouped['sum'] + 1) / total_negative)

            # Map WoE values to the original DataFrame
            df_woe[col.replace('_bin', '_woe')] = df_binned[col].map(grouped['woe']).fillna(0)
    df_woe[target] = df_binned[target]

    return df_woe


def train_and_evaluate_pipeline(df, target='default', time=2021, model=LogisticRegression(random_state=42), param_grid=None):
    # Step 1: Data preparation
    # Exclude the year 2021 as out-of-time data
    in_time_df = df[df['year'] < time]
    out_of_time_df = df[df['year'] >= time]

    X_in_time = in_time_df.drop(columns=[target, 'year'])  # Exclude 'year' from features
    y_in_time = in_time_df[target]

    X_out_of_time = out_of_time_df.drop(columns=[target, 'year'])  # Exclude 'year' from features
    y_out_of_time = out_of_time_df[target]

    # Divide in-time data into train and test sets
    X_in_time_train, X_in_time_test, y_in_time_train, y_in_time_test = train_test_split(
        X_in_time, y_in_time, test_size=0.3, stratify=y_in_time, random_state=42)

    # Step 2: Model training
    grid_search = GridSearchCV(model, param_grid, scoring='roc_auc', cv=5, verbose = 3)
    grid_search.fit(X_in_time_train, y_in_time_train)

    # Get top 5 models based on cross-validation
    top_models = grid_search.cv_results_['params']
    top_model_names = ['{}_{}'.format(model.__class__.__name__[:2], idx + 1) for idx in range(5)]  # Abbreviate model name

    # Best grid search
    best_model = grid_search.best_estimator_

    # Initialize results dataframe
    results_df = pd.DataFrame(columns=['Model_Specs', 'Model_Name', 'In_Time_Train_Gini', 'In_Time_Test_Gini',
                                        'Out_of_Time_Test_Gini', 'In_Time_Train_Confusion_Matrix',
                                        'In_Time_Test_Confusion_Matrix', 'Out_of_Time_Test_Confusion_Matrix'])

    # Append best model results
    in_time_train_gini, in_time_test_gini, out_of_time_test_gini, in_time_train_conf_matrix, in_time_test_conf_matrix, out_of_time_test_conf_matrix = evaluate_model(
        best_model, X_in_time_train, y_in_time_train, X_in_time_test, y_in_time_test, X_out_of_time, y_out_of_time)
    
    results_df = results_df.append({'Model_Specs': best_model,
                                    'Model_Name': top_model_names[0],
                                    'In_Time_Train_Gini': in_time_train_gini,
                                    'In_Time_Test_Gini': in_time_test_gini,
                                    'Out_of_Time_Test_Gini': out_of_time_test_gini,
                                    'In_Time_Train_Confusion_Matrix': in_time_train_conf_matrix,
                                    'In_Time_Test_Confusion_Matrix': in_time_test_conf_matrix,
                                    'Out_of_Time_Test_Confusion_Matrix': out_of_time_test_conf_matrix},
                                   ignore_index=True)

    # Append top 4 models to the results dataframe
    for idx, model_params in enumerate(top_models[1:5], start=1):  # Skip the first model (best_model)
        model_name = top_model_names[idx]  # Get model name
        model = model.set_params(**model_params)  # Set hyperparameters for the current model
        
        # Evaluate the current model
        in_time_train_gini, in_time_test_gini, out_of_time_test_gini, _, _, out_of_time_test_conf_matrix = evaluate_model(
            model, X_in_time_train, y_in_time_train, X_in_time_test, y_in_time_test, X_out_of_time, y_out_of_time)
        
        # Append results to the dataframe
        results_df = results_df.append({'Model_Specs': model_params,
                                        'Model_Name': model_name,
                                        'In_Time_Train_Gini': in_time_train_gini,
                                        'In_Time_Test_Gini': in_time_test_gini,
                                        'Out_of_Time_Test_Gini': out_of_time_test_gini,
                                        'In_Time_Train_Confusion_Matrix': None,  # Not applicable for top models
                                        'In_Time_Test_Confusion_Matrix': None,  # Not applicable for top models
                                        'Out_of_Time_Test_Confusion_Matrix': out_of_time_test_conf_matrix},
                                       ignore_index=True)

    # Printing
    print("In-Time Train Evaluation Results:")
    print(f"Train Gini: {in_time_train_gini:.4f}")
    print("Train Confusion Matrix:")
    print(in_time_train_conf_matrix)

    print("\nIn-Time Test Evaluation Results:")
    print(f"Test Gini: {in_time_test_gini:.4f}")
    print("Test Confusion Matrix:")
    print(in_time_test_conf_matrix)

    print("\nOut-of-Time Test Evaluation Results:")
    print(f"Test Gini: {out_of_time_test_gini:.4f}")
    print("Test Confusion Matrix:")
    print(out_of_time_test_conf_matrix)

    return results_df


def evaluate_model(model, X_train, y_train, X_test, y_test, X_out_of_time, y_out_of_time):
    # Model training
    model.fit(X_train, y_train)

    # Evaluation
    train_gini = 2 * roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]) - 1
    test_gini = 2 * roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) - 1
    out_of_time_test_gini = 2 * roc_auc_score(y_out_of_time, model.predict_proba(X_out_of_time)[:, 1]) - 1

    train_conf_matrix = confusion_matrix(y_train, model.predict(X_train))
    test_conf_matrix = confusion_matrix(y_test, model.predict(X_test))
    out_of_time_test_conf_matrix = confusion_matrix(y_out_of_time, model.predict(X_out_of_time))

    return train_gini, test_gini, out_of_time_test_gini, train_conf_matrix, test_conf_matrix, out_of_time_test_conf_matrix

def calculate_psi(df, time_column='year', variable_columns=None, epsilon=1e-6):
    if variable_columns is None:
        variable_columns = [col for col in df.columns if col.startswith('Var_')]

    # Create a new column indicating the period for each observation
    df['period'] = np.where(df[time_column] <= 2019, 1, 2)

    # Initialize a DataFrame to store PSI results
    psi_results = pd.DataFrame(index=variable_columns, columns=['PSI'])

    for col in variable_columns:
        # Calculate the distribution of the variable in period 1 (2015-2019)
        period1_distribution = df[df['period'] == 1][col].value_counts(normalize=True, dropna=False).sort_index()

        # Calculate the distribution of the variable in period 2 (2020-2021)
        period2_distribution = df[df['period'] == 2][col].value_counts(normalize=True, dropna=False).sort_index()

        # Add missing values to the distributions
        for index in period1_distribution.index:
            if index not in period2_distribution.index:
                period2_distribution[index] = epsilon

        for index in period2_distribution.index:
            if index not in period1_distribution.index:
                period1_distribution[index] = epsilon

        # Calculate PSI
        psi_diff = period2_distribution - period1_distribution
        log_ratio = np.log((period2_distribution + epsilon) / (period1_distribution + epsilon))
        psi = psi_diff * log_ratio
        psi_results.loc[col, 'PSI'] = psi.sum()
        
    # Drop the 'period' column before returning
    df.drop(columns=['period'], inplace=True)

    return psi_results

def check_collinearity(df, threshold=5):
    # Initialize a list to store variables to drop
    vars_to_drop = []

    while True:
        # Calculate VIF for each variable
        vif_data = pd.DataFrame()
        vif_data["Variable"] = df.columns
        vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]

        # Find variables with high collinearity
        high_collinearity_vars = vif_data[vif_data['VIF'] > threshold]

        if len(high_collinearity_vars) == 0:
            break  # No variables exceed the threshold, exit loop

        # Identify the variable with the highest VIF
        high_vif_var = high_collinearity_vars.loc[high_collinearity_vars['VIF'].idxmax(), 'Variable']

        # Drop the variable with the highest VIF
        df = df.drop(columns=[high_vif_var])

        # Add the dropped variable to the list
        vars_to_drop.append(high_vif_var)

    # Print variables dropped
    print("Variables dropped due to high collinearity:")
    print(vars_to_drop)

    # Plot VIF values after dropping variables
    plt.figure(figsize=(10, 6))
    plt.barh(vif_data['Variable'], vif_data['VIF'], color='skyblue')
    plt.xlabel('VIF')
    plt.title('Variance Inflation Factor (VIF)')
    plt.axvline(x=threshold, color='red', linestyle='--', label='Threshold')
    plt.legend()
    plt.show()

    return df

def model_to_shapley(df, model, target='default', time=2021):
    in_time_df = df[df['year'] < time]
    out_of_time_df = df[df['year'] >= time]

    X_in_time = in_time_df.drop(columns=[target, 'year'])  # Exclude 'year' from features
    y_in_time = in_time_df[target]

    X_out_of_time = out_of_time_df.drop(columns=[target, 'year'])  # Exclude 'year' from features
    y_out_of_time = out_of_time_df[target]

    # Divide in-time data into train and test sets
    X_in_time_train, X_in_time_test, y_in_time_train, y_in_time_test = train_test_split(
        X_in_time, y_in_time, test_size=0.3, stratify=y_in_time, random_state=42)
    
    # Fit the model
    model.fit(X_in_time_train, y_in_time_train)

    # Create a Linear explainer and calculate Shapley values
    explainer = shap.LinearExplainer(model, X_in_time_train, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_out_of_time)

    # Generate a SHAP summary plot for the out-of-time data
    plt.figure(figsize=(12,14))
    shap.summary_plot(shap_values, X_out_of_time, feature_names=X_out_of_time.columns.tolist(), show=False)

    # Modify the fontsize of the y-axis labels
    plt.yticks(fontsize=13)
    plt.title('SHAP Summary Plot', fontsize=14, fontweight='bold', loc='center')
    plt.show()