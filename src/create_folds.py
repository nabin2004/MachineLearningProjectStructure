import pandas as pd
from sklearn.model_selection import StratifiedKFold

def create_folds(data_path, target_column, num_folds=5, random_state=42):
    """
    Creates stratified folds for cross validation
    
    Args:
        data_path (str): Path to the dataset file (csv)
        target_column (str): Name of the target column
        num_folds (int): Number of folds to create
        random_state (int): Random seed for reproducibility
    
    Returns:
        DataFrame with an additional 'kfold' column
    """
    df = pd.read_csv(data_path)
    
    df["kfold"] = -1
    
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    
    for fold, (_, valid_idx) in enumerate(skf.split(X=df, y=df[target_column])):
        df.loc[valid_idx, "kfold"] = fold
    
    return df

if __name__ == "__main__":
    data_path = "../input/train.csv" 
    target_column = "smoking"      
    
    df_with_folds = create_folds(data_path, target_column)
    
    df_with_folds.to_csv("../input/train_folds.csv", index=False)
