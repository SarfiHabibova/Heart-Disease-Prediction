import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
   
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop"
    )
    return preprocessor


def plot_and_save_confusion_matrix(y_true, y_pred, out_path: str, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_and_save_roc_curve(y_true, y_proba, out_path: str, title: str) -> float:
   
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title} (AUC={auc:.3f})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return auc


def main():
    parser = argparse.ArgumentParser(description="Heart Failure Risk Prediction (Classification)")
    parser.add_argument("--data", type=str, default="data/heart.csv", help="Path to CSV dataset")
    parser.add_argument("--target", type=str, default="HeartDisease", help="Target column name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    figures_dir = "reports/figures"
    ensure_dir(figures_dir)

  
    df = load_data(args.data)

    print("Dataset shape:", df.shape)
    print("\nMissing values per column:\n", df.isna().sum())
    print("\nDuplicate rows:", df.duplicated().sum())

  
    df = df.drop_duplicates()

    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in columns: {df.columns.tolist()}")

  
    X = df.drop(columns=[args.target])
    y = df[args.target]

    
    if y.dtype == "object":
        y = y.map({"Yes": 1, "No": 0}).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=args.seed,
        stratify=y
    )

    preprocessor = build_preprocessor(X_train)

   
    models = {
        "logreg": LogisticRegression(max_iter=2000, random_state=args.seed),
        "dt": DecisionTreeClassifier(random_state=args.seed),
        "rf": RandomForestClassifier(random_state=args.seed),
    }

    
    param_grids = {
        "logreg": {
            "model__C": [0.1, 1.0, 10.0],
            "model__solver": ["liblinear"],  
        },
        "dt": {
            "model__max_depth": [3, 5, None],
            "model__min_samples_split": [2, 10],
        },
        "rf": {
            "model__n_estimators": [200, 500],
            "model__max_depth": [None, 5, 10],
            "model__min_samples_split": [2, 10],
        }
    }

    results = []
    best_estimators = {}

    for name, model in models.items():
        pipe = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", model)
        ])

        grid = GridSearchCV(
            estimator=pipe,
            param_grid=param_grids[name],
            scoring="f1",
            cv=5,
            n_jobs=-1
        )
        grid.fit(X_train, y_train)

        best_estimators[name] = grid.best_estimator_
        y_pred = grid.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"\n=== {name.upper()} ===")
        print("Best params:", grid.best_params_)
        print(f"Test Accuracy: {acc:.4f}")
        print(f"Test F1-score: {f1:.4f}")

        
        cm_path = os.path.join(figures_dir, f"confusion_matrix_{name}.png")
        plot_and_save_confusion_matrix(y_test, y_pred, cm_path, f"Confusion Matrix - {name.upper()}")

        
        auc = None
        if hasattr(grid.best_estimator_.named_steps["model"], "predict_proba"):
            y_proba = grid.best_estimator_.predict_proba(X_test)[:, 1]
            roc_path = os.path.join(figures_dir, f"roc_{name}.png")
            auc = plot_and_save_roc_curve(y_test, y_proba, roc_path, f"ROC - {name.upper()}")

        results.append({
            "model": name,
            "best_params": grid.best_params_,
            "test_accuracy": acc,
            "test_f1": f1,
            "test_auc": auc
        })

    
    if "rf" in best_estimators:
        rf_pipe = best_estimators["rf"]
        rf_model = rf_pipe.named_steps["model"]

        if hasattr(rf_model, "feature_importances_"):
            
            preprocess = rf_pipe.named_steps["preprocess"]
            num_features = preprocess.transformers_[0][2]
            cat_pipeline = preprocess.transformers_[1][1]
            cat_features = preprocess.transformers_[1][2]
            onehot = cat_pipeline.named_steps["onehot"]
            cat_feature_names = onehot.get_feature_names_out(cat_features).tolist()

            feature_names = list(num_features) + cat_feature_names
            importances = rf_model.feature_importances_

          
            idx = np.argsort(importances)[-15:]
            plt.figure()
            plt.barh(np.array(feature_names)[idx], importances[idx])
            plt.title("Random Forest - Top 15 Feature Importances")
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, "rf_feature_importance.png"), dpi=150)
            plt.close()

   
    res_df = pd.DataFrame(results).sort_values(by="test_f1", ascending=False)
    print("\n=== Summary (sorted by F1) ===")
    print(res_df)

    res_df.to_csv("reports/results_summary.csv", index=False)
    print("\nSaved: reports/results_summary.csv and plots in reports/figures/")


if __name__ == "__main__":
    main()
