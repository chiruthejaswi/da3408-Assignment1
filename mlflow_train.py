import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_openml          # real MNIST (needs internet)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss

# Tracking server 
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mnist-mlp-classifier")


# --- Data: real MNIST, 784-dim, 10 classes ---
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler().fit(X_train)
X_train, X_val = scaler.transform(X_train), scaler.transform(X_val)

configs = [
    {"lr": 0.001, "batch_size": 16,  "hidden": (128, 64)},
    {"lr": 0.001, "batch_size": 128, "hidden": (128, 64)},
    {"lr": 0.01,  "batch_size": 16,  "hidden": (128, 64)},
    {"lr": 0.01,  "batch_size": 128, "hidden": (128, 64)},
    {"lr": 0.1,   "batch_size": 16,  "hidden": (128, 64)},
    {"lr": 0.1,   "batch_size": 128, "hidden": (128, 64)},
]

for i, cfg in enumerate(configs, start=1):
    # with mlflow.start_run(): auto-closes the run even on error (Sec 1.2)
    with mlflow.start_run(run_name=f"mlp-run-{i}"):

        # ---- log hyperparameters (>=3 required by Exercise 1) ----
        mlflow.log_params({
            "model_type": "MLPClassifier",
            "learning_rate": cfg["lr"],
            "batch_size": cfg["batch_size"],
            "hidden_layer_sizes": cfg["hidden"],
            "max_iter": 60,
        })
        mlflow.set_tag("dataset", "MNIST")
        mlflow.set_tag("team", "aiops-lab")

        clf = MLPClassifier(
            hidden_layer_sizes=cfg["hidden"],
            learning_rate_init=cfg["lr"],
            batch_size=cfg["batch_size"],
            max_iter=60,
            random_state=42,
            solver="adam",
        )
        clf.fit(X_train, y_train)

        # ---- log metrics as a TIME SERIES, one point per epoch (Sec 1.2) ----
        for epoch, loss_val in enumerate(clf.loss_curve_):
            mlflow.log_metric("train_loss", loss_val, step=epoch)

        # ---- log final summary metrics (>=2 required by Exercise 1) ----
        final_train_loss = log_loss(y_train, clf.predict_proba(X_train))
        val_acc = accuracy_score(y_val, clf.predict(X_val))
        mlflow.log_metrics({
            "final_train_loss": final_train_loss,
            "val_accuracy": val_acc,
            "n_iter": clf.n_iter_,
        })

        # ---- log the model as an artifact, with a signature (Sec 3.4) ----
        from mlflow.models import infer_signature
        signature = infer_signature(X_train, clf.predict(X_train))
        mlflow.sklearn.log_model(
            clf, artifact_path="model",
            signature=signature, input_example=X_train[:5],
            serialization_format="pickle",   # <-- add this line
            )

        print(f"Run {i}: lr={cfg['lr']} batch={cfg['batch_size']} "
              f"-> train_loss={final_train_loss:.4f} val_acc={val_acc:.4f}")

# ---- programmatically find the best run (Sec 1.5 / Exercise 1 deliverable) ----
best_runs = mlflow.search_runs(
    experiment_names=["mnist-mlp-classifier"],
    order_by=["metrics.val_accuracy DESC"],
)
print("Best run_id:", best_runs.iloc[0].run_id,
      "val_accuracy:", best_runs.iloc[0]["metrics.val_accuracy"])