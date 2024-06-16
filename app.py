import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import xgboost as xgb
from matplotlib.colors import ListedColormap

# Function to plot decision boundaries
def plot_decision_boundaries(X, y, model, ax):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#0000FF'])
    
    ax.contourf(xx, yy, Z, alpha=0.8, cmap=cmap_light)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title("Decision Boundary")

# Streamlit UI
st.title("Classifier Hyperparameter Tuning and Decision Boundary Plotting")
st.write("Tune hyperparameters and visualize decision boundaries for various classifiers.")

classifier_name = st.sidebar.selectbox("Select Classifier", ["Random Forest", "SVC", "Decision Tree", "AdaBoost", "XGBoost", "GBM"])

# Generate a simple classification dataset
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, random_state=42)

# Sidebar controls for hyperparameters
st.sidebar.header("Classifier Hyperparameters")

# Initialize model variable outside the if-else blocks
model = None

if classifier_name == "Random Forest":
    n_estimators = st.sidebar.slider("Number of Trees", 10, 200, step=10, value=100)
    max_depth = st.sidebar.number_input("Max Depth of Tree", min_value=1, max_value=20, value=5, step=1)
    min_samples_split = st.sidebar.slider("Min Samples Split", 2, 10, step=1, value=2)
    min_samples_leaf = st.sidebar.slider("Min Samples Leaf", 1, 10, step=1, value=1)
    max_features = st.sidebar.selectbox("Max Features", ["sqrt", "log2"])
    bootstrap = st.sidebar.selectbox("Bootstrap", [True, False])
    criterion = st.sidebar.selectbox("Criterion", ["gini", "entropy"])
    max_leaf_nodes = st.sidebar.slider("Max Leaf Nodes", 2, 100, step=1, value=None)
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        criterion=criterion,
        max_leaf_nodes=max_leaf_nodes,
        random_state=42
    )

elif classifier_name == "SVC":
    C = st.sidebar.slider("C (Regularization parameter)", 0.01, 10.0, step=0.01, value=1.0)
    kernel = st.sidebar.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
    gamma = st.sidebar.selectbox("Gamma", ["scale", "auto"])
    degree = st.sidebar.slider("Degree (for poly kernel)", 1, 6, step=1, value=3)
    model = SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        degree=degree,
        random_state=42
    )

elif classifier_name == "Decision Tree":
    criterion = st.sidebar.selectbox("Criterion", ["gini", "entropy"])
    splitter = st.sidebar.selectbox("Splitter", ["best", "random"])
    max_depth = st.sidebar.number_input("Max Depth", min_value=1, max_value=20, value=5, step=1)
    min_samples_split = st.sidebar.slider("Min Samples Split", 2, 10, step=1, value=2)
    min_samples_leaf = st.sidebar.slider("Min Samples Leaf", 1, 10, step=1, value=1)
    max_features = st.sidebar.selectbox("Max Features", ["sqrt", "log2", None])
    max_leaf_nodes = st.sidebar.slider("Max Leaf Nodes", 2, 100, step=1, value=None)
    model = DecisionTreeClassifier(
        criterion=criterion,
        splitter=splitter,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        max_leaf_nodes=max_leaf_nodes,
        random_state=42
    )

elif classifier_name == "AdaBoost":
    n_estimators = st.sidebar.slider("Number of Estimators", 10, 200, step=10, value=50)
    learning_rate = st.sidebar.slider("Learning Rate", 0.01, 2.0, step=0.01, value=1.0)
    estimator = st.sidebar.selectbox("Estimator", ["Decision Tree", "SVC"])
    if estimator == "Decision Tree":
        base_model = DecisionTreeClassifier(max_depth=1, random_state=42)
    else:
        base_model = SVC(probability=True, kernel='linear', random_state=42)
    model = AdaBoostClassifier(
        estimator=base_model,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=42
    )

elif classifier_name == "XGBoost":
    n_estimators = st.sidebar.slider("Number of Trees", 10, 200, step=10, value=100)
    learning_rate = st.sidebar.slider("Learning Rate", 0.01, 1.0, step=0.01, value=0.1)
    max_depth = st.sidebar.number_input("Max Depth of Trees", min_value=0, max_value=100, value=50)
    min_child_weight = st.sidebar.slider("Min Child Weight", 1, 10, step=1, value=1)
    subsample = st.sidebar.slider("Subsample", 0.5, 1.0, step=0.1, value=1.0)
    colsample_bytree = st.sidebar.slider("Colsample by Tree", 0.5, 1.0, step=0.1, value=1.0)
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=42
    )

elif classifier_name == "GBM":
    n_estimators = st.sidebar.slider("Number of Trees", 10, 200, step=10, value=100)
    learning_rate = st.sidebar.slider("Learning Rate", 0.01, 1.0, step=0.01, value=0.1)
    max_depth = st.sidebar.number_input("Max Depth of Tree", min_value=1, max_value=20, value=5, step=1)
    min_samples_split = st.sidebar.slider("Min Samples Split", 2, 10, step=1, value=2)
    min_samples_leaf = st.sidebar.slider("Min Samples Leaf", 1, 10, step=1, value=1)
    subsample = st.sidebar.slider("Subsample", 0.5, 1.0, step=0.1, value=1.0)
    max_features = st.sidebar.selectbox("Max Features", [ "sqrt", "log2", None])
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        subsample=subsample,
        max_features=max_features,
        random_state=42
    )

# Train model
if st.sidebar.button("Train Algorithm"):
    model.fit(X, y)

# Calculate training accuracy
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    st.write(f"Training Accuracy: {accuracy:.2f}")

# Plot decision boundary
    st.subheader("Decision Boundary")
    fig, ax = plt.subplots()
    plot_decision_boundaries(X, y, model, ax)
    st.pyplot(fig)