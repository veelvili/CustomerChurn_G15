import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Function to load data
def load_data():
    try:
        train_df = pd.read_csv("bank_customer_churn_train_data_processed.csv")
        test_df = pd.read_csv("bank_customer_churn_test_data_processed.csv")
        return train_df, test_df
    except FileNotFoundError:
        st.error("Error: CSV files not found. Please check file paths.")
        return None, None

# Function to train and test the model
def train_random_forest(train_df, test_df):
    X_train = train_df.drop(columns=["Exited", "CustomerId"], errors="ignore")  # Exclude unnecessary columns
    y_train = train_df["Exited"]
    X_test = test_df.drop(columns=["Exited", "CustomerId"], errors="ignore")
    y_test = test_df["Exited"]

    best_rf = RandomForestClassifier(
        bootstrap=False,
        max_depth=20,
        min_samples_leaf=1,
        min_samples_split=2,
        n_estimators=200,
        random_state=42
    )
    best_rf.fit(X_train, y_train)

    y_pred = best_rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return best_rf, X_train.columns.tolist(), accuracy, cm, report

# Streamlit app
def main():
    st.title("Customer Churn Prediction App")
    st.write("This app predicts whether a bank customer will churn based on Random Forest modeling.")

    train_df, test_df = load_data()
    if train_df is None or test_df is None:
        return

    if st.button("Train and Evaluate Model"):
        with st.spinner("Training the Random Forest model..."):
            model, feature_names, accuracy, cm, report = train_random_forest(train_df, test_df)
            st.session_state["model"] = model
            st.session_state["feature_names"] = feature_names

        st.success("Model training complete!")
        st.subheader("Model Accuracy")
        st.write(f"Accuracy: {accuracy:.4f}")

        st.subheader("Confusion Matrix Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    st.header("Make Predictions")
    st.write("Input customer data to predict whether they will churn.")

    # Sidebar inputs
    st.sidebar.header("Input Features")
    inputs = {
        "CreditScore": st.sidebar.slider("Credit Score", 363, 850, 700),
        "Geography_France": st.sidebar.selectbox("Geography France", [0, 1]),
        "Geography_Germany": st.sidebar.selectbox("Geography Germany", [0, 1]),
        "Geography_Spain": st.sidebar.selectbox("Geography Spain", [0, 1]),
        "Gender": st.sidebar.selectbox("Gender (0: Female, 1: Male)", [0, 1]),
        "Age": st.sidebar.slider("Age", 21, 72, 35),
        "Tenure": st.sidebar.slider("Tenure", 0, 10, 5),
        "Balance": st.sidebar.slider("Balance", 0.0, 250898.09, 50000.0),
        "NumOfProducts": st.sidebar.slider("Number of Products", 1, 2, 1),
        "HasCrCard": st.sidebar.selectbox("Has Credit Card (0: No, 1: Yes)", [0, 1]),
        "IsActiveMember": st.sidebar.selectbox("Is Active Member (0: No, 1: Yes)", [0, 1]),
        "EstimatedSalary": st.sidebar.slider("Estimated Salary", 11.58, 199992.48, 50000.0),
        "Complain": st.sidebar.selectbox("Complain (0: No, 1: Yes)", [0, 1]),
        "Satisfaction Score": st.sidebar.slider("Satisfaction Score", 1, 5, 3),
        "Card Type": st.sidebar.slider("Card Type", 1, 4, 2),
        "Point Earned": st.sidebar.slider("Points Earned", 119, 1000, 500),
    }

    if st.sidebar.button("Predict"):
        if "model" in st.session_state and "feature_names" in st.session_state:
            # Align input data with training feature order
            customer_data = pd.DataFrame([inputs])
            missing_columns = set(st.session_state["feature_names"]) - set(customer_data.columns)
            for col in missing_columns:
                customer_data[col] = 0  # Fill missing columns with default values (e.g., 0)
            customer_data = customer_data[st.session_state["feature_names"]]

            probabilities = st.session_state["model"].predict_proba(customer_data)[0]
            st.sidebar.success(f"Probability of Not Exited: {probabilities[0]:.4f}")
            st.sidebar.success(f"Probability of Exited: {probabilities[1]:.4f}")
        else:
            st.sidebar.error("Model is not trained yet. Please train the model first.")

if __name__ == "__main__":
    main()
