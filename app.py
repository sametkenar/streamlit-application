import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns

class App:
    def __init__(self):
        self.dataset_name = None
        self.classifier_name = None
        self.enable_grid_search = False
        self.Init_Streamlit_Page()
        self.data = None
        self.params = dict()
        self.clf = None
        self.X, self.y = None, None
    
    def run(self):
        self.get_dataset()
        self.add_parameter_ui()
        if self.data is not None:
            self.generate()
    
    def Init_Streamlit_Page(self):
        st.title('Streamlit Application for Machine Learning Classification of Breast Cancer Wisconsin Dataset')

        st.write("""
        # How the application works:
        1. **Select a dataset**: You can either upload the data as CSV file, or use the preloaded 'Breast Cancer Wisconsin' dataset.
        2. **Select a classifier**: Choose 'KNN', 'SVM', or 'Gaussian Naive Bayes'.
        3. **Enable Grid Search**: If you want to use Grid Search for hyperparameter tuning,
                check this box. If activated, the best parameters will be determined and 
                the model will be trained with those parameters. If you prefer to manually 
                select the parameters and test the model yourself, **please uncheck this box**.
        4. **Set classifier parameters**: If you selected 'SVM', adjust the 'C' parameter using the slider. If you selected 'KNN', adjust the 'K' parameter.
        5. **Run the application**: The application will train the selected classifier on the chosen dataset, and display the evaluation metrics and confusion matrix.
        """)

        self.dataset_name = st.sidebar.selectbox(
            'Select Dataset',
            ('Upload CSV', 'Breast Cancer Wisconsin',)
        )
        st.write(f"## {self.dataset_name} Dataset")

        self.classifier_name = st.sidebar.selectbox(
            'Select classifier',
            ('KNN', 'SVM', 'Gaussian Naive Bayes')
        )
        self.enable_grid_search = st.sidebar.checkbox('Enable Grid Search')
        
    def get_dataset(self):
        if self.dataset_name == 'Breast Cancer Wisconsin':
            self.data = self.load_breast_cancer()
        elif self.dataset_name == 'Upload CSV':
            uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
            if uploaded_file is not None:
                st.write("filename:", uploaded_file.name)
                self.data = self.load_from_csv(uploaded_file)
        else:
            st.write('No dataset selected')
            
        if self.data is not None:
            st.write("Diagnosis Value Counts:")
            st.write(self.data['diagnosis'].value_counts())
            st.write('The first 10 rows of the dataset:')
            st.dataframe(self.data.head(10))
            st.write('The last 10 rows of the dataset with irrelevant columns removed:')
            if 'id' and 'Unnamed: 32' in self.data.columns:
                self.data.drop(columns=['id', 'Unnamed: 32'], axis=1, inplace=True)
            st.dataframe(self.data.tail(10))
            self.preprocess(self.data)
    
    def load_from_csv(self, uploaded_file): 
        data = pd.read_csv(uploaded_file)
        return data

    def load_breast_cancer(self):
        data = pd.read_csv('data.csv')
        return data
    
    def preprocess(self,df):
        df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
        self.X = df.drop('diagnosis', axis=1) 
        self.y = df['diagnosis'] 
        self.plot_correlation(df)
        self.plot_scatter(df)
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)
    
    def plot_correlation(self, df):
        st.subheader('Correlation Matrix')
        fig = plt.figure(figsize=(8, 6))
        sns.heatmap(df.corr(), annot=False, cmap='coolwarm', linewidths=0.5)
        st.pyplot(fig)
    
    def plot_scatter(self, df):
        st.subheader('Scatter Plot: radius_mean vs texture_mean')
        malignant_data = df[df['diagnosis'] == 1]  
        benign_data = df[df['diagnosis'] == 0] 
        fig2 = plt.figure(figsize=(10, 8))
        sns.scatterplot(x='radius_mean', y='texture_mean', data=malignant_data, label='Malignant', color='red')
        sns.scatterplot(x='radius_mean', y='texture_mean', data=benign_data, label='Benign', color='green')
        plt.title('radius_mean vs texture_mean')
        plt.legend()
        st.pyplot(fig2)

    def add_parameter_ui(self):
        if self.classifier_name == 'SVM':
            C = st.sidebar.slider('C', 0.01, 25.0)
            self.params['C'] = C
        elif self.classifier_name == 'KNN':
            K = st.sidebar.slider('K', 1, 25)
            self.params['K'] = K
        else:
            pass

    def get_classifier(self):
        if self.enable_grid_search:
            if self.classifier_name == 'SVM':
                param_grid = {'C': np.arange(0.01,5,0.1)}  
                self.clf = GridSearchCV(SVC(), param_grid,cv=KFold(n_splits=10))
            elif self.classifier_name == 'KNN':
                param_grid = {'n_neighbors': range(1, 25)}
                self.clf = GridSearchCV(KNeighborsClassifier(), param_grid,cv=KFold(n_splits=10),scoring='accuracy')
            elif self.classifier_name == 'Gaussian Naive Bayes':
                self.clf = GaussianNB()
        else:
            if self.classifier_name == 'SVM':
                self.clf  = SVC(C=self.params['C'])
            elif self.classifier_name == 'KNN':
                self.clf  = KNeighborsClassifier(n_neighbors=self.params['K'])
            else:
                self.clf  = GaussianNB()

    def generate(self):
        self.get_classifier()
        # CLASSIFICATION
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        if self.enable_grid_search and not self.classifier_name == 'Gaussian Naive Bayes':
            grid_search = self.clf.fit(X_train, y_train)
            st.write(f'### Best Parameter: {grid_search.best_params_}')
        
        else:
            self.clf.fit(X_train, y_train)
        y_pred = self.clf.predict(X_test)
        self.display_evaluation_metrics(y_test, y_pred)

    def display_evaluation_metrics(self, y_test, y_pred):
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        st.write(f'Classifier = {self.classifier_name}')
        st.write(f'Accuracy =', acc)
        st.write(f'Precision =', precision)
        st.write(f'Recall =', recall)
        st.write(f'F1 Score =', f1)
        st.write('## Confusion Matrix:')
        fig = plt.figure(figsize=(8, 6))
        group_names = ["True Negative","False Positive",'False Negative',"True Positive"]
        group_counts = ['{0:0.0f}'.format(value) for value in cm.flatten()]
        group_percentages = ['{0:.2%}'.format(value) for value in cm.flatten()/np.sum(cm)]
        labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
        labels = np.asarray(labels).reshape(2,2)
        sns.heatmap(cm, annot=labels, fmt='', cmap='Reds')
        plt.xlabel('y_pred')
        plt.ylabel('y_true')
        plt.title('Confusion Matrix')
        st.pyplot(fig)