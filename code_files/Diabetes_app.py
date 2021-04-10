# -*- coding: utf-8 -*-
import pandas as pd
import streamlit as st
from PIL import Image
import streamlit.components.v1 as components
import pickle
from joblib import load
import codecs
import matplotlib.pyplot as plt
import seaborn as sns


#The best model loaded

with open("C:/Users/zuzan/Documents/Diabetes/Diabetes/notebooks/model.pkl", 'rb') as file:
    model = pickle.load(file)
file.close()

#Standard Scaler model loaded
sc = load("C:/Users/zuzan/Documents/Diabetes/Diabetes/notebooks/scaler.joblib")

#Function to display weetviz report
def display_sweetviz(report_html, width = 1000, height=1000):
    report_file = codecs.open(report_html, 'r')
    page = report_file.read()
    components.html(page, width = width, height = height, scrolling = True)

#Function to display histogram. Function return created figure.
def hist_display(data,user_data, i):
    X = data.iloc[:,i]

    fig= plt.figure(figsize=(9,9))
    sns.set_style('darkgrid')
    ax = sns.distplot(X, vertical = True)
    low_border = ax.containers[0][0].get_xy()[1]

    for bar in ax.containers[0]:
        if  user_data.iloc[:,i].values[0] <= (low_border + bar.get_height()) and  user_data.iloc[:,i].values[0] >=low_border:

            bar.set_color('crimson')


        low_border = low_border + bar.get_height()

    return fig

    
#Function to get data inputed by user. Function return user data saved as Data Frame.   
def get_imput():
    
    age = st.sidebar.slider('Age', 21, 81, 21)
    weight = st.sidebar.slider('Weight (kg)', 40,120,55)
    height = st.sidebar.slider('Height (m)',1.4, 2.2, 1.75)
    pregnancies = st.sidebar.slider('Number of Pregnancies', 0, 17, 1)
    glucose = st.sidebar.slider('Plasma glucose concentration', 5, 200, 120)
    blood = st.sidebar.slider('Blood Pressure (mm Hg)', 5, 130, 70)
    skin = st.sidebar.slider('SkinThickness (mm)', 1, 60, 20)
    insulin = st.sidebar.slider('2-Hour serum insulin (mu U/ml)', 1, 900, 80)
    DiabetesPedigreeFunction = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 3.0, 0.5)
    
    

    data_from_user = {'Pregnancies': pregnancies,
                      'Glucose': glucose,                 
                      'BloodPressure': blood,
                      'SkinThickness': skin,
                      'Insulin': insulin,
                      'BMI': weight/(height)**2,
                      'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
                      'Age': age,  
                }
    
    user_features = pd.DataFrame(data_from_user, index =[0])
    return user_features

# Function to create windowns of appliaction
def main():
    #Created menu
    menu = ["Home", "What is Diabetes?","Test", "Statistics"]
    
    choice = st.sidebar.selectbox("Menu", menu)
    
    # Home view (deafult view)
    
    if choice == "Home":
        html_temp = """
        <div style = "background-color: rgb(153, 204, 255);padding:10px;border-radius:10px">
        <h1 style = "color:white; text-align:center;">DIABETES DETECTION APPLICATION
        </div>
        """
        components.html(html_temp)
        
        col1, col2 = st.beta_columns(2)
        with col1:
            image = Image.open('C:/Users/zuzan/Documents/Diabetes/Diabetes/code_files/images/diabetes.png')
            s = 600,600
            image.thumbnail(s)
            st.image(image)
        with col2:
            image = Image.open('C:/Users/zuzan/Documents/Diabetes/Diabetes/code_files/images/heart.png')
            s = 600,600
            image.thumbnail(s)
            st.image(image)  
            
   # "What is Diabetes?" - vido with short intro      
    elif  choice == "What is Diabetes?":
        html_temp4 = """
        <div style = "background-color: rgb(153, 204, 255);padding:10px;border-radius:10px">
        <h1 style = "color:white; text-align:center;">What is Diabetes?
        </div>
        """   
        components.html(html_temp4)
        st.video("https://youtu.be/ObpeolfZMPs")
        
        
   # Test view. User can input values and check if she might be suffering from Diabetes.
    elif choice == "Test":
    
        user_param = get_imput()
        
        X = sc.transform(user_param)
       
        prediction = 9


        if st.sidebar.button("Test your result"):
 
            prediction = model.predict(X)
        else:
            prediction = 9
            
        html_temp2 = """
        <div style = "background-color: rgb(153, 204, 255);padding:10px;border-radius:10px">
        <h1 style = "color:white; text-align:center;">Parameters
        </div>
        """
        html_temp3 = """
        <div style = "background-color: #ccccff;padding:10px;border-radius:10px">
        <h1 style = "color:white; text-align:center;">Negative Result
        </div>
        """
        html_temp5 = """
        <div style = "background-color: rgb(255, 102, 102);padding:10px;border-radius:10px">
        <h1 style = "color:white; text-align:center;">Positive Result
        </div>
        """
    
        
        if prediction == 0.0:
          components.html(html_temp3)
          st.subheader("You are not supposed to suffer from Diabetes.")
          st.write("")
          col1, col2 = st.beta_columns([3,1])
          
          with col1:            
              with st.beta_expander("Prevention"):
                  with st.beta_container():
                  
                      st.write("Remember about prevention. At present, type 1 diabetes cannot be prevented, but there are a number of factors that influence the development of type 2 diabetes")
                      st.write("Studies from different parts of the world have established that lifestyle modification with physical activity and/or healthy diet can delay or prevent the onset of type 2 diabetes.")
                      st.write("Taking a life course perspective is essential for preventing type 2 diabetes and its complications.")
              with st.beta_expander("Recommendations"):
                  with st.beta_container():
                     st.write("IDF recommendations for a healthy diet for the general population")
                     st.write("1. Choosing water, coffee or tea instead of fruit juice, soda, or other sugar sweetened beverages.")
                     st.write("2. Eating at least three servings of vegetable every day, including green leafy vegetables.")
                     st.write("3. Eating up to three servings of fresh fruit every day.")
                     st.write("4. Limiting alcohol intake to a maximum of two standard drinks per day.")
                     st.write("5. Choosing lean cuts of white meat, poultry or seafood instead of red or processed meat.")
                     st.write("6. Choosing peanut butter instead of chocolate spread or jam.")
                     st.write("7. Choosing whole-grain bread, rice, or pasta instead of white bread, rice, or pasta.")
                     st.write("8. Choosing unsaturated fats (olive oil, canola oil, corn oil, or sunflower oil) instead of saturated fats (butter, ghee, animal fat, coconut oil or palm oil.")
          with col2:
            image3 = Image.open('C:/Users/zuzan/Documents/Diabetes/Diabetes/code_files/images/right.png')
            s = 200,200
            image3.thumbnail(s)
            st.image(image3)  
            

            
        elif prediction == 1.0:
          
          components.html(html_temp5)
          st.subheader("You might be suffering from Diabetes")
          st.write("")
          col1, col2 = st.beta_columns([3,1])
          
          with col1:
              with st.beta_expander("Types of diabetes"):
                  with st.beta_container():
                      st.write("There are three main types of diabetes – type 1, type 2 and gestational.")
                      st.write("Type 1 diabetes can develop at any age, but occurs most frequently in children and adolescents.")
                      st.write("Type 2 diabetes is more common in adults and accounts for around 90% of all diabetes cases")
                      st.write("Gestational diabetes (GDM) is a type of diabetes that consists of high blood glucose during pregnancy and is associated with complications to both mother and child.")
              with st.beta_expander("Complications"):
                  with st.beta_container():
                      st.write("People with diabetes have an increased risk of developing a number of serious health problems.")
                      st.write("High blood glucose levels can lead to serious diseases affecting the heart and blood vessels, eyes, kidneys, nerves and teeth.")
                      st.write("People with diabetes also have a higher risk of developing infections. ")
                      st.write("diabetes is a leading cause of cardiovascular disease, blindness, kidney failure, and lower limb amputation.")
                      st.write("Maintaining blood glucose levels, blood pressure, and cholesterol at or close to normal can help delay or prevent diabetes complications. ")
                      st.write("People with diabetes need regular monitoring")

          with col2:
            image3 = Image.open('C:/Users/zuzan/Documents/Diabetes/Diabetes/code_files/images/wrong.png')
            s = 200,200
            image3.thumbnail(s)
            st.image(image3)    
              
        else:
          components.html(html_temp2)
        
          col1, col2 = st.beta_columns(2)
          
          with col1:
              with st.beta_container():
                  st.write("")
                  st.write("")
                  st.write("")
                  st.write("")
                  st.write("Pregnancies - number of times pregnant")
                  st.write("Glucose - plasma glucose concentration a 2 hours in an oral glucose tolerance test")
                  st.write("Blood Pressure - diastolic blood pressure (mm Hg)")
                  st.write("SkinThickness - triceps skin fold thickness (mm)")
                  st.write("Insulin - 2-Hour serum insulin (mu U/ml)")
                  st.write("BMI - body mass index (weight in kg/(height in m)^2)")
                  st.write("DiabetesPedigreeFunction - diabetes pedigree function")
                  st.write("Age - age (years)")
          with col2:
            image3 = Image.open('C:/Users/zuzan/Documents/Diabetes/Diabetes/code_files/images/the-world-day-of.png')
            s = 600,600
            image3.thumbnail(s)
            st.image(image3)    
              
  # Statistic view was created to check distribution of each variable and can compare user result with others.  
    else:
        html_temp6 = """
        <div style = "background-color: rgb(153, 204, 255);padding:10px;border-radius:10px">
        <h1 style = "color:white; text-align:center;">Statistics
        </div>
        """
        components.html(html_temp6)
        if st.sidebar.button("Show Report") and not st.sidebar.button("Show charts"):
            temp = 1
        else:
            temp = 0
        
        
        if temp == 1:
            display_sweetviz('C:/Users/zuzan/Documents/Diabetes/Diabetes/code_files/SWEETVIZ_REPORT.html')
        else:
            data_file = pd.read_csv('C:/Users/zuzan/Documents/Diabetes/Diabetes/code_files/data/export_dataframe.csv')

            user_param2 = get_imput()
            charts_number = len(data_file.columns) - 1
            cols = st.beta_columns(4)
            

            
            for i in range(charts_number):
                if i < 4:
                    with cols[i]:
                        st.write(data_file.columns[i])
                        st.pyplot(hist_display(data_file, user_param2, i))
                else:
                
                    with cols[i - 4]:
                        st.write(data_file.columns[i])
                        st.pyplot(hist_display(data_file, user_param2, i))       
        
      
if __name__ =='__main__':
    main()
    
        

    


    
    
    
    