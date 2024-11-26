# Core Pkgs
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pickle
from skimage.transform import resize
import pandas as pd
import plotly.express as px
from PIL import Image
import PIL.Image
from load_css import local_css

local_css("style.css")


def main():

    st.title("Metal surface defect detection")
    menu = ["Home","Quality Evaluation", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Home":
        st.subheader("Detection")

        model = pickle.load(open('App/models/cnn_model1.p', 'rb'))

        uploaded_file = st.file_uploader("Choose an Image..",type=['jpg', 'png','bmp','jpeg'])


        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.image(img,caption='Uploaded image')
            if st.button('Predict'):
                CATEGORIES = ['Crazing', 'Inclusion', 'Patches', 'Pitted', 'Rolled', 'Scratches']
                st.write('Result..')
                flat_data=[]
                img = np.array(img)
                img_resized = resize(img, (150, 150, 3))
                flat_data.append(img_resized.flatten())
                flat_data = np.array(flat_data)

                plt.imshow(img_resized)
                # print(flat_data)

                y_out = model.predict(flat_data)

                y_out = CATEGORIES[y_out[0]]
                st.write(f' PREDICTED OUTPUT:')
                res = f"<div><span class='highlight blue'>{y_out}</span> </div>"
                st.markdown(res, unsafe_allow_html=True)
                # st.write(brisque.score(img))


    elif choice == "Quality Evaluation":

        if st.button('info'):
            img = Image.open('App/fileDir/info_img.png')
            st.image(img,caption='info image')
            plt.imshow(img)
        excel_file = 'reports/clsf_report_train.xlsx'
        sheet_name = 'Sheet1'
        sheet_name2 ='Sheet2'
        sheet_name3 = 'Sheet3'

        #Models cmp
        st.header("Model Accuracy")

        df= pd.read_excel(excel_file,sheet_name=sheet_name3,usecols='A:B')
        st.dataframe(df)
        fig = px.bar(df,title='Classification Report for Train Dataset', x='Model', y='Accuracy')
        fig.update_traces(marker_color='#FF5733')
        st.plotly_chart(fig)

        # # CONFUSION MATRICES
        #
        # st.header("Confusion Matrix")
        #
        # st.image(Image.open('/Users/hrushika/Desktop/Projectworks/mj/App/fileDir/RanFrstCM.png'))
        # st.image(Image.open('/Users/hrushika/Desktop/Projectworks/mj/App/fileDir/KnnCM.png'))
        # st.image(Image.open('/Users/hrushika/Desktop/Projectworks/mj/App/fileDir/CnnCM.png'))

        #RFM
        st.header("Random Forest model")
        st.subheader("Classification Report for Train Dataset")

        df= pd.read_excel(excel_file,sheet_name=sheet_name,usecols='A:E')
        st.dataframe(df)
        fig = px.bar(df,title='Classification Report for Train Dataset', x='defects', y='f1-score')
        fig.update_traces(marker_color='#00FFEF')
        st.plotly_chart(fig)

        st.subheader("Classification Report for Test Dataset")
        excel_file2 = 'reports/clsf_report_test.xlsx'
        df= pd.read_excel(excel_file2,sheet_name=sheet_name,usecols='A:E')
        st.dataframe(df)
        fig = px.bar(df,title='Classification Report for Test Dataset', x='defects', y='f1-score')
        fig.update_traces(marker_color='#00FFEF')
        st.plotly_chart(fig)

        #KNN

        st.header("KNN model")
        st.subheader("Classification Report for Train Dataset")

        df= pd.read_excel(excel_file,sheet_name=sheet_name2,usecols='A:E')
        st.dataframe(df)
        fig = px.bar(df,title='Classification Report for Train Dataset', x='defects', y='f1-score')
        fig.update_traces(marker_color='#0FFF50')
        st.plotly_chart(fig)

        st.subheader("Classification Report for Test Dataset")
        excel_file2 = 'reports/clsf_report_test.xlsx'
        df= pd.read_excel(excel_file2,sheet_name=sheet_name2,usecols='A:E')
        st.dataframe(df)
        fig = px.bar(df,title='Classification Report for Test Dataset', x='defects', y='f1-score')
        fig.update_traces(marker_color='#0FFF50') #red #FF5733
        st.plotly_chart(fig)

        #CNN
        st.header("CNN model")
        st.subheader("Classification Report for Test Dataset")
        excel_file2 = 'reports/clsf_report_test.xlsx'
        df= pd.read_excel(excel_file2,sheet_name=sheet_name3,usecols='A:E')
        st.dataframe(df)
        fig = px.bar(df,title='Classification Report for Test Dataset', x='defects', y='f1-score')
        fig.update_traces(marker_color='#FF5733') #red #FF5733
        st.plotly_chart(fig)










        # x1 = pd.read_excel(excel_file, sheet_name=sheet_name,usecols='B:D')
        # # x2 = pd.read_excel(excel_file, sheet_name=sheet_name, usecols='C')
        # # x3 = pd.read_excel(excel_file, sheet_name=sheet_name, usecols='D')
        # hist_data = [x1]
        # group_labels = ['Crazing', 'Inclusion', 'Patches', 'Pitted', 'Rolled', 'Scratches','acu','ma','rate']
        # fig = ff.create_distplot(hist_data, group_labels, bin_size=[.1, .25, .5])
        #
        # # Plot!
        # st.plotly_chart(fig, use_container_width=True)


        # df_participants= pd.read_excel(excel_file,sheet_name=sheet_name,usecols='J:K')
        # st.dataframe(df)
        # df_participants.dropna(inplace=True)
        # st.dataframe(df_participants)
        # pie_chart = px.pie(df_participants,title='Classification Report for Test Dataset',values='defects1',names='pre1')
        # st.plotly_chart(pie_chart)
    else:
        st.subheader("About")
        # st.markdown("### ML Major Project")
        st.write("Metal defect detection is used to detect defects on the surface of the metal and to improve the quality of the metal surface. However, traditional image detection algorithms cannot meet the detection requirements because of small defect features and low contrast between background and features about metal surface defect datasets. A novel recognition algorithm for metal surface defects based on Random Forest Model\n")


if __name__ == '__main__':
    main()


# import joblib
# from sklearn.datasets import load_files
# import pyngrok
# from pyngrok import ngrok
# # EDA Pkgs
# import pandas as pd

# from keras.utils import np_utils
# from keras.preprocessing.image import array_to_img, img_to_array, load_img


# pipe_lr = joblib.load(open("models/metal_surface_def.pkl", "rb"))
# test_dir = 'input'
# uploaded_file = st.file_uploader("Choose an Image..",type=['jpg', 'png','bmp'])
# if uploaded_file is not None:
#     img = Image.open(uploaded_file)
#     st.image(img,caption='Uploaded image')
#     if st.button('Predict'):
#         CATEGORIES = ['Crazing', 'Inclusion', 'Patches', 'pitt', 'Rolled', 'Scratches']
#         st.write('Result..')
#         flat_data=[]
#         img = np.array(img)
#         img_resized = resize(img, (150, 150, 3))
#         flat_data.append(img_resized.flatten())
#         flat_data = np.array(flat_data)
#
#         plt.imshow(img_resized)
#         # print(flat_data)
#
#         y_out = model.predict(flat_data)
#
#         y_out = CATEGORIES[y_out[0]]
#         st.write(f' PREDICTED OUTPUT: {y_out}')









# def load_dataset(path):
#     data = load_files(path)
#     files = np.array(data['filenames'])
#     targets = np.array(data['target'])
#     target_labels = np.array(data['target_names'])
#     return files, targets, target_labels
#
#
# def predict_emotions(file):
#
#     results = pipe_lr.predict(file)
#     return results
#
# def load_image(img):
#     im = Image.open(img)
#     image = np.array(im)
#     return image
#
# def convert_image_to_array(files):
#     images_as_array=[]
#     for file in files:
#         # Convert to Numpy Array
#         images_as_array.append(img_to_array(load_img(file)))
#     return images_as_array
# t = "<div>Hello there my <span class='highlight blue'>name <span class='bold'>yo</span> </span> is <span class='highlight red'>Fanilo <span class='bold'>Name</span></span></div>"
#
# st.markdown(t, unsafe_allow_html=True)




# model = pickle.load(open('/Users/hrushika/Desktop/Projectworks/mj/App/models/rfm_model.p','rb'))s