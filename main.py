import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import RobustScaler


@st.cache_data(ttl=24*60*60)
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

       It gives the names of categorical, numerical and categorical but cardinal variables in the data set.
    Note: Categorical variables with numerical appearance are also included in categorical variables.

    Parameters
    ------
        dataframe: Dataframe
                The dataframe from which variable names are to be retrieved
        cat_th: int, optional
                threshold value for numeric but categorical variables
        car_th: int, optinal
                threshold value for categorical but cardinal variables

    Returns
    ------
        cat_cols: list
                Categorical variable list
        num_cols: list
                Numeric variable list
        cat_but_car: list
                Categorical but cardinal variable list

    Notes
    ------
        cat_cols + num_cols + cat_but_car = total number of variables
        num_but_cat is inside cat_cols

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    return cat_cols, num_cols, cat_but_car


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


@st.cache_data(ttl=24*60*60)
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def data_prep(dataframe):
    # Tüketilen kısımlarına göre item sınıflandırması

    dataframe.loc[(dataframe["item"] == "Potatoes") | (dataframe["item"] == "Sweet potatoes") | (
            dataframe["item"] == "Groundnuts, excluding shelled"), "consumption_class"] = "0"

    dataframe.loc[(dataframe["item"] == "Carrots and turnips") | (
            dataframe["item"] == "Cassava, fresh"), "consumption_class"] = "1"

    dataframe.loc[(dataframe["item"] == "Onions and shallots, dry (excluding dehydrated)") | (
            dataframe["item"] == "Green garlic") | (
                          dataframe["item"] == "Sugar cane"), "consumption_class"] = "2"

    dataframe.loc[(dataframe["item"] == "Cabbages") | (dataframe["item"] == "Unmanufactured tobacco") | (
            dataframe["item"] == "Lettuce and chicory") | (
                          dataframe["item"] == "Seed cotton, unginned"), "consumption_class"] = "3"

    dataframe.loc[(dataframe["item"] == "Tomatoes") | (dataframe["item"] == "Cucumbers and gherkins") | (
            dataframe["item"] == "Oranges"), "consumption_class"] = "4"
    dataframe.loc[(dataframe["item"] == "Watermelons") | (dataframe["item"] == "Pumpkins, squash and gourds") | (
            dataframe["item"] == "Peas, dry"), "consumption_class"] = "4"
    dataframe.loc[(dataframe["item"] == "Apples") | (dataframe["item"] == "Grapes") | (
            dataframe["item"] == "Bananas"), "consumption_class"] = 4
    dataframe.loc[(dataframe["item"] == "Lemons and limes") | (dataframe["item"] == "Peas, green") | (
            dataframe["item"] == "Plums and sloes"), "consumption_class"] = "4"
    dataframe.loc[(dataframe["item"] == "Pears") | (dataframe["item"] == "Strawberries") | (
            dataframe["item"] == "Peaches and nectarines"), "consumption_class"] = "4"
    dataframe.loc[(dataframe["item"] == "Mangoes, guavas and mangosteens") | (dataframe["item"] == "Pineapples") | (
            dataframe["item"] == "Eggplants (aubergines)"), "consumption_class"] = "4"
    dataframe.loc[(dataframe["item"] == "Coconuts, in shell") | (
            dataframe[
                "item"] == "Chillies and peppers, green (Capsicum spp. and Pimenta spp.)"), "consumption_class"] = "4"

    dataframe.loc[(dataframe["item"] == "Maize (corn)") | (dataframe["item"] == "Wheat") | (
            dataframe["item"] == "Rice"), "consumption_class"] = "5"
    dataframe.loc[(dataframe["item"] == "Beans, dry") | (dataframe["item"] == "Sorghum") | (
            dataframe["item"] == "Barley"), "consumption_class"] = "5"
    dataframe.loc[(dataframe["item"] == "Soya beans") | (dataframe["item"] == "Oats") | (
            dataframe["item"] == "Millet"), "consumption_class"] = "5"
    dataframe.loc[(dataframe["item"] == "Sunflower seed") | (dataframe["item"] == "Apricots") | (
            dataframe["item"] == "Coffee, green") | (
                          dataframe["item"] == "Sesame seed"), "consumption_class"] = "5"

    dataframe.loc[(dataframe["item"] == "Cauliflowers and broccoli"), "consumption_class"] = "6"

    # Yılların sınıflandırılmasıyla oluşturulan değişken

    dataframe.loc[((dataframe["year"] >= 1995) & (dataframe["year"] < 2000)), "YEAR_RANGES"] = "1995-2000"
    dataframe.loc[((dataframe["year"] >= 2000) & (dataframe["year"] < 2005)), "YEAR_RANGES"] = "2000-2005"
    dataframe.loc[((dataframe["year"] >= 2005) & (dataframe["year"] < 2010)), "YEAR_RANGES"] = "2005-2010"
    dataframe.loc[((dataframe["year"] >= 2010) & (dataframe["year"] < 2015)), "YEAR_RANGES"] = "2010-2015"
    dataframe.loc[((dataframe["year"] >= 2015) & (dataframe["year"] <= 2020)), "YEAR_RANGES"] = "2015-2020"

    # İklime göre item değişkeninin sınıflandırılması

    list_x = ["Potatoes", "Cabbages", "Onions and shallots, dry (excluding dehydrated)", "Carrots and turnips", "Wheat",
              "Barley", "Oranges",
              "Lettuce and chicory", "Cauliflowers and broccoli", "Peas, dry", "Green garlic", "Pears", "Oats",
              "Sesame seed", "Cassava, fresh"]

    dataframe["climalite_item"] = dataframe["item"].apply(lambda x: 0 if x in list_x else 1)

    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe, car_th=90)

    num_cols.remove("year")

    scaler = RobustScaler()
    dataframe[num_cols] = scaler.fit_transform(dataframe[num_cols])

    dataframe.replace(r"[^A-Za-z0-9_]+", "", regex=True, inplace=True)
    dataframe = one_hot_encoder(dataframe, cat_cols, drop_first=True)
    dataframe.columns = dataframe.columns.str.split().str.join("_").str.lower()

    return dataframe


st.set_page_config(layout="wide")


@st.cache_data
def get_data():
    dataframe = pd.read_csv("final_yield_df.csv")
    return dataframe


@st.cache_resource
def get_pipeline():
    model = joblib.load("voting_clf.pkl")
    return model


st.title(":green[CROP YIELD PREDICTION]")

main_page, data_page, model_page = st.tabs(["Ana Sayfa", "Veri Seti", "Model"])

# MAİN_PAGE

information_container = main_page.container()
information_container.image("anasayfa.jpeg", use_column_width=True)
information_container.subheader(":green[Mahsül Verimi Nedir?]")
information_container.markdown("""
Mahsül verimi,  tarım ürünlerinin belirli bir alanda veya dönemde ne kadar verimli bir şekilde yetiştirildiğini ve 
toplandığını ifade eder. Mahsül verimi, birçok faktörün etkisi altında değişebilir. 
Bu faktörler arasında iklim koşulları, toprak kalitesi, su kaynakları, gübreleme, zararlı organizmalarla mücadele, 
tarım teknikleri ve çeşitli tarım uygulamaları yer alır.""")

information_container.subheader(":green[Mahsül Verimi Tahminleme Modelinin Amacı Nedir?]")
information_container.markdown("""
Tarım, dünya nüfusunu beslemek ve gıda ihtiyacını karşılamak için kritik bir rol oynamaktadır. Ancak iklim değişikliği, 
doğal kaynakların sınırlı olması ve artan nüfus gibi faktörler, tarımın giderek daha karmaşık ve belirsiz hale gelmesine 
neden olmuştur.Mahsül Verimi Tahminleme modeli ile bu belirsizlikleri azaltmak mümkündür.""")

information_container.subheader(":green[Modelin İşleyişi]")
information_container.markdown("""
İklim verileri, toprak analizleri, sulama ve yağış seviyeleri, gübre ve ilaç kullanımı , bitki türü gibi çeşitli 
veriler, mahsül verimi tahminleme modellerinin temelini oluşturur.""")

# Data Page

df = get_data()
data_page.dataframe(df, use_container_width=True)
data_page.divider()

data_page_col1, data_page_col2 = data_page.columns(2)

fig = plt.figure(figsize=(20, 10))
sns.barplot(data=df, x="region", y="yield_value", )
data_page_col1.subheader("Kıtalara Göre Ortalama Mahsül")
data_page_col1.pyplot(fig)

fig2 = plt.figure(figsize=(20, 10))
sns.scatterplot(data=df, x="avg_pre", y="avg_temp", hue="region")
data_page_col2.subheader("Ülkelerin Ortalama Yağış ve Sıcaklığı ")
data_page_col2.pyplot(fig2)

fig3 = plt.figure(figsize=(20, 10))
sns.lineplot(data=df, x="avg_pre", y="yield_value")
data_page_col1.subheader("Yağışa Göre Ortalama Mahsül")
data_page_col1.pyplot(fig3)

fig4 = plt.figure(figsize=(20, 10))
sns.scatterplot(data=df, x="avg_temp", y="yield_value")
data_page_col2.subheader("Sıcaklığa Göre Ortalama Mahsül")
data_page_col2.pyplot(fig4)

fig5 = plt.figure(figsize=(20, 10))
sns.lineplot(data=df, x="year", y="yield_value", hue="region")
data_page_col1.subheader("Yıllara Göre Ortalama Mahsül")
data_page_col1.pyplot(fig5)

# model page
pipeline = get_pipeline()

# user inputs
user_input_col1, user_input_col2, result_col = model_page.columns([1, 1, 2])

user_country = user_input_col1.selectbox(label="Ülke",
                                         options=np.sort(df.country.unique()))

user_item = user_input_col2.selectbox(label="Ürün", options=np.sort(df[df.country == user_country]["item"].unique()))

user_pest_value = user_input_col1.slider(label="Pestisit(Ton)", min_value=0., max_value=40., step=0.5)

user_avg_temp = user_input_col2.slider(label="Sıcaklık(Celsius)", min_value=-10., max_value=35., step=0.5)

user_avg_pre = user_input_col1.number_input(label="Yağış(mm)", min_value=50., max_value=3500., value=1000., step=50.)

user_year = user_input_col2.number_input(label="Year", min_value=1997, max_value=2030, step=1)

# Prediction


user_input = pd.DataFrame({"region": df[df.country == user_country].iloc[0, 0],
                           "country": user_country,
                           "item": user_item,
                           "year": user_year,
                           "pest_value": user_pest_value,
                           "avg_pre": user_avg_pre,
                           "avg_temp": user_avg_temp,
                           }, index=[0])

if user_input_col2.button("Predict!"):
    result = pipeline.predict(
        pd.DataFrame(data_prep(pd.concat([df.drop("yield_value", axis=1), user_input])).iloc[-1]).T)
    result_col.header(f"Average Yield Value: {round(result[0], 2)} hg/ha", anchor=False)
    if result > 154234.4390617202:
        result_col.image("gulenciftci.jpeg", use_column_width=True)
    else:
        result_col.image("uzgunciftci.jpeg", use_column_width=True)
