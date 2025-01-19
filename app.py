import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
import sklearn

with open('scaler.pkl', 'rb') as f:
    scaler = pkl.load(f)

with open('onehot_encoder.pkl', 'rb') as f:
    onehot_encoder = pkl.load(f)

with open('rf_model.pkl', 'rb') as f:
    rf_model = pkl.load(f)

with open('station_mapping.pkl', 'rb') as f:
    station_mapping = pkl.load(f)

df = pd.read_csv("moscow_housing.csv")

st.title("Интеллектуальная система прогнозирования цен на недвижимость")
st.header("Введите данные недвижимости")

col1, col2 = st.columns(2)
col3, col4, col5 = st.columns(3)

with col1:
    area = np.array(st.number_input("Площадь дома (м²):", value=1.0, step=1.0))
    floor = np.array(st.number_input("Этаж:", value=1, step=1))
    station = st.selectbox(
        "Ближайшая станция метро:",
        df["Metro station"].str.lower().str.strip().unique().tolist()
    )


with col2:
    n_rooms = np.array(st.number_input("Количество комнат:", value=1, step=1))
    n_floors = np.array(st.number_input("Количество этажей в здании:", value=1, step=1))
    min_to_metro = np.array(st.number_input("Минут до метро:", value=1, step=1))

with col3:
    region = st.radio(
        "Регион:",
        options=['Москва', 'Область']
    )

with col4:
    renovation = st.radio(
        "Ремонт",
        options=["Без ремонта", "Косметический", "Европейский стиль", "Дизайнерский"]
    )

with col5:
    type = st.radio(
        "Тип",
        options=["Новостройка", "Старое здание"]
    )


region_mapping = {'Область': 0, 'Москва': 1}
type_mapping = {'Старое здание': 0, 'Новостройка': 1}
renovation_mapping = {"Без ремонта": "Without renovation", "Косметический": "Cosmetic", "Европейский стиль": "European-style renovation", "Дизайнерский": "Designer"}

station_encoded = station_mapping[station]
renovation_mapped = renovation_mapping.get(renovation)
region_encoded = region_mapping.get(region)
type_encoded = type_mapping.get(type)

relative_floor = floor / n_floors

if st.button('Прогнозировать цену'):
    renovation_encoded = pd.DataFrame(onehot_encoder.transform(np.array([[renovation_mapped]])).toarray(), columns=onehot_encoder.categories_[0])
    renovation_encoded.rename(columns={
        "Cosmetic": "cosmetic_renovation",
        "Designer": "designer_renovation",
        "European-style renovation": "euro_style_renovation",
        "Without renovation": "no_renovation"
    }, inplace=True)

    data_to_scale = pd.DataFrame({
        'n_rooms': [n_rooms],
        'area': [area],
        'n_floors': [n_floors],
        'floor': [floor],
        'station_encoded': [station_encoded],
        'min_to_metro': [min_to_metro]
    })
    scaled_data = pd.DataFrame(scaler.transform(data_to_scale), columns=data_to_scale.columns)

    input_data = pd.DataFrame({
        'type': [type_encoded],
        'region': [region_encoded],
        'relative_floor': [relative_floor]
    })

    input_data = pd.concat([input_data, scaled_data], axis=1)
    input_data = pd.concat([input_data, renovation_encoded], axis=1)

    data_order = ['type', 'min_to_metro', 'region', 'n_rooms', 'area', 'floor', 'n_floors', 'cosmetic_renovation',
                   'designer_renovation', 'euro_style_renovation', 'no_renovation', 'station_encoded', 'relative_floor']
    input_data = input_data[data_order]
    
    y_hat = rf_model.predict(input_data)
    
    min_price = y_hat - 2500000
    max_price = y_hat + 2500000
    
    min_formatted_value = f"{min_price[0]:,.0f}"
    max_formatted_value = f"{max_price[0]:,.0f}"

    st.markdown(
    f"""
    <h3 style="text-align: center; font-weight: semibold; font-size: 24px;">
     {min_formatted_value} - {max_formatted_value}
    </h3>
    """, 
    unsafe_allow_html=True
)