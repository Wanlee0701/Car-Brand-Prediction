import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import RandomOverSampler
# import joblib
# import os

# --- Đặt st.set_page_config() lên đầu ---
st.set_page_config(page_title="Dự đoán Hãng Xe", layout="wide")

# --- Các hàm tiền xử lý từ notebook (được điều chỉnh một chút) ---
def handle_missing_values_app(data):
    # ... (giữ nguyên phần còn lại của hàm) ...
    if 'Customer Name' in data.columns and data['Customer Name'].isnull().any():
        data = data.dropna(subset=['Customer Name'])
    return data

def detect_and_treat_outliers_iqr_app(df, column_name, factor=1.5):
    # ... (giữ nguyên phần còn lại của hàm) ...
    if column_name not in df.columns or not pd.api.types.is_numeric_dtype(df[column_name]):
        return df.copy()

    df_processed = df.copy()
    Q1 = df_processed[column_name].quantile(0.25)
    Q3 = df_processed[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR

    df_processed[column_name] = np.where(
        df_processed[column_name] < lower_bound, lower_bound,
        np.where(df_processed[column_name] > upper_bound, upper_bound, df_processed[column_name])
    )
    return df_processed

# --- Hàm tải dữ liệu, tiền xử lý và huấn luyện mô hình ---
@st.cache_resource
def load_data_and_train_model():
    # ... (giữ nguyên phần còn lại của hàm) ...
    try:
        data_orig_local = pd.read_csv('Car Sales.xlsx - car_data.csv')
    except FileNotFoundError:
        st.error("Lỗi: Không tìm thấy file dữ liệu 'Car Sales.xlsx - car_data.csv'. Hãy đảm bảo file này nằm cùng thư mục với app.py.")
        return None, None, None, None, None, None

    data = data_orig_local.copy()

    data = handle_missing_values_app(data)

    numerical_cols_for_outlier = ['Annual Income', 'Price ($)']
    for col_name in numerical_cols_for_outlier:
        if col_name in data.columns:
            data = detect_and_treat_outliers_iqr_app(data, col_name, factor=1.5)

    features_to_drop = ['Car_id', 'Date', 'Customer Name', 'Phone', 'Dealer_Name', 'Dealer_No ', 'Engine', 'Model', 'Gender']
    features_to_drop_existing = [col for col in features_to_drop if col in data.columns]

    y_series = data['Company']
    X_df = data.drop(columns=['Company'] + features_to_drop_existing, errors='ignore')

    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y_series)
    num_classes = len(le_target.classes_)

    if num_classes < 2:
        st.error("Cần ít nhất 2 hãng xe để thực hiện bài toán phân loại.")
        return None, None, None, None, None, None

    numerical_cols_model = X_df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols_model = X_df.select_dtypes(include=['object', 'category']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols_model),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols_model)
        ],
        remainder='passthrough'
    )

    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_df, y_encoded)

    final_model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                           ('classifier', LGBMClassifier(n_estimators=200, learning_rate=0.1, max_depth=40, num_leaves=15,objective='multiclass', num_classes=30))])

    final_model_pipeline.fit(X_resampled, y_resampled)

    unique_values = {col: X_df[col].unique().tolist() for col in categorical_cols_model}
    min_max_values = {col: (X_df[col].min(), X_df[col].max()) for col in numerical_cols_model}

    return final_model_pipeline, le_target, X_df.columns, unique_values, min_max_values, data_orig_local


# --- Tải mô hình và các đối tượng cần thiết ---
pipeline_objects = load_data_and_train_model()

if pipeline_objects and pipeline_objects[0] is not None:
    final_model_pipeline, le_target, feature_names_model, unique_categorical_values, min_max_numerical_values, data_orig = pipeline_objects

    # --- Các lệnh Streamlit khác bắt đầu từ đây ---
    st.title("🚗 Dự đoán Hãng Xe Ô Tô")
    st.markdown("Nhập các thông tin của xe để dự đoán hãng sản xuất.")

    with st.sidebar:
        st.header("Thông tin xe")
        user_inputs = {}

        if 'Annual Income' in feature_names_model:
            min_val_ai, max_val_ai = min_max_numerical_values.get('Annual Income', (0, 10000000))
            user_inputs['Annual Income'] = st.number_input(
                "Thu nhập hàng năm (Annual Income)",
                min_value=0,
                value=int(data_orig['Annual Income'].median()) if 'Annual Income' in data_orig else 500000,
                step=1000,
                help=f"Giá trị nên trong khoảng ({int(min_val_ai)} - {int(max_val_ai)}) dựa trên dữ liệu gốc."
            )

        if 'Transmission' in feature_names_model and 'Transmission' in unique_categorical_values:
            user_inputs['Transmission'] = st.selectbox("Loại hộp số (Transmission)", options=unique_categorical_values['Transmission'])
        
        if 'Color' in feature_names_model and 'Color' in unique_categorical_values:
            user_inputs['Color'] = st.selectbox("Màu sắc (Color)", options=unique_categorical_values['Color'])

        if 'Price ($)' in feature_names_model:
            min_val_p, max_val_p = min_max_numerical_values.get('Price ($)', (0, 100000))
            user_inputs['Price ($)'] = st.number_input(
                "Giá xe (Price $)",
                min_value=0,
                value=int(data_orig['Price ($)'].median()) if 'Price ($)' in data_orig else 20000,
                step=100,
                help=f"Giá trị nên trong khoảng ({int(min_val_p)} - {int(max_val_p)}) dựa trên dữ liệu gốc."
            )
        
        if 'Body Style' in feature_names_model and 'Body Style' in unique_categorical_values:
            user_inputs['Body Style'] = st.selectbox("Kiểu dáng xe (Body Style)", options=unique_categorical_values['Body Style'])

        if 'Dealer_Region' in feature_names_model and 'Dealer_Region' in unique_categorical_values:
            user_inputs['Dealer_Region'] = st.selectbox("Khu vực Đại lý (Dealer_Region)", options=unique_categorical_values['Dealer_Region'])

        predict_button = st.button("Dự đoán Hãng Xe", type="primary")

    if predict_button:
        if not user_inputs or len(user_inputs) < len(feature_names_model):
            st.warning("Vui lòng nhập đầy đủ thông tin cho tất cả các trường.")
        else:
            input_list = []
            for col in feature_names_model:
                if col in user_inputs:
                    input_list.append(user_inputs[col])
                else:
                    if col in data_orig.select_dtypes(include=np.number).columns:
                         input_list.append(data_orig[col].median())
                    elif col in data_orig.columns:
                         input_list.append(data_orig[col].mode()[0] if not data_orig[col].mode().empty else None)
                    else:
                        input_list.append(None)

            input_df = pd.DataFrame([input_list], columns=feature_names_model)

            st.subheader("Thông tin xe đã nhập:")
            st.dataframe(input_df)

            try:
                prediction_encoded = final_model_pipeline.predict(input_df)
                prediction_proba = final_model_pipeline.predict_proba(input_df)
                predicted_company = le_target.inverse_transform(prediction_encoded)[0]
                
                st.subheader("Kết quả Dự đoán:")
                st.success(f"Hãng xe được dự đoán là: **{predicted_company}**")

                st.markdown("---")
                st.subheader("Xác suất Dự đoán cho từng Hãng:")
                proba_df = pd.DataFrame(prediction_proba, columns=le_target.classes_)
                proba_df_transposed = proba_df.T.rename(columns={0: 'Xác suất'})
                proba_df_transposed = proba_df_transposed.sort_values(by='Xác suất', ascending=False)
                
                top_n = 5
                st.dataframe(proba_df_transposed.head(top_n).style.format("{:.2%}"))
                st.bar_chart(proba_df_transposed.head(top_n))

            except Exception as e:
                st.error(f"Đã xảy ra lỗi trong quá trình dự đoán: {e}")
                st.error("Vui lòng kiểm tra lại các giá trị đầu vào hoặc dữ liệu huấn luyện.")

    st.markdown("---")
    st.markdown("*Lưu ý: Đây là mô hình dự đoán dựa trên dữ liệu đã cung cấp và có thể không hoàn toàn chính xác trong mọi trường hợp.*")

else:
    st.error("Không thể tải và huấn luyện mô hình. Vui lòng kiểm tra thông báo lỗi.")