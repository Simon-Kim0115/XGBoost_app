import streamlit as st
import pandas as pd
import numpy as np
import os
from xgboost import XGBRegressor, XGBClassifier
import matplotlib.pyplot as plt

# ------------------------------------------------
# 한글 폰트 설정
# ------------------------------------------------
import platform
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ------------------------------------------------
# 1. 데이터 전처리 및 로드
# ------------------------------------------------
@st.cache_data
def preprocess_and_load_rf_data():

    processed_file = "C:/project/data/Manufacturing_AI_weekly_rf_final.csv" 
    
    if os.path.exists(processed_file):
        weekly_df = pd.read_csv(processed_file)
        weekly_df['order_date'] = pd.to_datetime(weekly_df['order_date'])
    else:
        df = pd.read_csv("Manufacturing_AI_with_product_name.csv", encoding='utf-8', low_memory=False)
        
        df = df.dropna(subset=['제품코드'])
        df = df[df['제품코드'].str.strip() != '']
        df['당월총주문량'] = pd.to_numeric(df['당월총주문량'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
        df['order_date'] = pd.to_datetime(df['order_date'])
        
        df_unique = df.drop_duplicates(subset=['제품코드', 'order_date']).sort_values(by=['제품코드', 'order_date'])
        df_unique['일별주문량'] = df_unique.groupby('제품코드')['당월총주문량'].diff()
        df_unique.loc[df_unique['일별주문량'].isnull() | (df_unique['일별주문량'] < 0), '일별주문량'] = df_unique['당월총주문량']
        
        daily_df = df_unique[['order_date', '제품코드', '일별주문량']].copy()
        weekly_list = []
        for prod_code, group in daily_df.groupby('제품코드'):
            group = group.set_index('order_date')
            weekly_group = group.resample('W')['일별주문량'].sum().reset_index()
            weekly_group['제품코드'] = prod_code
            weekly_list.append(weekly_group)
            
        weekly_df = pd.concat(weekly_list, ignore_index=True)
        weekly_df.rename(columns={'일별주문량': '주간주문량'}, inplace=True)
        
        weekly_df['년'] = weekly_df['order_date'].dt.isocalendar().year.astype(int)
        weekly_df['월'] = weekly_df['order_date'].dt.month
        weekly_df['주차'] = weekly_df['order_date'].dt.isocalendar().week.astype(int)
        
        grouped = weekly_df.groupby('제품코드')['주간주문량']
        weekly_df['전주주문량'] = grouped.transform(lambda x: x.shift(1).fillna(0))
        weekly_df['2주전주문량'] = grouped.transform(lambda x: x.shift(2).fillna(0))
        weekly_df['3주전주문량'] = grouped.transform(lambda x: x.shift(3).fillna(0))
        weekly_df['4주전주문량'] = grouped.transform(lambda x: x.shift(4).fillna(0))
        weekly_df['2주이동평균'] = grouped.transform(lambda x: x.rolling(window=2, min_periods=1).mean())
        weekly_df['4주이동평균'] = grouped.transform(lambda x: x.rolling(window=4, min_periods=1).mean())
        weekly_df['4주이동표준편차'] = grouped.transform(lambda x: x.rolling(window=4, min_periods=1).std().fillna(0))
        
        weekly_df = weekly_df.sort_values(['제품코드', 'order_date']).reset_index(drop=True)
        weekly_df.to_csv(processed_file, index=False, encoding='utf-8-sig')

    cat_cols = ['제품코드']
    num_cols = ['년', '월', '주차', '전주주문량', '2주전주문량', '3주전주문량', '4주전주문량', 
                '2주이동평균', '4주이동평균', '4주이동표준편차']
    target = '주간주문량'
    
    X_encoded = pd.get_dummies(weekly_df[num_cols + cat_cols], columns=cat_cols)
    y = weekly_df[target]
    
    unique_products = sorted(weekly_df['제품코드'].unique().tolist())
    return weekly_df, X_encoded, y, unique_products

# 오차율(WAPE) 계산 함수
def calculate_wape(y_true, y_pred):
    sum_true = np.sum(y_true)
    if sum_true == 0:
        return 0.0
    return (np.sum(np.abs(y_true - y_pred)) / sum_true) * 100

# 월/주차 라벨 생성 함수 (미래 예측용)
def get_month_week_label(dt):
    first_day = dt.replace(day=1)
    dom = dt.day
    adjusted_dom = dom + first_day.weekday()
    week_num = int(np.ceil(adjusted_dom / 7.0))
    return f"{dt.month}월 {week_num}주차"

# ------------------------------------------------
# 2. 대시보드 UI
# ------------------------------------------------
st.set_page_config(page_title="XGBoost 주간 예측", layout="wide")
st.title("🚀 XGBoost 주간 주문 예측 시스템")
st.write("주간 전체 물량 대비 오차 비율(WAPE)을 기반으로 정확도를 직관적으로 평가합니다.")

weekly_df, X, y, unique_products = preprocess_and_load_rf_data()

st.sidebar.header("예측 설정")
test_size = st.sidebar.slider("테스트 데이터 비율 (최근 주차 기준)", 0.1, 0.4, 0.2)
selected_product = st.sidebar.selectbox("상세 분석할 제품 코드", unique_products)

future_weeks = st.sidebar.number_input("📅 미래 예측 주수 입력 (주차)", min_value=1, max_value=52, value=12, step=1)
buffer_rate = st.sidebar.slider("🛡️ 권장 안전재고 버퍼 비율 (%)", min_value=0, max_value=30, value=15)

# 시계열 분할
train_indices, test_indices = [], []
for _, group in weekly_df.groupby('제품코드'):
    idx_list = group.index.tolist()
    split_point = int(len(idx_list) * (1 - test_size))
    if split_point == len(idx_list):
        split_point = max(0, len(idx_list) - 1)
    train_indices.extend(idx_list[:split_point])
    test_indices.extend(idx_list[split_point:])

X_train, X_test = X.loc[train_indices], X.loc[test_indices]
y_train, y_test = y.loc[train_indices], y.loc[test_indices]
train_df = weekly_df.loc[train_indices].copy()
test_df = weekly_df.loc[test_indices].copy()

# ------------------------------------------------
# 3. 모델 학습 (XGBoost)
# ------------------------------------------------
with st.spinner('고도화 XGBoost 모델 학습 진행 중...'):
    X_train_float = X_train.astype(float)
    X_test_float = X_test.astype(float)

    y_train_class = (y_train > 0).astype(int)
    clf = XGBClassifier(
        n_estimators=500, 
        max_depth=6, 
        learning_rate=0.05, 
        random_state=42, 
        n_jobs=-1, 
        eval_metric='logloss')
    clf.fit(X_train_float, y_train_class)
    prob_order = clf.predict_proba(X_test_float)[:, 1]

    mask_positive = y_train > 0
    reg = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1)
    reg.fit(X_train_float[mask_positive], np.log1p(y_train[mask_positive]))
    pred_qty = np.expm1(reg.predict(X_test_float))

    final_predictions = np.where(prob_order >= 0.3, pred_qty, 0)
    final_predictions = np.round(np.clip(final_predictions, a_min=0, a_max=None))

test_df['예측값'] = final_predictions

# ------------------------------------------------
# 4. 직관적 오차율(WAPE) 중심 성과 표시
# ------------------------------------------------
global_wape = calculate_wape(y_test, final_predictions)
global_accuracy = max(0.0, 100.0 - global_wape)

st.subheader("📊 전체 공장 통합 예측 정확도")
col_g1, col_g2 = st.columns(2)
col_g1.metric("전체 평균 주간 오차율 (WAPE)", f"{global_wape:,.2f} %")
col_g2.metric("전체 평균 예측 정확도", f"{global_accuracy:,.2f} %")

st.divider()
st.subheader(f"🎯 선택 제품 [{selected_product}] 오차율 분석")

prod_train = train_df[train_df['제품코드'] == selected_product]
prod_test = test_df[test_df['제품코드'] == selected_product]

if prod_test.empty:
    st.warning("선택한 제품의 테스트 데이터가 없습니다.")
else:
    prod_wape = calculate_wape(prod_test['주간주문량'], prod_test['예측값'])
    prod_accuracy = max(0.0, 100.0 - prod_wape)
    
    col_p1, col_p2 = st.columns(2)
    col_p1.metric("제품 주간 오차율 (WAPE)", f"{prod_wape:,.2f} %")
    col_p2.metric("제품 예측 정확도", f"{prod_accuracy:,.2f} %")

    # 시계열 차트
    fig, ax = plt.subplots(figsize=(15, 6))
    if not prod_train.empty:
        ax.plot(prod_train['order_date'], prod_train['주간주문량'], label='과거 학습 데이터', color='blue', alpha=0.3, marker='s')
    
    ax.plot(prod_test['order_date'], prod_test['주간주문량'], label='실제 주문량', color='green', alpha=0.8, marker='s', markersize=6)
    
    # 차트 라벨 수정
    ax.plot(prod_test['order_date'], prod_test['예측값'], label='XGBoost 예측량', color='darkorange', linestyle='--', linewidth=2, marker='^', markersize=8)

    ax.set_title(f"[{selected_product}] 주간 주문량 추이 (오차율: {prod_wape:.2f}%)", fontsize=15)
    ax.set_xlabel("날짜 (주차)")
    ax.set_ylabel("주간 총 주문량")
    ax.legend()
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig)

# ------------------------------------------------
# 5. 향후 N주(미래) 예측 및 생산 계획 (후처리 스무딩 적용)
# ------------------------------------------------
st.divider()
st.subheader(f"🚀 [{selected_product}] 향후 {future_weeks}주간 예측 및 필요 생산 계획 (추세 안정화)")

prod_full = weekly_df[weekly_df['제품코드'] == selected_product].sort_values('order_date').copy()

if prod_full.empty:
    st.warning("선택한 제품코드의 데이터가 없습니다.")
else:
    last_date = prod_full['order_date'].max()
    future_dates = [last_date + pd.Timedelta(weeks=i) for i in range(1, future_weeks + 1)]
    
    history_demand = list(prod_full['주간주문량'].values)
    future_records = []

    # 1. 기존 학습된 피처 로직 그대로 원본 예측 진행
    for f_date in future_dates:
        prev_1 = history_demand[-1] if len(history_demand) >= 1 else 0
        prev_2 = history_demand[-2] if len(history_demand) >= 2 else 0
        prev_3 = history_demand[-3] if len(history_demand) >= 3 else 0
        prev_4 = history_demand[-4] if len(history_demand) >= 4 else 0
        
        ma_2 = np.mean(history_demand[-2:]) if len(history_demand) >= 2 else np.mean(history_demand)
        ma_4 = np.mean(history_demand[-4:]) if len(history_demand) >= 4 else np.mean(history_demand)
        
        hist_4_slice = history_demand[-4:]
        std_4 = np.std(hist_4_slice, ddof=1) if len(hist_4_slice) > 1 else 0.0
        
        row_dict = {
            '년': [int(f_date.isocalendar().year)],
            '월': [f_date.month],
            '주차': [int(f_date.isocalendar().week)],
            '전주주문량': [prev_1],
            '2주전주문량': [prev_2],
            '3주전주문량': [prev_3],
            '4주전주문량': [prev_4],
            '2주이동평균': [ma_2],
            '4주이동평균': [ma_4],
            '4주이동표준편차': [std_4],
            '제품코드': [selected_product]
        }
        
        df_f = pd.DataFrame(row_dict)
        df_f_encoded = pd.get_dummies(df_f, columns=['제품코드'])
        df_f_encoded = df_f_encoded.reindex(columns=X.columns, fill_value=0)
        df_f_encoded_float = df_f_encoded.astype(float)
        
        p_prob = clf.predict_proba(df_f_encoded_float)[:, 1][0]
        if p_prob >= 0.3:
            p_val = max(0, np.expm1(reg.predict(df_f_encoded_float)[0]))
        else:
            p_val = 0.0
            
        future_records.append({
            '기준_일자': f_date,
            '기준_일자_str': f_date.strftime('%Y-%m-%d'),
            '월_주차': get_month_week_label(f_date),
            'raw_예상_수주량': p_val  # 스무딩 전 원본 예측값 임시 저장
        })
        history_demand.append(p_val)

    future_df = pd.DataFrame(future_records)
    
    # 2. 튀는 현상을 잡기 위해 최종 미래 예측값에만 4주 이동평균(Rolling) 필터 적용
    future_df['예상_수주량'] = future_df['raw_예상_수주량'].rolling(window=4, min_periods=1).mean().round(-2)
    
    future_df['안전재고_권장량'] = (future_df['예상_수주량'] * (buffer_rate / 100)).round(-2)
    future_df['최종_필요_생산량'] = future_df['예상_수주량'] + future_df['안전재고_권장량']

    # 탭 구조 UI (미래 예측 시각화 & 표)
    st.write("")
    tab1, tab2 = st.tabs(["📊 미래 시계열 트렌드 차트", "📋 주별 필요 생산 계획 수량 표"])

    with tab1:
        fig2, ax2 = plt.subplots(figsize=(15, 6))
        
        ax2.plot(prod_full['order_date'], prod_full['주간주문량'], label='과거 전체 실제 주문량', color='#059669', alpha=0.6, marker='s', markersize=4)
        
        last_hist_x = [prod_full['order_date'].iloc[-1]] + list(future_df['기준_일자'])
        last_hist_y = [prod_full['주간주문량'].iloc[-1]] + list(future_df['예상_수주량'])
        
        ax2.plot(last_hist_x, last_hist_y, label=f'향후 {future_weeks}주 미래 예측선 (추세 안정화 적용)', color='#dc2626', linestyle='--', linewidth=2.5, marker='o', markersize=8)

        ax2.set_title(f"[{selected_product}] 미래 {future_weeks}주 주문량 예측 트렌드 (XGBoost)", fontsize=15, pad=12)
        ax2.set_xlabel("날짜 (해당 주차 일요일)")
        ax2.set_ylabel("주간 총 주문량 (EA)")
        ax2.legend()
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig2)

    with tab2:
        st.markdown(f"#### 📋 **향후 {future_weeks}주간 주별 필요 생산 수량 상세 표**")
        
        display_df = pd.DataFrame({
            '기준 일자 (일요일)': future_df['기준_일자_str'],
            '주차 구분': future_df['월_주차'],
            '예상 수주량 (EA)': future_df['예상_수주량'].apply(lambda x: f"{x:,.0f} EA"),
            f'안전재고 ({buffer_rate}%) (EA)': future_df['안전재고_권장량'].apply(lambda x: f"{x:,.0f} EA"),
            '최종 필요 생산 계획 수량 (EA)': future_df['최종_필요_생산량'].apply(lambda x: f"{x:,.0f} EA")
        })
        st.dataframe(display_df, use_container_width=True, height=380)

    # 요약 카드 표시
    total_demand = future_df['예상_수주량'].sum()
    total_buffer = future_df['안전재고_권장량'].sum()
    total_production = future_df['최종_필요_생산량'].sum()

    st.markdown(f"""
    <div style="background-color: #eff6ff; padding: 24px; border-radius: 12px; border: 2px solid #2563eb; margin-top: 20px;">
        <div style="font-size: 20px; font-weight: bold; color: #1e40af; margin-bottom: 8px;">
            💡 [최종 결론] 향후 {future_weeks}주간 총 필요 생산 계획 수량 (XGBoost)
        </div>
        <p style="font-size: 15px; color: #374151; margin-bottom: 12px;">
            선택된 <b>[{selected_product}]</b> 제품에 대한 AI 주별 수주 예측 분석 결과입니다.
        </p>
        <ul style="font-size: 15px; color: #1f2937; line-height: 1.8;">
            <li><b>향후 {future_weeks}주간 예상 총 수주량:</b> <span style="font-weight:bold; color:#2563eb;">{total_demand:,.0f} EA</span></li>
            <li><b>안전재고 권장량 ({buffer_rate}% 버퍼 적용):</b> <span style="font-weight:bold; color:#d97706;">{total_buffer:,.0f} EA</span></li>
        </ul>
        <hr style="border: 0.5px solid #bfdbfe; margin: 12px 0;">
        <div style="font-size: 17px; font-weight: bold; color: #1e3a8a;">
            🚀 <b>최종 권장 생산량 결론:</b> 향후 {future_weeks}주간 총 <span style="font-size: 26px; color: #1d4ed8;">{total_production:,.0f} EA</span>의 수량을 생산 및 준비해야 합니다.
        </div>
    </div>
    """, unsafe_allow_html=True)
