import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os

# Page Config
st.set_page_config(
    page_title="Student Credit & Success Agent",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Feel
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main {
        background-color: #F9FAFB;
    }
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        color: #111827;
    }
    /* KPI Card Styling */
    div[data-testid="stMetric"], div[data-testid="metric-container"] {
        background-color: #FFFFFF !important;
        padding: 15px 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #E5E7EB;
        color: #1F2937;
    }
    div[data-testid="stMetric"] label {
        color: #6B7280 !important; /* Muted label color */
        font-weight: 500;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #111827 !important; /* Dark value color */
        font-weight: 700;
        font-size: 2rem;
    }
    
    div[data-testid="stSidebar"] {
        background-color: #1F2937;
    }
</style>
""", unsafe_allow_html=True)

# Helper: Load Resources
@st.cache_resource
def load_resources():
    try:
        import os
        print(f"DEBUG: Current CWD: {os.getcwd()}")
        if not os.path.exists("student_credit_model.pkl"):
            print("DEBUG: student_credit_model.pkl MISSING")
            return None, None, None, "Missing student_credit_model.pkl"
        
        model = joblib.load("student_credit_model.pkl")
        history = joblib.load("full_history_processed.pkl")
        latest_state = joblib.load("latest_student_state.pkl")
        return model, history, latest_state, None
    except Exception as e:
        print(f"DEBUG: Error loading resources: {e}")
        return None, None, None, str(e)

model, history, latest_state, err_msg = load_resources()

if model is None:
    st.error(f"⚠️ Model or Data not found. Error detail: {err_msg}")
    st.info(f"Current Directory: {os.getcwd()}")
    st.info("Please ensure 'student_credit_model.pkl' exists in this folder.")
    st.stop()

# Sidebar
with st.sidebar:
    st.title("🎓 Intelligent Agent")
    st.markdown("---")
    page = st.radio("Navigation", ["Data Monitoring", "Recommendation Agent", "Optimization & Insights"])
    st.markdown("---")
    st.info("System Ready. Connected to Student Data.")

# --- MODULE 1: MONITORING ---
if page == "Data Monitoring":
    st.title("📊 Cohort Monitoring Dashboard")
    
    st.markdown("### 🔍 Filters")
    col1, col2 = st.columns(2)
    with col1:
        years = sorted(history["NAM_TUYENSINH"].unique())
        selected_year = st.selectbox("Select Admission Year", ["All"] + list(years))
    
    # Filter Data
    filtered_df = history.copy()
    if selected_year != "All":
        filtered_df = filtered_df[filtered_df["NAM_TUYENSINH"] == selected_year]
    
    # KPIs
    st.markdown("### 📈 Chỉ Số Chính (KPIs)")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Tổng Sinh Viên", f"{filtered_df['MA_SO_SV'].nunique():,}")
    kpi2.metric("Tỷ Lệ Hoàn Thành TB", f"{filtered_df['ratio'].mean():.1%}")
    kpi3.metric("GPA Trung Bình", f"{filtered_df['GPA'].mean():.2f}")
    kpi4.metric("TC Hoàn Thành TB/Kỳ", f"{filtered_df['TC_HOANTHANH'].mean():.1f}")
    
    # Charts - Row 1
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### 🎯 Điểm Đầu Vào vs Tỷ Lệ Hoàn Thành")
        # Scatter plot to show correlation
        fig_scatter = px.scatter(
            filtered_df, 
            x="DIEM_TRUNGTUYEN", 
            y="ratio",
            color="PTXT",
            size="TC_HOANTHANH",
            hover_data=["MA_SO_SV"],
            title="Tác động của Điểm Trúng Tuyển đến Thành Công",
            color_discrete_sequence=px.colors.qualitative.Prism,
            opacity=0.6,
            labels={"DIEM_TRUNGTUYEN": "Điểm Trúng Tuyển", "ratio": "Tỷ Lệ Hoàn Thành", "PTXT": "Phương Thức XT"}
        )
        fig_scatter.update_layout(xaxis_title="Điểm Trúng Tuyển", yaxis_title="Tỷ Lệ Hoàn Thành")
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with c2:
        st.markdown("##### 📊 Hiệu Suất Trung Bình Theo Kỳ")
        x_col = "Term_ID" if "Term_ID" in filtered_df.columns else "term_sem"
        term_trend = filtered_df.groupby(x_col)[["ratio", "TC_HOANTHANH"]].mean().reset_index()
        
        # Dual axis plot
        fig_dual = go.Figure()
        fig_dual.add_trace(go.Bar(
            x=term_trend[x_col].astype(str),
            y=term_trend["TC_HOANTHANH"],
            name="Số TC Hoàn Thành",
            marker_color="#cbd5e1"
        ))
        fig_dual.add_trace(go.Scatter(
            x=term_trend[x_col].astype(str),
            y=term_trend["ratio"],
            name="Tỷ Lệ Hoàn Thành",
            yaxis="y2",
            line=dict(color="#2563eb", width=3)
        ))
        fig_dual.update_layout(
            title="Xu Hướng Qua Các Học Kỳ",
            yaxis=dict(title="Số Tín Chỉ"),
            yaxis2=dict(title="Tỷ Lệ", overlaying="y", side="right", range=[0, 1.1]),
            legend=dict(x=0, y=1.1, orientation="h"),
            xaxis=dict(title="Học Kỳ")
        )
        st.plotly_chart(fig_dual, use_container_width=True)

    # Charts - Row 2
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("##### ⚖️ Khối Lượng Học vs Kết Quả (GPA)")
        # Bin GPA or Scatter
        fig_load_gpa = px.scatter(
            filtered_df,
            x="TC_DANGKY",
            y="GPA",
            color="ratio",
            color_continuous_scale="RdYlGn",
            title="Đăng ký nhiều tín chỉ có ảnh hưởng GPA?",
            opacity=0.5,
            labels={"TC_DANGKY": "Số TC Đăng Ký", "GPA": "Điểm TB Học Kỳ (GPA)", "ratio": "Tỷ Lệ HT"}
        )
        fig_load_gpa.update_layout(xaxis_title="Số Tín Chỉ Đăng Ký", yaxis_title="Điểm TB Học Kỳ (GPA)")
        st.plotly_chart(fig_load_gpa, use_container_width=True)

    with c4:
        st.markdown("##### 🏆 Hiệu Suất Theo Nhóm Ngành")
        prog_perf = filtered_df.groupby("PTXT")["ratio"].agg(["mean", "count"]).reset_index()
        prog_perf = prog_perf[prog_perf["count"] > 10].sort_values("mean", ascending=False).head(10)
        
        fig_bar = px.bar(
            prog_perf,
            x="mean",
            y="PTXT",
            orientation='h',
            title="Top Nhóm/Ngành có Tỷ Lệ Hoàn Thành Cao",
            text_auto='.1%',
            color="mean",
            color_continuous_scale="Viridis",
            labels={"mean": "Tỷ Lệ Hoàn Thành TB", "PTXT": "Nhóm/Ngành"}
        )
        fig_bar.update_layout(xaxis_title="Tỷ Lệ Hoàn Thành TB", yaxis_title="Mã Nhóm/Ngành")
        st.plotly_chart(fig_bar, use_container_width=True)

    # Detailed Table
    st.subheader("🗂️ Student Detail View")
    search_id = st.text_input("Search Student ID", "")
    if search_id:
        student_view = filtered_df[filtered_df["MA_SO_SV"].astype(str).str.contains(search_id)]
    else:
        student_view = filtered_df.head(100)
    
    st.dataframe(student_view[["MA_SO_SV", "HOC_KY", "TC_DANGKY", "TC_HOANTHANH", "ratio", "GPA"]], 
                 use_container_width=True)

# --- MODULE 2: RECOMMENDATION AGENT ---
elif page == "Recommendation Agent":
    st.title("🤖 Trợ Lý Gợi Ý Tín Chỉ")
    st.markdown("Hệ thống AI gợi ý số tín chỉ tối ưu dựa trên năng lực và lịch sử học tập.")
    
    student_id = st.text_input("Nhập Mã Số Sinh Viên (MSSV):", "")
    
    if student_id and student_id in latest_state["MA_SO_SV"].values:
        student_data = latest_state[latest_state["MA_SO_SV"] == student_id].iloc[0]
        
        # Display Context
        with st.expander("Hồ Sơ Sinh Viên", expanded=True):
            cols = st.columns(4)
            cols[0].metric("Điểm Đầu Vào", student_data.get("DIEM_TRUNGTUYEN", 0))
            cols[1].metric("GPA Kỳ Trước", f"{student_data.get('lag1_gpa', 0):.2f}")
            cols[2].metric("Tỷ Lệ HT Kỳ Trước", f"{student_data.get('lag1_ratio', 0):.1%}")
            cols[3].metric("Năm Thứ", int(student_data.get("years_since_admission", 0)) + 1)
        
        # --- DEFINING THE CORE LOGIC FUNCTION (Centralized) ---
        def calculate_adjusted_prediction(row, c, raw_p):
            adj_p = raw_p

            # --- VÙNG BÌNH THƯỜNG (16 - 22 tín) ---
            # Đây là vùng "Sweet Spot" của sinh viên.
            # Trong vùng này, chúng ta tin tưởng model và chỉ áp dụng suy giảm tự nhiên cực nhẹ.
            # Không tác động gì nhiều.
            
            # --- VÙNG DƯỚI ( < 16 tín) --- 
            # Giảm tín -> Tăng dần nhẹ nhàng tỷ lệ đậu (Gradual Boost)
            if c < 16:
                dist = 16 - c
                # Tăng 1.5% cho mỗi tín chỉ giảm đi. Nhẹ nhàng, không gắt.
                adj_p += (dist * 0.015) 
                
                # Sàn an toàn nhẹ (để curve đi lên mượt)
                floor = 0.95 - (c * 0.01) # 10 tín -> min 0.85
                adj_p = max(adj_p, floor)

            # --- VÙNG TRÊN CAO ( > 22 tín) ---
            # Tăng tín -> Giảm dần nhẹ nhàng (Gradual Penalty)
            elif c > 22:
                dist = c - 22
                
                if c <= 26:
                    # Giai đoạn 1: 23-26 tín (Vùng nỗ lực) -> Giảm từ từ
                    # Mỗi tín chỉ giảm 2% khả năng đậu -> Để đỉnh tối ưu có thể rướn lên 23-24 nếu sinh viên giỏi
                    adj_p -= (dist * 0.02)
                
                else:
                    # Giai đoạn 2: > 26 tín (Vùng quá sức/Extreme) -> Giảm MẠNH
                    # Phạt mức 26 tín (4 * 0.02 = 0.08) + Phạt gắt cho phần dôi dư (0.07/tín)
                    base_penalty = (26 - 22) * 0.02 # Phạt của đoạn 22-26
                    extra_penalty = (c - 26) * 0.07 
                    adj_p -= (base_penalty + extra_penalty)

            # --- ĐIỀU CHỈNH THEO GPA (Năng lực cá nhân) ---
            # Chỉ tác động ở vùng cao để phân loại sinh viên
            if c > 20:
                gpa = float(row.get("lag1_gpa", 2.0))
                if gpa < 2.5: # Yếu
                    adj_p -= 0.03 # Giảm đều 3%
                elif gpa > 3.2: # Giỏi
                    adj_p += 0.02 # Hồi phục 2%

            # Floor an toàn cuối cùng
            return max(0.05, min(0.99, adj_p))

        # Simulation
        st.subheader("🔮 Mô Phỏng Kết Quả")
        credits_to_register = st.slider("Số Tín Chỉ Dự Kiến Đăng Ký:", 5, 35, 18)
        
        # Prepare inputs for prediction
        input_row = student_data.copy()
        input_row["TC_DANGKY"] = credits_to_register
        
        # Recalculate features
        c = credits_to_register
        if "lag1_tc" in input_row: input_row["load_stress"] = c / (float(input_row["lag1_tc"]) + 1e-9)
        if "lag1_gpa" in input_row: input_row["gpa_x_tc"] = float(input_row["lag1_gpa"]) * c
        if c <= 12: input_row["tc_bucket"] = 0
        elif c <= 20: input_row["tc_bucket"] = 1
        else: input_row["tc_bucket"] = 2
        
        # raw prediction
        prediction = model.predict(pd.DataFrame([input_row]))[0]
        
        # --- APPLY LOGIC TO SLIDER ---
        predicted_ratio = calculate_adjusted_prediction(input_row, credits_to_register, prediction)
        predicted_credits = predicted_ratio * credits_to_register
        
        # Result Display
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.metric("Dự Đoán Tỷ Lệ Hoàn Thành", f"{predicted_ratio:.1%}", 
                      delta=f"Rủi ro trượt: {(1-predicted_ratio):.1%}", delta_color="inverse")
        with col_res2:
            st.metric("Số Tín Chỉ Hoàn Thành Dự Kiến", f"{predicted_credits:.1f} / {credits_to_register}")
            
        # Recommendation Curve
        st.markdown("### 🚀 AI Khuyến Nghị & Tối Ưu")
        
        sim_credits = list(range(8, 31)) # Mở rộng range
        sim_results = []
        sim_ratios = []

        for sim_c in sim_credits:
            row = input_row.copy()
            row["TC_DANGKY"] = sim_c
            
            # Recalculate features for SIMULATION
            if "lag1_tc" in row: row["load_stress"] = sim_c / (float(row["lag1_tc"]) + 1e-9)
            if "lag1_gpa" in row: row["gpa_x_tc"] = float(row["lag1_gpa"]) * sim_c
            if sim_c <= 12: row["tc_bucket"] = 0
            elif sim_c <= 20: row["tc_bucket"] = 1
            else: row["tc_bucket"] = 2

            raw_p = model.predict(pd.DataFrame([row]))[0]
            
            # --- APPLY LOGIC TO SIMULATION (EXACT SAME FUNCTION) ---
            final_p = calculate_adjusted_prediction(row, sim_c, raw_p)
            
            sim_results.append(final_p * sim_c) # Yield
            sim_ratios.append(final_p)          # Ratio
            
        # Find optimal
        optimal_idx = np.argmax(sim_results)
        optimal_credits = sim_credits[optimal_idx]
        optimal_val = sim_results[optimal_idx]
        
        fig = go.Figure()
        
        # Trục 1: Số tín chỉ hoàn thành (Yield) - Dạng Bar hoặc Area để thấy "Khối lượng"
        fig.add_trace(go.Scatter(
            x=sim_credits, 
            y=sim_results, 
            mode='lines+markers', 
            name='TC Tích Lũy Dự Kiến',
            line=dict(color='#10B981', width=3),
            fill='tozeroy',
            fillcolor='rgba(16, 185, 129, 0.1)'
        ))
        
        # Trục 2: Tỷ lệ hoàn thành (Success Rate) - Để thấy rõ "Rủi ro" khi tăng tín
        fig.add_trace(go.Scatter(
            x=sim_credits, 
            y=sim_ratios, 
            name='Tỷ Lệ Đậu (%)',
            mode='lines',
            yaxis='y2',
            line=dict(color='#EF4444', width=2, dash='dot')
        ))
        
        # Highlight Optimal Point
        fig.add_vline(x=optimal_credits, line_dash="dash", line_color="#059669")
        fig.add_annotation(
            x=optimal_credits, y=optimal_val,
            text=f"Tối Ưu: {optimal_credits} tín",
            showarrow=True,
            arrowhead=1
        )

        fig.update_layout(
            title="⚖️ Cân Bằng: Năng Suất vs. An Toàn",
            xaxis=dict(title="Số Tín Chỉ Đăng Ký"),
            yaxis=dict(
                title=dict(text="TC Tích Lũy (Càng cao càng tốt)", font=dict(color="#10B981"))
            ),
            yaxis2=dict(
                title=dict(text="Tỷ Lệ Đậu (An Toàn)", font=dict(color="#EF4444")),
                overlaying="y",
                side="right",
                range=[0, 1.1]
            ),
            hovermode="x unified",
            legend=dict(x=0, y=1.1, orientation="h")
        )
        st.plotly_chart(fig, use_container_width=True)
        
        if optimal_credits > 22:
             msg = f"Model gợi ý **{optimal_credits} tín chỉ** để tối đa hóa tốc độ ra trường, NHƯNG rủi ro rớt môn khá cao ({1 - sim_ratios[optimal_idx]:.0%}). Cân nhắc giảm xuống 20-22 để an toàn hơn."
             st.warning(msg)
        else:
             st.success(f"💡 **Khuyến Nghị**: Đăng ký **{optimal_credits} tín chỉ** là điểm cân bằng tốt nhất giữa khối lượng học và khả năng qua môn.")
        
    elif student_id:
        st.error(f"Không tìm thấy MSSV '{student_id}' trong dữ liệu.")

# --- MODULE 3: OPTIMIZATION ---
elif page == "Optimization & Insights":
    st.title("⚡ Phân Tích & Tối Ưu Hóa")
    st.markdown("Hiểu rõ các yếu tố ảnh hưởng đến kết quả học tập.")
    
    # Feature Importance (Proxy extraction from pipeline)
    try:
        regressor = model.named_steps["m"]
        if hasattr(regressor, "feature_importances_"):
            importances = regressor.feature_importances_
            
            # Simplified visualization
            feat_fig = px.bar(
                x=range(len(importances)), 
                y=importances, 
                title="Mức Độ Ảnh Hưởng Của Các Yếu Tố (Mô Hình)", 
                labels={'x': "Chỉ số (Features)", 'y': "Độ quan trọng"}
            )
            st.plotly_chart(feat_fig, use_container_width=True)
            st.caption("Lưu ý: Các cột càng cao thể hiện yếu tố đó càng tác động mạnh đến khả năng hoàn thành tín chỉ.")
            
            st.markdown("""
            **🔍 Phân Tích Chuyên Sâu:**
            *   **Kết quả học tập quá khứ (GPA, Tỷ lệ hoàn thành):** Là dự báo chính xác nhất cho kỳ tiếp theo.
            *   **Điểm đầu vào:** Có ảnh hưởng nhưng giảm dần theo năm học.
            *   **Số tín chỉ đăng ký:** Có tác động phi tuyến tính (Đăng ký quá nhiều sẽ làm giảm tỷ lệ hoàn thành mạnh).
            """)
    except:
        st.info("Không thể trích xuất mức độ quan trọng của các yếu tố từ mô hình này.")

    # Educational Content
    st.subheader("📚 Lời Khuyên Cải Thiện")
    st.markdown("""
    1.  **Điều chỉnh vừa sức**: Nếu GPA kỳ trước thấp, hãy giảm bớt 2-3 tín chỉ so với dự định để tập trung cải thiện điểm số.
    2.  **Tránh quá tải**: Tỷ lệ rớt môn thường tăng vọt khi sinh viên đăng ký quá nhiều môn khó cùng lúc.
    3.  **Tìm 'Điểm Rơi Phong Độ'**: Sử dụng tab **Trợ Lý Gợi Ý** để tìm số lượng tín chỉ tối ưu nhất cho riêng bạn.
    """)
