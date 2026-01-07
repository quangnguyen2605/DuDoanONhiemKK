import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from data_loader import load_real_aqi_data
from data_generator import generate_aqi_data
from data_preprocessing import DataPreprocessor
from models import ModelTrainer
from evaluation import ModelEvaluator
from visualization import create_visualizations

# Page configuration
st.set_page_config(
    page_title="Hệ Thống Dự Đoán Ô Nhiễm Không Khí Hà Nội",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">🌫️ Hệ Thống Dự Đoán Ô Nhiễm Không Khí Hà Nội</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 1rem; color: #ffffff; 
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4); 
                border: 2px solid #667eea; margin-bottom: 2rem;">
        <h3 style="color: #ffffff; margin-bottom: 1rem; font-size: 1.4rem; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">
            📋 Tổng quan dự án
        </h3>
        <p style="color: #ffffff; margin: 0; font-size: 1.1rem; line-height: 1.6;">
            Ứng dụng demo này triển khai các thuật toán học máy để dự đoán chỉ số chất lượng không khí (AQI) và phân loại mức độ ô nhiễm tại Hà Nội.
        </p>
        <div style="margin-top: 1rem; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 0.8rem; border-left: 4px solid #f093fb;">
            <p style="color: #ffffff; margin: 0; font-size: 1rem; font-weight: bold;">
                🤖 Hệ thống so sánh 4 thuật toán:
            </p>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.5rem; margin-top: 0.5rem;">
                <div style="background: rgba(255,255,255,0.15); padding: 0.5rem; border-radius: 0.5rem; text-align: center;">
                    📈 Hồi quy tuyến tính
                </div>
                <div style="background: rgba(255,255,255,0.15); padding: 0.5rem; border-radius: 0.5rem; text-align: center;">
                    🌳 Cây quyết định (CART)
                </div>
                <div style="background: rgba(255,255,255,0.15); padding: 0.5rem; border-radius: 0.5rem; text-align: center;">
                    🎯 SVM
                </div>
                <div style="background: rgba(255,255,255,0.15); padding: 0.5rem; border-radius: 0.5rem; text-align: center;">
                    📊 Hồi quy logistic
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for navigation
    st.sidebar.title("Điều Hướng")
    
    # Add reload data button
    if st.sidebar.button("🔄 Tải Lại Dữ Liệu", type="secondary"):
        if 'data' in st.session_state:
            del st.session_state.data
        if 'data_source' in st.session_state:
            del st.session_state.data_source
        if 'search_applied' in st.session_state:
            del st.session_state.search_applied
        st.rerun()
    
    page = st.sidebar.selectbox("Chọn phần:", [
        "🏠 Dashboard Chính",
        "🔍 Tìm Kiếm Theo Thời Gian",
        "Tiền Xử Lý Dữ Liệu", 
        "Huấn Luyện Mô Hình",
        "Đánh Giá & So Sánh Mô Hình",
        "Dự Đoán Thời Gian Thực",
        "Kết Luận & Khuyến Nghị"
    ])
    
    # Generate or load data
    if 'data' not in st.session_state:
        with st.spinner("Đang tải dữ liệu AQI Hà Nội..."):
            # Try to load real data first
            real_data = load_real_aqi_data()
            if real_data is not None:
                st.session_state.data = real_data
                st.session_state.data_source = "Dữ liệu thật"
                st.success("✅ Đã tải dữ liệu AQI thật thành công!")
            else:
                # Fallback to synthetic data
                st.session_state.data = generate_aqi_data()
                st.session_state.data_source = "Dữ liệu giả lập"
                st.success("✅ Đã tạo dữ liệu giả lập thành công!")
    
    data = st.session_state.data
    data_source = st.session_state.get('data_source', 'Unknown')
    
    if page == "🏠 Dashboard Chính":
        main_dashboard_page(data)
    
    elif page == "🔍 Tìm Kiếm Theo Thời Gian":
        recent_data_page(data)
    
    elif page == "Tiền Xử Lý Dữ Liệu":
        preprocessing_page(data)
    
    elif page == "Huấn Luyện Mô Hình":
        model_training_page(data)
    
    elif page == "Đánh Giá & So Sánh Mô Hình":
        evaluation_page(data)
    
    elif page == "Dự Đoán Thời Gian Thực":
        prediction_page(data)
    
    elif page == "Kết Luận & Khuyến Nghị":
        conclusions_page()

def main_dashboard_page(data):
    st.markdown('<h2 class="sub-header">🏠 Dashboard Tổng Quan Hệ Thống</h2>', unsafe_allow_html=True)
    
    # Show data source
    data_source = st.session_state.get('data_source', 'Unknown')
    st.info(f"📂 Nguồn dữ liệu: {data_source}")
    
    # Key Metrics Dashboard
    st.markdown("### 📊 Chỉ Số Hiệu Suất Hệ Thống")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📈 Tổng Dữ Liệu", f"{len(data):,}", help="Tổng số bản ghi trong cơ sở dữ liệu")
    
    with col2:
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])
            date_range = f"{data['Date'].dt.year.min()}-{data['Date'].dt.year.max()}"
        else:
            date_range = "Không xác định"
        st.metric("📅 Khoảng Thời Gian", date_range, help="Phạm vi dữ liệu theo năm")
    
    with col3:
        avg_aqi = data['AQI'].mean()
        st.metric("🌫️ AQI Trung Bình", f"{avg_aqi:.1f}", help="Chỉ số chất lượng không khí trung bình")
    
    with col4:
        if 'training_results' in st.session_state:
            results = st.session_state.training_results
            total_models = len(results.get('regression', {})) + len(results.get('classification', {}))
            st.metric("🤖 Mô Hình Đã Huấn Luyện", f"{total_models}/4", help="Số mô hình đã huấn luyện")
        else:
            st.metric("🤖 Mô Hình Đã Huấn Luyện", "0/4", help="Chưa huấn luyện mô hình nào")
    
    # Current Status
    st.markdown("### 🎯 Trạng Thái Hiện Tại")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Chất Lượng Không Khí Hiện Tại")
        
        # Get most recent data point
        if 'Date' in data.columns:
            latest_data = data.sort_values('Date').iloc[-1]
            latest_aqi = latest_data['AQI']
            latest_level = latest_data['Pollution_Level']
            latest_time = latest_data['Date']
            
            # AQI level indicator
            if latest_aqi <= 50:
                level_color = "🟢"
                level_text = "Tốt"
                color_code = "green"
            elif latest_aqi <= 100:
                level_color = "🟡"
                level_text = "Trung Bình"
                color_code = "orange"
            elif latest_aqi <= 150:
                level_color = "🟠"
                level_text = "Kém"
                color_code = "red"
            elif latest_aqi <= 200:
                level_color = "🔴"
                level_text = "Xấu"
                color_code = "darkred"
            elif latest_aqi <= 300:
                level_color = "🟣"
                level_text = "Rất Xấu"
                color_code = "purple"
            else:
                level_color = "⚫"
                level_text = "Nguy Hiểm"
                color_code = "black"
            
            st.markdown(f"""
            <div style="padding: 1rem; border-radius: 10px; background-color: #f0f2f6; text-align: center;">
                <h3 style="margin: 0;">{level_color} {level_text}</h3>
                <h2 style="margin: 0; color: {color_code};">{latest_aqi:.1f}</h2>
                <p style="margin: 0; color: #666;">AQI</p>
                <p style="margin: 0; color: #666;">{latest_time.strftime('%d/%m/%Y %H:%M')}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Health advice
            health_advice = get_health_advice(latest_level)
            st.info(f"💡 **Khuyến Nghị Sức Khỏe:** {health_advice}")
        
        else:
            st.warning("Không có dữ liệu thời gian để hiển thị trạng thái hiện tại")
    
    with col2:
        st.markdown("#### 🤖 Trạng Thái Huấn Luyện Mô Hình")
        
        if 'training_results' in st.session_state:
            results = st.session_state.training_results
            
            st.markdown("**Mô Hình Hồi Quy:**")
            if results.get('regression'):
                for model_name, metrics in results['regression'].items():
                    st.write(f"✅ {model_name}: R² = {metrics['r2']:.3f}")
            else:
                st.write("❌ Chưa huấn luyện")
            
            st.markdown("**Mô Hình Phân Loại:**")
            if results.get('classification'):
                for model_name, metrics in results['classification'].items():
                    st.write(f"✅ {model_name}: F1 = {metrics['f1']:.3f}")
            else:
                st.write("❌ Chưa huấn luyện")
            
            # Best model recommendation
            st.markdown("**🏆 Mô Hình Tốt Nhất:**")
            best_reg = None
            best_clf = None
            
            if results.get('regression'):
                best_reg = max(results['regression'].items(), key=lambda x: x[1]['r2'])
                st.write(f"📈 Hồi Quy: {best_reg[0]} (R² = {best_reg[1]['r2']:.3f})")
            
            if results.get('classification'):
                best_clf = max(results['classification'].items(), key=lambda x: x[1]['f1'])
                st.write(f"🎯 Phân Loại: {best_clf[0]} (F1 = {best_clf[1]['f1']:.3f})")
        else:
            st.warning("🚨 Chưa huấn luyện mô hình nào")
            st.info("Vào mục 'Huấn Luyện Mô Hình' để bắt đầu")
    
    # Quick Actions
    st.markdown("### 🚀 Hành Động Nhanh")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔮 Dự Đoán Nhanh", use_container_width=True):
            if 'training_results' in st.session_state:
                st.success("✅ Chuyển đến trang dự đoán...")
                # In a real app, you'd use st.session_state.page or similar
                st.info("Vui lòng chọn 'Dự Đoán Thời Gian Thực' từ menu điều hướng")
            else:
                st.error("❌ Vui lòng huấn luyện mô hình trước!")
    
    with col2:
        if st.button("📊 Xem Chi Tiết", use_container_width=True):
            st.info("Vui lòng chọn 'Dữ Liệu 7 Ngày Gần Đây' từ menu điều hướng")
    
    with col3:
        if st.button("🤖 Huấn Luyện Mô Hình", use_container_width=True):
            st.info("Vui lòng chọn 'Huấn Luyện Mô Hình' từ menu điều hướng")
    
    # Recent Activity Summary
    st.markdown("### 📈 Tổng Kết Hoạt Động Gần Đây")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🌫️ Chất Lượng Không Khí 7 Ngày Qua")
        
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])
            latest_date = data['Date'].max()
            seven_days_ago = latest_date - pd.Timedelta(days=7)
            recent_data = data[data['Date'] >= seven_days_ago]
            
            # Calculate statistics
            avg_aqi_7d = recent_data['AQI'].mean()
            max_aqi_7d = recent_data['AQI'].max()
            min_aqi_7d = recent_data['AQI'].min()
            
            # Pollution level distribution
            level_counts = recent_data['Pollution_Level'].value_counts()
            most_common_level = level_counts.index[0]
            most_common_count = level_counts.iloc[0]
            
            st.write(f"- **AQI Trung Bình:** {avg_aqi_7d:.1f}")
            st.write(f"- **AQI Cao Nhất:** {max_aqi_7d:.1f}")
            st.write(f"- **AQI Thấp Nhất:** {min_aqi_7d:.1f}")
            st.write(f"- **Mức Độ Phổ Biến Nhất:** {most_common_level} ({most_common_count} lần)")
            
            # Mini chart
            fig = px.line(recent_data.tail(50), x='Date', y='AQI', 
                          title='AQI 7 Ngày Gần Đây', height=200)
            fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Không có dữ liệu thời gian")
    
    with col2:
        st.markdown("#### 📊 Phân Phối Mức Độ Ô Nhiễm")
        
        # Overall pollution level distribution
        level_counts = data['Pollution_Level'].value_counts()
        total_records = len(data)
        
        fig = px.pie(values=level_counts.values, names=level_counts.index, 
                    title='Phân Bố Mức Độ Ô Nhiễm Tổng Thể', height=300)
        fig.update_layout(showlegend=True, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("**Thống Kê Chi Tiết:**")
        for level, count in level_counts.items():
            percentage = (count / total_records) * 100
            st.write(f"- **{level}:** {count:,} bản ghi ({percentage:.1f}%)")
    
    # System Information
    st.markdown("### ℹ️ Thông Tin Hệ Thống")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📊 Dữ Liệu:**")
        st.write(f"- Nguồn: {data_source}")
        st.write(f"- Kích thước: {len(data):,} bản ghi")
        st.write(f"- Đặc trưng: {len(data.columns)} cột")
    
    with col2:
        st.markdown("**🤖 Mô Hình:**")
        st.write(f"- Hồi Quy: Linear Regression, Decision Tree")
        st.write(f"- Phân Loại: Logistic Regression, SVM")
        st.write(f"- Tổng cộng: 4 thuật toán")
    
    with col3:
        st.markdown("**🔧 Công Nghệ:**")
        st.write(f"- Framework: Streamlit")
        st.write(f"- ML Library: Scikit-learn")
        st.write(f"- Visualization: Plotly")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🌫️ <strong>Hệ Thống Dự Đoán Ô Nhiễm Không Khí Hà Nội</strong></p>
        <p>Phát triển với ❤️ bằng Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

def recent_data_page(data):
    st.markdown('<h2 class="sub-header">📊 Tìm Kiếm & Phân Tích Dữ Liệu Theo Thời Gian</h2>', unsafe_allow_html=True)
    
    # Show data source
    data_source = st.session_state.get('data_source', 'Unknown')
    st.info(f"📂 Nguồn dữ liệu: {data_source}")
    
    # Check if data has Date column
    if 'Date' not in data.columns:
        st.error("❌ Dữ liệu không có cột Date để tìm kiếm theo thời gian")
        st.info("Vui lòng đảm bảo file dữ liệu có cột 'Date' với định dạng datetime")
        return
    
    # Convert Date column
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Search options
    st.subheader("🔍 Tùy Chọn Tìm Kiếm")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Quick preset options
        st.markdown("**🚀 Tùy Chọn Nhanh:**")
        preset_option = st.selectbox(
            "Chọn khoảng thời gian:",
            ["7 Ngày Gần Đây", "30 Ngày Gần Đây", "Tháng Này", "Tháng Trước", "Năm Này", "Tùy Chọn"],
            index=0
        )
    
    with col2:
        st.markdown("**📅 Tùy Chọn Theo Ngày:**")
        if preset_option == "Tùy Chọn":
            start_date = st.date_input("Từ ngày:", data['Date'].min().date())
            end_date = st.date_input("Đến ngày:", data['Date'].max().date())
        else:
            start_date = None
            end_date = None
    
    with col3:
        st.markdown("**⏰ Tùy Chọn Theo Giờ:**")
        enable_hour_filter = st.checkbox("Lọc theo giờ", value=False)
        if enable_hour_filter:
            start_hour = st.slider("Giờ bắt đầu:", 0, 23, 0)
            end_hour = st.slider("Giờ kết thúc:", 0, 23, 23)
        else:
            start_hour = None
            end_hour = None
    
    # Add search button
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        search_button = st.button("🔍 Tìm Kiếm", type="primary")
    with col2:
        if st.button("🔄 Reset"):
            # Reset to default values
            st.session_state.search_applied = False
            st.rerun()
    with col3:
        st.write("")  # Empty column for spacing
    
    # Only apply filters when search button is clicked
    if search_button or 'search_applied' not in st.session_state:
        # Apply filters based on selection
        if preset_option == "7 Ngày Gần Đây":
            latest_date = data['Date'].max()
            seven_days_ago = latest_date - pd.Timedelta(days=7)
            filtered_data = data[data['Date'] >= seven_days_ago].copy()
            title_period = f"7 Ngày Gần Đây ({seven_days_ago.strftime('%Y-%m-%d')} đến {latest_date.strftime('%Y-%m-%d')})"
            
        elif preset_option == "30 Ngày Gần Đây":
            latest_date = data['Date'].max()
            thirty_days_ago = latest_date - pd.Timedelta(days=30)
            filtered_data = data[data['Date'] >= thirty_days_ago].copy()
            title_period = f"30 Ngày Gần Đây ({thirty_days_ago.strftime('%Y-%m-%d')} đến {latest_date.strftime('%Y-%m-%d')})"
            
        elif preset_option == "Tháng Này":
            current_date = pd.Timestamp.now()
            month_start = current_date.replace(day=1)
            filtered_data = data[data['Date'] >= month_start].copy()
            title_period = f"Tháng Này ({month_start.strftime('%Y-%m-%d')} đến {current_date.strftime('%Y-%m-%d')})"
            
        elif preset_option == "Tháng Trước":
            current_date = pd.Timestamp.now()
            if current_date.month == 1:
                prev_month = current_date.replace(year=current_date.year-1, month=12, day=1)
            else:
                prev_month = current_date.replace(month=current_date.month-1, day=1)
            
            if prev_month.month == 12:
                next_month = prev_month.replace(year=prev_month.year+1, month=1)
            else:
                next_month = prev_month.replace(month=prev_month.month+1)
            
            month_end = next_month - pd.Timedelta(days=1)
            filtered_data = data[(data['Date'] >= prev_month) & (data['Date'] <= month_end)].copy()
            title_period = f"Tháng Trước ({prev_month.strftime('%Y-%m-%d')} đến {month_end.strftime('%Y-%m-%d')})"
            
        elif preset_option == "Năm Này":
            current_year = pd.Timestamp.now().year
            year_start = pd.Timestamp(f"{current_year}-01-01")
            filtered_data = data[data['Date'] >= year_start].copy()
            title_period = f"Năm Này ({year_start.strftime('%Y-%m-%d')} đến {pd.Timestamp.now().strftime('%Y-%m-%d')})"
            
        elif preset_option == "Tùy Chọn":
            if start_date and end_date:
                start_datetime = pd.Timestamp.combine(start_date, pd.Timestamp.min.time())
                end_datetime = pd.Timestamp.combine(end_date, pd.Timestamp.max.time())
                filtered_data = data[(data['Date'] >= start_datetime) & (data['Date'] <= end_datetime)].copy()
                title_period = f"Tùy Chọn ({start_date.strftime('%Y-%m-%d')} đến {end_date.strftime('%Y-%m-%d')})"
            else:
                filtered_data = data.copy()
                title_period = "Toàn Bộ Dữ Liệu"
        else:
            filtered_data = data.copy()
            title_period = "Toàn Bộ Dữ Liệu"
        
        # Apply hour filter if enabled
        if enable_hour_filter and start_hour is not None and end_hour is not None:
            if start_hour <= end_hour:
                hour_mask = (data['Date'].dt.hour >= start_hour) & (data['Date'].dt.hour <= end_hour)
            else:
                # Handle case where time spans midnight (e.g., 22:00 to 06:00)
                hour_mask = (data['Date'].dt.hour >= start_hour) | (data['Date'].dt.hour <= end_hour)
            
            filtered_data = filtered_data[hour_mask]
            title_period += f" (Giờ: {start_hour}:00 - {end_hour}:00)"
        
        # Sort by date
        filtered_data = filtered_data.sort_values('Date')
        
        # Set export dates for filename generation
        if preset_option == "Tùy Chọn" and start_date and end_date:
            export_date_start = start_date
            export_date_end = end_date
        else:
            export_date_start = filtered_data['Date'].min().date()
            export_date_end = filtered_data['Date'].max().date()
        
        # Store in session state
        st.session_state.filtered_data = filtered_data
        st.session_state.title_period = title_period
        st.session_state.export_date_start = export_date_start
        st.session_state.export_date_end = export_date_end
        st.session_state.search_applied = True
    else:
        # Use stored data from session state
        filtered_data = st.session_state.get('filtered_data', data.copy().sort_values('Date'))
        title_period = st.session_state.get('title_period', "Toàn Bộ Dữ Liệu")
        export_date_start = st.session_state.get('export_date_start', data['Date'].min().date())
        export_date_end = st.session_state.get('export_date_end', data['Date'].max().date())
    
    # Display results
    st.markdown(f"### 📅 Phạm Vi Thời Gian: {title_period}")
    st.markdown(f"**Tổng số bản ghi:** {len(filtered_data):,}")
    
    # Always show data table first
    st.subheader("📋 Bảng Dữ Liệu")
    
    # Add export option
    col1, col2 = st.columns([4, 1])
    with col1:
        if len(filtered_data) > 0:
            st.info(f"Hiển thị tất cả {len(filtered_data):,} bản ghi tìm kiếm được")
        else:
            st.info("Không có dữ liệu để hiển thị")
    with col2:
        if len(filtered_data) > 0 and st.button("📥 Export CSV"):
            csv_data = filtered_data.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"aqi_data_{export_date_start.strftime('%Y%m%d')}_to_{export_date_end.strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    # Always show the table (even if empty)
    if len(filtered_data) > 0:
        st.dataframe(filtered_data.reset_index(drop=True), use_container_width=True)
    else:
        st.warning("⚠️ Không có dữ liệu trong khoảng thời gian đã chọn")
        st.info("💡 Thử chọn khoảng thời gian khác hoặc nhấn Reset để quay về mặc định")
    
    # Only show statistics and charts if there's data
    if len(filtered_data) > 0:
        # Data overview with filtered data
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Bản Ghi", f"{len(filtered_data):,}")
        with col2:
            st.metric("AQI Trung Bình", f"{filtered_data['AQI'].mean():.1f}")
        with col3:
            st.metric("AQI Cao Nhất", f"{filtered_data['AQI'].max():.1f}")
        with col4:
            st.metric("AQI Thấp Nhất", f"{filtered_data['AQI'].min():.1f}")
        
        # Search statistics
        st.subheader("📊 Thống Kê Tìm Kiếm")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Date range info
            min_date = filtered_data['Date'].min()
            max_date = filtered_data['Date'].max()
            days_span = (max_date - min_date).days + 1
            
            st.markdown("**📈 Thông Tin Khoảng Thời Gian:**")
            st.write(f"- **Từ:** {min_date.strftime('%d/%m/%Y %H:%M')}")
            st.write(f"- **Đến:** {max_date.strftime('%d/%m/%Y %H:%M')}")
            st.write(f"- **Số ngày:** {days_span}")
            st.write(f"- **Mật độ dữ liệu:** {len(filtered_data)/days_span:.1f} bản ghi/ngày")
        
        with col2:
            # Pollution level distribution
            pollution_levels = filtered_data['Pollution_Level'].value_counts()
            most_common_level = pollution_levels.index[0]
            most_common_count = pollution_levels.iloc[0]
            
            st.markdown("**🌫️ Phân Bổ Mức Độ Ô Nhiễm:**")
            st.write(f"- **Phổ biến nhất:** {most_common_level} ({most_common_count} lần)")
            st.write(f"- **Số mức độ:** {len(pollution_levels)}")
            
            # Calculate percentage
            total_records = len(filtered_data)
            for level, count in pollution_levels.head(3).items():
                percentage = (count / total_records) * 100
                st.write(f"- **{level}:** {count:,} ({percentage:.1f}%)")
        
        # Visualizations for filtered data
        st.subheader("📊 Phân Tích Dữ Liệu Theo Thời Gian")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # AQI trend over selected period
            fig = px.line(filtered_data, x='Date', y='AQI', 
                          title=f'AQI - {title_period}',
                          markers=True)
            fig.add_hline(y=100, line_dash="dash", line_color="orange", 
                         annotation_text="Mức Trung Bình")
            fig.add_hline(y=150, line_dash="dash", line_color="red", 
                         annotation_text="Mức Không Lành Mạnh")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Pollution level distribution
            pollution_levels = filtered_data['Pollution_Level'].value_counts()
            fig = px.pie(values=pollution_levels.values, names=pollution_levels.index, 
                        title=f'Phân Bổ Mức Độ Ô Nhiễm - {title_period}')
            st.plotly_chart(fig, use_container_width=True)
        
        # Additional analysis if data is sufficient
        if len(filtered_data) >= 24:  # At least 24 hours of data
            st.subheader("🕐️ Phân Tích Theo Giờ")
            
            # Hourly patterns
            filtered_data['Hour'] = filtered_data['Date'].dt.hour
            hourly_avg = filtered_data.groupby('Hour')['AQI'].mean()
            
            fig = px.bar(x=hourly_avg.index, y=hourly_avg.values, 
                         title=f'AQI Trung Bình Theo Giờ - {title_period}',
                         labels={'x': 'Giờ', 'y': 'AQI'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show hourly statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Giờ Ô Nhiễm Cao Nhất", 
                        f"{hourly_avg.idxmax()}h",
                        f"AQI: {hourly_avg.max():.1f}")
            with col2:
                st.metric("Giờ Ô Nhiễm Thấp Nhất", 
                        f"{hourly_avg.idxmin()}h",
                        f"AQI: {hourly_avg.min():.1f}")
            with col3:
                st.metric("Biến Động Giờ", 
                        f"{hourly_avg.std():.1f}",
                        "Độ lệch chuẩn")
        
        # Alert for high pollution
        max_aqi = filtered_data['AQI'].max()
        if max_aqi > 150:
            st.error(f"⚠️ Cảnh Báo: AQI cao nhất trong khoảng thời gian là **{max_aqi:.1f}** - Mức độ không lành mạnh!")
        elif max_aqi > 100:
            st.warning(f"⚠️ AQI cao nhất trong khoảng thời gian là **{max_aqi:.1f}** - Cần lưu ý sức khỏe!")
        else:
            st.success(f"✅ Chất lượng không khí trong khoảng thời gian khá tốt!")
        
        # Advanced statistics
        st.subheader("📈 Thống Kê Nâng Cao")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔬 Thống Kê Chi Tiết:**")
            st.write(f"- **Trung vị AQI:** {filtered_data['AQI'].median():.1f}")
            st.write(f"- **Độ lệch chuẩn:** {filtered_data['AQI'].std():.1f}")
            st.write(f"- **Phân vị 25%:** {filtered_data['AQI'].quantile(0.25):.1f}")
            st.write(f"- **Phân vị 75%:** {filtered_data['AQI'].quantile(0.75):.1f}")
            
            # Trend analysis
            if len(filtered_data) > 1:
                first_half = filtered_data[:len(filtered_data)//2]
                second_half = filtered_data[len(filtered_data)//2:]
                trend = second_half['AQI'].mean() - first_half['AQI'].mean()
                trend_direction = "📈 Tăng" if trend > 0 else "📉 Giảm" if trend < 0 else "➡️ Ổn định"
                st.write(f"- **Xu hướng:** {trend_direction} ({abs(trend):.1f})")
        
        with col2:
            st.markdown("**🎯 Phân Tích Mức Độ:**")
            
            # Calculate percentages for each level
            level_stats = []
            for level in ['Tốt', 'Trung Bình', 'Kém', 'Xấu', 'Rất Xấu', 'Nguy Hiểm']:
                count = len(filtered_data[filtered_data['Pollution_Level'] == level])
                percentage = (count / len(filtered_data)) * 100 if len(filtered_data) > 0 else 0
                if count > 0:
                    level_stats.append(f"- **{level}:** {count:,} ({percentage:.1f}%)")
            
            for stat in level_stats:
                st.write(stat)
            
            # Health impact summary
            unhealthy_count = len(filtered_data[filtered_data['AQI'] > 100])
            unhealthy_percentage = (unhealthy_count / len(filtered_data)) * 100
            st.write(f"- **Thời gian không lành mạnh:** {unhealthy_percentage:.1f}%")
        
        # Correlation analysis for the selected period
        if len(filtered_data) > 10:
            st.subheader("🔗 Phân Tích Tương Quan")
            
            # Select numeric columns for correlation
            numeric_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 'Humidity', 'Wind_Speed', 'AQI']
            available_cols = [col for col in numeric_cols if col in filtered_data.columns]
            
            if len(available_cols) > 1:
                correlation_matrix = filtered_data[available_cols].corr()
                
                # Create heatmap
                fig = px.imshow(correlation_matrix, 
                              text_auto=True, 
                              aspect="auto",
                              title=f"Ma Trận Tương Quan - {title_period}")
                st.plotly_chart(fig, use_container_width=True)
                
                # Show top correlations with AQI
                if 'AQI' in correlation_matrix.columns:
                    aqi_correlations = correlation_matrix['AQI'].sort_values(ascending=False)
                    st.markdown("**🎯 Tương Quan Với AQI:**")
                    for col, corr in aqi_correlations.items():
                        if col != 'AQI' and abs(corr) > 0.1:
                            st.write(f"- **{col}:** {corr:.3f}")
        
        # Export functionality
        st.subheader("📤 Xuất Dữ Liệu")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Export Full CSV"):
                csv_data = filtered_data.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"aqi_full_{export_date_start.strftime('%Y%m%d')}_to_{export_date_end.strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📈 Export Summary"):
                summary_data = {
                    'Metric': ['Total Records', 'Average AQI', 'Max AQI', 'Min AQI', 'Median AQI', 'Std Dev'],
                    'Value': [len(filtered_data), filtered_data['AQI'].mean(), filtered_data['AQI'].max(), 
                             filtered_data['AQI'].min(), filtered_data['AQI'].median(), filtered_data['AQI'].std()]
                }
                summary_df = pd.DataFrame(summary_data)
                csv_summary = summary_df.to_csv(index=False)
                st.download_button(
                    label="Download Summary",
                    data=csv_summary,
                    file_name=f"aqi_summary_{export_date_start.strftime('%Y%m%d')}_to_{export_date_end.strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        with col3:
            if st.button("📋 Export Statistics"):
                stats_text = f"""
                AQI Analysis Report - {title_period}
                
                Total Records: {len(filtered_data):,}
                Date Range: {filtered_data['Date'].min().strftime('%Y-%m-%d %H:%M')} to {filtered_data['Date'].max().strftime('%Y-%m-%d %H:%M')}
                
                AQI Statistics:
                - Average: {filtered_data['AQI'].mean():.2f}
                - Maximum: {filtered_data['AQI'].max():.2f}
                - Minimum: {filtered_data['AQI'].min():.2f}
                - Median: {filtered_data['AQI'].median():.2f}
                - Standard Deviation: {filtered_data['AQI'].std():.2f}
                
                Pollution Level Distribution:
                """
                for level, count in filtered_data['Pollution_Level'].value_counts().items():
                    percentage = (count / len(filtered_data)) * 100
                    stats_text += f"\n                - {level}: {count:,} ({percentage:.1f}%)"
                
                st.download_button(
                    label="Download Report",
                    data=stats_text,
                    file_name=f"aqi_report_{export_date_start.strftime('%Y%m%d')}_to_{export_date_end.strftime('%Y%m%d')}.txt",
                    mime="text/plain"
                )

def preprocessing_page(data):
    st.markdown('<h2 class="sub-header">🔧 Tiền Xử Lý Dữ Liệu</h2>', unsafe_allow_html=True)
    
    if 'preprocessor' not in st.session_state:
        st.session_state.preprocessor = DataPreprocessor()
    
    preprocessor = st.session_state.preprocessor
    
    # Show preprocessing steps
    st.subheader("📋 Quy Trình Tiền Xử Lý")
    
    steps = [
        "1. Xử lý giá trị thiếu",
        "2. Loại bỏ ngoại lệ bằng phương pháp IQR",
        "3. Chuẩn hóa đặc trưng (StandardScaler)",
        "4. Mã hóa biến phân loại",
        "5. Kỹ thuật đặc trưng (tạo biến tương tác)"
    ]
    
    for step in steps:
        st.markdown(f"- {step}")
    
    # Apply preprocessing
    if st.button("Áp Dụng Tiền Xử Lý"):
        with st.spinner("Đang tiền xử lý dữ liệu..."):
            X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf = preprocessor.fit_transform(data)
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train_reg = y_train_reg
            st.session_state.y_test_reg = y_test_reg
            st.session_state.y_train_clf = y_train_clf
            st.session_state.y_test_clf = y_test_clf
            
            st.success("Tiền xử lý hoàn tất!")
            
            # Show preprocessing results
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Kích Thước Tập Huấn Luyện", f"{X_train.shape[0]} mẫu")
                st.metric("Đặc Trưng Sau Xử Lý", f"{X_train.shape[1]}")
            with col2:
                st.metric("Kích Thức Tập Kiểm Tra", f"{X_test.shape[0]} mẫu")
                st.metric("Thời Gian Xử Lý", "< 1 giây")

def model_training_page(data):
    st.markdown('<h2 class="sub-header">🤖 Huấn Luyện Mô Hình</h2>', unsafe_allow_html=True)
    
    # Check if data is preprocessed
    if 'X_train' not in st.session_state:
        st.warning("Vui lòng hoàn thành tiền xử lý dữ liệu trước!")
        return
    
    if 'trainer' not in st.session_state:
        st.session_state.trainer = ModelTrainer()
    
    trainer = st.session_state.trainer
    
    # Model selection
    st.subheader("📋 Chọn Mô Hình Để Huấn Luyện")
    
    # Hiển thị tất cả 4 mô hình có sẵn
    st.markdown("### 🤖 4 Thuật Toán Học Máy Có Sẵn:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📈 Mô Hình Hồi Quy (Dự Đoán AQI):**")
        lr_reg = st.checkbox("Hồi Quy Tuyến Tính", value=True)
        dt_reg = st.checkbox("Cây Quyết Định (CART)", value=True)
        
        st.markdown("**🎯 Mô Hình Phân Loại (Mức Độ Ô Nhiễm):**")
        lr_clf = st.checkbox("Hồi Quy Logistic", value=True)
        svm_clf = st.checkbox("SVM", value=True)
    
    with col2:
        st.markdown("**📊 Mô Tả Thuật Toán:**")
        st.info("""
        **Hồi Quy Tuyến Tính:**
        - Đơn giản, dễ diễn giải
        - Huấn luyện nhanh nhất
        - Phù hợp dự đoán AQI liên tục
        
        **Cây Quyết Định (CART):**
        - Xử lý mối quan hệ phi tuyến
        - Dễ trực quan hóa
        - Hiểu tầm quan trọng đặc trưng
        
        **Hồi Quy Logistic:**
        - Đầu ra xác suất
        - Ổn định và đáng tin cậy
        - Tốt cho cảnh báo
        
        **SVM:**
        - Độ chính xác cao nhất
        - Tốt cho dữ liệu phức tạp
        - Chống overfitting tốt
        """)
    
    # Collect selected models
    regression_models = []
    classification_models = []
    
    if lr_reg:
        regression_models.append("Hồi Quy Tuyến Tính")
    if dt_reg:
        regression_models.append("Cây Quyết Định (CART)")
    if lr_clf:
        classification_models.append("Hồi Quy Logistic")
    if svm_clf:
        classification_models.append("SVM")
    
    # Show selected models summary
    st.markdown(f"### 🎯 Đã Chọn: {len(regression_models)} mô hình hồi quy, {len(classification_models)} mô hình phân loại")
    
    # Ensure we have models selected
    if not regression_models and not classification_models:
        st.warning("⚠️ Vui lòng chọn ít nhất một mô hình!")
        return
    
    # Training parameters
    st.subheader("⚙️ Tham Số Huấn Luyện")
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider("Kích Thước Kiểm Tra", 0.1, 0.4, 0.2, 0.05)
        random_state = st.number_input("Trạng Thái Ngẫu Nhiên", value=42)
        cv_folds = st.slider("Số Lần Cross-Validation", 3, 10, 5)
        enable_grid_search = st.checkbox("Bật Tìm Kiếm Lưới", value=False)
    
    with col2:
        st.markdown("**📊 Mô Tả Tham Số:**")
        st.info("""
        **Kích Thước Kiểm Tra:** % dữ liệu dùng để kiểm tra
        
        **Cross-Validation:** Số lần chia dữ liệu để đánh giá
        
        **Tìm Kiếm Lưới:** Tự động tìm tham số tốt nhất
        
        **Trạng Thái Ngẫu Nhiên:** Đảm bảo kết quả có thể lặp lại
        """)
    
    # Advanced hyperparameter tuning
    if enable_grid_search:
        st.subheader("🔧 Điều Chỉnh Siêu Tham Số Nâng Cao")
        
        # Create tabs for each model type
        tab1, tab2 = st.tabs(["📈 Mô Hình Hồi Quy", "🎯 Mô Hình Phân Loại"])
        
        with tab1:
            st.markdown("### Hồi Quy Tuyến Tính")
            st.info("Hồi quy tuyến tính không có siêu tham số cần điều chỉnh")
            
            st.markdown("### Cây Quyết Định (CART)")
            col1, col2 = st.columns(2)
            with col1:
                dt_max_depth = st.selectbox("Chiều Sâu Tối Đa", [3, 5, 7, 10, None], index=1)
                dt_min_samples_split = st.selectbox("Mẫu Tách Nhỏ Nhất", [2, 5, 10], index=1)
            with col2:
                dt_min_samples_leaf = st.selectbox("Lá Nhỏ Nhất", [1, 2, 4], index=1)
                dt_max_features = st.selectbox("Đặc Trưng Tối Đa", ["sqrt", "log2", None], index=0)
            
            # Store CART parameters
            st.session_state.cart_params = {
                'max_depth': [dt_max_depth],
                'min_samples_split': [dt_min_samples_split],
                'min_samples_leaf': [dt_min_samples_leaf],
                'max_features': [dt_max_features]
            }
        
        with tab2:
            st.markdown("### Hồi Quy Logistic")
            col1, col2 = st.columns(2)
            with col1:
                lr_c = st.multiselect("C (Độ Ngược)", [0.1, 1, 10, 100], default=[1], key="lr_c")
                lr_penalty = st.multiselect("Penalty", ["l1", "l2"], default=["l2"], key="lr_penalty")
            with col2:
                lr_solver = st.multiselect("Solver", ["liblinear", "saga"], default=["liblinear"], key="lr_solver")
                lr_max_iter = st.slider("Max Iterations", 100, 2000, 1000, 100)
            
            # Store Logistic Regression parameters
            st.session_state.lr_params = {
                'C': lr_c,
                'penalty': lr_penalty,
                'solver': lr_solver,
                'max_iter': [lr_max_iter]
            }
            
            st.markdown("### SVM")
            col1, col2 = st.columns(2)
            with col1:
                svm_c = st.multiselect("C (Độ Ngược)", [0.1, 1, 10, 100], default=[1], key="svm_c")
                svm_kernel = st.multiselect("Kernel", ["linear", "rbf", "poly"], default=["rbf"], key="svm_kernel")
            with col2:
                svm_gamma = st.multiselect("Gamma", ["scale", "auto"], default=["scale"], key="svm_gamma")
                svm_degree = st.slider("Degree (cho polynomial)", 2, 5, 3) if "poly" in svm_kernel else 3
            
            # Store SVM parameters
            st.session_state.svm_params = {
                'C': svm_c,
                'kernel': svm_kernel,
                'gamma': svm_gamma,
                'degree': [svm_degree] if "poly" in svm_kernel else [3]
            }
    
    else:
        st.info("💡 Bật 'Tìm Kiếm Lưới' để điều chỉnh siêu tham số nâng cao cho từng mô hình")
    
    # Train models
    if st.button("🚀 Huấn Luyện Các Mô Hình Đã Chọn"):
        with st.spinner("Đang huấn luyện mô hình... Quá trình có thể mất vài phút..."):
            # Collect custom parameters
            custom_params = {}
            if enable_grid_search:
                if 'cart_params' in st.session_state:
                    custom_params['cart_params'] = st.session_state.cart_params
                if 'lr_params' in st.session_state:
                    custom_params['lr_params'] = st.session_state.lr_params
                if 'svm_params' in st.session_state:
                    custom_params['svm_params'] = st.session_state.svm_params
            
            results = trainer.train_models(
                st.session_state.X_train,
                st.session_state.X_test,
                st.session_state.y_train_reg,
                st.session_state.y_test_reg,
                st.session_state.y_train_clf,
                st.session_state.y_test_clf,
                regression_models,
                classification_models,
                cv_folds=cv_folds,
                enable_grid_search=enable_grid_search,
                custom_params=custom_params
            )
            
            st.session_state.training_results = results
            st.success("Huấn luyện mô hình hoàn tất!")
            
            # Display training results
            st.subheader("📊 Kết Quả Huấn Luyện 4 Mô Hình")
            
            # Show data source
            st.info(f"📂 Huấn luyện trên: {st.session_state.get('data_source', 'Unknown')}")
            
                        
            # Create a comprehensive summary table
            st.markdown("### 🏆 Bảng Tổng Kết Tất Cả Mô Hình")
            
            # Prepare data for summary table
            summary_data = []
            
            # Add regression models
            if regression_models:
                for model_name in regression_models:
                    if model_name in results['regression']:
                        metrics = results['regression'][model_name]
                        summary_data.append({
                            'STT': len(summary_data) + 1,
                            'Tên Mô Hình': model_name,
                            'Loại': 'Hồi Quy',
                            'Nhiệm Vụ': 'Dự Đoán AQI',
                            'RMSE': f"{metrics['rmse']:.3f}",
                            'R²': f"{metrics['r2']:.3f}",
                            'MAE': f"{metrics['mae']:.3f}",
                            'Accuracy': '-',
                            'F1-Score': '-',
                            'Đánh Giá': 'Tốt' if metrics['r2'] > 0.9 else 'Khá' if metrics['r2'] > 0.8 else 'Trung Bình'
                        })
            
            # Add classification models
            if classification_models:
                for model_name in classification_models:
                    if model_name in results['classification']:
                        metrics = results['classification'][model_name]
                        summary_data.append({
                            'STT': len(summary_data) + 1,
                            'Tên Mô Hình': model_name,
                            'Loại': 'Phân Loại',
                            'Nhiệm Vụ': 'Phân Loại Mức Độ',
                            'RMSE': '-',
                            'R²': '-',
                            'MAE': '-',
                            'Accuracy': f"{metrics['accuracy']:.3f}",
                            'F1-Score': f"{metrics['f1']:.3f}",
                            'Đánh Giá': 'Xuất Sắc' if metrics['f1'] > 0.95 else 'Tốt' if metrics['f1'] > 0.9 else 'Khá'
                        })
            
            # Display summary table
            if summary_data:
                df_summary = pd.DataFrame(summary_data)
                st.dataframe(df_summary, use_container_width=True, hide_index=True)
                
                # Highlight best models
                st.markdown("### 🥇 Mô Hình Xuất Sắc Nhất")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Best regression model
                    best_reg = None
                    best_reg_score = 0
                    for model_name in regression_models:
                        if model_name in results['regression']:
                            r2 = results['regression'][model_name]['r2']
                            if r2 > best_reg_score:
                                best_reg_score = r2
                                best_reg = model_name
                    
                    if best_reg:
                        st.success(f"📈 **Hồi Quy Tốt Nhất:** {best_reg}\nR² = {best_reg_score:.3f}")
                
                with col2:
                    # Best classification model
                    best_clf = None
                    best_clf_score = 0
                    for model_name in classification_models:
                        if model_name in results['classification']:
                            f1 = results['classification'][model_name]['f1']
                            if f1 > best_clf_score:
                                best_clf_score = f1
                                best_clf = model_name
                    
                    if best_clf:
                        st.success(f"🎯 **Phân Loại Tốt Nhất:** {best_clf}\nF1-Score = {best_clf_score:.3f}")
                
                # Overall recommendation
                if best_reg and best_clf:
                    st.markdown("---")
                    st.markdown("### 🏆 KHUYẾN NGHỊ TỔNG THỂ")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    padding: 1.5rem; border-radius: 1rem; color: #ffffff; 
                                    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);">
                            <h4 style="color: #ffffff; margin-bottom: 1rem;">🎯 NÊN SỬ DỤNG:</h4>
                            <ul style="list-style: none; padding: 0; margin: 0;">
                                <li style="margin-bottom: 0.5rem;">✅ <strong>{}</strong> cho dự đoán AQI chính xác</li>
                                <li style="margin-bottom: 0.5rem;">✅ <strong>{}</strong> cho phân loại mức độ ô nhiễm</li>
                                <li style="margin-bottom: 0.5rem;">✅ Kết hợp cả 2 để hệ thống hoàn chỉnh</li>
                            </ul>
                        </div>
                        """.format(best_reg, best_clf), unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                                    padding: 1.5rem; border-radius: 1rem; color: #ffffff; 
                                    box-shadow: 0 4px 15px rgba(240, 147, 251, 0.4);">
                            <h4 style="color: #ffffff; margin-bottom: 1rem;">🚀 TRƯỜNG HỢP SỬ DỤNG:</h4>
                            <ul style="list-style: none; padding: 0; margin: 0;">
                                <li style="margin-bottom: 0.5rem;">📊 <strong>Phân tích</strong>: {} để hiểu quan hệ</li>
                                <li style="margin-bottom: 0.5rem;">⚡ <strong>Nhanh nhất</strong>: {} cho dự đoán tức thì</li>
                                <li style="margin-bottom: 0.5rem;">🛡️ <strong>Production</strong>: {} cho hệ thống ổn định</li>
                            </ul>
                        </div>
                        """.format(
                            "Decision Tree" if "Decision Tree" in regression_models else best_reg,
                            "Linear Regression" if "Linear Regression" in regression_models else best_reg,
                            "Logistic Regression" if "Logistic Regression" in classification_models else best_clf
                        ), unsafe_allow_html=True)
                    
                    # Final recommendation box
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                                padding: 2rem; border-radius: 1rem; color: #ffffff; 
                                box-shadow: 0 4px 15px rgba(79, 172, 254, 0.4); 
                                border: 2px solid #4facfe; margin-top: 1rem;">
                        <h3 style="color: #ffffff; margin-bottom: 1rem; text-align: center; font-size: 1.4rem;">
                            🏆 KHUYẾN NGHỊ CUỐI CÙNG
                        </h3>
                        <div style="text-align: center; font-size: 1.1rem; line-height: 1.6;">
                            <p style="margin-bottom: 1rem;"><strong>Để có hệ thống dự đoán ô nhiễm không khí hoàn chỉnh và hiệu quả nhất:</strong></p>
                            <div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 0.8rem; margin: 1rem 0;">
                                <p style="margin: 0; font-weight: bold;">
                                    🎯 Sử dụng <strong>{}</strong> để dự đoán giá trị AQI chính xác<br>
                                    🎯 Sử dụng <strong>{}</strong> để phân loại mức độ ô nhiễm đáng tin cậy
                                </p>
                            </div>
                            <p style="margin: 0; font-style: italic;">
                                💡 Kết hợp cả hai mô hình này sẽ cho bạn hệ thống dự đoán toàn diện nhất!
                            </p>
                        </div>
                    </div>
                    """.format(best_reg, best_clf), unsafe_allow_html=True)
            
            # Detailed results for each model
            st.markdown("### 📈 Chi Tiết Chi Tiết Từng Mô Hình")
            
            # Regression models details
            if regression_models:
                st.markdown("#### 📊 Mô Hình Hồi Quy (Dự Đoán AQI)")
                for model_name in regression_models:
                    if model_name in results['regression']:
                        metrics = results['regression'][model_name]
                        with st.expander(f"🔍 {model_name} - Chi Tiết", expanded=True):
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("RMSE", f"{metrics['rmse']:.3f}")
                            with col2:
                                st.metric("R²", f"{metrics['r2']:.3f}")
                            with col3:
                                st.metric("MAE", f"{metrics['mae']:.3f}")
                            with col4:
                                st.metric("CV RMSE", f"{metrics['cv_rmse_mean']:.3f}")
                            
                            # Feature importance if available
                            if metrics.get('feature_importance'):
                                st.markdown("**🎯 Tầm Quan Trọng Đặc Trưng (Top 5):**")
                                importance = metrics['feature_importance']
                                top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
                                for feature, score in top_features:
                                    st.write(f"• {feature}: {score:.3f}")
            
            # Classification models details
            if classification_models:
                st.markdown("#### 🎯 Mô Hình Phân Loại (Mức Độ Ô Nhiễm)")
                for model_name in classification_models:
                    if model_name in results['classification']:
                        metrics = results['classification'][model_name]
                        with st.expander(f"🔍 {model_name} - Chi Tiết", expanded=True):
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Độ Chính Xác", f"{metrics['accuracy']:.3f}")
                            with col2:
                                st.metric("Precision", f"{metrics['precision']:.3f}")
                            with col3:
                                st.metric("Recall", f"{metrics['recall']:.3f}")
                            with col4:
                                st.metric("F1-Score", f"{metrics['f1']:.3f}")
                            
                            # Feature importance if available
                            if metrics.get('feature_importance'):
                                st.markdown("**🎯 Tầm Quan Trọng Đặc Trưng (Top 5):**")
                                importance = metrics['feature_importance']
                                top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
                                for feature, score in top_features:
                                    st.write(f"• {feature}: {score:.3f}")

def evaluation_page(data):
    st.markdown('<h2 class="sub-header">📊 Đánh Giá & So Sánh Mô Hình</h2>', unsafe_allow_html=True)
    
    if 'training_results' not in st.session_state:
        st.warning("Vui lòng huấn luyện mô hình trước!")
        return
    
    results = st.session_state.training_results
    
    # Initialize evaluator
    if 'evaluator' not in st.session_state:
        st.session_state.evaluator = ModelEvaluator()
    
    evaluator = st.session_state.evaluator
    
    # Comprehensive comparison
    st.subheader("🏆 So Sánh Hiệu Suất Mô Hình")
    
    # Regression comparison
    if results['regression']:
        st.markdown("### 📈 So Sánh Mô Hình Hồi Quy")
        evaluator.compare_regression_models(results['regression'])
    
    # Classification comparison
    if results['classification']:
        st.markdown("### 🎯 So Sánh Mô Hình Phân Loại")
        evaluator.compare_classification_models(results['classification'])
    
    # Best model recommendation
    st.subheader("🥇 Khuyến Nghị Mô Hình Tốt Nhất")
    evaluator.recommend_best_models(results)
    
    # Enhanced recommendation display
    if 'regression' in results and results['regression'] and 'classification' in results and results['classification']:
        st.markdown("---")
        st.markdown("### 🏆 KHUYẾN NGHỊ SỬ DỤNG THỰC TẾ")
        
        # Find best models
        best_reg_model = min(results['regression'].keys(), key=lambda x: results['regression'][x]['rmse'])
        best_clf_model = max(results['classification'].keys(), key=lambda x: results['classification'][x]['f1'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 1.5rem; border-radius: 1rem; color: #ffffff; 
                        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);">
                <h4 style="color: #ffffff; margin-bottom: 1rem;">🎯 NÊN CHỌN:</h4>
                <div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 0.8rem; margin-bottom: 1rem;">
                    <p style="margin: 0; font-weight: bold; font-size: 1.1rem;">
                        📈 <strong>{}</strong><br>
                        🎯 <strong>{}</strong>
                    </p>
                </div>
                <p style="margin: 0; font-size: 0.9rem;">
                    💡 Đây là sự kết hợp tốt nhất cho hệ thống dự đoán hoàn chỉnh
                </p>
            </div>
            """.format(best_reg_model, best_clf_model), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                        padding: 1.5rem; border-radius: 1rem; color: #ffffff; 
                        box-shadow: 0 4px 15px rgba(240, 147, 251, 0.4);">
                <h4 style="color: #ffffff; margin-bottom: 1rem;">🚀 LÝ DO CHỌN:</h4>
                <ul style="list-style: none; padding: 0; margin: 0; font-size: 0.9rem;">
                    <li style="margin-bottom: 0.5rem;">✨ {} có RMSE thấp nhất ({:.2f})</li>
                    <li style="margin-bottom: 0.5rem;">🎯 {} có F1-score cao nhất ({:.3f})</li>
                    <li style="margin-bottom: 0.5rem;">🛡️ Cả hai đều ổn định qua cross-validation</li>
                    <li style="margin: 0;">⚡ Cân bằng giữa độ chính xác và tốc độ</li>
                </ul>
            </div>
            """.format(
                best_reg_model, 
                results['regression'][best_reg_model]['rmse'],
                best_clf_model,
                results['classification'][best_clf_model]['f1']
            ), unsafe_allow_html=True)
        
        # Final recommendation summary
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 2rem; border-radius: 1rem; color: #ffffff; 
                    box-shadow: 0 4px 15px rgba(79, 172, 254, 0.4); 
                    border: 2px solid #4facfe; margin-top: 1rem;">
            <h3 style="color: #ffffff; margin-bottom: 1rem; text-align: center; font-size: 1.4rem;">
                🏆 KHUYẾN NGHỊ CUỐI CÙNG CHO HỆ THỐNG
            </h3>
            <div style="text-align: center; font-size: 1.1rem; line-height: 1.6;">
                <p style="margin-bottom: 1rem;"><strong>Để xây dựng hệ thống dự đoán ô nhiễm không khí hiệu quả nhất:</strong></p>
                <div style="background: rgba(255,255,255,0.2); padding: 1.2rem; border-radius: 0.8rem; margin: 1rem 0;">
                    <p style="margin: 0; font-weight: bold;">
                        🎯 <strong>Dự đoán AQI</strong>: Sử dụng <strong>{}</strong><br>
                        🎯 <strong>Phân loại mức độ</strong>: Sử dụng <strong>{}</strong><br>
                        🎯 <strong>Hệ thống hoàn chỉnh</strong>: Kết hợp cả hai mô hình
                    </p>
                </div>
                <p style="margin: 0; font-style: italic;">
                    💡 Cấu hình này cho độ chính xác cao nhất trong khi vẫn giữ hiệu suất tốt!
                </p>
            </div>
        </div>
        """.format(best_reg_model, best_clf_model), unsafe_allow_html=True)
    
    # Detailed analysis
    st.subheader("🔍 Phân Tích Chi Tiết")
    
    if st.button("Tạo Phân Tích Chi Tiết"):
        with st.spinner("Đang phân tích hiệu suất mô hình..."):
            # Create visualizations
            figs = evaluator.create_detailed_visualizations(results)
            
            for i, fig in enumerate(figs):
                st.plotly_chart(fig, use_container_width=True)

def prediction_page(data):
    st.markdown('<h2 class="sub-header">🔮 Dự Đoán Thời Gian Thực</h2>', unsafe_allow_html=True)
    
    if 'training_results' not in st.session_state:
        st.warning("Vui lòng huấn luyện mô hình trước!")
        return
    
    # Input form for prediction
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 1.5rem; border-radius: 1rem; color: #ffffff; 
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4); 
                border: 2px solid #667eea; margin-bottom: 2rem;">
        <h3 style="color: #ffffff; margin-bottom: 1rem; font-size: 1.4rem; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">
            📝 Nhập Tham Số Môi Trường
        </h3>
        <p style="color: #ffffff; margin: 0; font-size: 1rem; line-height: 1.6;">
            Nhập các chỉ số môi trường để dự đoán chất lượng không khí và mức độ ô nhiễm
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 1rem; border-radius: 0.8rem; color: #ffffff; 
                    box-shadow: 0 4px 15px rgba(240, 147, 251, 0.4); 
                    border: 2px solid #f093fb; margin-bottom: 1rem;">
            <h4 style="color: #ffffff; margin-bottom: 0.8rem; font-size: 1.1rem; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">
                🌫️ Chỉ Số Ô Nhiễm
            </h4>
        </div>
        """, unsafe_allow_html=True)
        
        pm25 = st.number_input("PM2.5 (μg/m³)", min_value=0.0, max_value=500.0, value=50.0, step=1.0)
        pm10 = st.number_input("PM10 (μg/m³)", min_value=0.0, max_value=600.0, value=75.0, step=1.0)
        no2 = st.number_input("NO₂ (μg/m³)", min_value=0.0, max_value=200.0, value=40.0, step=1.0)
        so2 = st.number_input("SO₂ (μg/m³)", min_value=0.0, max_value=150.0, value=20.0, step=1.0)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 1rem; border-radius: 0.8rem; color: #ffffff; 
                    box-shadow: 0 4px 15px rgba(79, 172, 254, 0.4); 
                    border: 2px solid #4facfe; margin-bottom: 1rem;">
            <h4 style="color: #ffffff; margin-bottom: 0.8rem; font-size: 1.1rem; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">
                🌤️ Thời Tiết & Khí Hậu
            </h4>
        </div>
        """, unsafe_allow_html=True)
        
        co = st.number_input("CO (mg/m³)", min_value=0.0, max_value=50.0, value=5.0, step=0.1)
        o3 = st.number_input("O₃ (μg/m³)", min_value=0.0, max_value=300.0, value=80.0, step=1.0)
        temperature = st.number_input("Nhiệt Độ (°C)", min_value=-10.0, max_value=50.0, value=25.0, step=0.5)
        humidity = st.number_input("Độ Ẩm (%)", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
    
    # Additional parameters
    st.markdown("""
    <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                padding: 1rem; border-radius: 0.8rem; color: #333333; 
                box-shadow: 0 4px 15px rgba(250, 112, 154, 0.4); 
                border: 2px solid #fa709a; margin-bottom: 1rem;">
        <h4 style="color: #333333; margin-bottom: 0.8rem; font-size: 1.1rem; text-shadow: 1px 1px 2px rgba(255,255,255,0.5);">
            💨 Thông Số Khí Tượng
        </h4>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        wind_speed = st.number_input("Tốc Độ Gió (m/s)", min_value=0.0, max_value=20.0, value=2.5, step=0.1)
    with col2:
        pressure = st.number_input("Áp Suất (hPa)", min_value=900.0, max_value=1100.0, value=1013.0, step=1.0)
    with col3:
        rainfall = st.number_input("Lượng Mưa (mm)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    
    # Model selection for prediction
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 1.5rem; border-radius: 1rem; color: #ffffff; 
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4); 
                border: 2px solid #667eea; margin-bottom: 2rem;">
        <h3 style="color: #ffffff; margin-bottom: 1rem; font-size: 1.4rem; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">
            🤖 Chọn Mô Hình Để Dự Đoán
        </h3>
        <p style="color: #ffffff; margin: 0; font-size: 1rem; line-height: 1.6;">
            Lựa chọn thuật toán phù hợp để thực hiện dự đoán AQI và phân loại mức độ ô nhiễm
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    regression_model = st.selectbox(
        "Mô Hình Hồi Quy (Dự Đoán AQI)",
        ["Hồi Quy Tuyến Tính", "Cây Quyết Định (CART)"]
    )
    
    classification_model = st.selectbox(
        "Mô Hình Phân Loại (Mức Độ Ô Nhiễm)",
        ["Hồi Quy Logistic", "SVM"]
    )
    
    # Make prediction
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0;">
    """, unsafe_allow_html=True)
    
    if st.button("🔮 Thực Hiện Dự Đoán", use_container_width=True):
        st.markdown("""
        </div>
        """, unsafe_allow_html=True)
        
        # Create input data with all required features
        input_data = pd.DataFrame({
            'PM2.5': [pm25], 'PM10': [pm10], 'NO2': [no2], 'SO2': [so2], 'CO': [co], 'O3': [o3],
            'Temperature': [temperature], 'Humidity': [humidity], 'Wind_Speed': [wind_speed],
            'Pressure': [pressure], 'Rainfall': [rainfall]
        })
        
        # Get current datetime for temporal features
        from datetime import datetime
        current_time = datetime.now()
        
        # Add temporal features that were created during preprocessing
        input_data['Hour'] = current_time.hour
        input_data['DayOfWeek'] = current_time.weekday()
        input_data['Month'] = current_time.month
        
        # Add cyclical features
        input_data['Hour_sin'] = np.sin(2 * np.pi * input_data['Hour'] / 24)
        input_data['Hour_cos'] = np.cos(2 * np.pi * input_data['Hour'] / 24)
        input_data['Month_sin'] = np.sin(2 * np.pi * input_data['Month'] / 12)
        input_data['Month_cos'] = np.cos(2 * np.pi * input_data['Month'] / 12)
        
        # Add season
        season_mapping = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                          3: 'Spring', 4: 'Spring', 5: 'Spring',
                          6: 'Summer', 7: 'Summer', 8: 'Summer',
                          9: 'Fall', 10: 'Fall', 11: 'Fall'}
        input_data['Season'] = season_mapping[current_time.month]
        
        # Add pollution ratios and indices
        input_data['PM25_PM10_Ratio'] = input_data['PM2.5'] / (input_data['PM10'] + 1e-6)
        input_data['Traffic_Pollution_Index'] = input_data['NO2'] + input_data['CO']
        input_data['Industrial_Pollution_Index'] = input_data['SO2']
        input_data['Total_Pollution'] = input_data[['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']].sum(axis=1)
        input_data['Max_Pollutant'] = input_data[['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']].max(axis=1)
        input_data['Pollution_Std'] = input_data[['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']].std(axis=1)
        
        # Add weather interaction features
        input_data['Temp_Humidity_Interaction'] = input_data['Temperature'] * input_data['Humidity']
        input_data['Wind_Pollution_Interaction'] = input_data['Wind_Speed'] / (input_data['PM2.5'] + 1e-6)
        
        # Add encoded categorical features
        input_data['Season_Encoded'] = 0  # Default encoding
        if input_data['Season'].iloc[0] == 'Winter':
            input_data['Season_Encoded'] = 3
        elif input_data['Season'].iloc[0] == 'Spring':
            input_data['Season_Encoded'] = 0
        elif input_data['Season'].iloc[0] == 'Summer':
            input_data['Season_Encoded'] = 2
        else:  # Fall
            input_data['Season_Encoded'] = 1
        
        print(f"🔍 Input data shape: {input_data.shape}")
        print(f"🔍 Input columns: {list(input_data.columns)}")
        
        # Ensure input data has all required columns that were used during training
        if 'preprocessor' in st.session_state:
            preprocessor = st.session_state.preprocessor
            required_columns = preprocessor.feature_columns
            
            # Add missing columns with default values
            for col in required_columns:
                if col not in input_data.columns:
                    if col in ['Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos']:
                        input_data[col] = 0.0
                    elif col in ['Season_Encoded']:
                        input_data[col] = 1
                    elif col in ['Temp_Humidity_Interaction', 'Wind_Pollution_Interaction']:
                        input_data[col] = 0.0
                    elif col in ['PM25_PM10_Ratio', 'Traffic_Pollution_Index', 'Industrial_Pollution_Index']:
                        input_data[col] = 0.0
                    elif col in ['Total_Pollution', 'Max_Pollutant', 'Pollution_Std']:
                        input_data[col] = 0.0
                    else:
                        input_data[col] = 0.0
            
            # Reorder columns to match training data
            input_data = input_data[required_columns]
            
            print(f"🔍 Final input data shape: {input_data.shape}")
            print(f"🔍 Final input columns: {list(input_data.columns)}")
        
        # Get predictions
        trainer = st.session_state.trainer
        
        try:
            # Apply the same preprocessing as training data
            if 'preprocessor' in st.session_state:
                # Use the fitted preprocessor to transform the input data
                input_data_scaled = st.session_state.preprocessor.scaler.transform(input_data)
                input_data_scaled = pd.DataFrame(input_data_scaled, columns=input_data.columns)
                print(f"🔍 Scaled input data shape: {input_data_scaled.shape}")
            else:
                # Fallback: simple scaling
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                input_data_scaled = scaler.fit_transform(input_data)
                input_data_scaled = pd.DataFrame(input_data_scaled, columns=input_data.columns)
                print(f"🔍 Fallback scaling applied")
            
            # Regression prediction
            aqi_pred = trainer.predict_regression(input_data_scaled, regression_model)
            
            # Classification prediction
            pollution_pred = trainer.predict_classification(input_data_scaled, classification_model)
            
            # Convert numeric prediction back to label
            pollution_level_map = {
                0: "Tốt",
                1: "Trung Bình", 
                2: "Kém",
                3: "Xấu",
                4: "Rất Xấu",
                5: "Nguy Hiểm"
            }
            
            # Get the most common prediction if it's an array
            if isinstance(pollution_pred, np.ndarray):
                pollution_pred_value = pollution_pred[0]
            else:
                pollution_pred_value = pollution_pred
            
            pollution_label = pollution_level_map.get(int(pollution_pred_value), "Không xác định")
            
            # Display results
            st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 2rem; border-radius: 1rem; color: #ffffff; 
                        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4); 
                        border: 2px solid #667eea; margin-bottom: 2rem;">
                <h3 style="color: #ffffff; margin-bottom: 1.5rem; font-size: 1.4rem; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">
                    📊 Kết Quả Dự Đoán
                </h3>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                            padding: 1.5rem; border-radius: 1rem; color: #ffffff; 
                            box-shadow: 0 4px 15px rgba(240, 147, 251, 0.4); 
                            border: 2px solid #f093fb; margin-bottom: 1rem;">
                    <h4 style="color: #ffffff; margin-bottom: 1rem; font-size: 1.2rem; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">
                        📈 Dự Đoán AQI
                    </h4>
                    <div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 0.8rem; text-align: center;">
                        <div style="font-size: 2rem; font-weight: bold; margin-bottom: 0.5rem;">
                            {aqi_value:.1f}
                        </div>
                        <div style="font-size: 1rem;">
                            Chỉ số AQI
                        </div>
                    </div>
                </div>
                """.format(aqi_value=aqi_pred[0]), unsafe_allow_html=True)
                
                # AQI level indicator
                if aqi_pred[0] <= 50:
                    level_color = "#4CAF50"
                    level_text = "Tốt"
                    level_emoji = "🟢"
                elif aqi_pred[0] <= 100:
                    level_color = "#FFC107"
                    level_text = "Trung Bình"
                    level_emoji = "🟡"
                elif aqi_pred[0] <= 150:
                    level_color = "#FF9800"
                    level_text = "Kém"
                    level_emoji = "🟠"
                elif aqi_pred[0] <= 200:
                    level_color = "#F44336"
                    level_text = "Xấu"
                    level_emoji = "🔴"
                elif aqi_pred[0] <= 300:
                    level_color = "#9C27B0"
                    level_text = "Rất Xấu"
                    level_emoji = "🟣"
                else:
                    level_color = "#424242"
                    level_text = "Nguy Hiểm"
                    level_emoji = "⚫"
                
                st.markdown(f"""
                <div style="background: {level_color}; padding: 1rem; border-radius: 1rem; 
                            color: white; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
                    <div style="font-size: 1.5rem; font-weight: bold; margin-bottom: 0.5rem;">
                        {level_emoji} {level_text}
                    </div>
                    <div style="font-size: 0.9rem;">
                        Mức độ chất lượng không khí
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                            padding: 1.5rem; border-radius: 1rem; color: #ffffff; 
                            box-shadow: 0 4px 15px rgba(79, 172, 254, 0.4); 
                            border: 2px solid #4facfe; margin-bottom: 1rem;">
                    <h4 style="color: #ffffff; margin-bottom: 1rem; font-size: 1.2rem; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">
                        🎯 Phân Loại Mức Độ Ô Nhiễm
                    </h4>
                    <div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 0.8rem; text-align: center;">
                        <div style="font-size: 1.5rem; font-weight: bold; margin-bottom: 0.5rem;">
                            {pollution_level}
                        </div>
                        <div style="font-size: 0.9rem;">
                            Phân loại ô nhiễm
                        </div>
                    </div>
                </div>
                """.format(pollution_level=pollution_label), unsafe_allow_html=True)
                
                # Health recommendations
                health_advice = get_health_advice(pollution_label)
                st.markdown("""
                <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                            padding: 1.5rem; border-radius: 1rem; color: #333333; 
                            box-shadow: 0 4px 15px rgba(250, 112, 154, 0.4); 
                            border: 2px solid #fa709a;">
                    <h4 style="color: #333333; margin-bottom: 1rem; font-size: 1.2rem; text-shadow: 1px 1px 2px rgba(255,255,255,0.5);">
                        💡 Khuyến Nghị Sức Khỏe
                    </h4>
                    <div style="background: rgba(51,51,51,0.1); padding: 1rem; border-radius: 0.8rem; color: #333333;">
                        {health_advice_text}
                    </div>
                </div>
                """.format(health_advice_text=health_advice), unsafe_allow_html=True)
            
            # Visualization
            st.subheader("📊 Trực Quan Hóa Dự Đoán")
            
            # Create gauge chart for AQI
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = aqi_pred[0],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Giá Trị AQI"},
                delta = {'reference': 100},
                gauge = {
                    'axis': {'range': [None, 500]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgreen"},
                        {'range': [50, 100], 'color': "yellow"},
                        {'range': [100, 150], 'color': "orange"},
                        {'range': [150, 200], 'color': "red"},
                        {'range': [200, 300], 'color': "purple"},
                        {'range': [300, 500], 'color': "darkred"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 150
                    }
                }
            ))
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Lỗi dự đoán: {str(e)}")

def get_health_advice(pollution_level):
    advice = {
        "Tốt": "Chất lượng không khí tốt. Hãy tận hưởng hoạt động ngoài trời!",
        "Trung Bình": "Chất lượng không khí chấp nhận được. Người nhạy cảm nên cân nhắc hạn chế hoạt động ngoài trời kéo dài.",
        "Kém": "Nhóm người nhạy cảm có thể gặp tác động sức khỏe. Hạn chế hoạt động ngoài trời kéo dài.",
        "Xấu": "Mọi người có thể bắt đầu gặp tác động sức khỏe. Hạn chế hoạt động ngoài trời kéo dài.",
        "Rất Xấu": "Cảnh báo sức khỏe tình trạng khẩn cấp. Mọi người nên tránh hoạt động ngoài trời.",
        "Nguy Hiểm": "Tình trạng khẩn cấp. Mọi người nên tránh các hoạt động ngoài trời."
    }
    return advice.get(pollution_level, "Không có khuyến nghị cụ thể.")

def conclusions_page():
    st.markdown('<h2 class="sub-header">📋 Kết Luận & Khuyến Nghị</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 1rem; color: #ffffff; 
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4); 
                border: 2px solid #667eea; margin-bottom: 2rem;">
        <h3 style="color: #ffffff; margin-bottom: 1.5rem; font-size: 1.4rem; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">
            🎯 Thành Tựu Dự Án
        </h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem;">
            <div style="background: rgba(255,255,255,0.1); padding: 1.2rem; border-radius: 0.8rem; 
                        border-left: 4px solid #f093fb;">
                <h4 style="color: #ffffff; margin-bottom: 0.8rem; font-size: 1.1rem;">
                    🤖 Thuật Toán Học Máy
                </h4>
                <p style="color: #ffffff; margin: 0; font-size: 0.95rem; line-height: 1.5;">
                    Triển khai thành công 4 thuật toán học máy cho dự đoán ô nhiễm không khí
                </p>
            </div>
            <div style="background: rgba(255,255,255,0.1); padding: 1.2rem; border-radius: 0.8rem; 
                        border-left: 4px solid #4facfe;">
                <h4 style="color: #ffffff; margin-bottom: 0.8rem; font-size: 1.1rem;">
                    🔧 Tiền Xử Lý Dữ Liệu
                </h4>
                <p style="color: #ffffff; margin: 0; font-size: 0.95rem; line-height: 1.5;">
                    Pipeline tiền xử lý dữ liệu toàn diện với kỹ thuật đặc trưng
                </p>
            </div>
            <div style="background: rgba(255,255,255,0.1); padding: 1.2rem; border-radius: 0.8rem; 
                        border-left: 4px solid #fa709a;">
                <h4 style="color: #ffffff; margin-bottom: 0.8rem; font-size: 1.1rem;">
                    📊 Đánh Giá Mô Hình
                </h4>
                <p style="color: #ffffff; margin: 0; font-size: 0.95rem; line-height: 1.5;">
                    Đánh giá mô hình mạnh mẽ sử dụng nhiều chỉ số
                </p>
            </div>
            <div style="background: rgba(255,255,255,0.1); padding: 1.2rem; border-radius: 0.8rem; 
                        border-left: 4px solid #00f2fe;">
                <h4 style="color: #ffffff; margin-bottom: 0.8rem; font-size: 1.1rem;">
                    🌐 Giao Diện Web
                </h4>
                <p style="color: #ffffff; margin: 0; font-size: 0.95rem; line-height: 1.5;">
                    Giao diện web tương tác cho dự đoán thời gian thực
                </p>
            </div>
            <div style="background: rgba(255,255,255,0.1); padding: 1.2rem; border-radius: 0.8rem; 
                        border-left: 4px solid #fee140;">
                <h4 style="color: #ffffff; margin-bottom: 0.8rem; font-size: 1.1rem;">
                    ⚡ So Sánh Toàn Diện
                </h4>
                <p style="color: #ffffff; margin: 0; font-size: 0.95rem; line-height: 1.5;">
                    So sánh toàn diện để xác định thuật toán tối ưu
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Algorithm performance summary
    st.subheader("📊 Tóm Tắt Hiệu Suất Thuật Toán")
    
    performance_data = {
        "Thuật Toán": ["Hồi Quy Tuyến Tính", "Cây Quyết Định", "SVM", "Hồi Quy Logistic"],
        "Nhiệm Vụ": ["Hồi Quy", "Hồi Quy", "Phân Loại", "Phân Loại"],
        "Điểm Mạnh": [
            "Đơn giản, dễ diễn giải, huấn luyện nhanh",
            "Mối quan hệ phi tuyến, dễ trực quan hóa",
            "Độ chính xác cao, tốt cho mẫu phức tạp",
            "Đầu ra xác suất, dự đoán nhanh"
        ],
        "Trường Hợp Tốt Nhất": [
            "Ước tính AQI nhanh",
            "Hiểu tầm quan trọng đặc trưng",
            "Phân loại rủi ro cao",
            "Hệ thống cảnh báo thời gian thực"
        ]
    }
    
    df_performance = pd.DataFrame(performance_data)
    st.dataframe(df_performance, use_container_width=True)
    
    # Recommendations
    st.subheader("🥇 Khuyến Nghị")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 1rem; color: #ffffff; 
                    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4); 
                    border: 2px solid #667eea; margin-bottom: 1rem;">
            <h3 style="color: #ffffff; margin-bottom: 1rem; font-size: 1.3rem; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">
                🏆 Độ Chính Xác Cao Nhất
            </h3>
            <p style="font-size: 1.1rem; font-weight: bold; margin-bottom: 1rem; color: #ffffff;">
                <span style="background: rgba(255,255,255,0.9); color: #333333; padding: 0.3rem 0.8rem; 
                           border-radius: 0.5rem; display: inline-block; text-shadow: none;">
                    SVM cho nhiệm vụ phân loại
                </span>
            </p>
            <ul style="list-style: none; padding: 0; margin: 0;">
                <li style="margin-bottom: 0.5rem; padding-left: 1.5rem; position: relative; color: #ffffff;">
                    ✨ F1-score cao nhất
                </li>
                <li style="margin-bottom: 0.5rem; padding-left: 1.5rem; position: relative; color: #ffffff;">
                    🛡️ Chống overfitting tốt
                </li>
                <li style="padding-left: 1.5rem; position: relative; color: #ffffff;">
                    🎯 Tốt cho mẫu phức tạp
                </li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 1.5rem; border-radius: 1rem; color: #ffffff; 
                    box-shadow: 0 4px 15px rgba(240, 147, 251, 0.4); 
                    border: 2px solid #f093fb;">
            <h3 style="color: #ffffff; margin-bottom: 1rem; font-size: 1.3rem; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">
                ⚡ Nhanh Nhất
            </h3>
            <p style="font-size: 1.1rem; font-weight: bold; margin-bottom: 1rem; color: #ffffff;">
                <span style="background: rgba(255,255,255,0.9); color: #333333; padding: 0.3rem 0.8rem; 
                           border-radius: 0.5rem; display: inline-block; text-shadow: none;">
                    Hồi Quy Tuyến Tính cho hồi quy
                </span>
            </p>
            <ul style="list-style: none; padding: 0; margin: 0;">
                <li style="margin-bottom: 0.5rem; padding-left: 1.5rem; position: relative; color: #ffffff;">
                    🚀 Huấn luyện nhanh nhất
                </li>
                <li style="margin-bottom: 0.5rem; padding-left: 1.5rem; position: relative; color: #ffffff;">
                    💻 Yêu cầu tính toán thấp
                </li>
                <li style="padding-left: 1.5rem; position: relative; color: #ffffff;">
                    🎛️ Dễ triển khai
                </li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 1.5rem; border-radius: 1rem; color: #ffffff; 
                    box-shadow: 0 4px 15px rgba(79, 172, 254, 0.4); 
                    border: 2px solid #4facfe; margin-bottom: 1rem;">
            <h3 style="color: #ffffff; margin-bottom: 1rem; font-size: 1.3rem; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">
                🔍 Dễ Diễn Giải Nhất
            </h3>
            <p style="font-size: 1.1rem; font-weight: bold; margin-bottom: 1rem; color: #ffffff;">
                <span style="background: rgba(255,255,255,0.9); color: #333333; padding: 0.3rem 0.8rem; 
                           border-radius: 0.5rem; display: inline-block; text-shadow: none;">
                    Cây Quyết Định cho phân tích
                </span>
            </p>
            <ul style="list-style: none; padding: 0; margin: 0;">
                <li style="margin-bottom: 0.5rem; padding-left: 1.5rem; position: relative; color: #ffffff;">
                    📖 Dễ hiểu
                </li>
                <li style="margin-bottom: 0.5rem; padding-left: 1.5rem; position: relative; color: #ffffff;">
                    🎨 Quy tắc quyết định trực quan
                </li>
                <li style="padding-left: 1.5rem; position: relative; color: #ffffff;">
                    📊 Tầm quan trọng đặc trưng
                </li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                    padding: 1.5rem; border-radius: 1rem; color: #333333; 
                    box-shadow: 0 4px 15px rgba(250, 112, 154, 0.4); 
                    border: 2px solid #fa709a;">
            <h3 style="color: #333333; margin-bottom: 1rem; font-size: 1.3rem; text-shadow: 1px 1px 2px rgba(255,255,255,0.5);">
                🚀 Tốt Cho Sản Xuất
            </h3>
            <p style="font-size: 1.1rem; font-weight: bold; margin-bottom: 1rem; color: #333333;">
                <span style="background: rgba(51,51,51,0.9); color: #ffffff; padding: 0.3rem 0.8rem; 
                           border-radius: 0.5rem; display: inline-block; text-shadow: none;">
                    Hồi Quy Logistic cho triển khai
                </span>
            </p>
            <ul style="list-style: none; padding: 0; margin: 0;">
                <li style="margin-bottom: 0.5rem; padding-left: 1.5rem; position: relative; color: #333333;">
                    ⚡ Hiệu suất ổn định
                </li>
                <li style="margin-bottom: 0.5rem; padding-left: 1.5rem; position: relative; color: #333333;">
                    🔧 Bảo trì thấp
                </li>
                <li style="padding-left: 1.5rem; position: relative; color: #333333;">
                    🎯 Dự đoán đáng tin cậy
                </li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Future improvements
    st.subheader("🔮 Cải Tiến Trong Tương Lai")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 1rem; color: #ffffff; 
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4); 
                border: 2px solid #667eea;">
        <h3 style="color: #ffffff; margin-bottom: 1.5rem; font-size: 1.4rem; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">
            📈 Cải Tiến Tiềm Năng
        </h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem;">
            <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 0.8rem; 
                        border-left: 4px solid #f093fb;">
                <h4 style="color: #ffffff; margin-bottom: 0.5rem; font-size: 1.1rem;">
                    🧠 Học Sâu
                </h4>
                <p style="color: #ffffff; margin: 0; font-size: 0.9rem; line-height: 1.4;">
                    Triển khai LSTM/GRU cho dự đoán chuỗi thời gian
                </p>
            </div>
            <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 0.8rem; 
                        border-left: 4px solid #4facfe;">
                <h4 style="color: #ffffff; margin-bottom: 0.5rem; font-size: 1.1rem;">
                    🎯 Phương Pháp Ensemble
                </h4>
                <p style="color: #ffffff; margin: 0; font-size: 0.9rem; line-height: 1.4;">
                    Kết hợp nhiều mô hình để độ chính xác cao hơn
                </p>
            </div>
            <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 0.8rem; 
                        border-left: 4px solid #fa709a;">
                <h4 style="color: #ffffff; margin-bottom: 0.5rem; font-size: 1.1rem;">
                    📡 Dữ Liệu Thời Gian Thực
                </h4>
                <p style="color: #ffffff; margin: 0; font-size: 0.9rem; line-height: 1.4;">
                    Tích hợp trạm giám sát không khí trực tiếp
                </p>
            </div>
            <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 0.8rem; 
                        border-left: 4px solid #00f2fe;">
                <h4 style="color: #ffffff; margin-bottom: 0.5rem; font-size: 1.1rem;">
                    🗺️ Mở Rộng Địa Lý
                </h4>
                <p style="color: #ffffff; margin: 0; font-size: 0.9rem; line-height: 1.4;">
                    Bao gồm các thành phố khác của Việt Nam
                </p>
            </div>
            <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 0.8rem; 
                        border-left: 4px solid #fee140;">
                <h4 style="color: #ffffff; margin-bottom: 0.5rem; font-size: 1.1rem;">
                    📱 Ứng Dụng Di Động
                </h4>
                <p style="color: #ffffff; margin: 0; font-size: 0.9rem; line-height: 1.4;">
                    Phát triển ứng dụng iOS/Android cho công chúng
                </p>
            </div>
            <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 0.8rem; 
                        border-left: 4px solid #f5576c;">
                <h4 style="color: #ffffff; margin-bottom: 0.5rem; font-size: 1.1rem;">
                    🚨 Hệ Thống Cảnh Báo
                </h4>
                <p style="color: #ffffff; margin: 0; font-size: 0.9rem; line-height: 1.4;">
                    Tự động cảnh báo ô nhiễm
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Final message
    st.markdown("""
    <div style="text-align: center; margin-top: 2rem;">
        <h3>🎉 Cảm ơn bạn đã sử dụng Hệ Thống Dự Đoán Ô Nhiễm Không Khí Hà Nội!</h3>
        <p>Dự án này thể hiện ứng dụng thực tế của học máy trong giám sát môi trường và bảo vệ sức khỏe công chúng.</p>
        <p><strong>Thuật Toán Tốt Nhất:</strong> Dựa trên đánh giá toàn diện, <strong>SVM</strong> cho hiệu suất tối ưu cho phân loại, trong khi <strong>Hồi Quy Tuyến Tính</strong> cung cấp khả năng hồi quy hiệu quả.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
