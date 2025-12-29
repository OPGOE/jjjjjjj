import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# 页面配置
st.set_page_config(
    page_title="销售仪表板",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
.main {
    padding-top: 1rem;
}
.metric-container {
    background: white;
    padding: 1rem;
    border-radius: 0.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    text-align: center;
    margin-bottom: 1rem;
}
.big-number {
    font-size: 2rem;
    font-weight: bold;
    color: #1f77b4;
}
.metric-label {
    font-size: 1rem;
    color: #666;
    margin-bottom: 0.5rem;
}
.filter-tag {
    background-color: #ff4b4b;
    color: white;
    padding: 8px 15px;
    border-radius: 20px;
    margin: 3px;
    display: inline-block;
    font-size: 14px;
    font-weight: 500;
}
.sidebar .sidebar-content {
    background-color: #f0f2f6;
}
/* 隐藏multiselect的默认样式 */
.stMultiSelect > div > div {
    display: none;
}
</style>
""", unsafe_allow_html=True)

# 侧边栏
with st.sidebar:
    st.header("请筛选数据:")
    
    # 城市选择（多选标签）
    st.subheader("请选择城市:")
    
    # 使用checkbox来模拟多选标签
    col1, col2 = st.columns(2)
    
    with col1:
        city_taiyuan = st.checkbox("太原", value=True, key="city_taiyuan")
        city_datong = st.checkbox("大同", value=True, key="city_datong")
        city_changzhi = st.checkbox("长治", value=False, key="city_changzhi")
    
    with col2:
        city_linfen = st.checkbox("临汾", value=True, key="city_linfen")
        city_yuncheng = st.checkbox("运城", value=False, key="city_yuncheng")
        city_jinzhong = st.checkbox("晋中", value=False, key="city_jinzhong")
    
    # 收集选中的城市
    selected_cities = []
    if city_taiyuan: selected_cities.append("太原")
    if city_linfen: selected_cities.append("临汾")
    if city_datong: selected_cities.append("大同")
    if city_yuncheng: selected_cities.append("运城")
    if city_changzhi: selected_cities.append("长治")
    if city_jinzhong: selected_cities.append("晋中")
    
    # 显示选中的城市标签（红色标签样式）
    st.markdown("**已选择的城市:**")
    if selected_cities:
        tag_html = ""
        for city in selected_cities:
            tag_html += f'<span style="background-color: #ff4b4b; color: white; padding: 6px 12px; border-radius: 15px; margin: 3px; display: inline-block; font-size: 12px;">{city} ×</span> '
        st.markdown(tag_html, unsafe_allow_html=True)
    else:
        st.markdown("*未选择任何城市*")
    
    st.markdown("---")
    
    # 顾客类型选择
    st.subheader("请选择顾客类型:")
    
    customer_member = st.checkbox("会员用户", value=True, key="customer_member")
    customer_normal = st.checkbox("普通用户", value=True, key="customer_normal")
    customer_vip = st.checkbox("VIP用户", value=False, key="customer_vip")
    
    # 收集选中的顾客类型
    selected_customer_types = []
    if customer_member: selected_customer_types.append("会员用户")
    if customer_normal: selected_customer_types.append("普通用户")
    if customer_vip: selected_customer_types.append("VIP用户")
    
    # 显示选中的顾客类型标签
    st.markdown("**已选择的顾客类型:**")
    if selected_customer_types:
        customer_tag_html = ""
        for customer_type in selected_customer_types:
            customer_tag_html += f'<span style="background-color: #ff4b4b; color: white; padding: 6px 12px; border-radius: 15px; margin: 3px; display: inline-block; font-size: 12px;">{customer_type} ×</span> '
        st.markdown(customer_tag_html, unsafe_allow_html=True)
    else:
        st.markdown("*未选择任何顾客类型*")
    
    st.markdown("---")
    
    # 性别选择
    st.subheader("请选择性别:")
    
    gender_male = st.checkbox("男性", value=True, key="gender_male")
    gender_female = st.checkbox("女性", value=True, key="gender_female")
    
    # 收集选中的性别
    selected_genders = []
    if gender_male: selected_genders.append("男性")
    if gender_female: selected_genders.append("女性")
    
    # 显示选中的性别标签
    st.markdown("**已选择的性别:**")
    if selected_genders:
        gender_tag_html = ""
        for gender in selected_genders:
            gender_tag_html += f'<span style="background-color: #ff4b4b; color: white; padding: 6px 12px; border-radius: 15px; margin: 3px; display: inline-block; font-size: 12px;">{gender} ×</span> '
        st.markdown(gender_tag_html, unsafe_allow_html=True)
    else:
        st.markdown("*未选择任何性别*")

# 创建模拟数据
@st.cache_data
def create_sample_data():
    # 创建详细的销售数据
    np.random.seed(42)
    n_records = 1000
    
    # 生成基础数据
    cities = ["太原", "临汾", "大同", "运城", "长治", "晋中"]
    customer_types = ["会员用户", "普通用户", "VIP用户"]
    genders = ["男性", "女性"]
    products = ['食品健康', '电子配件', '时尚配饰', '家居园艺', '运动户外', '美容护理']
    hours = ['10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
    
    # 生成详细销售记录
    sales_data = pd.DataFrame({
        '城市': np.random.choice(cities, n_records),
        '顾客类型': np.random.choice(customer_types, n_records),
        '性别': np.random.choice(genders, n_records),
        '产品类型': np.random.choice(products, n_records),
        '小时': np.random.choice(hours, n_records),
        '销售额': np.random.uniform(50, 500, n_records).round(2),
        '数量': np.random.randint(1, 10, n_records)
    })
    
    return sales_data

# 获取数据
sales_data = create_sample_data()

# 根据筛选条件过滤数据
def filter_data(data, selected_cities, selected_customer_types, selected_genders):
    filtered_data = data.copy()
    
    # 城市筛选
    if selected_cities:
        filtered_data = filtered_data[filtered_data['城市'].isin(selected_cities)]
    
    # 顾客类型筛选
    if selected_customer_types:
        filtered_data = filtered_data[filtered_data['顾客类型'].isin(selected_customer_types)]
    
    # 性别筛选
    if selected_genders:
        filtered_data = filtered_data[filtered_data['性别'].isin(selected_genders)]
    
    return filtered_data

# 应用筛选
filtered_sales_data = filter_data(
    sales_data, 
    selected_cities,
    selected_customer_types, 
    selected_genders
)

# 基于筛选后的数据生成图表数据
hour_df = filtered_sales_data.groupby('小时')['销售额'].sum().reset_index()
product_df = filtered_sales_data.groupby('产品类型')['销售额'].sum().reset_index().sort_values('销售额', ascending=False)

# 计算筛选后的指标
total_sales_filtered = filtered_sales_data['销售额'].sum()
avg_sales_filtered = filtered_sales_data['销售额'].mean()
total_records_filtered = len(filtered_sales_data)

# 主标题
st.title("📊 销售仪表板")

# 显示筛选状态
st.info(f"📊 当前显示数据：共 {total_records_filtered} 条记录，总销售额 ¥{total_sales_filtered:,.2f}")

# 显示筛选条件摘要
filter_summary = []
if selected_cities:
    filter_summary.append(f"城市: {', '.join(selected_cities)}")
if selected_customer_types:
    filter_summary.append(f"顾客类型: {', '.join(selected_customer_types)}")
if selected_genders:
    filter_summary.append(f"性别: {', '.join(selected_genders)}")

if filter_summary:
    st.caption(f"🔍 当前筛选条件: {' | '.join(filter_summary)}")
else:
    st.caption("🔍 未应用任何筛选条件")

# 核心指标行
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**总销售额:**")
    st.markdown(f"# RMB ¥ {total_sales_filtered:,.2f}")
    st.caption(f"筛选后记录数：{total_records_filtered}")

with col2:
    st.markdown("**顾客评分的平均值:**")
    st.markdown("# 7.0 ⭐⭐⭐⭐⭐⭐⭐")
    st.caption("基于用户反馈")

with col3:
    st.markdown("**每单的平均销售额:**")
    st.markdown(f"# RMB ¥ {avg_sales_filtered:.2f}")
    st.caption(f"基于 {total_records_filtered} 条记录")

st.divider()

# 图表行
col_left, col_right = st.columns(2)

# 左侧：按小时销售额的柱状图
with col_left:
    st.subheader("按小时销售额的柱状图")
    
    # 使用Streamlit内置柱状图
    chart_data = hour_df.set_index('小时')
    st.bar_chart(chart_data, height=400)

# 右侧：按产品类型销售额的横向柱状图
with col_right:
    st.subheader("按产品类型销售额的柱状图")
    
    # 创建横向显示的数据
    st.dataframe(product_df, use_container_width=True)
    
    # 使用柱状图
    chart_data2 = product_df.set_index('产品类型')
    st.bar_chart(chart_data2, height=400)

# 数据表格展示
st.divider()
col_table1, col_table2 = st.columns(2)

with col_table1:
    st.subheader("📊 小时销售数据")
    st.dataframe(hour_df, use_container_width=True)

with col_table2:
    st.subheader("📊 产品销售数据")
    st.dataframe(product_df, use_container_width=True)

# 添加筛选后的详细数据表
st.divider()
st.subheader("📋 筛选后的详细销售数据")

# 显示前20条记录
display_data = filtered_sales_data.head(20).copy()
display_data['销售额'] = display_data['销售额'].round(2)

st.dataframe(
    display_data, 
    use_container_width=True,
    column_config={
        "销售额": st.column_config.NumberColumn(
            "销售额",
            help="单笔销售金额",
            format="¥%.2f"
        ),
        "数量": st.column_config.NumberColumn(
            "数量",
            help="销售数量"
        )
    }
)

if len(filtered_sales_data) > 20:
    st.caption(f"显示前20条记录，共有 {len(filtered_sales_data)} 条记录符合筛选条件")

# 页脚
st.markdown("---")
st.caption(f"数据更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")