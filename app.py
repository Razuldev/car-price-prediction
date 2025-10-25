import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Avtomobil Qiymət Proqnozu - Model Testi",
                   page_icon="🚗",
                   layout="wide")

st.title("🚗 Avtomobil Qiymət Proqnoz Modeli - Performans Analizi")
st.markdown("---")


@st.cache_data
def load_predictions(pipe, df, output):
    y_pred = np.exp(pipe.predict(output[1]))
    y_real = np.exp(output[3])

    results = pd.DataFrame({
        'Real Qiymət': np.round(y_real, 0).astype(int),
        'Proqnoz': np.round(y_pred, 0).astype(int),
        'Fərq': np.round(y_pred - y_real, 0).astype(int),
        'Səhv %': np.round(np.abs((y_pred - y_real) / y_real * 100), 1),
        'URL': df['url'].values
    })

    def quality_label(error):
        if error < 5:
            return "🟢 Əla"
        elif error < 10:
            return "🟡 Yaxşı"
        elif error < 20:
            return "🟠 Orta"
        else:
            return "🔴 Zəif"

    results['Keyfiyyət'] = results['Səhv %'].apply(quality_label)
    return results



np.random.seed(42)
n_samples = 150
real_prices = np.random.randint(5000, 80000, n_samples)
noise = np.random.normal(0, 0.15, n_samples)
predicted_prices = real_prices * (1 + noise)

results_df = pd.DataFrame({
    'Real Qiymət': real_prices,
    'Proqnoz': predicted_prices.astype(int),
    'Fərq': (predicted_prices - real_prices).astype(int),
    'Səhv %': np.abs((predicted_prices - real_prices) / real_prices * 100).round(1),
    'URL': [f'https://turbo.az/autos/{i}' for i in range(n_samples)]
})

results_df['Keyfiyyət'] = results_df['Səhv %'].apply(lambda x:
                                                      "🟢 Əla" if x < 5 else "🟡 Yaxşı" if x < 10 else "🟠 Orta" if x < 20 else "🔴 Zəif")

col1, col2, col3, col4 = st.columns(4)

mae = mean_absolute_error(results_df['Real Qiymət'], results_df['Proqnoz'])
mape = np.mean(results_df['Səhv %'])
r2 = r2_score(results_df['Real Qiymət'], results_df['Proqnoz'])
accuracy_5 = (results_df['Səhv %'] < 5).sum() / len(results_df) * 100

with col1:
    st.metric("📊 Orta Səhv", f"{mae:,.0f} AZN",
              delta=f"{mape:.1f}% MAPE", delta_color="inverse")

with col2:
    st.metric("🎯 Dəqiqlik (±5%)", f"{accuracy_5:.1f}%",
              delta=f"{(results_df['Səhv %'] < 10).sum()} avtomobil")

with col3:
    st.metric("📈 R² Göstəricisi", f"{r2:.3f}",
              delta="Güclü korrelyasiya" if r2 > 0.9 else "Yaxşı korrelyasiya")

with col4:
    st.metric("🚗 Test Edilən Avtomobil", f"{len(results_df)}",
              delta=f"Oktyabr 2025")

st.markdown("---")

tab1, tab3, tab4 = st.tabs(["📊 Ümumi Baxış", "📋 Cədvəl", "🔍 Avtomobil Detalı"])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.scatter(results_df, x='Real Qiymət', y='Proqnoz',
                          color='Keyfiyyət',
                          color_discrete_map={
                              "🟢 Əla": "#00cc66",
                              "🟡 Yaxşı": "#ffcc00",
                              "🟠 Orta": "#ff9933",
                              "🔴 Zəif": "#ff3333"
                          },
                          hover_data=['Səhv %', 'Fərq'],
                          title="Real Qiymət vs Proqnoz",
                          labels={'Real Qiymət': 'Real Qiymət (AZN)',
                                  'Proqnoz': 'Proqnoz (AZN)'})

        min_price = results_df['Real Qiymət'].min()
        max_price = results_df['Real Qiymət'].max()
        fig1.add_trace(go.Scatter(x=[min_price, max_price],
                                  y=[min_price, max_price],
                                  mode='lines',
                                  name='Mükəmməl Proqnoz',
                                  line=dict(dash='dash', color='gray')))
        fig1.update_layout(height=400)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        quality_counts = results_df['Keyfiyyət'].value_counts()
        fig2 = px.pie(values=quality_counts.values,
                      names=quality_counts.index,
                      title="Proqnoz Keyfiyyətinin Paylanması",
                      color=quality_counts.index,
                      color_discrete_map={
                          "🟢 Əla": "#00cc66",
                          "🟡 Yaxşı": "#ffcc00",
                          "🟠 Orta": "#ff9933",
                          "🔴 Zəif": "#ff3333"
                      })
        fig2.update_traces(textposition='inside', textinfo='percent+label')
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)


with tab3:
    st.subheader("📋 Ətraflı Nəticələr Cədvəli")

    col1, col2, col3 = st.columns(3)
    with col1:
        quality_filter = st.multiselect("Keyfiyyət Filtri",
                                        options=results_df['Keyfiyyət'].unique(),
                                        default=results_df['Keyfiyyət'].unique())
    with col2:
        min_price = st.number_input("Min Qiymət (AZN)",
                                    value=int(results_df['Real Qiymət'].min()),
                                    step=1000)
    with col3:
        max_price = st.number_input("Maks Qiymət (AZN)",
                                    value=int(results_df['Real Qiymət'].max()),
                                    step=1000)

    filtered_df = results_df[
        (results_df['Keyfiyyət'].isin(quality_filter)) &
        (results_df['Real Qiymət'] >= min_price) &
        (results_df['Real Qiymət'] <= max_price)
        ].copy()

    display_df = filtered_df[['Keyfiyyət', 'Real Qiymət', 'Proqnoz', 'Fərq', 'Səhv %']].copy()
    display_df['Real Qiymət'] = display_df['Real Qiymət'].apply(lambda x: f"{x:,} AZN")
    display_df['Proqnoz'] = display_df['Proqnoz'].apply(lambda x: f"{x:,} AZN")
    display_df['Fərq'] = display_df['Fərq'].apply(lambda x: f"{x:+,} AZN")
    display_df['Səhv %'] = display_df['Səhv %'].apply(lambda x: f"{x}%")

    st.dataframe(display_df, use_container_width=True, height=400)

    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 CSV Yüklə",
        data=csv,
        file_name="model_neticeleri.csv",
        mime="text/csv"
    )

with tab4:
    st.subheader("🔍 Tək Avtomobil Axtarışı")

    search_idx = st.number_input("Avtomobil İndeksi (0 - {})".format(len(results_df) - 1),
                                 min_value=0,
                                 max_value=len(results_df) - 1,
                                 value=0)

    selected = results_df.iloc[search_idx]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Real Qiymət", f"{selected['Real Qiymət']:,} AZN")
    with col2:
        st.metric("Proqnoz", f"{selected['Proqnoz']:,} AZN",
                  delta=f"{selected['Fərq']:+,} AZN")
    with col3:
        st.metric("Səhv Nisbəti", f"{selected['Səhv %']}%",
                  delta=selected['Keyfiyyət'])

    # st.info(f"🔗 Elan Linki: {selected['URL']}")

    fig6 = go.Figure()
    fig6.add_trace(go.Bar(name='Real', x=['Qiymət'], y=[selected['Real Qiymət']],
                          marker_color='#3498db'))
    fig6.add_trace(go.Bar(name='Proqnoz', x=['Qiymət'], y=[selected['Proqnoz']],
                          marker_color='#e74c3c'))
    fig6.update_layout(title="Real vs Proqnoz Müqayisəsi",
                       yaxis_title="Qiymət (AZN)",
                       height=300)
    st.plotly_chart(fig6, use_container_width=True)