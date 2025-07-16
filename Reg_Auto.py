import streamlit as st
import pandas as pd
import numpy as np
import itertools
import time
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.ar_model import ar_select_order
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.spatial.distance import mahalanobis
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pathlib import Path

st.set_page_config(layout="wide")
left, center, right = st.columns([1, 8, 1])

with center:

    tab1, tab2, tab3 = st.tabs(["단순 OLS", "SARIMAX(잔차의 AR보정)", "횡단유사국면"])

    with tab1:
        st.title("📊 다변량 회귀 분석 (OLS)")

        uploaded_file = st.file_uploader("엑셀 파일 업로드", type=["xlsx"], key="ols_file")
        if uploaded_file:
            df = pd.read_excel(uploaded_file)
            df.columns = df.columns.str.strip()
            df.dropna(inplace=True)

            date_col = next((col for col in df.columns if "date" in col.lower()), None)
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col])
                df.sort_values(by=date_col, inplace=True)
                df.set_index(date_col, inplace=True)

                min_date = df.index.min()
                max_date = df.index.max()
                st.markdown(f"🕒 데이터의 시작 날짜는 **{min_date.date()}**, 최신 날짜는 **{max_date.date()}**입니다.")
                start_date = st.date_input("분석 시작일", min_value=min_date.date(), max_value=max_date.date(),
                                           value=min_date.date(), key="ols_start")
                end_date = st.date_input("분석 종료일", min_value=min_date.date(), max_value=max_date.date(),
                                         value=max_date.date(), key="ols_end")

                df = df.loc[start_date:end_date]

            target = st.selectbox("Target 변수 선택", df.columns, key="ols_target")
            possible_inputs = [col for col in df.columns if col != target]
            excluded = st.multiselect("분석에서 제외할 변수 선택", possible_inputs, key="ols_exclude")
            valid_inputs = [col for col in possible_inputs if col not in excluded]

            k = st.number_input("입력 변수 조합 수 (k)", min_value=1, max_value=len(valid_inputs), value=2, key="ols_k")
            r2_threshold = st.slider("최소 R-squared 임계값", 0.0, 1.0, 0.7, key="ols_r2")

            if st.button("OLS 회귀분석 실행"):
                results = []

                combinations = list(itertools.combinations(valid_inputs, k))
                total = len(combinations)
                progress_bar = st.progress(0)
                progress_text = st.empty()

                start_time = time.time()

                for i, combo in enumerate(combinations):
                    try:
                        subset_df = df[[target] + list(combo)].dropna()
                        if subset_df.empty:
                            continue

                        X = sm.add_constant(subset_df[list(combo)])
                        y = subset_df[target]
                        model = sm.OLS(y, X).fit()

                        r2 = model.rsquared
                        pvalues = model.pvalues[1:]
                        vif = [variance_inflation_factor(X.values, j) for j in range(1, X.shape[1])]
                        dw = sm.stats.durbin_watson(model.resid)

                        if r2 >= r2_threshold and all(p < 0.05 for p in pvalues) and all(v < 10 for v in vif):
                            last_actual = y.iloc[-1]
                            last_pred = model.predict(X).iloc[-1]
                            effective_start = subset_df.index.min().date()

                            results.append({
                                "입력변수": ", ".join(combo),
                                "R-squared": round(r2, 4),
                                "Durbin-Watson": round(dw, 3),
                                "계수": str({k: round(v, 4) for k, v in model.params.items()}),
                                "분석 시작일": str(effective_start),
                                "마지막 실제값": round(last_actual, 4),
                                "마지막 예측값": round(last_pred, 4)
                            })
                    except Exception as e:
                        st.warning(f"조합 {combo}에서 오류 발생: {e}")
                    finally:
                        progress = (i + 1) / total
                        elapsed = time.time() - start_time
                        avg_per_iter = elapsed / (i + 1)
                        eta = avg_per_iter * (total - (i + 1))
                        mins, secs = divmod(int(eta), 60)
                        progress_text.text(f"진행률: {int(progress * 100)}% ⏳ 예상 남은 시간: {mins}분 {secs}초")
                        progress_bar.progress(progress)

                if results:
                    df_results = pd.DataFrame(results)
                    df_results = df_results.sort_values(by="R-squared", ascending=False).reset_index(drop=True)
                    st.success(f"{len(df_results)}개의 유효한 OLS 모델이 발견되었습니다.")
                    st.dataframe(df_results)
                else:
                    st.error("조건을 만족하는 모델이 없습니다.")
        else:
            st.info("엑셀 파일을 먼저 업로드해 주세요.")

    with tab2:

        st.title("📊 다변량 회귀 분석 (SARIMAX 기반, AR 자동 포함)")

        uploaded_file = st.file_uploader("엑셀 파일 업로드", type=["xlsx"])
        if uploaded_file:
            df = pd.read_excel(uploaded_file)
            df.columns = df.columns.str.strip()
            df.dropna(inplace=True)

            date_col = None
            for col in df.columns:
                if "date" in col.lower():
                    date_col = col
                    break

            if date_col:
                df[date_col] = pd.to_datetime(df[date_col])
                df.sort_values(by=date_col, inplace=True)
                df.set_index(date_col, inplace=True)

                try:
                    inferred_freq = pd.infer_freq(df.index)
                    if inferred_freq:
                        df = df.asfreq(inferred_freq)
                except:
                    pass

                min_date = df.index.min()
                max_date = df.index.max()
                st.markdown(f"🕒 데이터의 시작 날짜는 **{min_date.date()}**, 최신 날짜는 **{max_date.date()}**입니다.")
                start_date = st.date_input("분석 시작일", min_value=min_date.date(), max_value=max_date.date(),
                                           value=min_date.date())
                end_date = st.date_input("분석 종료일", min_value=min_date.date(), max_value=max_date.date(),
                                         value=max_date.date())

                df = df.loc[start_date:end_date]

            target = st.selectbox("Target 변수 선택", df.columns)
            possible_inputs = [col for col in df.columns if col != target]
            excluded = st.multiselect("분석에서 제외할 변수 선택", possible_inputs)
            valid_inputs = [col for col in possible_inputs if col not in excluded]

            k = st.number_input("입력 변수 조합 수 (k)", min_value=1, max_value=len(valid_inputs), value=2)
            r2_threshold = st.slider("최소 R-squared 임계값", 0.0, 1.0, 0.7)

            if st.button("회귀분석 실행"):
                st.info("모든 조합에 대해 SARIMAX 모델을 적합합니다. AR(p) 차수는 자동 선택됩니다.")
                results = []

                combinations = list(itertools.combinations(valid_inputs, k))
                total = len(combinations)
                progress_bar = st.progress(0)
                progress_text = st.empty()

                start_time = time.time()

                for i, combo in enumerate(combinations):
                    try:
                        subset_df = df[[target] + list(combo)].dropna()
                        if subset_df.empty:
                            continue

                        X = sm.add_constant(subset_df[list(combo)])
                        y = subset_df[target]
                        effective_start = subset_df.index.min().date()

                        ar_order = ar_select_order(y, maxlag=5, ic='aic', old_names=False).ar_lags
                        p = max(ar_order) if ar_order is not None and len(ar_order) > 0 else 0

                        model = SARIMAX(endog=y, exog=X, order=(p, 0, 0),
                                        enforce_stationarity=False, enforce_invertibility=False)
                        fit = model.fit(disp=False)

                        if not fit.mle_retvals.get("converged", True):
                            continue  # 수렴 실패한 모델은 제외

                        y_pred = fit.fittedvalues
                        ss_res = ((y - y_pred) ** 2).sum()
                        ss_tot = ((y - y.mean()) ** 2).sum()
                        r2_posthoc = 1 - ss_res / ss_tot

                        pvalues = fit.pvalues
                        significant = all([p < 0.05 for name, p in pvalues.items() if name != 'const'])

                        vif = [variance_inflation_factor(X.values, j) for j in range(1, X.shape[1])]
                        no_multicollinearity = all(v < 10 for v in vif)

                        if r2_posthoc >= r2_threshold and significant and no_multicollinearity:
                            last_actual = y.iloc[-1]
                            last_pred = y_pred.iloc[-1]

                            results.append({
                                "입력변수": ", ".join(combo),
                                "AR 차수": f"AR({p})",
                                "R-squared": round(r2_posthoc, 4),
                                "AIC": round(fit.aic, 2),
                                "BIC": round(fit.bic, 2),
                                "계수": str({k: round(v, 4) for k, v in fit.params.items()}),
                                "분석 시작일": str(effective_start),
                                "마지막 실제값": round(last_actual, 4),
                                "마지막 예측값": round(last_pred, 4)
                            })
                    except Exception as e:
                        st.warning(f"조합 {combo}에서 오류 발생: {e}")
                    finally:
                        progress = (i + 1) / total
                        elapsed = time.time() - start_time
                        avg_per_iter = elapsed / (i + 1)
                        eta = avg_per_iter * (total - (i + 1))
                        mins, secs = divmod(int(eta), 60)
                        progress_text.text(f"진행률: {int(progress * 100)}% ⏳ 예상 남은 시간: {mins}분 {secs}초")
                        progress_bar.progress(progress)

                if results:
                    df_results = pd.DataFrame(results)
                    df_results = df_results.sort_values(by="R-squared", ascending=False).reset_index(drop=True)
                    st.success(f"{len(df_results)}개의 유효한 모델이 발견되었습니다. R-squared 기준 내림차순으로 정렬됨.")
                    st.dataframe(df_results)
                else:
                    st.error("조건을 만족하는 모델이 없습니다.")
        else:
            st.info("엑셀 파일을 먼저 업로드해 주세요.")

    with tab3:

        st.title("🔎 횡단 유사국면 분석 (Z-score 기반)")

        uploaded_file = st.file_uploader("📁 엑셀 파일 업로드", type=["xlsx"], key="z_file")
        if uploaded_file:
            df_raw = pd.read_excel(uploaded_file, sheet_name=0)

            # DATE 컬럼 처리
            date_col = next(c for c in df_raw.columns if str(c).lower().startswith("date"))
            df_raw[date_col] = pd.to_datetime(df_raw[date_col])
            df_raw = df_raw.sort_values(date_col).set_index(date_col)

            # 숫자형만 추출
            df = df_raw.select_dtypes(include=[np.number]).dropna(axis=1, how="all")

            # 기준일 선택
            max_date = df.index.max()
            selected_date = st.date_input("🔍 분석 기준일", value=max_date, min_value=df.index.min(), max_value=max_date)

            # 윈도우 설정
            window_years = st.slider("⏳ Z-score 분석기간 (년)", 5, 15, 10)
            window_months = window_years * 12

            # 제외 변수
            exclude_cols = st.multiselect("❌ 분석에서 제외할 변수", df.columns)
            used_cols = [c for c in df.columns if c not in exclude_cols]
            df = df[used_cols]

            # Top-N 개수 설정
            top_n = st.slider("🎯 유사 시점 개수 (Top N)", 5, 30, 10)

            # 실행 버튼
            if st.button("📌 유사도 분석 실행"):
                roll_mean = df.rolling(window=window_months, min_periods=window_months).mean()
                roll_std = df.rolling(window=window_months, min_periods=window_months).std(ddof=0)
                z = (df - roll_mean) / roll_std
                z = z.dropna()

                # 기준 시점 벡터 추출
                current_date = pd.Timestamp(selected_date)
                if current_date not in z.index:
                    st.warning("선택한 날짜가 Z-score 계산 결과에 없음 → 가장 가까운 이전 날짜로 자동 선택됩니다.")
                    current_date = z.index[z.index < current_date][-1]

                current_vec = z.loc[current_date].to_numpy()
                hist_z = z[z.index < current_date - pd.DateOffset(months=12)]
                hist_mat = hist_z.to_numpy()

                # Mahalanobis 거리
                try:
                    cov_inv = np.linalg.pinv(np.cov(z.T, ddof=0))


                    def mahalanobis_all(mat, vec, cov_inv):
                        diff = mat - vec
                        return np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))


                    maha_distances = mahalanobis_all(hist_mat, current_vec, cov_inv)
                    series_maha = pd.Series(maha_distances, index=hist_z.index, name="Mahalanobis")

                    # Euclidean 거리
                    eucl_distances = np.linalg.norm(hist_mat - current_vec, axis=1)
                    series_eucl = pd.Series(eucl_distances, index=hist_z.index, name="Euclidean")

                    # Top-N 시점
                    top_maha = series_maha.nsmallest(top_n)
                    top_eucl = series_eucl.nsmallest(top_n)

                    # 공통 시점
                    common_dates = sorted(set(top_maha.index) & set(top_eucl.index))
                    heat_df = z.loc[common_dates + [current_date]]
                    heat_df.index = heat_df.index.strftime("%Y-%m-%d")

                    # 히트맵
                    st.markdown("### 🔥 Z-Score Heatmap – 공통 유사시점 vs 현재")
                    import matplotlib.pyplot as plt
                    import seaborn as sns

                    fig, ax = plt.subplots(figsize=(14, 6))
                    sns.heatmap(
                        heat_df.T, annot=True, fmt=".2f",
                        cmap="RdBu_r", center=0, linewidths=.5,
                        cbar_kws={"label": "Z-score"}, ax=ax
                    )
                    ax.set_title("Z-score Heatmap – Similar Dates vs Current", fontsize=14)
                    st.pyplot(fig)

                    # 결과 테이블
                    st.markdown("### 📋 공통 유사 시점 리스트")
                    result_table = pd.DataFrame({
                        "Mahalanobis": top_maha.loc[common_dates].round(4),
                        "Euclidean": top_eucl.loc[common_dates].round(4)
                    }).reset_index().rename(columns={"index": "Date"})
                    st.dataframe(result_table)

                except Exception as e:
                    st.error(f"Mahalanobis 거리 계산 중 오류 발생: {e}")


