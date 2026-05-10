# -*- coding: utf-8 -*-
# model_trainer.py  ——  修复版
"""
模型训练模块
功能：训练多个机器学习模型并对比性能
输入：data_with_features.csv
输出：models/*.pkl, model_comparison.csv, predictions.csv

修复记录：
  Fix1: 增加 MAPE 指标，评估更全面
  Fix2: 保存 feature_cols.json（供 app.py 对齐特征）
  Fix3: 修复 save_results 中 importance_df 可能 undefined 的 bug
  Fix4: 添加 inf 值处理
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """模型训练器"""

    def __init__(self, data_file='data_with_features.csv', target='pm25'):
        print("=" * 70)
        print("🌬️  时空呼吸 · 模型训练模块")
        print("=" * 70)

        print(f"\n📂 读取数据：{data_file}")
        self.df     = pd.read_csv(data_file)
        self.target = target

        print(f"   ✅ 读取成功，数据维度：{self.df.shape}")

        self._prepare_data()

        self.models      = {}
        self.results     = {}
        self.predictions = {}
        self.importance_df = None     # ✅ Fix3：初始化防止 undefined

    def _prepare_data(self):
        """准备训练数据（时序顺序切分）"""
        print("\n⚙️  准备训练数据...")

        if 'city' in self.df.columns:
            print(f"   检测到 {self.df['city'].nunique()} 个城市/站点")

        # ─── v2 严格反泄漏特征过滤 ──────────────────────
        # 黑名单: 任何"等价于直接给答案"的特征，全部剔除
        # 1) pm25 本身和归一化版本（v1 留下的隐患）
        # 2) BHI 系列（基于 pm25 计算，等价于答案）
        # 3) 元数据（city, source 等）
        STRICT_BLACKLIST = {
            'timestamp', 'city', self.target,
            'time_period', 'season', 'wind_direction',
            'source', 'city_en', 'station', 'unit',
            'latitude', 'longitude',
            # v2 新增黑名单（泄漏特征）：
            'pm25_normalized', 'pm25_global_mean', 'pm25_global_std',
            'pm25_deviation_from_mean',
            'pm25_24h_mean',                   # 现已修为 shift 版，但保留黑名单防回滚
            # BHI 系列：基于当前 pm25 计算，等价于答案
            'bhi', 'bhi_level', 'bhi_ipm', 'bhi_it', 'bhi_ie',
            'breathing_health_index',
        }

        # 额外规则：任何包含 'pm25' 但 *不是* lag/rolling/diff/change 的列
        # （这些前缀的列都是基于历史值的安全特征）
        SAFE_PM25_PREFIXES = ('pm25_lag_', 'pm25_rolling_', 'pm25_diff_',
                              'pm25_pct_change_', 'pm25_change_vs_')

        def is_safe_feature(col):
            if col in STRICT_BLACKLIST:
                return False
            # 含 pm25 的列必须是已知安全前缀
            if 'pm25' in col.lower():
                return col.startswith(SAFE_PM25_PREFIXES)
            return True

        self.feature_cols = [
            col for col in self.df.columns
            if is_safe_feature(col)
            and pd.api.types.is_numeric_dtype(self.df[col])
        ]

        # 双重检查：确保没有特征与 target 完全相关（相关系数 > 0.99 = 泄漏）
        leaked = []
        for col in self.feature_cols:
            try:
                corr = self.df[[col, self.target]].corr().iloc[0, 1]
                if abs(corr) > 0.99:
                    leaked.append((col, corr))
            except Exception:
                pass
        if leaked:
            print(f"\n   🚨 检测到 {len(leaked)} 个高度相关特征（疑似泄漏）：")
            for col, corr in leaked:
                print(f"      {col}: corr = {corr:.4f}  → 已剔除")
            self.feature_cols = [c for c in self.feature_cols if c not in [x[0] for x in leaked]]

        print(f"   特征数量：{len(self.feature_cols)}")

        X = self.df[self.feature_cols]
        y = self.df[self.target]

        # ✅ Fix4：处理 inf 值
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())

        split_idx    = int(len(X) * 0.8)
        self.X_train = X.iloc[:split_idx]
        self.X_test  = X.iloc[split_idx:]
        self.y_train = y.iloc[:split_idx]
        self.y_test  = y.iloc[split_idx:]

        print(f"   训练集：{len(self.X_train):,} 样本（前80%）")
        print(f"   测试集：{len(self.X_test):,} 样本（后20%）")

    def train_ridge(self):
        print("\n🤖 [1/5] 训练 Ridge Regression（基准）...")
        model = Ridge(alpha=1.0, random_state=42)
        model.fit(self.X_train, self.y_train)
        self.models['Ridge'] = model
        self._evaluate_model('Ridge', model)

    def train_random_forest(self):
        print("\n🤖 [2/5] 训练 Random Forest...")
        model = RandomForestRegressor(
            n_estimators=100, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, max_features='sqrt',
            random_state=42, n_jobs=-1,
        )
        model.fit(self.X_train, self.y_train)
        self.models['RandomForest'] = model
        self._evaluate_model('RandomForest', model)

    def train_gradient_boosting(self):
        print("\n🤖 [3/5] 训练 Gradient Boosting...")
        model = GradientBoostingRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            subsample=0.8, min_samples_split=5, min_samples_leaf=2,
            random_state=42,
        )
        model.fit(self.X_train, self.y_train)
        self.models['GradientBoosting'] = model
        self._evaluate_model('GradientBoosting', model)

    def train_xgboost(self):
        print("\n🤖 [4/5] 训练 XGBoost...")
        model = xgb.XGBRegressor(
            objective='reg:squarederror', n_estimators=200,
            max_depth=6, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=3,
            gamma=0.1, reg_alpha=0.1, reg_lambda=1,
            random_state=42, n_jobs=-1, verbosity=0,
        )
        model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_test, self.y_test)],
            verbose=False,
        )
        self.models['XGBoost'] = model
        self._evaluate_model('XGBoost', model)

    def train_lightgbm(self):
        print("\n🤖 [5/5] 训练 LightGBM...")
        model = lgb.LGBMRegressor(
            objective='regression', n_estimators=200,
            max_depth=6, learning_rate=0.05, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20, reg_alpha=0.1, reg_lambda=1,
            random_state=42, n_jobs=-1, verbose=-1,
        )
        model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_test, self.y_test)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        self.models['LightGBM'] = model
        self._evaluate_model('LightGBM', model)

    def _evaluate_model(self, name, model):
        """评估模型，含 MAPE"""
        y_train_pred = model.predict(self.X_train)
        y_test_pred  = model.predict(self.X_test)
        self.predictions[name] = y_test_pred

        # ✅ Fix1：MAPE
        mask_train = self.y_train > 1e-6
        mask_test  = self.y_test  > 1e-6
        mape_train = (
            np.mean(np.abs((self.y_train[mask_train] - y_train_pred[mask_train])
                           / self.y_train[mask_train])) * 100
        )
        mape_test = (
            np.mean(np.abs((self.y_test[mask_test] - y_test_pred[mask_test])
                           / self.y_test[mask_test])) * 100
        )

        results = {
            'train_mae':  mean_absolute_error(self.y_train, y_train_pred),
            'train_rmse': np.sqrt(mean_squared_error(self.y_train, y_train_pred)),
            'train_r2':   r2_score(self.y_train, y_train_pred),
            'train_mape': mape_train,
            'test_mae':   mean_absolute_error(self.y_test, y_test_pred),
            'test_rmse':  np.sqrt(mean_squared_error(self.y_test, y_test_pred)),
            'test_r2':    r2_score(self.y_test, y_test_pred),
            'test_mape':  mape_test,
        }
        self.results[name] = results

        print(f"   训练集 — MAE:{results['train_mae']:>6.2f}  "
              f"RMSE:{results['train_rmse']:>6.2f}  "
              f"R²:{results['train_r2']:>5.3f}  "
              f"MAPE:{results['train_mape']:>5.1f}%")
        print(f"   测试集 — MAE:{results['test_mae']:>6.2f}  "
              f"RMSE:{results['test_rmse']:>6.2f}  "
              f"R²:{results['test_r2']:>5.3f}  "
              f"MAPE:{results['test_mape']:>5.1f}%")

    def train_all_models(self):
        print("\n" + "🚀" * 35)
        print("开始训练所有模型")
        print("🚀" * 35)

        self.train_ridge()
        self.train_random_forest()
        self.train_gradient_boosting()
        self.train_xgboost()
        self.train_lightgbm()

        print("\n" + "=" * 70)
        print("所有模型训练完成")
        print("=" * 70)

    def print_comparison(self):
        print("\n" + "📊" * 35)
        print("模型性能对比")
        print("📊" * 35 + "\n")

        comparison_df = pd.DataFrame(self.results).T
        comparison_df = comparison_df[
            ['test_mae', 'test_rmse', 'test_r2', 'test_mape']
        ]
        comparison_df.columns = ['MAE', 'RMSE', 'R²', 'MAPE%']
        comparison_df = comparison_df.sort_values('R²', ascending=False)

        print(comparison_df.to_string())

        best_model = comparison_df.index[0]
        print(f"\n🏆 最佳模型：{best_model} "
              f"(R² = {comparison_df.loc[best_model, 'R²']:.3f}, "
              f"MAE = {comparison_df.loc[best_model, 'MAE']:.2f})")

        return comparison_df

    def save_models(self):
        print("\n💾 保存模型...")
        os.makedirs('models', exist_ok=True)

        for name, model in self.models.items():
            path = f'models/{name}_model.pkl'
            joblib.dump(model, path)
            print(f"   ✅ {path}")

    def save_results(self):
        print("\n💾 保存结果...")

        # 1. 对比表
        comparison_df = pd.DataFrame(self.results).T
        comparison_df.to_csv('model_comparison.csv')
        print("   ✅ model_comparison.csv")

        # 2. 预测结果（最佳模型）
        best_model = max(self.results.items(), key=lambda x: x[1]['test_r2'])[0]
        predictions_df = pd.DataFrame({
            'true_value':      self.y_test.values,
            'predicted_value': self.predictions[best_model],
            'error':           self.predictions[best_model] - self.y_test.values,
        })
        if 'timestamp' in self.df.columns:
            test_ts = self.df['timestamp'].iloc[-len(self.y_test):]
            predictions_df.insert(0, 'timestamp', test_ts.values)
        predictions_df.to_csv('predictions.csv', index=False)
        print("   ✅ predictions.csv")

        # 3. 特征重要性
        if best_model in ['RandomForest', 'XGBoost', 'LightGBM']:
            model = self.models[best_model]
            if hasattr(model, 'feature_importances_'):
                self.importance_df = pd.DataFrame({
                    'feature':    self.feature_cols,
                    'importance': model.feature_importances_,
                }).sort_values('importance', ascending=False)
                self.importance_df.to_csv('feature_importance.csv', index=False)
                print("   ✅ feature_importance.csv")

        # ✅ Fix2：保存特征列名供 app.py 使用
        os.makedirs('models', exist_ok=True)
        with open('models/feature_cols.json', 'w', encoding='utf-8') as f:
            json.dump(self.feature_cols, f, ensure_ascii=False, indent=2)
        print("   ✅ models/feature_cols.json")


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║           🌬️  时空呼吸 · 模型训练模块                         ║
    ║                 Model Training Module                        ║
    ║                                                               ║
    ║     功能：训练5个机器学习模型并选出最优                         ║
    ║     指标：MAE / RMSE / R² / MAPE（四项指标）                  ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)

    if not os.path.exists('data_with_features.csv'):
        print("❌ 错误：找不到 data_with_features.csv")
        print("   请先运行：python feature_engineer.py")
        return

    trainer = ModelTrainer('data_with_features.csv')
    trainer.train_all_models()
    comparison_df = trainer.print_comparison()
    trainer.save_models()
    trainer.save_results()

    print("\n" + "🎉" * 35)
    print("模型训练全部完成！")
    print("🎉" * 35)

    best = comparison_df.index[0]
    print(f"""
    ✅ 最佳模型：{best}
    📄 model_comparison.csv（含MAPE指标的完整对比）
    📄 predictions.csv
    📄 feature_importance.csv
    📁 models/feature_cols.json（app.py需要）

    🚀 下一步：
       streamlit run app.py
    """)


if __name__ == "__main__":
    main()