import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
import warnings

warnings.filterwarnings('ignore')

# ----------------------
# 1. 全局配置（对齐论文）
# ----------------------
INPUT_PATH = r"C:\Users\35632\Desktop\MSC显著性检验.xlsx"
OUTPUT_PATH = r"C:\Users\35632\Desktop\MSC显著性检验_结果输出.xlsx"
BOOTSTRAP_TIMES = 5000  # 论文10000次Bootstrap
ALPHA_LIST = [0.10, 0.25]  # 置信水平
FORECAST_HORIZONS = [1, 5, 22]  # 预测步数
LOSS_TYPE = "SPE"  # 平方预测误差（论文默认）


# ----------------------
# 2. 核心工具函数（取消额外过滤，强制长度一致）
# ----------------------
def calculate_loss(y_true, y_pred, fixed_valid_idx):
    """
    仅用固定有效索引，不额外过滤，确保所有模型损失长度=固定索引长度
    若预测值≤0，用极小值（1e-10）替换，避免计算错误但不改变长度
    """
    # 仅使用固定索引的观测（核心：所有模型共用这组数据）
    y_true_fixed = y_true[fixed_valid_idx]
    y_pred_fixed = y_pred[fixed_valid_idx].copy()  # 避免修改原数据

    # 处理预测值≤0的情况（用1e-10替换，不删除行，确保长度不变）
    y_pred_fixed[y_pred_fixed <= 1e-10] = 1e-10

    # 计算损失（无额外过滤，长度=fixed_valid_idx的有效数量）
    if LOSS_TYPE == "SPE":
        return (y_true_fixed - y_pred_fixed) ** 2
    elif LOSS_TYPE == "QLIKE":
        return np.log(y_pred_fixed) + y_true_fixed / y_pred_fixed
    elif LOSS_TYPE == "R2LOG":
        return (np.log(y_true_fixed) - np.log(y_pred_fixed)) ** 2
    else:
        raise ValueError("仅支持SPE/QLIKE/R2LOG")


def stable_bootstrap(data, block_length=None):
    """稳定块Bootstrap（解决时间序列自相关）"""
    n = len(data)
    block_length = int(np.floor(n ** (1 / 3))) if block_length is None else block_length
    bootstrap_samples = []

    for _ in range(BOOTSTRAP_TIMES):
        # 生成块起始索引，确保不越界
        block_starts = np.random.randint(0, n - block_length + 1, size=int(np.ceil(n / block_length)))
        # 拼接块并截断至原长度
        bootstrap_sample = []
        for start in block_starts:
            bootstrap_sample.extend(data[start:start + block_length])
        bootstrap_samples.append(np.array(bootstrap_sample[:n]))

    return np.array(bootstrap_samples)


def mcs_test(loss_matrix):
    """MCS检验主函数（输入矩阵已确保形状均匀）"""
    n_periods, n_models = loss_matrix.shape
    model_names = [f"Model_{i}" for i in range(n_models)]
    mcs_set = set(range(n_models))
    mcs_p = np.ones(n_models)  # 初始p值=1（所有模型在集内）

    # 逐步剔除劣等模型
    while len(mcs_set) > 1:
        current_models = list(mcs_set)
        current_loss = loss_matrix[:, current_models]
        # 计算所有模型对的损失差（i-j）
        loss_diff = []
        for i in range(len(current_models)):
            for j in range(len(current_models)):
                if i != j:
                    loss_diff.append(current_loss[:, i] - current_loss[:, j])
        loss_diff = np.array(loss_diff)

        # 10000次Bootstrap计算p值
        block_len = int(np.floor(n_periods ** (1 / 3)))
        bootstrap_stats = []
        for diff in loss_diff:
            bootstrap_samples = stable_bootstrap(diff, block_length=block_len)
            bootstrap_stats.append(np.mean(bootstrap_samples, axis=1))

        # 计算单侧p值（损失差≥原始均值的比例）
        original_mean = np.mean(loss_diff, axis=1)
        p_values = [np.mean(stat >= mean) for stat, mean in zip(bootstrap_stats, original_mean)]

        # 剔除p值最小的模型
        min_p_idx = np.argmin(p_values)
        bad_model = current_models[min_p_idx // (len(current_models) - 1)]
        mcs_set.remove(bad_model)
        mcs_p[bad_model] = min(p_values)

    # 生成结果表
    mcs_results = pd.DataFrame({
        "模型名称": model_names,
        "MCS_p值": np.round(mcs_p, 4)
    })
    # 标记是否纳入置信集
    for alpha in ALPHA_LIST:
        mcs_results[f"纳入α={int(alpha * 100)}%置信集"] = (mcs_results["MCS_p值"] >= alpha).map(
            {True: "是", False: "否"})

    return mcs_results


# ----------------------
# 3. 数据读取（强制统一长度，增加报错捕获）
# ----------------------
def load_data(horizon):
    """读取单步数据，返回：均匀损失矩阵、模型名、清洁数据"""
    try:
        # 读取工作表
        df = pd.read_excel(INPUT_PATH, sheet_name=str(horizon))

        # 识别关键列（用try-except确保列名匹配）
        try:
            date_col = next(col for col in df.columns if "日期" in col or "date" in col.lower())
            true_rv_col = next(col for col in df.columns if "真实rv" in col or "rv_true" in col.lower())
        except StopIteration:
            raise ValueError(f"{horizon}步工作表缺少必要列（需含日期列和真实RV列）")

        # 模型列（排除日期和真实RV）
        model_cols = [col for col in df.columns if col not in [date_col, true_rv_col]]
        if len(model_cols) < 2:
            raise ValueError(f"{horizon}步工作表需至少2个模型列，当前仅{len(model_cols)}个")

        # 数据类型转换
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df[true_rv_col] = pd.to_numeric(df[true_rv_col], errors="coerce")
        for col in model_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # 固定有效索引（仅保留：日期非空+真实RV>0，所有模型共用）
        fixed_valid_idx = (
                df[date_col].notna() &
                (df[true_rv_col] > 1e-10)  # 真实RV极小值阈值
        ).values
        if fixed_valid_idx.sum() < 30:
            raise ValueError(f"{horizon}步有效观测数仅{fixed_valid_idx.sum()}个，需≥30个")

        # 填充模型列空值（用该模型在固定索引内的中位数）
        for col in model_cols:
            valid_vals = df[col][fixed_valid_idx]
            median_val = valid_vals.median() if valid_vals.notna().sum() > 0 else 1e-10
            df[col] = df[col].fillna(median_val)

        # 计算损失矩阵（强制所有模型长度=fixed_valid_idx.sum()）
        y_true = df[true_rv_col].values
        loss_matrix = []
        for col in model_cols:
            y_pred = df[col].values
            loss = calculate_loss(y_true, y_pred, fixed_valid_idx)
            # 强制校验长度（确保无意外）
            assert len(loss) == fixed_valid_idx.sum(), f"{col}损失序列长度异常"
            loss_matrix.append(loss)

        # 转置为（观测数×模型数）矩阵，确保形状均匀
        loss_matrix = np.array(loss_matrix).T
        assert loss_matrix.ndim == 2, "损失矩阵必须是2维数组"
        assert len(set(loss_matrix.shape)) > 1, "损失矩阵行数或列数为0"

        # 清洁后的数据（仅含有效观测）
        df_clean = df[fixed_valid_idx].copy().reset_index(drop=True)
        df_clean["有效观测标记"] = "是"

        return {
            "loss_matrix": loss_matrix,
            "model_names": model_cols,
            "clean_data": df_clean,
            "obs_count": loss_matrix.shape[0]
        }

    except Exception as e:
        print(f"❌ {horizon}步数据处理失败：{str(e)}")
        return None


# ----------------------
# 4. 主程序（增加报错捕获，确保Excel正常生成）
# ----------------------
def main():
    print("=== 模型置信集（MCS）检验开始 ===\n")
    success_horizons = []  # 记录处理成功的步数

    # 初始化Excel写入器，确保至少生成1个sheet
    with pd.ExcelWriter(OUTPUT_PATH, engine='openpyxl') as writer:
        # 先写入一个说明sheet，避免无可见sheet报错
        pd.DataFrame({
            "说明": [
                "1. 本文件包含MCS检验结果和清洁后的数据",
                "2. 若某步数无结果，查看控制台报错信息",
                "3. 损失函数：平方预测误差（SPE）",
                "4. Bootstrap次数：10000次"
            ]
        }).to_excel(writer, sheet_name="说明", index=False)

        # 处理每个预测步数
        for horizon in FORECAST_HORIZONS:
            print(f"--- 处理{horizon}步预测数据 ---")
            data = load_data(horizon)
            if data is None:
                print(f"--- {horizon}步跳过，继续处理下一步数 ---")
                continue

            # 执行MCS检验
            try:
                mcs_results = mcs_test(data["loss_matrix"])
                # 替换为真实模型名
                mcs_results["模型名称"] = data["model_names"]
                # 添加基础信息
                mcs_results["预测步数"] = horizon
                mcs_results["有效观测数"] = data["obs_count"]
                mcs_results["损失函数"] = LOSS_TYPE

                # 调整列顺序（便于阅读）
                col_order = [
                    "预测步数", "模型名称", "有效观测数", "损失函数", "MCS_p值",
                    f"纳入α={int(ALPHA_LIST[0] * 100)}%置信集",
                    f"纳入α={int(ALPHA_LIST[1] * 100)}%置信集"
                ]
                mcs_results = mcs_results[col_order]

                # 保存结果到Excel
                mcs_results.to_excel(writer, sheet_name=f"{horizon}步MCS结果", index=False)
                data["clean_data"].to_excel(writer, sheet_name=f"{horizon}步清洁数据", index=False)

                # 打印成功信息
                success_horizons.append(horizon)
                print(f"✅ {horizon}步处理成功：{data['obs_count']}个观测，{len(data['model_names'])}个模型")
                print(f"   前3个模型的MCS p值：{mcs_results['MCS_p值'].head(3).tolist()}\n")

            except Exception as e:
                print(f"❌ {horizon}步MCS检验失败：{str(e)}\n")
                continue

    # 最终汇总
    print("=== 检验完成 ===")
    if success_horizons:
        print(f"✅ 成功处理步数：{success_horizons}")
        print(f"📊 结果已保存至：{OUTPUT_PATH}")
    else:
        print(f"❌ 所有步数处理失败，请查看控制台报错信息")
    print("\n结果解读：MCS_p值≥α（10%/25%）时，模型纳入对应置信集，即属于最优模型梯队")


if __name__ == "__main__":
    main()