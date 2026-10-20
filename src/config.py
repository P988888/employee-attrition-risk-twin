"""项目全局配置：路径、字段分组、模型超参。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUTPUT = ROOT / "output"
FIGURES = OUTPUT / "figures"
MODELS = OUTPUT / "models"
DASHBOARD = OUTPUT / "dashboard"

for _d in (DATA, OUTPUT, FIGURES, MODELS, DASHBOARD):
    _d.mkdir(parents=True, exist_ok=True)

RAW_CSV = DATA / "ibm_hr_attrition_raw.csv"
SIM_CSV = DATA / "simulated_employees.csv"
MERGED_CSV = DATA / "hr_merged.csv"
SCORE_CSV = OUTPUT / "employee_risk_scores.csv"
DASH_JSON = DASHBOARD / "data.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# 三段式划分：训练 / 验证 / 测试
# 验证集用于「选模型 + 定阈值」，测试集只在最后评估时使用一次，
# 避免用测试集调阈值造成的乐观偏差。三者合计 0.6 / 0.2 / 0.2。
VAL_SIZE = 0.2
TEST_SIZE_FINAL = 0.2

# 唯一值恒为 1 的列，对建模无信息量
CONSTANT_COLS = ["EmployeeCount", "Over18", "StandardHours"]

# 直接泄漏/无意义的 ID 列
DROP_COLS = CONSTANT_COLS + ["EmployeeNumber"]

TARGET = "Attrition"

NUMERIC_COLS = [
    "Age", "DailyRate", "DistanceFromHome", "Education", "HourlyRate",
    "JobInvolvement", "JobLevel", "JobSatisfaction", "MonthlyIncome",
    "MonthlyRate", "NumCompaniesWorked", "PercentSalaryHike",
    "PerformanceRating", "RelationshipSatisfaction", "StockOptionLevel",
    "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance",
    "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]

CATEGORICAL_COLS = [
    "BusinessTravel", "Department", "EducationField", "Gender",
    "JobRole", "MaritalStatus", "OverTime",
]

# 派生特征（特征工程）
ENGINEERED_COLS = [
    "IncomePerLevel",        # 薪酬相对职级的水平
    "TenureRatio",           # 当前司龄 / 总工龄
    "PromotionGap",          # 距上次晋升年限占司龄比
    "ManagerStability",      # 与现任经理共事年限占司龄比
    "SatisfactionScore",     # 满意度综合分
    "CareerStagnation",      # 职业停滞指数
    "CompanyHopRate",        # 跳槽频率
]
