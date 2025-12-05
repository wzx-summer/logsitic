import numpy as np
from sklearn.linear_model import v


filename=r"C:\Users\E507\Desktop\logsitic\logistic.py"
#=====================
# 1. 数据读取函数
#=====================
def load_dataset(filename):
    data = np.loadtxt(filename)
    X = data[:, :-1]   # 特征
    y = data[:, -1]    # 标签
    return X, y

#=====================
# 2. 缺失值处理函数
#   （缺失值替换为该列均值）
#=====================
def replace_nan_with_mean(X):
    for i in range(X.shape[1]):
        col = X[:, i]
        # 选择非0的数作为有效特征
        valid = col[col != 0]
        if len(valid) > 0:
            mean_val = np.mean(valid)
            col[col == 0] = mean_val
            X[:, i] = col
    return X

#=====================
# 3. 主流程
#=====================
# 读取训练集
def main():
    X,y=load_data(path)
    print("===使用梯度下降训练 Logistic 回归 ===")
    w_gd=gradient_descent_logistic(X,y,lr=0.001,n_iters=8000)
    print(f"\n[GD] learned w={w_gd}")
    clf=LogisticRegression(solver="lbfgs",max_iter=5000)
    clf.fit(X,y)
    print(f"[sklearn] coef_={clf.coef_},intercept_={clf.intercept_}")
    y_pred_gd=(sigmoid(np.c_[np.ones((X.shape[0],1)),X]@w_gd)>=0.5).astype(int)
    acc_gd=np.mean(y_pred_gd==y)
    y_pred_sk=clf.predict(X)
    acc_sk=np.mean(y_pred_sk==y)
    print(f"\n训练集准确率:")
    print(f"自写 GD Logistic 回归 accuracy={acc_gd:.4f}")
    print(f"sklearn LogisticRegression accuracy={acc_sk:.4f}")
    plot_compare_boumdaries(X,y,w_gd,clf)

# 读取测试集


#=====================
# 4. 构建并训练逻辑回归模型
#=====================


#=====================
# 5. 测试集预测
#=====================


#=====================
# 6. 计算准确率
#=====================

