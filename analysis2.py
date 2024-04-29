import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Генеруємо синтетичні дані для прикладу
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Розділяємо дані на тренувальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Навчаємо модель лінійної регресії на тренувальних даних
model = LinearRegression()
model.fit(X_train, y_train)

# Перевіряємо точність моделі на тестових даних
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# Прогнозуємо значення на тестовому наборі
y_pred = model.predict(X_test)

# Візуалізація результатів
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression')
plt.show()
