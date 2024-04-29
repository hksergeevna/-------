# Імпорт необхідних бібліотек
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Згенеруємо уявлені дані
# Припустимо, що у нас є дані про рівень освіти (X) та ВВП (y)
# Для цього прикладу ми просто згенеруємо випадкові дані
np.random.seed(0)
X = 10 * np.random.rand(100, 1)  # Рівень освіти
y = 3 * X + np.random.randn(100, 1)  # ВВП, з деяким шумом

# Розділимо дані на навчальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Ініціалізуємо та навчаємо модель лінійної регресії
model = LinearRegression()
model.fit(X_train, y_train)

# Зробимо прогнози для тестового набору
y_pred = model.predict(X_test)

# Оцінимо якість моделі за допомогою середньоквадратичної помилки та коефіцієнта детермінації
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Середньоквадратична помилка (MSE):", mse)
print("Коефіцієнт детермінації (R^2):", r2)

# Зробимо прогноз для нових даних
new_X = np.array([[5], [7]])  # Нові дані про рівень освіти
predicted_y = model.predict(new_X)
print("Прогнозоване ВВП для нових даних:", predicted_y)
