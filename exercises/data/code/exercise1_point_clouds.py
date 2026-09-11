"""
Exercise 1 - Point Clouds: Geometry and Spread in 2D
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

rng = np.random.default_rng(seed=42)
figures = Path(__file__).resolve().parents[1] / "figures"

means = np.array([[2, 3], [5, 6], [8, 1], [15, 4]])
stds = np.array([[0.8, 2.5], [1.2, 1.9], [0.9, 0.9], [0.5, 2.0]])
n = 100  # Number of points per class
#A
class0 = rng.normal(
    loc=means[0],
    scale=stds[0],
    size=(n, 2)
)
class1 = rng.normal(
    loc=means[1],
    scale=stds[1],
    size=(n, 2)
)
class2 = rng.normal(
    loc=means[2],
    scale=stds[2],
    size=(n, 2)
)
class3 = rng.normal(
    loc=means[3],
    scale=stds[3],
    size=(n, 2)
)

figure0 = plt.scatter(class0[:, 0], class0[:, 1], color='blue', label='Class 0')
figure1 = plt.scatter(class1[:, 0], class1[:, 1], color='orange', label='Class 1')
figure2 = plt.scatter(class2[:, 0], class2[:, 1], color='red', label='Class 2')
figure3 = plt.scatter(class3[:, 0], class3[:, 1], color='green', label='Class 3')
mean0_fig = plt.scatter(means[0][0], means[0][1], color='black', marker='x', s=100, label='Class Mean')
mean1_fig = plt.scatter(means[1][0], means[1][1], color='black', marker='x', s=100)
mean2_fig = plt.scatter(means[2][0], means[2][1], color='black', marker='x', s=100)
mean3_fig = plt.scatter(means[3][0], means[3][1], color='black', marker='x', s=100)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Synthetic Gaussian Data with Different Means and Standard Deviations')
plt.legend()
plt.savefig(figures / 'synthetic_gaussian_data.png', dpi=300, bbox_inches='tight')
xx, yy = np.meshgrid(
    np.linspace(-1, 17, 400),
    np.linspace(-2, 11, 400)
)
grid_points = np.c_[xx.ravel(), yy.ravel()]
distances = np.linalg.norm(
    grid_points[:, np.newaxis, :] - means[np.newaxis, :, :],
    axis=2
)
decision = np.argmin(distances, axis=1)
decision = decision.reshape(xx.shape)
plt.contour(
    xx,
    yy,
    decision,
    levels=[0.5, 1.5, 2.5],
    colors="black",
    linewidths=2,
    linestyles="--"
)
plt.savefig(figures / 'synthetic_gaussian_data_with_line.png', dpi=300, bbox_inches='tight')
plt.show()


# B
scales = [0.5, 1.0, 2.0, 4.0]

def calculate_scaled_classes(scales):
    scaled_classes = []
    for scale in scales:
        class0_scaled = rng.normal(
            loc=means[0],
            scale=stds[0] * scale,
            size=(n, 2)
        )
        class1_scaled = rng.normal(
            loc=means[1],
            scale=stds[1] * scale,
            size=(n, 2)
        )
        class2_scaled = rng.normal(
            loc=means[2],
            scale=stds[2] * scale,
            size=(n, 2)
        )
        class3_scaled = rng.normal(
            loc=means[3],
            scale=stds[3] * scale,
            size=(n, 2)
        )
        scaled_classes.append((class0_scaled, class1_scaled, class2_scaled, class3_scaled))
    return scaled_classes

scaled_classes = calculate_scaled_classes(scales)
s_names = {
    0: 0.5,
    1: 1.0,
    2: 2.0,
    3: 4.0
}
dimensions = {
    0: [0,0],
    1: [0,1],
    2: [1,0],
    3: [1,1]
}

fig2, axes = plt.subplots(2, 2, figsize=(12, 10))
for i in range(len(scaled_classes)):
    axes[dimensions[i][0]][dimensions[i][1]].scatter(scaled_classes[i][0][:,0], scaled_classes[i][0][:,1], color='blue', label='Class 0')
    axes[dimensions[i][0]][dimensions[i][1]].scatter(scaled_classes[i][1][:,0], scaled_classes[i][1][:,1], color='orange', label='Class 1')
    axes[dimensions[i][0]][dimensions[i][1]].scatter(scaled_classes[i][2][:,0], scaled_classes[i][2][:,1], color='red', label='Class 2')
    axes[dimensions[i][0]][dimensions[i][1]].scatter(scaled_classes[i][3][:,0], scaled_classes[i][3][:,1], color='green', label='Class 3')
    axes[dimensions[i][0]][dimensions[i][1]].scatter(means[0][0], means[0][1], color='black', marker='x', s=100, label='Class Mean')
    axes[dimensions[i][0]][dimensions[i][1]].scatter(means[1][0], means[1][1], color='black', marker='x', s=100)
    axes[dimensions[i][0]][dimensions[i][1]].scatter(means[2][0], means[2][1], color='black', marker='x', s=100)
    axes[dimensions[i][0]][dimensions[i][1]].scatter(means[3][0], means[3][1], color='black', marker='x', s=100)
    axes[dimensions[i][0]][dimensions[i][1]].set_title(f"S = {s_names[i]}")
    axes[dimensions[i][0]][dimensions[i][1]].legend()
    axes[dimensions[i][0]][dimensions[i][1]].grid(True)
fig2.suptitle("Synthetic Gaussian Data Scale with All Scale Factors")
fig2.savefig(figures /'synthetic_gaussian_data_scale_all.png', dpi=300, bbox_inches='tight')
plt.show()
# for scale, (class0_scaled, class1_scaled, class2_scaled, class3_scaled) in zip(scales, scaled_classes):
#     plt.figure(figsize=(8, 6))
#     plt.scatter(class0_scaled[:, 0], class0_scaled[:, 1], color='blue', label='Class 0')
#     plt.scatter(class1_scaled[:, 0], class1_scaled[:, 1], color='orange', label='Class 1')
#     plt.scatter(class2_scaled[:, 0], class2_scaled[:, 1], color='red', label='Class 2')
#     plt.scatter(class3_scaled[:, 0], class3_scaled[:, 1], color='green', label='Class 3')
#     plt.scatter(means[0][0], means[0][1], color='black', marker='x', s=100, label='Class Mean')
#     plt.scatter(means[1][0], means[1][1], color='black', marker='x', s=100)
#     plt.scatter(means[2][0], means[2][1], color='black', marker='x', s=100)
#     plt.scatter(means[3][0], means[3][1], color='black', marker='x', s=100)
#     plt.xlabel('Feature 1')
#     plt.ylabel('Feature 2')
#     plt.title(f'Synthetic Gaussian Data with Scale {scale}')
#     plt.legend()
#     plt.savefig(figures / f'synthetic_gaussian_data_scale_{scale}.png', dpi=300, bbox_inches='tight')
#     plt.show()

s = 1

# calculo da razao de separacao de cada par de classes
sigma0 = (stds[0][0] + stds[0][1])/2
sigma1 = (stds[1][0] + stds[1][1])/2
sigma2 = (stds[2][0] + stds[2][1])/2
sigma3 = (stds[3][0] + stds[3][1])/2
r01 = np.sqrt((means[0][0] - means[1][0])**2 + (means[0][1] - means[1][1])**2)/(sigma0 + sigma1)
r02 = np.sqrt((means[0][0] - means[2][0])**2 + (means[0][1] - means[2][1])**2)/(sigma0 + sigma2)
r03 = np.sqrt((means[0][0] - means[3][0])**2 + (means[0][1] - means[3][1])**2)/(sigma0 + sigma3)
r12 = np.sqrt((means[1][0] - means[2][0])**2 + (means[1][1] - means[2][1])**2)/(sigma1 + sigma2)
r13 = np.sqrt((means[1][0] - means[3][0])**2 + (means[1][1] - means[3][1])**2)/(sigma1 + sigma3)
r23 = np.sqrt((means[2][0] - means[3][0])**2+ (means[2][1] - means[3][1])**2)/(sigma2 + sigma3)

rij_min = np.min([r01, r02, r03, r12, r13, r23])
rij_min_s2 = rij_min / 2
print(f"Menor razão de separação entre classes: {rij_min:.4f}")
print(f"Menor razão de separação mínima entre classes (dividida por 2): {rij_min_s2:.4f}")

# calculo da razao de mistura de cada par de classes
mixing_ratios = []
for scale_index in range(4):
    mixed = 0
    total = 0

    classes_of_scale = scaled_classes[scale_index]

    for number_classes in range(len(classes_of_scale)):
        for point in classes_of_scale[number_classes]:
            d0 = np.linalg.norm(point - means[0])
            d1 = np.linalg.norm(point - means[1])
            d2 = np.linalg.norm(point - means[2])
            d3 = np.linalg.norm(point - means[3])

            distances = [d0, d1, d2, d3]

            closest_center = np.argmin(distances)
            if closest_center != number_classes:
                mixed += 1
            
            total += 1

    mixing_ratio = mixed / total
    mixing_ratios.append(mixing_ratio)
    print(f"Escala: {scales[scale_index]}")
    print(f"Razão de mistura entre classes: {mixing_ratio:.4f}")

#figura 3

plt.figure(figsize=(8, 6))
plt.plot(scales, mixing_ratios, marker='o')
plt.xlabel('Escala')
plt.ylabel('Razão de Mistura')
plt.title('Mistura entre Classes vs Escala')
plt.grid(True)
plt.savefig(figures / 'mixing_ratio_vs_scale.png', dpi=300, bbox_inches='tight')
plt.show()