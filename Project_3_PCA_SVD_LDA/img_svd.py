import numpy as np
import matplotlib.pyplot as plt
import cv2

# Read image (greyscale)
img = cv2.imread('dome.jpg', 0)

# Perform SVD
U, S, V = np.linalg.svd(img)

print(U.shape, S.shape, V.shape)

orig_img_size = U.shape[0]

print(U[:, :orig_img_size].shape)
print(np.diag(S[:orig_img_size]).shape)
print(V[:orig_img_size, :].shape)


B = U[:, :orig_img_size].dot(np.diag(S[:orig_img_size]).dot(V[:orig_img_size, :]))
plt.figure(figsize=(8, 8))
plt.imshow(B, cmap='gray'), plt.axis('off'), plt.title("Αρχική εικόνα με " + str(orig_img_size) + " διαστάσεις")
plt.show()

B = U[:, :360].dot(np.diag(S[:360]).dot(V[:360, :]))
plt.figure(figsize=(8, 8))
plt.imshow(B, cmap='gray'), plt.axis('off'), plt.title("Εικόνα με " + str(360) + " διαστάσεις")
plt.show()

B = U[:, :100].dot(np.diag(S[:100]).dot(V[:100, :]))
plt.figure(figsize=(8, 8))
plt.imshow(B, cmap='gray'), plt.axis('off'), plt.title("Εικόνα με " + str(100) + " διαστάσεις")
plt.show()
print("\n")
B = U[:, :10].dot(np.diag(S[:10]).dot(V[:10, :]))
# print(U[:, :10].shape)
# print(np.diag(S[:10]).shape)
# print(V[:10, :].shape)
plt.figure(figsize=(8, 8))
plt.imshow(B, cmap='gray'), plt.axis('off'), plt.title("Εικόνα με " + str(10) + " διαστάσεις")
plt.show()
