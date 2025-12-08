import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the left and right images
left_img = cv2.imread('left.JPG')
right_img = cv2.imread('right.JPG')

# Convert to grayscale
left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

print("Images loaded successfully!")
print(f"Left image shape: {left_img.shape}")
print(f"Right image shape: {right_img.shape}")

# Initialize ORB detector
orb = cv2.ORB_create(nfeatures=1000)

# Detect keypoints and compute descriptors
keypoints1, descriptors1 = orb.detectAndCompute(left_gray, None)
keypoints2, descriptors2 = orb.detectAndCompute(right_gray, None)

print(f"\nDetected {len(keypoints1)} keypoints in left image")
print(f"Detected {len(keypoints2)} keypoints in right image")

# Create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors
matches = bf.match(descriptors1, descriptors2)

# Sort matches by distance (best matches first)
matches = sorted(matches, key=lambda x: x.distance)

# Keep only the best matches (top 50)
num_good_matches = min(50, len(matches))
good_matches = matches[:num_good_matches]

print(f"\nTotal matches found: {len(matches)}")
print(f"Using top {num_good_matches} matches for triangulation")

# Draw matches
img_matches = cv2.drawMatches(left_img, keypoints1, right_img, keypoints2, 
                               good_matches, None, 
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Save the matched keypoints visualization
cv2.imwrite('matched_keypoints.jpg', img_matches)
print("\nMatched keypoints saved as 'matched_keypoints.jpg'")

# Display the matches
plt.figure(figsize=(15, 8))
plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
plt.title('Matched Keypoints between Left and Right Images')
plt.axis('off')
plt.tight_layout()
plt.savefig('matched_keypoints_display.png', dpi=150, bbox_inches='tight')
plt.show()

# Extract matched keypoints
pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

# Camera intrinsic parameters (simplified assumption)
# You can adjust these based on your camera specifications
# Typical values for smartphone cameras
h, w = left_gray.shape
focal_length = w  # Approximate focal length
cx = w / 2  # Principal point x
cy = h / 2  # Principal point y

# Camera matrix (intrinsic parameters)
K = np.array([[focal_length, 0, cx],
              [0, focal_length, cy],
              [0, 0, 1]])

# Baseline distance between cameras (in meters)
# For handheld stereo, typically 0.05 to 0.15 meters
baseline = 0.1  # 10 cm - adjust based on your actual setup

# Essential matrix estimation
E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

# Recover pose (rotation and translation)
_, R, t, mask = cv2.recoverPose(E, pts1, pts2, K, mask=mask)

# Create projection matrices
# Left camera is at origin
P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])

# Right camera with rotation R and translation t
P2 = K @ np.hstack([R, t])

# Triangulate points
# Convert 2D points to homogeneous coordinates
pts1_filtered = pts1[mask.ravel() == 1]
pts2_filtered = pts2[mask.ravel() == 1]

# Triangulation
points_4d = cv2.triangulatePoints(P1, P2, pts1_filtered.T, pts2_filtered.T)

# Convert from homogeneous to 3D coordinates
points_3d = points_4d[:3, :] / points_4d[3, :]
points_3d = points_3d.T

print("\n" + "="*60)
print("3D COORDINATES OF MATCHED POINTS")
print("="*60)
print(f"{'Point #':<10} {'X (m)':<15} {'Y (m)':<15} {'Z (m)':<15}")
print("-"*60)

for i, point in enumerate(points_3d):
    print(f"{i+1:<10} {point[0]:<15.4f} {point[1]:<15.4f} {point[2]:<15.4f}")
    if i >= 19:  # Print first 20 points
        print(f"... and {len(points_3d) - 20} more points")
        break

print("="*60)

# Visualize 3D points
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot 3D points
ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
           c='blue', marker='o', s=20, alpha=0.6)

ax.set_xlabel('X (meters)')
ax.set_ylabel('Y (meters)')
ax.set_zlabel('Z (meters)')
ax.set_title('3D Reconstruction of Matched Points')

plt.savefig('3d_points.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n3D visualization saved as '3d_points.png'")
print(f"\nTotal 3D points reconstructed: {len(points_3d)}")

# Save 3D coordinates to file
np.savetxt('3d_coordinates.txt', points_3d, 
           header='X(m)\tY(m)\tZ(m)', 
           fmt='%.4f', 
           delimiter='\t')
print("3D coordinates saved to '3d_coordinates.txt'")

print("\n✓ Program completed successfully!")
