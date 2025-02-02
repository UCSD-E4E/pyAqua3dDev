# from typing import Tuple

# import cv2
# import numpy as np
# import skimage.exposure
# import torch
# import torch.nn.functional as f
# from scipy.optimize import minimize
# from skimage.util import img_as_float, img_as_ubyte
# from sklearn.cluster import KMeans


# class SeaThru:
#     def __init__(
#         self,
#         device: str | torch.device,
#         backscatter_kmeans_clusters=10,
#         coarse_attenuation_coefficient_p=0.1,
#         local_illuminant_map_f=2.0,
#         h_hsv_scalar=1.0,
#         s_hsv_scalar=1.6,
#         v_hsv_scalar=1.8,
#         gamma_adjustment=0.55,
#     ):
#         self.__device = device
#         self.__backscatter_kmeans_clusters = backscatter_kmeans_clusters
#         self.__coarse_attenuation_coefficient_p = coarse_attenuation_coefficient_p
#         self.__local_illuminant_map_f = local_illuminant_map_f
#         self.__h_hsv_scalar = h_hsv_scalar
#         self.__s_hsv_scalar = s_hsv_scalar
#         self.__v_hsv_scalar = v_hsv_scalar
#         self.__gamma_adjustment = gamma_adjustment

#     def __get_depth_clusters(self, depth: np.ndarray[float]) -> np.ndarray[np.uint8]:
#         height, width = depth.shape
#         depth_flat = depth.flatten()

#         kmeans = KMeans(
#             n_clusters=self.__backscatter_kmeans_clusters, random_state=0, n_init="auto"
#         ).fit(depth_flat[depth_flat != 0].reshape(-1, 1))
#         labels = np.zeros(depth_flat.shape, dtype=np.uint8)
#         labels[depth_flat != 0] = kmeans.labels_
#         labels[depth_flat == 0] = 255  # Use 255 to represent nan
#         means = np.array(kmeans.cluster_centers_).flatten()
#         means_args = np.argsort(means)

#         labels_new = np.zeros_like(labels)
#         labels_new[labels == 255] = self.__backscatter_kmeans_clusters + 1

#         for current_label in range(self.__backscatter_kmeans_clusters):
#             new_label = np.nonzero(means_args == current_label)[0]
#             labels_new[labels == current_label] = new_label

#         labels = labels_new.reshape((height, width))

#         return labels

#     def __get_dark_pixels(
#         self, image: np.ndarray[float], depth: np.ndarray[float]
#     ) -> Tuple[np.ndarray[float], np.ndarray[float]]:
#         labels = self.__get_depth_clusters(depth)

#         dark_pixels = []
#         z_values = []

#         for i in range(self.__backscatter_kmeans_clusters):
#             mask = labels == i

#             b = image[:, :, 0]
#             g = image[:, :, 1]
#             r = image[:, :, 2]

#             b_filtered = b[mask].flatten()
#             g_filtered = g[mask].flatten()
#             r_filtered = r[mask].flatten()

#             pixels = np.array(
#                 [[b, g, r] for b, g, r in zip(b_filtered, g_filtered, r_filtered)]
#             )
#             idx = np.nonzero(np.all(pixels <= np.percentile(pixels, 1, axis=0), axis=1))
#             selected_pixels = pixels[idx]

#             filtered_tif = depth[mask].flatten()
#             z = filtered_tif[idx]

#             dark_pixels.extend(selected_pixels.tolist())
#             z_values.extend(z.tolist())

#         dark_pixels = np.array(dark_pixels)
#         z_values = np.array(z_values)

#         return dark_pixels, z_values

#     def __backscatter_estimation(
#         self, image: np.ndarray[float], depth: np.ndarray[float]
#     ) -> np.ndarray[float]:
#         def estimate_backscatter(
#             B_inf: float, beta_B: float, J_prime: float, beta_D_prime: float, z
#         ):
#             return B_inf * (1 - np.exp(-beta_B * z)) + (
#                 J_prime * np.exp(-beta_D_prime * z)
#             )

#         def optimize_estimate_backscatter(
#             arguments: Tuple[float, float, float, float],
#             B_hat: np.ndarray,
#             z: np.ndarray,
#         ):
#             B_inf, beta_B, J_prime, beta_D_prime = arguments

#             return np.sum(
#                 (B_hat - estimate_backscatter(B_inf, beta_B, J_prime, beta_D_prime, z))
#                 ** 2
#             )

#         height, width, _ = image.shape
#         dark_pixels, z_values = self.__get_dark_pixels(image, depth)
#         depth_flat = depth.flatten()

#         result = minimize(
#             optimize_estimate_backscatter,
#             [1, 1, 1, 1],
#             args=(dark_pixels[:, 0], z_values),
#             bounds=[(0, 1), (0, 5), (0, 1), (0, 5)],
#         )
#         b_args = result.x

#         B_b = estimate_backscatter(
#             b_args[0], b_args[1], b_args[2], b_args[3], depth_flat
#         ).reshape((height, width))

#         result = minimize(
#             optimize_estimate_backscatter,
#             [1, 1, 1, 1],
#             args=(dark_pixels[:, 1], z_values),
#             bounds=[(0, 1), (0, 5), (0, 1), (0, 5)],
#         )
#         g_args = result.x

#         B_g = estimate_backscatter(
#             g_args[0], g_args[1], g_args[2], g_args[3], depth_flat
#         ).reshape((height, width))

#         result = minimize(
#             optimize_estimate_backscatter,
#             [1, 1, 1, 1],
#             args=(dark_pixels[:, 2], z_values),
#             bounds=[(0, 1), (0, 5), (0, 1), (0, 5)],
#         )
#         r_args = result.x

#         B_r = estimate_backscatter(
#             r_args[0], r_args[1], r_args[2], r_args[3], depth_flat
#         ).reshape((height, width))

#         backscatter = np.zeros((height, width, 3), dtype=np.float64)

#         backscatter[:, :, 0] = B_b
#         backscatter[:, :, 1] = B_g
#         backscatter[:, :, 2] = B_r

#         return backscatter

#     def __compute_direct_signal(
#         self, image: np.ndarray[float], backscatter: np.ndarray
#     ) -> np.ndarray[float]:
#         direct_signal = image - backscatter

#         sig_min_b = direct_signal[:, :, 0].min()
#         sig_min_g = direct_signal[:, :, 1].min()
#         sig_min_r = direct_signal[:, :, 2].min()

#         if sig_min_b > 0:
#             sig_min_b = 0

#         if sig_min_g > 0:
#             sig_min_g = 0

#         if sig_min_r > 0:
#             sig_min_r = 0

#         direct_signal[:, :, 0] = direct_signal[:, :, 0] - sig_min_b
#         direct_signal[:, :, 1] = direct_signal[:, :, 1] - sig_min_g
#         direct_signal[:, :, 2] = direct_signal[:, :, 2] - sig_min_r

#         return direct_signal

#     def __pad_tensor(
#         self, tensor: torch.Tensor, top: int, right: int, left: int, bottom: int
#     ) -> torch.Tensor:
#         dim = len(tensor.shape)

#         if len(tensor.shape) == 2:
#             tensor = tensor[:, :, None]

#         height, width, channels = tensor.shape

#         new_height = height + top + bottom
#         new_width = width + right + left

#         bottom_idx = new_height - bottom
#         right_idx = new_width - right

#         padded = torch.zeros((new_height, new_width, channels), device=self.__device)
#         padded[top:bottom_idx, left:right_idx, :] = tensor

#         if dim == 2:
#             return padded[:, :]
#         else:
#             return padded

#     def __pad_direction(self, tensor: torch.Tensor):
#         if len(tensor.shape) == 2:
#             tensor = tensor[:, :, None]

#         height, width, channels = tensor.shape

#         direction = torch.zeros(
#             (height + 2, width + 2, 4, channels), device=self.__device
#         )
#         direction[:, :, 0, :] = self.__pad_tensor(tensor, 2, 1, 1, 0)
#         direction[:, :, 1, :] = self.__pad_tensor(tensor, 1, 2, 0, 1)
#         direction[:, :, 2, :] = self.__pad_tensor(tensor, 0, 1, 1, 2)
#         direction[:, :, 3, :] = self.__pad_tensor(tensor, 1, 0, 2, 1)

#         return direction

#     def __estimate_coarse_attenuation_coefficient(
#         self, direct_signal: np.ndarray[float], depth: np.ndarray[float]
#     ) -> np.ndarray[float]:
#         direct_signal_tensor = self.__pad_tensor(
#             torch.as_tensor(direct_signal, device=self.__device), 1, 1, 1, 1
#         )
#         a_values_tensor = direct_signal_tensor
#         depths_tensor = torch.as_tensor(depth, device=self.__device)
#         diff_depth_map_tensor = torch.abs(
#             torch.dstack([self.__pad_tensor(depths_tensor, 1, 1, 1, 1)] * 4)
#             - self.__pad_direction(depths_tensor).squeeze()
#         )
#         depths_tensor.detach()
#         del depths_tensor
#         softmax = f.softmax(-diff_depth_map_tensor, dim=-1)
#         diff_depth_map_tensor.detach()
#         del diff_depth_map_tensor
#         weights = torch.stack([softmax] * 3, dim=-1)
#         softmax.detach()
#         del softmax

#         for _ in range(100):
#             a_values_tensor = self.__pad_direction(a_values_tensor[1:-1, 1:-1, :])
#             a_values_tensor = torch.sum(weights * a_values_tensor, axis=2)
#             a_values_tensor = (
#                 (1 - self.__coarse_attenuation_coefficient_p) * a_values_tensor
#                 + self.__coarse_attenuation_coefficient_p * direct_signal_tensor
#             )

#         a_values = a_values_tensor[1:-1, 1:-1, :].detach().cpu().numpy()

#         a_values_tensor.detach()
#         del a_values_tensor
#         direct_signal_tensor.detach()
#         del direct_signal_tensor
#         weights.detach()
#         del weights

#         return a_values

#     def __estimate_local_illuminant_map(
#         self, direct_signal: np.ndarray[float], depth: np.ndarray[float]
#     ) -> np.ndarray[float]:
#         coarse_attenuation_coefficient = self.__estimate_coarse_attenuation_coefficient(
#             direct_signal, depth
#         )

#         local_illuminant_map = (
#             self.__local_illuminant_map_f * coarse_attenuation_coefficient
#         )

#         return local_illuminant_map

#     def __refine_beta_D(
#         self, local_illuminant_map: np.ndarray[float], depth: np.ndarray[float]
#     ):
#         def compute_beta_D(
#             a: float, b: float, c: float, d: float, z: np.ndarray
#         ) -> np.ndarray:
#             return a * np.exp(b * z) + c * np.exp(d * z)

#         def optimize_compute_beta_D(
#             args: Tuple[float, float, float, float], E: np.ndarray, z: np.ndarray
#         ) -> float:
#             a, b, c, d = args
#             return np.sum((z - np.log(E / compute_beta_D(a, b, c, d, z))) ** 2)

#         result = minimize(
#             optimize_compute_beta_D,
#             [100, -100, 100, -100],
#             args=(local_illuminant_map[:, :, 0], depth),
#             bounds=[(0, None), (None, 0), (0, None), (None, 0)],
#         )
#         local_illuminant_params_b = result.x

#         result = minimize(
#             optimize_compute_beta_D,
#             [100, -100, 100, -100],
#             args=(local_illuminant_map[:, :, 1], depth),
#             bounds=[(0, None), (None, 0), (0, None), (None, 0)],
#         )
#         local_illuminant_params_g = result.x

#         result = minimize(
#             optimize_compute_beta_D,
#             [100, -100, 100, -100],
#             args=(local_illuminant_map[:, :, 2], depth),
#             bounds=[(0, None), (None, 0), (0, None), (None, 0)],
#         )
#         local_illuminant_params_r = result.x

#         beta_d = np.zeros_like(local_illuminant_map)

#         beta_d[:, :, 0] = compute_beta_D(
#             local_illuminant_params_b[0],
#             local_illuminant_params_b[1],
#             local_illuminant_params_b[2],
#             local_illuminant_params_b[3],
#             depth,
#         )
#         beta_d[:, :, 1] = compute_beta_D(
#             local_illuminant_params_g[0],
#             local_illuminant_params_g[1],
#             local_illuminant_params_g[2],
#             local_illuminant_params_g[3],
#             depth,
#         )
#         beta_d[:, :, 2] = compute_beta_D(
#             local_illuminant_params_r[0],
#             local_illuminant_params_r[1],
#             local_illuminant_params_r[2],
#             local_illuminant_params_r[3],
#             depth,
#         )

#         return beta_d

#     def __compute_J(
#         self,
#         depth: np.ndarray[float],
#         direct_signal: np.ndarray[float],
#         beta_d: np.ndarray[float],
#     ) -> np.ndarray[float]:
#         J = np.zeros_like(direct_signal)

#         J[:, :, 0] = direct_signal[:, :, 0] * np.exp(beta_d[:, :, 0] * depth)
#         J[:, :, 1] = direct_signal[:, :, 1] * np.exp(beta_d[:, :, 1] * depth)
#         J[:, :, 2] = direct_signal[:, :, 2] * np.exp(beta_d[:, :, 2] * depth)

#         return J

#     def __white_balance(self, image: np.ndarray, percentile=50) -> np.ndarray[float]:
#         """
#         Adjust the color balance of an image based on the white patch method.
#         """
#         access_wp2 = (image * 1.0 / np.percentile(image, percentile, axis=(0, 1))).clip(
#             0, 1
#         )
#         return access_wp2

#     def __hsv_correction(self, image: np.ndarray[np.uint8]) -> np.ndarray[np.uint8]:
#         hsv = img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
#         hsv[:, :, 0] *= self.__h_hsv_scalar
#         hsv[:, :, 1] *= self.__s_hsv_scalar
#         hsv[:, :, 2] *= self.__v_hsv_scalar

#         hsv[hsv[:, :, 0] >= 1, 0] = 1
#         hsv[hsv[:, :, 1] >= 1, 1] = 1
#         hsv[hsv[:, :, 2] >= 1, 2] = 1

#         return cv2.cvtColor(img_as_ubyte(hsv), cv2.COLOR_HSV2BGR)

#     def process(
#         self, linear_image: np.ndarray[np.uint8], depth: np.ndarray[float]
#     ) -> np.ndarray[np.uint8]:
#         height, width, _ = linear_image.shape
#         depth_height, depth_width = depth.shape

#         # Our depth and image need to be the same size.
#         if depth_height != height or depth_width != width:
#             depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)

#         image = img_as_float(linear_image)
#         backscatter = self.__backscatter_estimation(image, depth)

#         direct_signal = self.__compute_direct_signal(image, backscatter)
#         local_illuminant_map = self.__estimate_local_illuminant_map(
#             direct_signal, depth
#         )
#         beta_D = self.__refine_beta_D(local_illuminant_map, depth)

#         J = self.__compute_J(depth, direct_signal, beta_D)
#         img = img_as_ubyte(self.__white_balance(J, 100))
#         img = self.__hsv_correction(img)
#         img = skimage.exposure.adjust_gamma(img, gamma=self.__gamma_adjustment)
#         img = skimage.exposure.equalize_adapthist(img)

#         return img_as_ubyte(img)
