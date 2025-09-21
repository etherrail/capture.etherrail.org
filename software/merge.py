import numpy as np

def merge_images(images):
	heights = [img.rotated.shape[0] for img in images]
	widths  = [img.rotated.shape[1] for img in images]
	offsets = [int(img.offset_x) for img in images]
	shifts = [int(img.shift) for img in images]

	H = max(heights)
	W = max(o + w for o, w in zip(offsets, widths))
	if W <= 0 or H <= 0:
		return np.zeros((1,1,4), dtype=np.uint8)

	# Accumulators
	color_sum = np.zeros((H, W, 3), dtype=np.float32)  # premultiplied by weight
	alpha_sum = np.zeros((H, W, 1), dtype=np.float32)  # alpha weighted
	weight_sum = np.zeros((H, W, 1), dtype=np.float32) # scalar weights

	for img, off, shift in zip(images, offsets, shifts):
		src = img.rotated
		mask = img.focus_map
		h, w = src.shape[:2]

		# --- Clip horizontally to canvas ---
		x0 = max(off, 0)
		x1 = min(off + w, W)
		if x1 <= x0:
			continue
		sx0 = 0 if off >= 0 else -off
		sx1 = sx0 + (x1 - x0)

		# --- Clip vertically (top-aligned) ---
		y0 = max(shift, 0)
		y1 = min(shift + h, H)
		if y1 <= y0:
			continue
		sy0 = 0 if shift >= 0 else -shift
		sy1 = sy0 + (y1 - y0)

		# Extract ROIs
		src_roi  = src[sy0:sy1, sx0:sx1].astype(np.float32)
		mask_roi = mask[sy0:sy1, sx0:sx1].astype(np.float32)

		bgr = src_roi[..., :3]          # (..,3) float32 0..255
		a   = src_roi[..., 3:4] / 255.0 # (..,1) 0..1
		m   = (mask_roi / 255.0)[..., None]  # (..,1) 0..1

		# Weight = focus * alpha  (so transparent areas contribute less)
		wgt = m * a

		# Accumulate color and alpha with weights
		color_sum[y0:y1, x0:x1] += bgr * wgt          # premultiplied by wgt
		alpha_sum[y0:y1, x0:x1] += a * wgt            # average alpha with same weights
		weight_sum[y0:y1, x0:x1] += wgt

	# Finalize
	# --- Finalize without shape gotchas ---
	eps = 1e-8
	mask = weight_sum > eps                      # (H,W,1) boolean
	den  = np.where(mask, weight_sum, 1.0)       # (H,W,1), avoids /0 and keeps channel dim

	out_bgr = np.zeros_like(color_sum, dtype=np.float32)  # (H,W,3)
	out_a   = np.zeros_like(alpha_sum, dtype=np.float32)  # (H,W,1)

	# Elementwise division with broadcasting; only compute where mask is true
	np.divide(color_sum, den, out=out_bgr, where=mask)
	np.divide(alpha_sum, den, out=out_a,   where=mask)

	out = np.concatenate([out_bgr, out_a * 255.0], axis=-1)
	out = np.clip(out, 0, 255).astype(np.uint8)

	return out
