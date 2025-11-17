import matplotlib.pyplot as plt
import numpy as np
import inspect

def show_images_auto(*images, cmap=None, figsize=(12, 6), max_cols=4):
    """
    Display multiple images (1~N) automatically in a clean grid layout.

    Args:
        *images: One or more numpy arrays (each can be (H, W) or (H, W, 3)).
                 Example: show_images_auto(img1, img2, img3)
        cmap (str, optional): Colormap for grayscale images.
        figsize (tuple, optional): Figure size in inches, default (12, 6).
        max_cols (int, optional): Maximum number of columns per row.
    """
    # --- Extract argument names from the calling scope to use as titles ---
    frame = inspect.currentframe().f_back
    call_line = inspect.getframeinfo(frame).code_context[0]
    arg_names = []
    try:
        # Extract inside parentheses e.g. show_images_auto(img1, img2)
        inside = call_line.split("show_images_auto(")[1].split(")")[0]
        arg_names = [name.strip() for name in inside.split(",")]
    except Exception:
        arg_names = [f"Image {i}" for i in range(len(images))]

    # --- Convert to list of np.ndarrays ---
    images = [np.array(img) for img in images]
    num_images = len(images)

    if num_images == 0:
        print("No images to display.")
        return

    # --- Compute grid layout ---
    cols = min(num_images, max_cols)
    rows = int(np.ceil(num_images / cols))

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).reshape(-1)  # Flatten in case of multi-row layout

    # --- Plot each image ---
    for i, img in enumerate(images):
        ax = axes[i]
        if img.ndim == 2:  # Grayscale
            ax.imshow(img, cmap=cmap or "gray")
        else:              # RGB
            ax.imshow(img.astype(np.uint8))
        
        title = arg_names[i] if i < len(arg_names) else f"Image {i}"
        ax.set_title(title)
        ax.axis("off")

    # --- Hide unused subplots ---
    for j in range(num_images, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

def make_dummy_images():
    """Create a set of dummy grayscale and RGB images for testing."""
    rng = np.random.default_rng(42)

    # Grayscale: gradient + noise
    gray1 = np.linspace(0, 255, 480*640, dtype=np.uint8).reshape(480, 640)
    gray2 = (gray1.T).copy()  # transpose to vary the gradient direction
    gray3 = (gray1 * 0.5 + rng.normal(0, 10, gray1.shape)).clip(0, 255).astype(np.uint8)

    # RGB: simple color blocks with a circle/rectangle overlay
    rgb1 = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb1[..., 0] = 255  # Red
    rgb1[200:280, 280:360, :] = 255  # White square in the middle

    rgb2 = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb2[..., 1] = 255  # Green
    rr, cc = np.ogrid[:480, :640]
    circle = (rr - 240)**2 + (cc - 320)**2 <= 60**2
    rgb2[circle] = [0, 0, 255]  # Blue circle

    rgb3 = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb3[..., 2] = 255  # Blue
    rgb3[100:380, 150:490, 1] = 200  # Add green channel rectangular tint

    return gray1, gray2, gray3, rgb1, rgb2, rgb3


def test_show_images_auto_single():
    """Test with a single image."""
    gray1, *_ = make_dummy_images()
    show_images_auto(gray1)  # Title should be the variable name 'gray1'


def test_show_images_auto_three():
    """Test with three images (mix of grayscale and RGB)."""
    gray1, gray2, _, rgb1, _, _ = make_dummy_images()
    show_images_auto(gray1, gray2, rgb1)  # Should auto-layout in one row (max_cols=4)


def test_show_images_auto_six():
    """Test with six images to trigger multi-row grid."""
    gray1, gray2, gray3, rgb1, rgb2, rgb3 = make_dummy_images()
    # With default max_cols=4 -> grid will be 2 rows x 3 columns
    show_images_auto(gray1, gray2, gray3, rgb1, rgb2, rgb3)


if __name__ == "__main__":
    # Run all tests interactively
    print("Running test: single image")
    test_show_images_auto_single()

    print("Running test: three images")
    test_show_images_auto_three()

    print("Running test: six images")
    test_show_images_auto_six()

    print("All tests executed.")