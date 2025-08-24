import matplotlib.pyplot as plt
import imageio

image_list = [
    "input.jpg",
    "output_0000.png",
    "output_0010.png",
    "output_0020.png",  #
    "output_0030.png",
    "output_0050.png",
    "output_0070.png",
    "output_0090.png",  #
    "output_0110.png",
    "output_0130.png",
    "output_0150.png",
    "output_0170.png",  #
    "output_0190.png",
    "output_0250.png",
    "output_0300.png",
    "output_0990.png",
]

# plot images in `image_list` in a 4x4 grid
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
for ax, img_path in zip(axes.flatten(), image_list):
    img = plt.imread(img_path)
    ax.imshow(img)
    ax.axis("off")
plt.tight_layout()
plt.savefig("progression.png")

image_list = []
for i in range(700):
    if i % 10 == 0:
        image_list.append(f"output_{i:04d}.png")

# Create GIF that loops for ever
frames = [imageio.imread(img) for img in image_list]
imageio.mimwrite("progression.gif", frames, fps=10, loop=0)
