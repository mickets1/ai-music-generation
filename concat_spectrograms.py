from PIL import Image
import os

def concatenate_images(input_folder, output_folder, output_filename):
    # List all image files in the input folder
    images = [Image.open(os.path.join(input_folder, file)) for file in os.listdir(input_folder) if file.endswith(('png', 'jpg', 'jpeg'))]

    # Check if there are any images to concatenate
    if not images:
        print("No images found in the input folder.")
        return

    # Ensure all images have the same height
    max_height = max(image.height for image in images)
    images = [image.resize((image.width, max_height)) for image in images]

    # Calculate the width of the combined image
    combined_width = sum(image.width for image in images)

    # Create a new blank image with the combined width and max height
    combined_image = Image.new('RGB', (combined_width, max_height))

    # Paste each image into the combined image
    x_offset = 0
    for image in images:
        combined_image.paste(image, (x_offset, 0))
        x_offset += image.width

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Save the combined image to the output folder
    output_path = os.path.join(output_folder, output_filename)
    combined_image.save(output_path)
    print(f"Combined image saved to {output_path}")

# Example usage
input_folder = 'INPUT_FOLDER'
output_folder = 'OUTPUT_FOLDER'
output_filename = 'combined_image.png'

concatenate_images(input_folder, output_folder, output_filename)
