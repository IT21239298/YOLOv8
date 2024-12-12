from simple_image_download import simple_image_download as simp
import magic
import os

# Create response object
response = simp.simple_image_download()

# Initialize magic object for file type checking
magic_obj = magic.Magic(mime=True)

def download_and_verify_images(keyword, num_images):
    # Download images
    response.download(keyword, num_images)
    
    # Path to downloaded images
    download_path = f'simple_images/{keyword}'
    
    # Verify each downloaded image
    if os.path.exists(download_path):
        for filename in os.listdir(download_path):
            file_path = os.path.join(download_path, filename)
            try:
                # Check file type using python-magic
                file_type = magic_obj.from_file(file_path)
                if not file_type.startswith('image/'):
                    print(f"Removing non-image file: {filename}")
                    os.remove(file_path)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Keywords for construction workers
keywords = [
    "construction workers building",
    "building site workers",
    "construction workers safety",
    "construction workers working"
]

# Download and verify images for each keyword
for keyword in keywords:
    try:
        print(f"Downloading images for: {keyword}")
        download_and_verify_images(keyword, 50)
        print(f"Completed downloading and verifying images for: {keyword}")
    except Exception as e:
        print(f"Error with keyword {keyword}: {e}")