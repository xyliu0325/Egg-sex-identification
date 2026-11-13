import os
import shutil


def extract_images(folder_path):
    image_files = [f for f in os.listdir(folder_path) if
                   os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Image files found in the directory: {image_files}")

    if not image_files:
        print("No image files were found in the directory.")
        return

    female_prefix = [
        '33']

    male_prefix = [
        '8']

    female_folder_path = os.path.join(folder_path, "Female")
    male_folder_path = os.path.join(folder_path, "Male")
    os.makedirs(female_folder_path, exist_ok=True)
    os.makedirs(male_folder_path, exist_ok=True)

    female_count = 0
    male_count = 0

    for file_name in image_files:
        prefix = ''.join([ch for ch in file_name if ch.isdigit()])

        if not prefix:
            print(f"Skipping filename without a numeric identifier: {file_name}")
            continue

        print(f"Numeric identifier for {file_name}: {prefix}")

        if prefix in female_prefix:
            src_path = os.path.join(folder_path, file_name)
            dst_path = os.path.join(female_folder_path, file_name)
            if not os.path.exists(dst_path):
                shutil.copyfile(src_path, dst_path)
                female_count += 1
                print(f"Copied {file_name} to the Female directory.")
            else:
                print(f"Skipped {file_name}: the file already exists in the Female directory.")
        elif prefix in male_prefix:
            src_path = os.path.join(folder_path, file_name)
            dst_path = os.path.join(male_folder_path, file_name)
            if not os.path.exists(dst_path):
                shutil.copyfile(src_path, dst_path)
                male_count += 1
                print(f"Copied {file_name} to the Male directory.")
            else:
                print(f"Skipped {file_name}: the file already exists in the Male directory.")

    print(f"Image organization completed.\n"
          f"Copied {female_count} female sample images to {female_folder_path}\n"
          f"Copied {male_count} male sample images to {male_folder_path}")


folder_path = r"F:\20261113\batch4\1"
extract_images(folder_path)