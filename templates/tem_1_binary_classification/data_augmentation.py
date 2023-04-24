from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np


def generate_new_train_images(folder, to_augment_class, to_compare_class, image_size=(256, 256)):
    first_dir = os.path.join(folder, to_augment_class)
    second_dir = os.path.join(folder, to_compare_class)

    first_count = len(os.listdir(first_dir))
    second_count = len(os.listdir(second_dir))

    # Create a data generator for the NORMAL images with random transformations
    # datagen = ImageDataGenerator(
    #     rotation_range=20,
    #     width_shift_range=0.1,
    #     height_shift_range=0.1,
    #     shear_range=0.2,
    #     zoom_range=0.2,
    #     horizontal_flip=True,
    #     fill_mode='nearest'
    # )
    
    datagen = ImageDataGenerator(
        rotation_range=20,
        shear_range=0.2,
        zoom_range=0.1,
        fill_mode='nearest'
    )

    # Calculate the number of additional NORMAL images needed to balance the dataset
    augmentation_factor = int(np.ceil(second_count / first_count))
    additional_normal_count = augmentation_factor * first_count - first_count

    # Generate additional NORMAL images and save them to the NORMAL subfolder
    normal_generator = datagen.flow_from_directory(
        folder,
        batch_size=1,
        target_size=image_size,
        save_to_dir=first_dir,
        save_format='jpeg',
        save_prefix='IM_AUGMENTED',
        classes=[to_augment_class]
    )

    for i in range(additional_normal_count):
        normal_generator.next()

    # Check the number of images in each subfolder again to ensure balance
    new_first_count = len(os.listdir(first_dir))
    new_second_count = len(os.listdir(second_dir))

    print(f'New {to_augment_class} count: {new_first_count}')
    print(f'New {to_compare_class} count: {new_second_count}')


def generate_augmented_images(folder, to_augment_class, image_size=(256, 256)):
    dir = os.path.join(folder, to_augment_class)
    count = len(os.listdir(dir))

    # Create folder + subfolder for the augmented images
    augmented_dir = os.path.join(f"{folder}_augmented", to_augment_class)

    if not os.path.exists(f"{folder}_augmented"):
        os.mkdir(f"{folder}_augmented")

    if not os.path.exists(augmented_dir):
        os.mkdir(f"{folder}_augmented/{to_augment_class}")

    augmented_count = len(os.listdir(augmented_dir))

    # Create a data generator for the NORMAL images with random transformations
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Generate additional NORMAL images and save them to the NORMAL subfolder
    generator = datagen.flow_from_directory(
        folder,
        batch_size=1,
        target_size=image_size,
        save_to_dir=augmented_dir,
        save_format='jpeg',
        save_prefix='IM_AUGMENTED',
        classes=[to_augment_class]
    )

    # Calculate the number of additional images to generate inside train_augmented dir before reaching 5000
    additional_count = 5000 - len(os.listdir(augmented_dir))
    print(f'Additional count of images to generate: {additional_count}')

    for i in range(additional_count):
        generator.next()

    # Check the number of images in each subfolder again to ensure balance
    new_count = len(os.listdir(augmented_dir))

    print(f'Given {to_augment_class} count: {count}')
    print(f'Augmented {to_augment_class} count: {new_count}')