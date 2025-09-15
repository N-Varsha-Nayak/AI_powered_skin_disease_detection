import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import shutil

# Constants
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
CLASSES = ['Acne and Rosacea Photos', 'Eczema Photos', 'Melanoma Skin Cancer Nevi and Moles', 'Hair Loss Photos Alopecia and other Hair Diseases', 'Psoriasis pictures Lichen Planus and related diseases']

def create_model():
    """Create and compile the model"""
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(len(CLASSES), activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def organize_dataset(data_dir, train_dir, val_dir):
    """Organize dataset into train and validation directories"""
    # Create directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # For each class
    for class_name in CLASSES:
        # Create class directories in train and validation
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        
        # Get all images for this class
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} not found")
            continue
            
        image_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Split into train and validation
        train_files, val_files = train_test_split(
            image_files, test_size=0.2, random_state=42
        )
        
        # Copy files to respective directories
        for file in train_files:
            src = os.path.join(class_dir, file)
            dst = os.path.join(train_dir, class_name, file)
            shutil.copy2(src, dst)
            
        for file in val_files:
            src = os.path.join(class_dir, file)
            dst = os.path.join(val_dir, class_name, file)
            shutil.copy2(src, dst)

def create_data_generators():
    """Create train and validation data generators"""
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )
    
    return train_datagen, val_datagen

def train_model():
    """Main training function"""
    # Setup directories
    base_dir = 'dataset'
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    
    # Organize dataset
    print("Organizing dataset...")
    organize_dataset(base_dir, train_dir, val_dir)
    
    # Create data generators
    print("Creating data generators...")
    train_datagen, val_datagen = create_data_generators()
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASSES
    )
    
    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASSES
    )
    
    # Create and compile model
    print("Creating model...")
    model = create_model()
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            'models/best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    # Train the model
    print("Starting training...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks
    )
    
    # Fine-tune the model
    print("Fine-tuning the model...")
    # Unfreeze some layers for fine-tuning
    for layer in model.layers[-20:]:
        layer.trainable = True
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train again with unfrozen layers
    history = model.fit(
        train_generator,
        epochs=20,
        validation_data=validation_generator,
        callbacks=callbacks
    )
    
    # Save the final model
    model.save('models/final_model.h5')
    print("Training completed!")

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('dataset', exist_ok=True)
    
    # Start training
    train_model() 